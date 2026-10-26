"""Decision backends that run the SDK agents: offline (policy model) and LLM.

``SdkBackend`` implements ``domain.ports.DecisionBackend``. Both factories build
real ``agents.Agent`` objects from the same ``AgentSpec`` roster and run them
through ``Runner.run_sync`` with the same JSON payload, so both backends share
one information boundary. The offline backend fails loudly on any error; the
LLM backend falls back to the rule-based policy and records why.
"""

import logging
import os
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Final

import openai
import pydantic
from agents import (
    Agent,
    AgentsException,
    Model,
    ModelSettings,
    RunConfig,
    RunContextWrapper,
    RunResult,
    Runner,
    ToolCallItem,
    ToolCallOutputItem,
)

from emergency_sim.domain.agent_io import AgentDecision, TokenUsage, ToolCallTrace
from emergency_sim.domain.base import AgentIOModel
from emergency_sim.domain.scenario import HospitalSpec
from emergency_sim.mas.context import AgentContext
from emergency_sim.mas.policy_model import PolicyModel
from emergency_sim.mas.specs import AgentSpec, agent_specs, specs_by_name
from emergency_sim.policies.params import PolicyParams

LOGGER = logging.getLogger(__name__)

LLM_MODEL_ENV: Final = "EMS_LLM_MODEL"
FALLBACK_LLM_MODEL: Final = "gpt-4.1-mini"
DEFAULT_LLM_MODEL: Final = os.environ.get(LLM_MODEL_ENV, FALLBACK_LLM_MODEL)
"""LLM model name resolved at import time (``make_llm_backend`` re-reads the env)."""

API_KEY_ENV: Final = "OPENAI_API_KEY"
OFFLINE_LABEL: Final = "offline"
OFFLINE_MAX_TURNS: Final = 4
LLM_MAX_TURNS: Final = 6
LLM_REQUEST_TIMEOUT_S: Final = 60.0
"""Per-request timeout for the LLM backend; a timeout falls back to the policy."""
MAX_ERROR_CHARS: Final = 300


class BackendUnavailableError(RuntimeError):
    """A backend cannot be built (for example the LLM backend without an API key)."""


class OutputTypeError(Exception):
    """The run finished but its final output is not the agent's output type."""


FALLBACK_ERRORS: Final = (
    AgentsException, openai.OpenAIError, pydantic.ValidationError, TimeoutError, OutputTypeError,
)
"""Errors that trigger the policy fallback when ``fallback_on_error`` is set."""


def default_llm_model() -> str:
    """Returns the LLM model name from ``EMS_LLM_MODEL`` (default ``gpt-4.1-mini``)."""
    return os.environ.get(LLM_MODEL_ENV, FALLBACK_LLM_MODEL)


def build_agent(
    spec: AgentSpec, model: Model | str, model_settings: ModelSettings | None = None
) -> Agent[AgentContext]:
    """Builds one SDK agent from its spec.

    Args:
        spec: The agent spec.
        model: A ``Model`` instance (offline/test) or a model name (LLM).
        model_settings: Optional model settings.

    Returns:
        The agent, with strict structured output ``spec.output_type``.
    """
    return Agent[AgentContext](
        name=spec.name,
        instructions=spec.instructions,
        tools=list(spec.tools),
        output_type=spec.output_type,
        model=model,
        model_settings=model_settings or ModelSettings(),
    )


def build_agents(
    specs: Sequence[AgentSpec],
    model_for: Callable[[AgentSpec], Model | str],
    model_settings: ModelSettings | None = None,
) -> dict[str, Agent[AgentContext]]:
    """Builds one SDK agent per spec.

    Args:
        specs: Agent specs.
        model_for: Chooses the model of each agent.
        model_settings: Optional model settings shared by all agents.

    Returns:
        Agents keyed by name.
    """
    return {s.name: build_agent(s, model_for(s), model_settings) for s in specs}


def tool_traces(result: RunResult) -> tuple[ToolCallTrace, ...]:
    """Extracts the function-tool calls of a run, in call order.

    Args:
        result: A finished run.

    Returns:
        One trace per tool call, paired with its output by ``call_id``.
    """
    outputs = {
        str(_get(item.raw_item, "call_id")): str(item.output)
        for item in result.new_items if isinstance(item, ToolCallOutputItem)
    }
    traces = []
    for item in result.new_items:
        if isinstance(item, ToolCallItem):
            call_id = str(_get(item.raw_item, "call_id"))
            name, arguments = str(_get(item.raw_item, "name")), str(_get(item.raw_item, "arguments"))
            traces.append(ToolCallTrace(name=name, arguments=arguments, output=outputs.get(call_id, "")))
    return tuple(traces)


def _get(raw: object, key: str) -> object:
    return raw.get(key) if isinstance(raw, dict) else getattr(raw, key, None)


def token_usage(wrapper: RunContextWrapper[AgentContext]) -> TokenUsage:
    """Converts the run's accumulated SDK usage into the domain record.

    Args:
        wrapper: The run's context wrapper.

    Returns:
        Requests and token counts.
    """
    usage = wrapper.usage
    return TokenUsage(
        requests=usage.requests, input_tokens=usage.input_tokens, output_tokens=usage.output_tokens
    )


class SdkBackend:
    """``DecisionBackend`` that runs SDK agents through ``Runner.run_sync``."""

    def __init__(
        self,
        label: str,
        specs: Sequence[AgentSpec],
        agents: Mapping[str, Agent[AgentContext]],
        params: PolicyParams,
        *,
        run_config: RunConfig,
        max_turns: int,
        fallback_on_error: bool,
        measure_latency: bool = False,
    ) -> None:
        """Initialises the backend.

        Args:
            label: Backend label written to logs (``"offline"``, ``"llm-<model>"``).
            specs: Agent specs (one per agent name).
            agents: SDK agents keyed by name; must cover every spec.
            params: Policy parameters (tools and fallback policy).
            run_config: Run configuration (tracing, workflow name).
            max_turns: Maximum model turns per call.
            fallback_on_error: Replace failed model calls with the policy.
            measure_latency: Record wall-clock latency (LLM only; offline logs
                must stay byte-identical).

        Raises:
            ValueError: If an agent is missing for a spec.
        """
        self._label = label
        self._specs = specs_by_name(specs)
        missing = sorted(set(self._specs) - set(agents))
        if missing:
            raise ValueError(f"no SDK agent for {missing}")
        self._agents = dict(agents)
        self._params = params
        self._run_config = run_config
        self._max_turns = max_turns
        self._fallback_on_error = fallback_on_error
        self._measure_latency = measure_latency

    @property
    def label(self) -> str:
        """Backend label: ``"offline"``, ``"llm-<model>"`` or a test label."""
        return self._label

    @property
    def agent_names(self) -> tuple[str, ...]:
        """Names of the agents this backend can run, in roster order."""
        return tuple(self._specs)

    def spec(self, agent_name: str) -> AgentSpec:
        """Returns the spec of an agent.

        Args:
            agent_name: Agent name.

        Returns:
            Its spec.

        Raises:
            KeyError: If the agent is unknown.
        """
        if agent_name not in self._specs:
            raise KeyError(f"unknown agent {agent_name!r}; known: {sorted(self._specs)}")
        return self._specs[agent_name]

    def decide(self, agent_name: str, payload: AgentIOModel) -> AgentDecision:
        """Runs the named agent on a payload.

        Args:
            agent_name: Agent name, e.g. ``"dispatch"`` or ``"hospital_H1"``.
            payload: The agent's ``*Input`` model.

        Returns:
            The structured output and its provenance.

        Raises:
            KeyError: If the agent is unknown.
            TypeError: If the payload is not the agent's input type.
            Exception: Any run error when ``fallback_on_error`` is False.
        """
        spec = self.spec(agent_name)
        if not isinstance(payload, spec.input_type):
            raise TypeError(f"{agent_name} expects {spec.input_type.__name__}, got {type(payload).__name__}")
        start = time.perf_counter() if self._measure_latency else None
        try:
            return self._run(spec, payload, start)
        except FALLBACK_ERRORS as error:
            if not self._fallback_on_error:
                raise
            return self._fallback(spec, payload, error, start)

    def _run(self, spec: AgentSpec, payload: AgentIOModel, start: float | None) -> AgentDecision:
        result = Runner.run_sync(
            self._agents[spec.name],
            payload.model_dump_json(),
            context=AgentContext(payload, self._params),
            max_turns=self._max_turns,
            run_config=self._run_config,
        )
        output = result.final_output
        if not isinstance(output, spec.output_type):
            got, expected = type(output).__name__, spec.output_type.__name__
            raise OutputTypeError(f"{spec.name} returned {got}, not {expected}")
        return AgentDecision(
            output=output,
            source="model",
            error=None,
            tool_calls=tool_traces(result),
            usage=token_usage(result.context_wrapper),
            latency_ms=_elapsed_ms(start),
        )

    def _fallback(
        self, spec: AgentSpec, payload: AgentIOModel, error: Exception, start: float | None
    ) -> AgentDecision:
        summary = f"{type(error).__name__}: {error}"[:MAX_ERROR_CHARS]
        LOGGER.warning(
            "agent %s failed on backend %s, using policy fallback: %s", spec.name, self._label, summary
        )
        return AgentDecision(
            output=spec.policy(payload, self._params),
            source="policy_fallback",
            error=summary,
            tool_calls=(),
            usage=None,
            latency_ms=_elapsed_ms(start),
        )


def _elapsed_ms(start: float | None) -> float | None:
    return None if start is None else round((time.perf_counter() - start) * 1000.0, 1)


def make_offline_backend(hospitals: Sequence[HospitalSpec], params: PolicyParams) -> SdkBackend:
    """Builds the deterministic offline backend (no network, tracing disabled).

    Every agent's model is a ``PolicyModel`` instance (never a model name), so
    no provider can be reached. Tracing is disabled for this backend's runs
    only (through its ``RunConfig``), not for the whole process. Errors are not
    caught: offline bugs fail loudly.

    Args:
        hospitals: Scenario hospitals (one hospital agent each).
        params: Policy parameters.

    Returns:
        The backend, labelled ``"offline"``.
    """
    specs = agent_specs(hospitals, params)
    agents = build_agents(specs, lambda spec: PolicyModel(spec, params))
    return SdkBackend(
        OFFLINE_LABEL,
        specs,
        agents,
        params,
        run_config=RunConfig(tracing_disabled=True, workflow_name="ems-offline"),
        max_turns=OFFLINE_MAX_TURNS,
        fallback_on_error=False,
    )


def make_llm_backend(
    hospitals: Sequence[HospitalSpec],
    params: PolicyParams,
    *,
    model: str | None = None,
    trace: bool = False,
    temperature: float | None = None,
) -> SdkBackend:
    """Builds the LLM backend (OpenAI via the Agents SDK) with policy fallback.

    Building it makes no network call; the first ``decide`` does.

    Args:
        hospitals: Scenario hospitals (one hospital agent each).
        params: Policy parameters (tools and fallback policy).
        model: Model name; defaults to ``EMS_LLM_MODEL`` or ``gpt-4.1-mini``.
        trace: Export SDK traces to OpenAI.
        temperature: Sampling temperature; None keeps the provider default.

    Returns:
        The backend, labelled ``"llm-<model>"``.

    Raises:
        BackendUnavailableError: If ``OPENAI_API_KEY`` is not set.
    """
    if not os.environ.get(API_KEY_ENV):
        raise BackendUnavailableError(f"{API_KEY_ENV} is not set; the LLM backend needs an OpenAI API key")
    model_name = model or default_llm_model()
    specs = agent_specs(hospitals, params)
    settings = ModelSettings(temperature=temperature, timeout=LLM_REQUEST_TIMEOUT_S)
    agents = build_agents(specs, lambda _spec: model_name, settings)
    return SdkBackend(
        f"llm-{model_name}",
        specs,
        agents,
        params,
        run_config=RunConfig(tracing_disabled=not trace, workflow_name="ems-llm"),
        max_turns=LLM_MAX_TURNS,
        fallback_on_error=True,
        measure_latency=True,
    )
