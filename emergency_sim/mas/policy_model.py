"""Deterministic offline ``Model`` that answers with an agent's rule-based policy.

``PolicyModel`` plugs into the real SDK ``Runner`` like any LLM: on its first
turn it asks for one "evidence" tool call (so offline runs exercise the SDK
tool loop and record genuine tool traces), and on its second turn it returns
the policy's output as the final JSON message, which the SDK validates against
the agent's ``output_type``. It never touches the network.
"""

import json
from collections.abc import AsyncIterator
from typing import Any

from agents import (
    AgentOutputSchemaBase,
    Handoff,
    Model,
    ModelResponse,
    ModelSettings,
    ModelTracing,
    Tool,
    TResponseInputItem,
    Usage,
)
from agents.items import TResponseStreamEvent
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)
from openai.types.responses.response_prompt_param import ResponsePromptParam

from emergency_sim.domain.base import AgentIOModel
from emergency_sim.mas.specs import AgentSpec
from emergency_sim.policies.params import PolicyParams


def _field(item: Any, key: str) -> Any:
    """Reads a field of an input item that may be a dict or a model object."""
    return item.get(key) if isinstance(item, dict) else getattr(item, key, None)


def first_user_text(model_input: str | list[TResponseInputItem]) -> str:
    """Extracts the payload JSON: the input string, or the first user item's text.

    Args:
        model_input: The ``input`` passed to ``Model.get_response``.

    Returns:
        The first user message text.

    Raises:
        ValueError: If there is no user item with string content.
    """
    if isinstance(model_input, str):
        return model_input
    for item in model_input:
        content = _field(item, "content")
        if _field(item, "role") == "user" and isinstance(content, str):
            return content
    raise ValueError("no user message with string content in the model input")


def has_tool_output(model_input: str | list[TResponseInputItem]) -> bool:
    """Whether a function tool output is already in the conversation.

    Args:
        model_input: The ``input`` passed to ``Model.get_response``.

    Returns:
        True once the evidence call has been answered.
    """
    if isinstance(model_input, str):
        return False
    return any(_field(item, "type") == "function_call_output" for item in model_input)


def _tool_call_response(tool_name: str, arguments: dict[str, str]) -> ModelResponse:
    call_id = f"{tool_name}-evidence"
    call = ResponseFunctionToolCall(
        id=call_id, call_id=call_id, type="function_call", name=tool_name,
        arguments=json.dumps(arguments, sort_keys=True),
    )
    return ModelResponse(output=[call], usage=Usage(requests=1), response_id=None)


def _final_response(agent_name: str, text: str) -> ModelResponse:
    message = ResponseOutputMessage(
        id=f"{agent_name}-final", type="message", role="assistant", status="completed",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[], logprobs=[])],
    )
    return ModelResponse(output=[message], usage=Usage(requests=1), response_id=None)


class PolicyModel(Model):
    """Deterministic offline Model: answers with the agent's rule-based policy.

    Attributes:
        spec: The agent this model answers for.
        params: Policy parameters passed to the policy.
    """

    def __init__(self, spec: AgentSpec, params: PolicyParams) -> None:
        """Initialises the model.

        Args:
            spec: The agent this model answers for.
            params: Policy parameters passed to the policy.
        """
        self.spec = spec
        self.params = params

    def respond(self, model_input: str | list[TResponseInputItem]) -> ModelResponse:
        """Computes the next turn: one evidence tool call, then the final JSON message.

        Args:
            model_input: The conversation so far.

        Returns:
            The model response.

        Raises:
            pydantic.ValidationError: If the payload does not match the spec's
                input type.
            ValueError: If the input holds no user message.
        """
        payload: AgentIOModel = self.spec.input_type.model_validate_json(first_user_text(model_input))
        evidence = self.spec.evidence(payload)
        if evidence is not None and not has_tool_output(model_input):
            return _tool_call_response(*evidence)
        text = self.spec.policy(payload, self.params).model_dump_json()
        return _final_response(self.spec.name, text)

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: ResponsePromptParam | None,
    ) -> ModelResponse:
        """Returns the next deterministic turn (see ``respond``); other arguments are ignored.

        Args:
            system_instructions: Ignored (the policy is the instructions).
            input: The conversation so far.
            model_settings: Ignored.
            tools: Ignored (the evidence tool is chosen by the spec).
            output_schema: Ignored (the SDK validates the final text).
            handoffs: Ignored (agents have no handoffs).
            tracing: Ignored.
            previous_response_id: Ignored.
            conversation_id: Ignored.
            prompt: Ignored.

        Returns:
            The model response.
        """
        return self.respond(input)

    def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: ResponsePromptParam | None,
    ) -> AsyncIterator[TResponseStreamEvent]:
        """Not supported: the offline backend never streams.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("PolicyModel does not stream; use Runner.run or Runner.run_sync")
