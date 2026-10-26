"""LLM-style failures fall back to the policy; the LLM factory needs an API key.

Models are ``agents.testing.ScriptedModel`` instances, so nothing reaches the
network.
"""

import httpx2  # openai 3.x ships httpx2 (not httpx)
import openai
import pytest
from agents import ModelBehaviorError, RunConfig
from agents.testing import ScriptedModel, assistant_message, function_call

from emergency_sim.domain.agent_io import HospitalInput
from emergency_sim.mas import BackendUnavailableError, SdkBackend, make_llm_backend
from emergency_sim.mas.backend import build_agents
from emergency_sim.mas.policy_model import PolicyModel
from emergency_sim.mas.specs import AgentSpec, agent_specs
from emergency_sim.policies.hospital import hospital
from emergency_sim.policies.params import PolicyParams
from tests.test_mas_offline_runner import HOSPITALS, hospital_payload

PARAMS = PolicyParams()
AGENT = "hospital_H1"


def scripted_backend(model: ScriptedModel, *, fallback: bool = True, max_turns: int = 6) -> SdkBackend:
    """A backend whose hospital_H1 agent uses ``model``; the others use the policy model."""
    specs = agent_specs(HOSPITALS, PARAMS)

    def model_for(spec: AgentSpec) -> ScriptedModel | PolicyModel:
        return model if spec.name == AGENT else PolicyModel(spec, PARAMS)

    return SdkBackend(
        "test-scripted", specs, build_agents(specs, model_for), PARAMS,
        run_config=RunConfig(tracing_disabled=True, workflow_name="ems-test"),
        max_turns=max_turns, fallback_on_error=fallback,
    )


@pytest.fixture
def payload() -> HospitalInput:
    return hospital_payload("H1")


def test_malformed_json_falls_back_to_policy(payload: HospitalInput) -> None:
    decision = scripted_backend(ScriptedModel([[assistant_message("not json")]])).decide(AGENT, payload)
    assert decision.source == "policy_fallback"
    assert decision.error is not None and "ModelBehaviorError" in decision.error
    assert decision.output == hospital(payload, PARAMS)
    assert decision.tool_calls == () and decision.usage is None


def test_connection_error_falls_back_to_policy(payload: HospitalInput) -> None:
    error = openai.APIConnectionError(request=httpx2.Request("POST", "https://example.invalid"))
    decision = scripted_backend(ScriptedModel([error])).decide(AGENT, payload)
    assert decision.source == "policy_fallback"
    assert decision.error is not None and decision.error.startswith("APIConnectionError")
    assert decision.output == hospital(payload, PARAMS)


def test_schema_violation_falls_back(payload: HospitalInput) -> None:
    bad = '{"responses": [{"proposal_id": "P-0001", "response": "MAYBE"}]}'
    decision = scripted_backend(ScriptedModel([[assistant_message(bad)]])).decide(AGENT, payload)
    assert decision.source == "policy_fallback" and "ModelBehaviorError" in (decision.error or "")


def test_max_turns_falls_back(payload: HospitalInput) -> None:
    looping = [[function_call("capacity_check", {"proposal_id": "P-0001"}, call_id=f"c{i}")] for i in range(3)]
    decision = scripted_backend(ScriptedModel(looping), max_turns=2).decide(AGENT, payload)
    assert decision.source == "policy_fallback" and "MaxTurnsExceeded" in (decision.error or "")


def test_valid_json_is_used_as_model_output(payload: HospitalInput) -> None:
    answer = hospital(payload, PARAMS).model_dump_json()
    model = ScriptedModel([
        [function_call("capacity_check", {"proposal_id": "P-0001"}, call_id="call-1")],
        [assistant_message(answer)],
    ])
    decision = scripted_backend(model).decide(AGENT, payload)
    assert decision.source == "model" and decision.error is None
    assert decision.output == hospital(payload, PARAMS)
    assert [t.name for t in decision.tool_calls] == ["capacity_check"]
    assert '"reserve_breached": true' in decision.tool_calls[0].output
    model.assert_complete()


def test_errors_propagate_without_fallback(payload: HospitalInput) -> None:
    with pytest.raises(ModelBehaviorError):
        scripted_backend(ScriptedModel([[assistant_message("not json")]]), fallback=False).decide(AGENT, payload)


def test_llm_backend_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(BackendUnavailableError):
        make_llm_backend(HOSPITALS, PARAMS)


def test_llm_backend_builds_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.delenv("EMS_LLM_MODEL", raising=False)
    assert make_llm_backend(HOSPITALS, PARAMS).label == "llm-gpt-4.1-mini"
    monkeypatch.setenv("EMS_LLM_MODEL", "gpt-test-model")
    backend = make_llm_backend(HOSPITALS, PARAMS, temperature=0.0)
    assert backend.label == "llm-gpt-test-model"
    assert len(backend.agent_names) == 6
    assert make_llm_backend(HOSPITALS, PARAMS, model="explicit").label == "llm-explicit"
