"""All six agents run through the real SDK ``Runner`` with the offline ``PolicyModel``.

No network and no API key: every agent's model is a ``PolicyModel`` instance.
"""

import json
from collections.abc import Mapping

import pytest
from agents import MessageOutputItem, RunConfig, RunContextWrapper, Runner, ToolCallItem, ToolCallOutputItem

from emergency_sim.domain.agent_io import (
    AllocationInput,
    AssessmentInput,
    DispatchInput,
    HospitalInput,
)
from emergency_sim.domain.base import AgentIOModel
from emergency_sim.domain.enums import AgentRole, CaseKind, VictimStatus
from emergency_sim.domain.scenario import HospitalSpec, Scenario
from emergency_sim.mas import SdkBackend, make_offline_backend
from emergency_sim.mas import tools as mas_tools
from emergency_sim.mas.backend import build_agent
from emergency_sim.mas.context import AgentContext
from emergency_sim.mas.policy_model import PolicyModel
from emergency_sim.mas.specs import AgentSpec, agent_specs, specs_by_name
from emergency_sim.policies import estimates
from emergency_sim.policies.allocation import DECLINE_OPTION_ID, grant_option_id
from emergency_sim.policies.params import PolicyParams
from tests import factories
from tests.test_policies import (
    make_allocation_input,
    make_ambulance,
    make_assessment_input,
    make_board,
    make_case,
    make_dispatch_input,
    make_hospital_input,
    make_loaded,
    make_need,
    make_option,
    make_report,
    make_request,
    make_victim,
)

PARAMS = PolicyParams()
HOSPITALS = (
    factories.hospital("H1", 6.0, 13.5, ("trauma", "burns", "neuro", "cardiac", "general"), er_capacity=14, reserve=2),
    factories.hospital("H2", 14.5, 12.0, ("cardiac", "neuro", "trauma", "general"), er_capacity=10),
    factories.hospital("H3", 12.0, 4.0, ("pediatric", "general"), er_capacity=8),
)
AGENT_NAMES = ("incident_assessment", "dispatch", "hospital_H1", "hospital_H2", "hospital_H3", "resource_allocation")


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture(scope="module")
def backend() -> SdkBackend:
    return make_offline_backend(HOSPITALS, PARAMS)


def assessment_payload() -> AssessmentInput:
    cardiac = make_report("INC-02", "cardiac_event", "They are not responding; a bystander started CPR.", since=1.0)
    triaged = make_report("INC-01", "fall", victims=[make_victim("INC-01-V1", "YELLOW", "trauma", "hip fracture")])
    return make_assessment_input(cardiac, triaged)


def dispatch_payload(*, loaded: bool = True) -> DispatchInput:
    needs = [make_need("INC-01", 5.0, 5.0, 95, suspected_red=True, als_needed=1, est_patients_remaining=2),
             make_need("INC-02", 12.0, 5.0, 40)]
    units = [make_ambulance("AMB-01", "ALS", 6.0, 6.0), make_ambulance("AMB-02", "BLS", 11.0, 5.0),
             make_ambulance("AMB-03", "BLS", 4.0, 4.0)]
    board = [make_board(h.id, h.x, h.y, [c.value for c in h.capabilities], free=2) for h in HOSPITALS]
    victims = [make_victim("INC-09-V1", "RED", "trauma", "stab wound", VictimStatus.LOADED)]
    units_loaded = [make_loaded("AMB-07", 8.0, 8.0, victims)] if loaded else []
    return make_dispatch_input(needs, units, board, units_loaded, diversion=True, escalation=True)


def hospital_payload(hospital_id: str) -> HospitalInput:
    red = make_request("P-0002", [make_victim("V1", "RED", status=VictimStatus.LOADED)])
    yellow = make_request("P-0001", [make_victim("V2", "YELLOW", status=VictimStatus.LOADED)])
    return make_hospital_input([red, yellow], free=1, reserve=1, caps=("general",)).model_copy(
        update={"hospital_id": hospital_id}
    )


def allocation_payload() -> AllocationInput:
    contention = make_case("CASE-0002", CaseKind.UNIT_CONTENTION, [make_option("OPT-1", 2.1), make_option("OPT-2", 2.6)])
    float_case = make_case("CASE-0001", CaseKind.FLOAT_STAFF,
                           [make_option(grant_option_id("DR-F1"), 2.4), make_option(DECLINE_OPTION_ID, 0.3)])
    return make_allocation_input(contention, float_case)


def payloads() -> Mapping[str, tuple[AgentIOModel, str]]:
    """Agent name -> (payload, expected evidence tool)."""
    return {
        "incident_assessment": (assessment_payload(), mas_tools.INCIDENT_TYPE_PROFILE),
        "dispatch": (dispatch_payload(), mas_tools.RANK_HOSPITALS_FOR_UNIT),
        "hospital_H1": (hospital_payload("H1"), mas_tools.CAPACITY_CHECK),
        "hospital_H2": (hospital_payload("H2"), mas_tools.CAPACITY_CHECK),
        "hospital_H3": (hospital_payload("H3"), mas_tools.CAPACITY_CHECK),
        "resource_allocation": (allocation_payload(), mas_tools.SCORE_CASE_OPTIONS),
    }


def _spec(name: str) -> AgentSpec:
    return specs_by_name(agent_specs(HOSPITALS, PARAMS))[name]


# --- Roster ---------------------------------------------------------------------


def test_roster_has_six_agents_with_distinct_instructions() -> None:
    specs = agent_specs(HOSPITALS, PARAMS)
    assert tuple(s.name for s in specs) == AGENT_NAMES
    assert [s.role for s in specs] == [AgentRole.ASSESSMENT, AgentRole.DISPATCH] + [AgentRole.HOSPITAL] * 3 + [AgentRole.ALLOCATION]
    assert len({s.instructions for s in specs}) == 6
    for spec in specs:
        lines = spec.instructions.splitlines()
        assert 15 <= len(lines) <= 25, (spec.name, len(lines))
        assert "Reply only with the JSON object" in spec.instructions
        assert spec.tools and all(t.name in mas_tools.TOOL_FUNCTIONS for t in spec.tools)
    for spec, hospital in zip(specs[2:5], HOSPITALS, strict=True):
        assert hospital.id in spec.instructions and hospital.name in spec.instructions


def test_roster_follows_real_scenario_hospitals(real_scenarios: Mapping[str, Scenario]) -> None:
    hospitals: tuple[HospitalSpec, ...] = real_scenarios["surge"].hospitals
    backend = make_offline_backend(hospitals, PARAMS)
    assert backend.agent_names == AGENT_NAMES
    assert backend.label == "offline"


# --- Offline backend through Runner -----------------------------------------------


@pytest.mark.parametrize("name", AGENT_NAMES)
def test_offline_decide_equals_policy(backend: SdkBackend, name: str) -> None:
    payload, tool_name = payloads()[name]
    decision = backend.decide(name, payload)
    assert decision.output == _spec(name).policy(payload, PARAMS)
    assert decision.source == "model" and decision.error is None
    assert [t.name for t in decision.tool_calls] == [tool_name]
    assert decision.usage is not None and decision.usage.requests == 2
    assert decision.latency_ms is None


def test_tool_trace_holds_the_estimate(backend: SdkBackend) -> None:
    payload, _ = payloads()["hospital_H2"]
    trace = backend.decide("hospital_H2", payload).tool_calls[0]
    assert json.loads(trace.arguments) == {"proposal_id": "P-0001"}
    check = estimates.capacity_check(payload, "P-0001")  # type: ignore[arg-type]
    assert json.loads(trace.output)["reserve_breached"] is check.reserve_breached is True


def test_dispatch_without_loaded_units_ranks_units(backend: SdkBackend) -> None:
    decision = backend.decide("dispatch", dispatch_payload(loaded=False))
    assert [t.name for t in decision.tool_calls] == [mas_tools.RANK_UNITS_FOR_INCIDENT]
    assert json.loads(decision.tool_calls[0].arguments) == {"ambulance_type": "ANY", "incident_id": "INC-01"}


def test_empty_payload_answers_without_a_tool_call(backend: SdkBackend) -> None:
    decision = backend.decide("resource_allocation", make_allocation_input())
    assert decision.tool_calls == () and decision.usage is not None and decision.usage.requests == 1
    assert decision.output.model_dump() == {"decisions": []}


def test_offline_runs_are_deterministic(backend: SdkBackend) -> None:
    payload, _ = payloads()["dispatch"]
    assert backend.decide("dispatch", payload) == backend.decide("dispatch", payload)


@pytest.mark.parametrize("name", AGENT_NAMES)
def test_runner_items_and_raw_responses(name: str) -> None:
    payload, _ = payloads()[name]
    spec = _spec(name)
    agent = build_agent(spec, PolicyModel(spec, PARAMS))
    result = Runner.run_sync(
        agent, payload.model_dump_json(), context=AgentContext(payload, PARAMS), max_turns=4,
        run_config=RunConfig(tracing_disabled=True, workflow_name="ems-test"),
    )
    assert [type(item) for item in result.new_items] == [ToolCallItem, ToolCallOutputItem, MessageOutputItem]
    assert len(result.raw_responses) == 2
    assert result.final_output == spec.policy(payload, PARAMS)


def test_decide_rejects_unknown_agent_and_wrong_payload(backend: SdkBackend) -> None:
    with pytest.raises(KeyError):
        backend.decide("hospital_H9", hospital_payload("H9"))
    with pytest.raises(TypeError):
        backend.decide("dispatch", hospital_payload("H1"))


def test_policy_model_never_streams() -> None:
    spec = _spec("dispatch")
    with pytest.raises(NotImplementedError):
        PolicyModel(spec, PARAMS).stream_response(
            None, "{}", None, [], None, [], None,  # type: ignore[arg-type]
            previous_response_id=None, conversation_id=None, prompt=None,
        )


# --- Tools are read-only views of the payload ---------------------------------------


def _wrapper(payload: AgentIOModel) -> RunContextWrapper[AgentContext]:
    return RunContextWrapper(context=AgentContext(payload, PARAMS))


def test_tools_report_errors_instead_of_raising() -> None:
    dispatch_ctx = _wrapper(dispatch_payload())
    assert "error" in json.loads(mas_tools.rank_units_for_incident(dispatch_ctx, "INC-99", "ALS"))
    assert "error" in json.loads(mas_tools.rank_hospitals_for_unit(dispatch_ctx, "AMB-01"))
    assert "error" in json.loads(mas_tools.capacity_check(dispatch_ctx, "P-0001"))  # wrong payload type
    assert "error" in json.loads(mas_tools.score_case_options(_wrapper(allocation_payload()), "CASE-9"))


def test_tools_compute_the_shared_estimates() -> None:
    dispatch_ctx = _wrapper(dispatch_payload())
    ranked = json.loads(mas_tools.rank_units_for_incident(dispatch_ctx, "INC-01", "ALS"))
    assert [u["unit_id"] for u in ranked["units"]] == ["AMB-01"]
    options = json.loads(mas_tools.score_case_options(_wrapper(allocation_payload()), "CASE-0002"))["options"]
    assert [o["option_id"] for o in options] == ["OPT-2", "OPT-1"]
    deadline = json.loads(mas_tools.nominal_deadline(dispatch_ctx, "RED", "cardiac arrest"))
    assert deadline["deadline_min"] == 10.0
    profile = json.loads(mas_tools.incident_type_profile(dispatch_ctx, "stroke"))
    assert profile["needed_capabilities"] == ["neuro"]
