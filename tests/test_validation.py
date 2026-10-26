"""Tests of the semantic validation with policy fallback (``control/validation.py``)."""

from emergency_sim.control.conditions import NEGOTIATION
from emergency_sim.control.needs import incident_needs
from emergency_sim.control.observations import (
    assessment_input,
    dispatch_input,
    hospital_input,
    triaged_victim,
)
from emergency_sim.control.validation import (
    validate_allocation,
    validate_assessment,
    validate_dispatch,
    validate_hospital,
)
from emergency_sim.domain.agent_io import (
    AdmissionRequest,
    AdmissionResponse,
    AllocationInput,
    AllocationOutput,
    CaseDecision,
    CaseOption,
    ContentionFlag,
    DispatchInput,
    DispatchOutput,
    DiversionRequest,
    EscalationCase,
    HospitalInput,
    HospitalOutput,
    TransportPlan,
    UnitOrder,
)
from emergency_sim.domain.enums import CaseKind, Condition, CounterCondition, IncidentStatus, UnitStatus
from emergency_sim.policies.allocation import allocate
from emergency_sim.policies.assessment import assess
from emergency_sim.policies.dispatch import dispatch
from emergency_sim.policies.hospital import hospital
from emergency_sim.policies.params import PolicyParams
from tests.control_fakes import (
    hospital_view,
    incident_view,
    public_view,
    unit_view,
    victim_view,
)
from tests.control_fakes import staff_view as staff

PARAMS = PolicyParams()


def _dispatch_input(condition: Condition = Condition.FULL) -> DispatchInput:
    victims = [victim_view("V9", "INC-09", cap="trauma", unit_id="A9")]
    view = public_view(
        incidents=[incident_view("INC-01", reported=2), incident_view("INC-09", victims=victims,
                                                                       status=IncidentStatus.CLEARED)],
        units=[unit_view("A1", x=6.0), unit_view("B1", unit_type="BLS", x=7.0),
               unit_view("A2", status=UnitStatus.EN_ROUTE_SCENE, incident_id="INC-01"),
               unit_view("A9", status=UnitStatus.AWAITING_DESTINATION, load=("V9",))],
        hospitals=[hospital_view("H1"), hospital_view("H2", caps=("trauma",))],
    )
    return dispatch_input(view, incident_needs(view, None, PARAMS), NEGOTIATION[condition])


def _plan(ranking: list[str]) -> TransportPlan:
    return TransportPlan(unit_id="A9", hospital_ranking=ranking, rationale="r")


def _output(**sections: object) -> DispatchOutput:
    values: dict[str, object] = {"orders": [], "diversions": [], "contentions": [],
                                 "transports": [_plan(["H2"])]}
    values.update(sections)
    return DispatchOutput.model_validate(values)


def test_valid_dispatch_output_is_kept() -> None:
    inp = _dispatch_input()
    out = _output(orders=[UnitOrder(unit_id="A1", incident_id="INC-01", rationale="go")])
    result = validate_dispatch(out, inp, dispatch(inp, PARAMS), PARAMS)
    assert result.output == out and result.notes == ()


def test_busy_unit_order_replaces_orders_section_only() -> None:
    inp = _dispatch_input()
    policy = dispatch(inp, PARAMS)
    out = _output(orders=[UnitOrder(unit_id="A2", incident_id="INC-01", rationale="busy")])
    result = validate_dispatch(out, inp, policy, PARAMS)
    assert result.output.orders == policy.orders
    assert result.output.transports == out.transports
    (note,) = result.notes
    assert (note.scope, note.key, note.rule) == ("section", "orders", "unit_not_available")


def test_disallowed_and_inconsistent_sections_are_replaced() -> None:
    inp = _dispatch_input(Condition.B1)
    policy = dispatch(inp, PARAMS)
    out = _output(
        diversions=[DiversionRequest(unit_id="A2", to_incident_id="INC-01", rationale="x")],
        contentions=[ContentionFlag(unit_id="A1", incident_ids=["INC-01", "INC-01"], rationale="x")],
        transports=[_plan(["H1"])],
    )
    result = validate_dispatch(out, inp, policy, PARAMS)
    rules = {n.key: n.rule for n in result.notes}
    assert rules == {"diversions": "diversion_not_allowed", "contentions": "contention_not_allowed",
                     "transports": "hospital_not_capable"}
    assert result.output.transports == policy.transports


def test_missing_loaded_unit_and_pre_triage_cap() -> None:
    inp = _dispatch_input()
    out = _output(transports=[], orders=[
        UnitOrder(unit_id="A1", incident_id="INC-01", rationale="x"),
        UnitOrder(unit_id="B1", incident_id="INC-01", rationale="x"),
    ])
    capped = inp.model_copy(update={"incidents": [inp.incidents[0].model_copy(update={"ambulances_assigned": 2})]})
    rules = {n.key: n.rule for n in validate_dispatch(out, capped, dispatch(inp, PARAMS), PARAMS).notes}
    assert rules == {"orders": "pre_triage_cap", "transports": "loaded_units_mismatch"}


def test_cross_section_conflict_uses_whole_policy_output() -> None:
    base = _dispatch_input()
    second = base.incidents[0].model_copy(update={"incident_id": "INC-02"})
    inp = base.model_copy(update={"incidents": [*base.incidents, second]})
    policy = dispatch(inp, PARAMS)
    out = _output(
        orders=[UnitOrder(unit_id="A1", incident_id="INC-01", rationale="x")],
        contentions=[ContentionFlag(unit_id="A1", incident_ids=["INC-01", "INC-02"], rationale="x")],
    )
    result = validate_dispatch(out, inp, policy, PARAMS)
    assert result.output == policy
    assert [(n.scope, n.rule) for n in result.notes] == [("call", "cross_section_conflict")]


def test_assessment_items_replaced_individually() -> None:
    view = public_view(incidents=[incident_view("INC-01"), incident_view("INC-02"), incident_view("INC-03")])
    inp = assessment_input(view, ["INC-01", "INC-02", "INC-03"])
    policy = assess(inp, PARAMS)
    good, bad = policy.reports[0], policy.reports[1].model_copy(update={"urgency": 140})
    unknown = policy.reports[2].model_copy(update={"incident_id": "INC-99"})
    out = policy.model_copy(update={"reports": [good, bad, unknown]})
    result = validate_assessment(out, inp, policy)
    assert result.output.reports == policy.reports
    assert [(n.key, n.rule) for n in result.notes] == [
        ("INC-02", "urgency_out_of_range"), ("INC-99", "unknown_item"), ("INC-03", "missing_item")]
    dup = policy.model_copy(update={"reports": [good, good, *policy.reports[1:]]})
    assert [n.rule for n in validate_assessment(dup, inp, policy).notes] == ["duplicate_item"]


def _hospital_input(condition: Condition) -> HospitalInput:
    view = public_view(hospitals=[hospital_view("H1", caps=("general",), staff=[
        staff("DR-1", "doctor", "H1"), staff("RN-1", "nurse", "H1")])])
    requests = [
        AdmissionRequest(proposal_id=f"M-000{i}", incident_id="INC-01", unit_id=f"A{i}",
                         victims=[triaged_victim(victim_view(f"V{i}", cap=cap))], eta_min=5.0)
        for i, cap in ((1, "general"), (2, "trauma"), (3, "general"))
    ]
    return hospital_input(view, "H1", requests, NEGOTIATION[condition])


def _response(pid: str, response: str, condition: CounterCondition | None = None,
              skill: str | None = None) -> AdmissionResponse:
    return AdmissionResponse(proposal_id=pid, response=response, reason="OK",
                             counter_condition=condition, counter_skill=skill, rationale="r")


def test_hospital_items_checked_against_allowed_responses_and_skills() -> None:
    inp = _hospital_input(Condition.B1)
    policy = hospital(inp, PARAMS)
    out = HospitalOutput(responses=[
        _response("M-0001", "COUNTER", CounterCondition.RESERVE_RELEASE),
        _response("M-0002", "ACCEPT"),
        _response("M-0003", "ACCEPT"),
        _response("M-0042", "ACCEPT"),
    ])
    result = validate_hospital(out, inp, policy)
    assert [(n.key, n.rule) for n in result.notes] == [
        ("M-0001", "response_not_allowed"), ("M-0002", "accept_not_capable"), ("M-0042", "unknown_item")]
    assert result.output.responses[:2] == policy.responses[:2]
    assert result.output.responses[2] == out.responses[2]


def test_counter_skill_consistency_in_full() -> None:
    inp = _hospital_input(Condition.FULL)
    policy = hospital(inp, PARAMS)
    out = HospitalOutput(responses=[
        _response("M-0001", "COUNTER", CounterCondition.FLOAT_DOCTOR, "trauma"),
        _response("M-0002", "COUNTER", CounterCondition.RESERVE_RELEASE, "trauma"),
        _response("M-0003", "COUNTER", CounterCondition.FLOAT_NURSE, "emergency"),
    ])
    result = validate_hospital(out, inp, policy)
    assert [n.rule for n in result.notes] == ["counter_skill_inconsistent"] * 2
    assert result.output.responses[2] == out.responses[2]


def _option(oid: str, value: float, weight: float = 2.0) -> CaseOption:
    return CaseOption(option_id=oid, summary=oid, triage_weight=weight, slack_min=10.0,
                      travel_min=5.0, utility=value)


def test_allocation_unknown_option_and_double_grant() -> None:
    cases = [
        EscalationCase(case_id=f"CASE-000{i}", kind=CaseKind.FLOAT_STAFF, incident_id="INC-01",
                       hospital_id=f"H{i}", summary="s",
                       options=[_option("GRANT:DR-F1", 1.5), _option("DECLINE", 1.0)])
        for i in (1, 2)
    ]
    inp = AllocationInput(t_min=10.0, cases=cases, float_pool=[])
    policy = allocate(inp, PARAMS)
    out = AllocationOutput(decisions=[
        CaseDecision(case_id="CASE-0001", chosen_option_id="GRANT:DR-F1", rationale="r"),
        CaseDecision(case_id="CASE-0002", chosen_option_id="GRANT:DR-F1", rationale="r"),
    ])
    result = validate_allocation(out, inp, policy)
    assert [d.chosen_option_id for d in result.output.decisions] == ["GRANT:DR-F1", "DECLINE"]
    assert [n.rule for n in result.notes] == ["double_grant"]
    bad = out.model_copy(update={"decisions": [out.decisions[0].model_copy(update={"chosen_option_id": "X"})]})
    result = validate_allocation(bad, inp, policy)
    assert [(n.key, n.rule) for n in result.notes] == [("CASE-0001", "unknown_option"),
                                                      ("CASE-0002", "missing_item")]


def test_bls_to_red_only_scene_replaces_orders() -> None:
    inp = _dispatch_input()
    red_only = inp.incidents[0].model_copy(
        update={"triaged": True, "red_waiting": 1, "est_patients_remaining": 1}
    )
    inp = inp.model_copy(update={"incidents": [red_only]})
    policy = dispatch(inp, PARAMS)
    out = _output(orders=[UnitOrder(unit_id="B1", incident_id="INC-01", rationale="x")])
    result = validate_dispatch(out, inp, policy, PARAMS)
    assert {n.key: n.rule for n in result.notes}["orders"] == "bls_for_red_only"
    assert result.output.orders == policy.orders
