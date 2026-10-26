"""Protocol tests of the agent coordinator (``control/coordinator.py``) with fakes."""

from collections.abc import Sequence

import pytest

from emergency_sim.control.conditions import NEGOTIATION, NegotiationConfig
from emergency_sim.control.coordinator import AgentCoordinator
from emergency_sim.control.observations import DeteriorationLog
from emergency_sim.domain.actions import Action, AssignTransport, DispatchUnit, DivertUnit, LendStaff
from emergency_sim.domain.agent_io import DispatchOutput, UnitOrder
from emergency_sim.domain.base import AgentIOModel
from emergency_sim.domain.enums import (
    AgentRole,
    CaseKind,
    Condition,
    IncidentStatus,
    MessageType,
    RejectReason,
    Resolution,
    TriggerKind,
    UnitStatus,
    VictimStatus,
)
from emergency_sim.domain.messages import CommitAction, Message, ResolveAction
from emergency_sim.domain.ports import Controller
from emergency_sim.domain.views import DecisionPoint, HospitalView, IncidentView, PublicView, StaffView, Trigger
from emergency_sim.eventlog import MemoryEventSink
from emergency_sim.policies.params import PolicyParams
from tests.control_fakes import (
    FakePort,
    PolicyBackend,
    Rejector,
    agent_calls,
    hospital_view,
    incident_view,
    messages,
    public_view,
    staff_view,
    unit_view,
    victim_view,
)

PARAMS = PolicyParams()
CPR = "He collapsed and is not responding, a neighbour is doing CPR."
REPLY_TYPES = frozenset({MessageType.ACCEPT, MessageType.REJECT, MessageType.COUNTER,
                         MessageType.REQUEST, MessageType.COMMIT})
NEGOTIATION_ONLY = frozenset({MessageType.COUNTER, MessageType.ESCALATE, MessageType.REQUEST,
                              MessageType.COMMIT})


def _point(*triggers: Trigger, t: float = 10.0) -> DecisionPoint:
    return DecisionPoint(t_min=t, triggers=triggers)


LOADED = _point(Trigger(kind=TriggerKind.PATIENTS_LOADED, unit_id="A1"))


def _run(
    view: PublicView,
    condition: Condition | NegotiationConfig,
    point: DecisionPoint = LOADED,
    *,
    backend: PolicyBackend | None = None,
    reject: Rejector | None = None,
) -> tuple[FakePort, MemoryEventSink, PolicyBackend]:
    config = NEGOTIATION[condition] if isinstance(condition, Condition) else condition
    port, sink, backend = FakePort(view, reject=reject), MemoryEventSink(), backend or PolicyBackend()
    AgentCoordinator(backend, config, sink, PARAMS).handle(point, port)
    return port, sink, backend


def _transport_world(
    hospitals: Sequence[HospitalView],
    *,
    triage: str = "YELLOW",
    cap: str = "general",
    extra_incidents: Sequence[IncidentView] = (),
    float_pool: Sequence[StaffView] = (),
) -> PublicView:
    victims = [victim_view("V1", triage=triage, cap=cap, unit_id="A1")]
    return public_view(
        incidents=[incident_view(victims=victims, status=IncidentStatus.CLEARED, x=0.0), *extra_incidents],
        units=[unit_view("A1", x=0.0, status=UnitStatus.AWAITING_DESTINATION, incident_id="INC-01",
                         load=("V1",))],
        hospitals=hospitals, float_pool=float_pool,
    )


def _types(msgs: Sequence[Message]) -> list[MessageType]:
    return [m.type for m in msgs]


def _resolutions(msgs: Sequence[Message]) -> list[Resolution]:
    return [m.proposed_action.resolution for m in msgs if isinstance(m.proposed_action, CommitAction)]


def _reserve_world(extra_incidents: Sequence[IncidentView] = ()) -> PublicView:
    return _transport_world(
        [hospital_view("H1", x=1.0, free=2, reserve=2), hospital_view("H2", x=3.0)],
        extra_incidents=extra_incidents,
    )


def test_is_a_controller() -> None:
    assert isinstance(AgentCoordinator(PolicyBackend(), NEGOTIATION[Condition.FULL], MemoryEventSink(), PARAMS),
                      Controller)


def test_b1_reserve_reject_then_accept_and_no_negotiation_messages() -> None:
    port, sink, backend = _run(_reserve_world(), Condition.B1)
    msgs = messages(sink)
    assert _types(msgs) == [MessageType.PROPOSE, MessageType.REJECT, MessageType.PROPOSE, MessageType.ACCEPT]
    assert not NEGOTIATION_ONLY & set(_types(msgs))
    assert _resolutions(msgs) == [Resolution.ACCEPTED_AFTER_REJECT]
    assert port.applied == [("hospital_H2", AssignTransport(unit_id="A1", hospital_id="H2", reserve_beds=True))]
    assert [a for a, _ in backend.calls] == ["dispatch", "hospital_H1", "hospital_H2"]
    assert {m.conversation_id for m in msgs} == {"NEG-0001"}
    assert [m.round for m in msgs] == [1, 1, 2, 2]


def test_full_reserve_counter_escalate_commit_granted() -> None:
    port, sink, _ = _run(_reserve_world(), Condition.FULL)
    msgs = messages(sink)
    assert _types(msgs) == [MessageType.PROPOSE, MessageType.COUNTER, MessageType.ESCALATE,
                            MessageType.COMMIT, MessageType.COMMIT]
    escalate = msgs[2].proposed_action
    assert isinstance(escalate, ResolveAction) and escalate.case_kind is CaseKind.RESERVE_RELEASE
    assert _resolutions(msgs) == [Resolution.COUNTER_GRANTED] * 2
    assert {m.recipient for m in msgs[3:]} == {"hospital_H1", "dispatch"}
    assert port.applied == [("resource_allocation",
                             AssignTransport(unit_id="A1", hospital_id="H1", reserve_beds=True))]
    roles = [c.role for c in agent_calls(sink)]
    assert roles == [AgentRole.DISPATCH, AgentRole.HOSPITAL, AgentRole.ALLOCATION]


def test_full_reserve_held_when_red_demand_exceeds_beds_left() -> None:
    reds = [victim_view(f"R{i}", "INC-02", triage="RED", status=VictimStatus.WAITING) for i in (1, 2)]
    world = _reserve_world(extra_incidents=[incident_view("INC-02", x=1.0, victims=reds)])
    port, sink, _ = _run(world, Condition.FULL)
    msgs = [m for m in messages(sink) if m.round > 0]
    assert _types(msgs) == [MessageType.PROPOSE, MessageType.COUNTER, MessageType.ESCALATE,
                            MessageType.REJECT, MessageType.PROPOSE, MessageType.ACCEPT]
    assert _resolutions(msgs) == [Resolution.COUNTER_DECLINED, Resolution.ACCEPTED_AFTER_REJECT]
    assert port.applied_actions(AssignTransport) == [
        AssignTransport(unit_id="A1", hospital_id="H2", reserve_beds=True)]


def _no_trauma_doctor(hid: str, x: float) -> HospitalView:
    return hospital_view(hid, x=x, caps=("general", "trauma"))


def test_float_request_lends_staff_and_assigns_transport() -> None:
    world = _transport_world(
        [_no_trauma_doctor("H1", 1.0), hospital_view("H2", x=8.0, caps=("trauma",), staff=[
            staff_view("DR-H2", "doctor", "H2", ("emergency", "trauma")), staff_view("RN-H2", "nurse", "H2")])],
        cap="trauma", float_pool=[staff_view("DR-F1", "doctor", "FLOAT", ("emergency", "trauma"))],
    )
    port, sink, _ = _run(world, Condition.FULL)
    msgs = messages(sink)
    assert _types(msgs) == [MessageType.PROPOSE, MessageType.COUNTER, MessageType.REQUEST,
                            MessageType.COMMIT, MessageType.COMMIT]
    assert port.applied == [
        ("resource_allocation", LendStaff(staff_id="DR-F1", hospital_id="H1")),
        ("resource_allocation", AssignTransport(unit_id="A1", hospital_id="H1", reserve_beds=True)),
    ]
    commit = msgs[-1].proposed_action
    assert isinstance(commit, CommitAction) and commit.option_id == "GRANT:DR-F1"
    assert commit.resolution is Resolution.COUNTER_GRANTED


def test_float_counter_without_float_staff_is_auto_declined_without_ra_call() -> None:
    world = _transport_world([_no_trauma_doctor("H1", 1.0), _no_trauma_doctor("H2", 2.0)], cap="trauma")
    _, sink, backend = _run(world, Condition.FULL)
    ra_calls = [p for a, p in backend.calls if a == "resource_allocation"]
    assert len(ra_calls) == 1  # only the final NO_ACCEPTING_HOSPITAL case needs a choice
    assert [c.kind for c in ra_calls[0].cases] == [CaseKind.NO_ACCEPTING_HOSPITAL]  # type: ignore[attr-defined]
    declines = [m for m in messages(sink) if m.type is MessageType.REJECT and m.sender == "resource_allocation"]
    assert declines and all(m.rationale == "no feasible alternative" for m in declines)


def test_two_hospitals_compete_for_one_float_doctor() -> None:
    victims = [victim_view("V1", cap="trauma", unit_id="A1"), victim_view("V2", cap="trauma", unit_id="A2")]
    world = public_view(
        incidents=[incident_view(victims=victims, status=IncidentStatus.CLEARED, x=5.0)],
        units=[unit_view("A1", x=0.0, status=UnitStatus.AWAITING_DESTINATION, incident_id="INC-01", load=("V1",)),
               unit_view("A2", x=10.0, status=UnitStatus.AWAITING_DESTINATION, incident_id="INC-01",
                         load=("V2",))],
        hospitals=[_no_trauma_doctor("H1", 1.0), _no_trauma_doctor("H2", 9.0)],
        float_pool=[staff_view("DR-F1", "doctor", "FLOAT", ("emergency", "trauma"))],
    )
    port, sink, backend = _run(world, Condition.FULL, _point(
        Trigger(kind=TriggerKind.PATIENTS_LOADED, unit_id="A1"),
        Trigger(kind=TriggerKind.PATIENTS_LOADED, unit_id="A2")))
    round1 = [m for m in messages(sink) if m.round == 1]
    assert _types(round1).count(MessageType.REQUEST) == 2
    assert sorted(_resolutions(round1)) == sorted(
        [Resolution.COUNTER_GRANTED, Resolution.COUNTER_GRANTED, Resolution.COUNTER_DECLINED])
    assert port.applied_actions(LendStaff) == [LendStaff(staff_id="DR-F1", hospital_id="H1")]
    ra_calls = [p for a, p in backend.calls if a == "resource_allocation"]
    assert len(ra_calls[0].cases) == 2  # type: ignore[attr-defined]


def _all_full_world() -> PublicView:
    return _transport_world([hospital_view(f"H{i}", x=float(i), free=0) for i in (1, 2, 3, 4)])


def test_b1_all_reject_forced_admission_within_max_rounds() -> None:
    port, sink, _ = _run(_all_full_world(), Condition.B1)
    msgs = messages(sink)
    assert _types(msgs).count(MessageType.PROPOSE) == 3
    assert max(m.round for m in msgs) == 3
    last = msgs[-1]
    assert last.type is MessageType.INFORM and last.recipient == "hospital_H1"
    assert _resolutions(msgs) == [Resolution.FORCED_ADMISSION]
    assert port.applied == [("dispatch", AssignTransport(unit_id="A1", hospital_id="H1", reserve_beds=False))]


def test_full_all_reject_no_accepting_case_ra_commit() -> None:
    port, sink, _ = _run(_all_full_world(), Condition.FULL)
    msgs = messages(sink)
    assert _types(msgs)[-2:] == [MessageType.ESCALATE, MessageType.COMMIT]
    escalate = msgs[-2].proposed_action
    assert isinstance(escalate, ResolveAction) and escalate.case_kind is CaseKind.NO_ACCEPTING_HOSPITAL
    assert len(escalate.option_ids) == 4
    assert _resolutions(msgs) == [Resolution.RA_COMMIT]
    ((actor, action),) = port.applied
    assert actor == "resource_allocation" and isinstance(action, AssignTransport) and not action.reserve_beds


def test_max_rounds_config_respected() -> None:
    config = NegotiationConfig(False, False, False, False, max_rounds=2)
    _, sink, _ = _run(_all_full_world(), config)
    assert _types(messages(sink)).count(MessageType.PROPOSE) == 2


def test_message_ids_sequential_and_replies_linked() -> None:
    _, sink, _ = _run(_reserve_world(), Condition.FULL)
    msgs = messages(sink)
    assert [m.message_id for m in msgs] == [f"M-{i:04d}" for i in range(1, len(msgs) + 1)]
    seen: dict[str, Message] = {}
    for m in msgs:
        if m.type in REPLY_TYPES or (m.type is MessageType.ESCALATE and m.sender.startswith("hospital_")):
            assert m.in_reply_to in seen
            assert seen[m.in_reply_to].conversation_id == m.conversation_id
        seen[m.message_id] = m


def test_illegal_dispatch_section_replaced_by_policy_with_note() -> None:
    world = public_view(
        incidents=[incident_view("INC-01", x=2.0, reported=1)],
        units=[unit_view("A1", x=1.0), unit_view("A2", x=0.0, status=UnitStatus.EN_ROUTE_SCENE,
                                                 incident_id="INC-09")],
    )

    def busy_order(payload: AgentIOModel) -> AgentIOModel:
        return DispatchOutput(orders=[UnitOrder(unit_id="A2", incident_id="INC-01", rationale="busy")],
                              diversions=[], contentions=[], transports=[])

    port, sink, _ = _run(world, Condition.FULL, _point(Trigger(kind=TriggerKind.NEW_INCIDENT,
                                                               incident_id="INC-01")),
                         backend=PolicyBackend(overrides={"dispatch": busy_order}))
    (call,) = [c for c in agent_calls(sink) if c.agent == "dispatch"]
    assert [(n.scope, n.key, n.rule) for n in call.fallbacks] == [("section", "orders", "unit_not_available")]
    assert call.source == "model"
    assert [a.unit_id for a in port.applied_actions(DispatchUnit)] == ["A1"]
    assert all(a.unit_id != "A2" for _, a, _ in port.attempts if isinstance(a, DispatchUnit))


def test_engine_rejected_accept_moves_to_next_hospital() -> None:
    world = _transport_world([hospital_view("H1", x=1.0), hospital_view("H2", x=3.0)])

    def beds_raced(action: Action) -> RejectReason | None:
        is_h1 = isinstance(action, AssignTransport) and action.hospital_id == "H1"
        return RejectReason.INSUFFICIENT_BEDS if is_h1 else None

    port, sink, _ = _run(world, Condition.B1, reject=beds_raced)
    msgs = messages(sink)
    assert _types(msgs) == [MessageType.PROPOSE, MessageType.INFORM, MessageType.PROPOSE, MessageType.ACCEPT]
    assert _resolutions(msgs) == [Resolution.ENGINE_REJECTED, Resolution.ACCEPTED_AFTER_REJECT]
    assert msgs[1].in_reply_to == msgs[0].message_id
    assert port.applied_actions(AssignTransport)[0].hospital_id == "H2"


def test_assessment_and_dispatch_phase_messages() -> None:
    world = public_view(
        incidents=[incident_view("INC-01", x=2.0, report=CPR, incident_type="cardiac_event")],
        units=[unit_view("A1", x=4.0), unit_view("B1", unit_type="BLS", x=1.0)],
    )
    port, sink, _ = _run(world, Condition.FULL, _point(Trigger(kind=TriggerKind.NEW_INCIDENT,
                                                               incident_id="INC-01")))
    msgs = messages(sink)
    assert _types(msgs) == [MessageType.PRIORITY_REPORT, MessageType.INFORM]
    assert (msgs[0].sender, msgs[0].recipient) == ("incident_assessment", "dispatch")
    assert (msgs[1].sender, msgs[1].recipient) == ("dispatch", "all")
    assert all(m.round == 0 and m.conversation_id == "NEG-0001" for m in msgs)
    assert ("dispatch", DispatchUnit(unit_id="A1", incident_id="INC-01")) in port.applied


def _two_cardiac_world() -> PublicView:
    return public_view(
        incidents=[incident_view("INC-01", x=2.0, report=CPR, incident_type="cardiac_event"),
                   incident_view("INC-02", x=8.0, report=CPR, incident_type="cardiac_event")],
        units=[unit_view("A1", x=4.0)],
    )


NEW_TWO = _point(Trigger(kind=TriggerKind.NEW_INCIDENT, incident_id="INC-01"),
                 Trigger(kind=TriggerKind.NEW_INCIDENT, incident_id="INC-02"))


def test_full_unit_contention_escalated_and_committed() -> None:
    port, sink, _ = _run(_two_cardiac_world(), Condition.FULL, NEW_TWO)
    msgs = messages(sink)
    escalate = next(m for m in msgs if m.type is MessageType.ESCALATE)
    assert isinstance(escalate.proposed_action, ResolveAction)
    assert escalate.proposed_action.case_kind is CaseKind.UNIT_CONTENTION
    commit = msgs[-1]
    assert commit.type is MessageType.COMMIT and commit.in_reply_to == escalate.message_id
    assert _resolutions(msgs) == [Resolution.RA_COMMIT]
    ((actor, action),) = port.applied
    assert actor == "resource_allocation" and isinstance(action, DispatchUnit) and action.unit_id == "A1"


def test_b1_dispatches_contested_unit_greedily() -> None:
    port, sink, _ = _run(_two_cardiac_world(), Condition.B1, NEW_TWO)
    assert MessageType.ESCALATE not in _types(messages(sink))
    assert port.applied == [("dispatch", DispatchUnit(unit_id="A1", incident_id="INC-01"))]


def test_full_diversion_of_en_route_als() -> None:
    world = public_view(
        incidents=[incident_view("INC-01", x=0.0, report="Slipped on ice, sore wrist."),
                   incident_view("INC-02", x=3.0, report=CPR, incident_type="cardiac_event")],
        units=[unit_view("A2", x=2.0, status=UnitStatus.EN_ROUTE_SCENE, incident_id="INC-01")],
    )
    port, sink, _ = _run(world, Condition.FULL, _point(Trigger(kind=TriggerKind.NEW_INCIDENT,
                                                               incident_id="INC-02")))
    assert port.applied == [("resource_allocation", DivertUnit(unit_id="A2", incident_id="INC-02"))]
    assert _resolutions(messages(sink)) == [Resolution.RA_COMMIT]
    _, sink_nd, _ = _run(world, Condition.FULL_NO_DIVERSION, _point(Trigger(
        kind=TriggerKind.NEW_INCIDENT, incident_id="INC-02")))
    assert MessageType.ESCALATE not in _types(messages(sink_nd))


def test_backend_fallback_is_recorded_on_the_call() -> None:
    backend = PolicyBackend(fallback_agents=frozenset({"dispatch"}))
    _, sink, _ = _run(_reserve_world(), Condition.B1, backend=backend)
    (call,) = [c for c in agent_calls(sink) if c.agent == "dispatch"]
    assert call.source == "policy_fallback"
    assert [(n.scope, n.rule) for n in call.fallbacks] == [("call", "backend_fallback")]
    assert call.triggers == (TriggerKind.PATIENTS_LOADED,)


def test_idle_point_makes_no_calls() -> None:
    world = public_view(units=[unit_view("A1")], hospitals=[hospital_view("H1")])
    _, sink, backend = _run(world, Condition.FULL, _point(Trigger(kind=TriggerKind.UNIT_FREED, unit_id="A1")))
    assert backend.calls == [] and sink.records == ()


@pytest.mark.parametrize("condition", list(NEGOTIATION))
def test_runs_are_deterministic(condition: Condition) -> None:
    def dump() -> list[str]:
        _, sink, _ = _run(_all_full_world(), condition)
        return [r.model_dump_json() for r in sink.records]

    assert dump() == dump()


def test_rejects_non_positive_rounds() -> None:
    from emergency_sim.control import ProtocolError

    with pytest.raises(ProtocolError):
        AgentCoordinator(PolicyBackend(), NegotiationConfig(True, True, True, True, max_rounds=0),
                         MemoryEventSink(), PARAMS)



def test_loading_patients_triggers_a_fresh_assessment() -> None:
    # Regression: the report kept "urgency 100, suspected_red" after the only
    # RED was loaded, so the GREEN left behind outranked other scenes and
    # Resource Allocation weighted it as RED.
    triaged = public_view(
        incidents=[incident_view(victims=[victim_view("V1", triage="RED", status=VictimStatus.WAITING),
                                          victim_view("V2", triage="GREEN", status=VictimStatus.WAITING)])],
        units=[unit_view("A1", status=UnitStatus.ON_SCENE, incident_id="INC-01")],
    )
    loaded = public_view(
        t=14.0,
        incidents=[incident_view(victims=[
            victim_view("V1", triage="RED", status=VictimStatus.LOADED, unit_id="A1"),
            victim_view("V2", triage="GREEN", status=VictimStatus.WAITING)])],
        units=[unit_view("A1", status=UnitStatus.AWAITING_DESTINATION, incident_id="INC-01", load=("V1",))],
        hospitals=[hospital_view("H1")],
    )
    sink = MemoryEventSink()
    coordinator = AgentCoordinator(PolicyBackend(), NEGOTIATION[Condition.B1], sink, PARAMS)
    coordinator.handle(_point(Trigger(kind=TriggerKind.SCENE_TRIAGED, incident_id="INC-01")), FakePort(triaged))
    coordinator.handle(_point(Trigger(kind=TriggerKind.PATIENTS_LOADED, unit_id="A1", incident_id="INC-01",
                                      victim_ids=("V1",)), t=14.0), FakePort(loaded))
    reports = [m.proposed_action for m in messages(sink) if m.type is MessageType.PRIORITY_REPORT]
    assert [(r.urgency, r.suspected_red) for r in reports] == [(100, True), (30, False)]


def test_deterioration_age_reaches_the_assessment_payload() -> None:
    victims = [victim_view("V1", triage="RED", status=VictimStatus.WAITING, deteriorated=True)]
    world = public_view(t=50.0, incidents=[incident_view(victims=victims)], units=[unit_view("A1")])
    det = Trigger(kind=TriggerKind.DETERIORATION, incident_id="INC-01", victim_ids=("V1",))
    backend = PolicyBackend()
    coordinator = AgentCoordinator(backend, NEGOTIATION[Condition.B1], MemoryEventSink(), PARAMS)
    coordinator.handle(_point(det, t=50.0), FakePort(world))
    later = world.model_copy(update={"t_min": 58.0})
    coordinator.handle(_point(det, t=58.0), FakePort(later))
    payloads = [p for name, p in backend.calls if name == "incident_assessment"]
    ages = [p.incidents[0].triaged_victims[0].minutes_since_deterioration for p in payloads]
    assert ages == [0.0, 8.0]


def test_deterioration_log_keeps_the_first_sighting() -> None:
    log = DeteriorationLog()
    fresh = public_view(t=10.0, incidents=[incident_view(victims=[victim_view("V1", status=VictimStatus.WAITING)])])
    assert log.observe(fresh) == {}
    turned = public_view(t=20.0, incidents=[incident_view(
        victims=[victim_view("V1", triage="RED", status=VictimStatus.WAITING, deteriorated=True)])])
    assert log.observe(turned) == {"V1": 0.0}
    assert log.observe(turned.model_copy(update={"t_min": 32.5})) == {"V1": 12.5}


def _refuse_transports(action: Action) -> RejectReason | None:
    return RejectReason.INSUFFICIENT_BEDS if isinstance(action, AssignTransport) else None


def test_b1_refused_forced_admission_is_an_engine_rejected_inform() -> None:
    port, sink, _ = _run(_all_full_world(), Condition.B1, reject=_refuse_transports)
    last = messages(sink)[-1]
    assert last.type is MessageType.INFORM
    assert isinstance(last.proposed_action, CommitAction)
    assert last.proposed_action.resolution is Resolution.ENGINE_REJECTED
    assert last.proposed_action.actions == (
        AssignTransport(unit_id="A1", hospital_id="H1", reserve_beds=False),
    )
    assert port.applied == []


def test_full_refused_ra_destination_is_an_inform_not_a_commit() -> None:
    port, sink, _ = _run(_all_full_world(), Condition.FULL, reject=_refuse_transports)
    msgs = messages(sink)
    assert MessageType.COMMIT not in _types(msgs)
    last = msgs[-1]
    assert last.type is MessageType.INFORM and last.sender == "resource_allocation"
    assert isinstance(last.proposed_action, CommitAction)
    assert last.proposed_action.resolution is Resolution.ENGINE_REJECTED
    (attempted,) = last.proposed_action.actions
    assert isinstance(attempted, AssignTransport) and attempted.unit_id == "A1"
    assert port.applied == []


def test_reserved_destination_retries_without_reservation() -> None:
    # Beds are free but no hospital has a trauma doctor (and there is no float
    # staff), so every proposal is turned down and RA picks a destination with
    # a reservation; the engine refuses the reservation and RA retries without.
    def refuse_reservations(action: Action) -> RejectReason | None:
        reserving = isinstance(action, AssignTransport) and action.reserve_beds
        return RejectReason.INSUFFICIENT_BEDS if reserving else None

    world = _transport_world([_no_trauma_doctor(f"H{i}", float(i)) for i in (1, 2, 3, 4)], cap="trauma")
    port, sink, _ = _run(world, Condition.FULL, reject=refuse_reservations)
    transports = [a for _, a, _ in port.attempts if isinstance(a, AssignTransport)]
    assert [a.reserve_beds for a in transports] == [True, False]
    assert transports[0].hospital_id == transports[1].hospital_id
    assert _resolutions(messages(sink))[-1] is Resolution.RA_COMMIT
    ((_, action),) = port.applied
    assert isinstance(action, AssignTransport) and not action.reserve_beds


def test_engine_refused_float_grant_is_an_inform_and_unit_moves_on() -> None:
    world = _transport_world(
        [_no_trauma_doctor("H1", 1.0), hospital_view("H2", x=8.0, caps=("trauma",), staff=[
            staff_view("DR-H2", "doctor", "H2", ("emergency", "trauma")), staff_view("RN-H2", "nurse", "H2")])],
        cap="trauma", float_pool=[staff_view("DR-F1", "doctor", "FLOAT", ("emergency", "trauma"))],
    )

    def refuse_lending(action: Action) -> RejectReason | None:
        return RejectReason.STAFF_ALREADY_LENT if isinstance(action, LendStaff) else None

    port, sink, _ = _run(world, Condition.FULL, reject=refuse_lending)
    msgs = messages(sink)
    refused = next(m for m in msgs if m.sender == "resource_allocation" and m.type is MessageType.INFORM)
    assert isinstance(refused.proposed_action, CommitAction)
    assert refused.proposed_action.resolution is Resolution.ENGINE_REJECTED
    assert refused.proposed_action.option_id == "GRANT:DR-F1"
    assert any(isinstance(a, LendStaff) for a in refused.proposed_action.actions)
    assert Resolution.COUNTER_DECLINED not in _resolutions(msgs)
    assert port.applied_actions(LendStaff) == []
    assert port.applied_actions(AssignTransport)[0].hospital_id == "H2"
