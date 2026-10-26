"""Action validation (plan section 4.6) and deterioration rules (section 4.5).

Every illegal action must be answered with the documented ``RejectReason``,
be logged as ``ActionRecord(ok=False)`` by ``apply`` and leave the observable
state exactly as it was (deep comparison of ``view()``).
"""

import pytest

from emergency_sim.domain.actions import Action, AssignTransport, DispatchUnit, DivertUnit, LendStaff
from emergency_sim.domain.enums import RejectReason, SimEventType, Triage, TriggerKind, VictimStatus
from emergency_sim.domain.log_records import ActionRecord
from emergency_sim.domain.scenario import Scenario
from emergency_sim.engine import Engine
from emergency_sim.eventlog import MemoryEventSink
from tests import factories as f
from tests.test_engine_lifecycle import (
    ACTOR,
    advance_until,
    dispatch,
    events_of,
    incident_of,
    make_engine,
    run_to_end,
    transport,
    unit_of,
    victim_of,
)


def _rules_world() -> Scenario:
    """H1 (trauma, general), H3 (pediatric); assorted units and staff; INC-02 reported at t=50."""
    return f.make_scenario(
        hospitals=[
            f.hospital("H1", caps=("trauma", "general"), er_capacity=4, occupied=3),
            f.hospital("H3", 9.0, 9.0, caps=("pediatric", "general")),
        ],
        ambulances=[
            f.ambulance("AMB-01", "ALS"),
            f.ambulance("AMB-02", "BLS", 1.0, 1.0),
            f.ambulance("AMB-03", "BLS", status="out_of_service"),
        ],
        staff=[
            f.doctor("DR-1", "H1", ("emergency", "trauma")),
            f.nurse("RN-1", "H1"),
            f.doctor("DR-F", "FLOAT", ("emergency", "trauma")),
            f.nurse("RN-F", "FLOAT", shift_end=0),
        ],
        responders=[f.responder("FR-1", ("scene_control",), 2.0, 2.0)],
        incidents=[
            f.incident("INC-01", 0, 7.0, 5.0, [f.victim("V1", "RED", "trauma"), f.victim("V2", "GREEN")]),
            f.incident("INC-02", 50, 3.0, 3.0, [f.victim("V3", "GREEN")]),
            f.incident("INC-03", 0, 19.0, 19.0, [f.victim("V4", "GREEN")]),
        ],
    )


def _assert_rejected_without_side_effects(
    engine: Engine, sink: MemoryEventSink, action: Action, reason: RejectReason
) -> None:
    before_view, before_len = engine.view(), len(sink.records)

    checked = engine.check(action)
    assert len(sink.records) == before_len  # check() never logs
    result = engine.apply(action, ACTOR)

    assert (checked.ok, checked.reason) == (False, reason)
    assert (result.ok, result.reason) == (False, reason) and result.detail
    assert engine.view() == before_view
    new = sink.records[before_len:]
    assert len(new) == 1 and isinstance(new[0], ActionRecord)
    assert (new[0].ok, new[0].reason, new[0].actor, new[0].action) == (False, reason, ACTOR, action)
    assert engine.check_invariants() == ()


AT_START = [
    (DispatchUnit(unit_id="AMB-99", incident_id="INC-01"), RejectReason.UNKNOWN_UNIT),
    (DispatchUnit(unit_id="AMB-03", incident_id="INC-01"), RejectReason.UNIT_NOT_AVAILABLE),
    (DispatchUnit(unit_id="AMB-01", incident_id="INC-99"), RejectReason.UNKNOWN_INCIDENT),
    (DispatchUnit(unit_id="AMB-01", incident_id="INC-02"), RejectReason.UNKNOWN_INCIDENT),
    (DivertUnit(unit_id="AMB-99", incident_id="INC-01"), RejectReason.UNKNOWN_UNIT),
    (DivertUnit(unit_id="FR-1", incident_id="INC-01"), RejectReason.NOT_AN_AMBULANCE),
    (DivertUnit(unit_id="AMB-01", incident_id="INC-01"), RejectReason.UNIT_NOT_EN_ROUTE),
    (AssignTransport(unit_id="AMB-99", hospital_id="H1", reserve_beds=False), RejectReason.UNKNOWN_UNIT),
    (AssignTransport(unit_id="FR-1", hospital_id="H1", reserve_beds=False), RejectReason.NOT_AN_AMBULANCE),
    (AssignTransport(unit_id="AMB-01", hospital_id="H1", reserve_beds=False), RejectReason.NOT_AWAITING_DESTINATION),
    (LendStaff(staff_id="DR-99", hospital_id="H1"), RejectReason.UNKNOWN_STAFF),
    (LendStaff(staff_id="DR-1", hospital_id="H3"), RejectReason.STAFF_NOT_FLOAT),
    (LendStaff(staff_id="RN-F", hospital_id="H1"), RejectReason.STAFF_OFF_SHIFT),
    (LendStaff(staff_id="DR-F", hospital_id="H9"), RejectReason.UNKNOWN_HOSPITAL),
]


@pytest.mark.parametrize(("action", "reason"), AT_START, ids=[f"{a.kind}-{r}" for a, r in AT_START])
def test_illegal_action_at_start(action: Action, reason: RejectReason) -> None:
    engine, sink = make_engine(_rules_world())
    engine.advance()

    _assert_rejected_without_side_effects(engine, sink, action, reason)


@pytest.fixture
def loaded_engine() -> tuple[Engine, MemoryEventSink]:
    """AMB-01 loaded at the cleared INC-01; AMB-02 en route to the far INC-03."""
    engine, sink = make_engine(_rules_world())
    engine.advance()
    dispatch(engine, "AMB-01", "INC-01")
    dispatch(engine, "AMB-02", "INC-03")
    advance_until(engine, TriggerKind.PATIENTS_LOADED)
    return engine, sink


LOADED = [
    (AssignTransport(unit_id="AMB-01", hospital_id="H9", reserve_beds=False), RejectReason.UNKNOWN_HOSPITAL),
    (AssignTransport(unit_id="AMB-01", hospital_id="H3", reserve_beds=False), RejectReason.HOSPITAL_NOT_CAPABLE),
    (AssignTransport(unit_id="AMB-01", hospital_id="H1", reserve_beds=True), RejectReason.INSUFFICIENT_BEDS),
    (AssignTransport(unit_id="AMB-02", hospital_id="H1", reserve_beds=False), RejectReason.NOT_AWAITING_DESTINATION),
    (DispatchUnit(unit_id="AMB-01", incident_id="INC-03"), RejectReason.UNIT_NOT_AVAILABLE),
    (DispatchUnit(unit_id="FR-1", incident_id="INC-01"), RejectReason.INCIDENT_NOT_OPEN),
    (DivertUnit(unit_id="AMB-01", incident_id="INC-03"), RejectReason.UNIT_NOT_EN_ROUTE),
    (DivertUnit(unit_id="AMB-02", incident_id="INC-03"), RejectReason.SAME_TARGET),
    (DivertUnit(unit_id="AMB-02", incident_id="INC-01"), RejectReason.INCIDENT_NOT_OPEN),
]


@pytest.mark.parametrize(("action", "reason"), LOADED, ids=[f"{a.kind}-{r}" for a, r in LOADED])
def test_illegal_action_with_a_loaded_unit(
    loaded_engine: tuple[Engine, MemoryEventSink], action: Action, reason: RejectReason
) -> None:
    engine, sink = loaded_engine

    _assert_rejected_without_side_effects(engine, sink, action, reason)


def test_staff_can_be_lent_only_once() -> None:
    engine, sink = make_engine(_rules_world())
    engine.advance()
    assert engine.apply(LendStaff(staff_id="DR-F", hospital_id="H1"), ACTOR).ok

    _assert_rejected_without_side_effects(
        engine, sink, LendStaff(staff_id="DR-F", hospital_id="H3"), RejectReason.STAFF_ALREADY_LENT
    )


def test_legal_actions_are_logged_as_ok(loaded_engine: tuple[Engine, MemoryEventSink]) -> None:
    engine, sink = loaded_engine
    action = AssignTransport(unit_id="AMB-01", hospital_id="H1", reserve_beds=False)

    result = engine.apply(action, "hospital_H1")

    assert result.ok and result.reason is None
    record = [r for r in sink.records if isinstance(r, ActionRecord)][-1]
    assert (record.ok, record.actor, record.action, record.t_min) == (True, "hospital_H1", action, engine.now)


# --- deterioration -----------------------------------------------------------


def _deterioration_world(*, required: tuple[str, ...] = (), unit_type: str = "ALS", det: int = 10) -> Scenario:
    return f.make_scenario(
        hospitals=[f.hospital("H1", caps=("general",))],
        ambulances=[f.ambulance("AMB-01", unit_type)],
        staff=[f.doctor("DR-1", "H1"), f.nurse("RN-1", "H1")],
        incidents=[
            f.incident("INC-01", 0, 7.0, 5.0, [f.victim("V1", "YELLOW", deteriorates=det)], responders=required),
            f.incident("INC-02", 20, 3.0, 3.0, [f.victim("V2", "GREEN")]),
        ],
    )


def test_visible_deterioration_on_a_triaged_scene() -> None:
    engine, sink = make_engine(_deterioration_world(required=("scene_control",)))
    engine.advance()
    dispatch(engine, "AMB-01", "INC-01")  # staged: no scene control

    point = advance_until(engine, TriggerKind.DETERIORATION)

    trigger = next(t for t in point.triggers if t.kind is TriggerKind.DETERIORATION)
    assert (point.t_min, trigger.incident_id, trigger.victim_ids) == (10.0, "INC-01", ("V1",))
    view = victim_of(engine, "V1")
    assert (view.triage, view.needs_als, view.deteriorated, view.status) == (
        Triage.RED, True, True, VictimStatus.WAITING,
    )
    assert events_of(sink, SimEventType.VICTIM_DETERIORATED)[0].detail == "visible"
    run_to_end(engine)
    outcome = engine.outcomes().victims[0]
    assert (outcome.initial_triage, outcome.final_triage) == (Triage.YELLOW, Triage.RED)
    assert (outcome.deteriorated_at_min, outcome.deadline_min) == (10.0, 40.0)


def test_hidden_deterioration_before_triage() -> None:
    engine, sink = make_engine(_deterioration_world())
    engine.advance()

    point = engine.advance()

    assert point is not None and point.t_min == 20.0
    assert [t.kind for t in point.triggers] == [TriggerKind.NEW_INCIDENT]
    assert incident_of(engine, "INC-01").victims == ()
    assert events_of(sink, SimEventType.VICTIM_DETERIORATED)[0].detail.startswith("hidden")
    dispatch(engine, "AMB-01", "INC-01")
    advance_until(engine, TriggerKind.SCENE_TRIAGED)
    view = victim_of(engine, "V1")
    assert (view.triage, view.deteriorated) == (Triage.RED, True)


def test_deterioration_in_a_bls_does_not_reroute() -> None:
    engine, _ = make_engine(_deterioration_world(unit_type="BLS", det=5))
    engine.advance()
    dispatch(engine, "AMB-01", "INC-01")

    point = advance_until(engine, TriggerKind.DETERIORATION)
    trigger = next(t for t in point.triggers if t.kind is TriggerKind.DETERIORATION)
    assert trigger.unit_id == "AMB-01"
    loaded = advance_until(engine, TriggerKind.PATIENTS_LOADED)

    assert next(t for t in loaded.triggers if t.kind is TriggerKind.PATIENTS_LOADED).victim_ids == ("V1",)
    assert unit_of(engine, "AMB-01").load_victim_ids == ("V1",)


def test_no_deterioration_once_care_started() -> None:
    engine, sink = make_engine(_deterioration_world(det=60))
    engine.advance()
    dispatch(engine, "AMB-01", "INC-01")
    advance_until(engine, TriggerKind.PATIENTS_LOADED)
    transport(engine, "AMB-01", "H1", reserve=True)
    run_to_end(engine)

    assert events_of(sink, SimEventType.VICTIM_DETERIORATED) == []
    assert engine.outcomes().victims[0].final_triage is Triage.YELLOW
