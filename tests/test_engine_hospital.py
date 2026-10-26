"""ER beds, reservations, full-on-arrival, queue, staff selection (plan section 4.4)."""

from collections.abc import Sequence

from emergency_sim import datagen
from emergency_sim.domain.actions import AssignTransport, LendStaff
from emergency_sim.domain.enums import RejectReason, SimEventType, TriggerKind, UnitStatus, VictimStatus
from emergency_sim.domain.scenario import AmbulanceSpec, HospitalSpec, IncidentSpec, Scenario, StaffSpec
from emergency_sim.domain.views import HospitalView
from emergency_sim.engine import Engine, care
from emergency_sim.eventlog import MemoryEventSink
from tests import factories as f
from tests.test_engine_lifecycle import (
    ACTOR,
    advance_until,
    dispatch,
    events_of,
    make_engine,
    run_to_end,
    transport,
    unit_of,
    victim_of,
)

RED_TRAUMA = "INC-01-V1"
GREEN_GENERAL = "INC-01-V2"
DEFAULT_STAFF = (f.doctor("DR-1", "H1", ("emergency", "trauma")), f.nurse("RN-1", "H1"))
DEFAULT_VICTIMS = (f.victim(RED_TRAUMA, "RED", "trauma"), f.victim(GREEN_GENERAL, "GREEN"))


def _world(
    *, er_capacity: int = 4, occupied: int = 0, staff: Sequence[StaffSpec] | None = None,
    extra_hospitals: Sequence[HospitalSpec] = (), incidents: Sequence[IncidentSpec] | None = None,
    ambulances: Sequence[AmbulanceSpec] | None = None,
) -> Scenario:
    """H1 (trauma, general) at (5, 5); INC-01 at (7, 5) with a RED trauma and a GREEN general victim."""
    return f.make_scenario(
        hospitals=[f.hospital("H1", caps=("trauma", "general"), er_capacity=er_capacity, occupied=occupied),
                   *extra_hospitals],
        ambulances=ambulances or [f.ambulance("AMB-01", "ALS")],
        staff=staff if staff is not None else DEFAULT_STAFF,
        incidents=incidents or [f.incident("INC-01", 0, 7.0, 5.0, DEFAULT_VICTIMS)],
    )


def _h(engine: Engine, hospital_id: str = "H1") -> HospitalView:
    return next(h for h in engine.view().hospitals if h.hospital_id == hospital_id)


def _loaded(scenario: Scenario) -> tuple[Engine, MemoryEventSink]:
    engine, sink = make_engine(scenario)
    engine.advance()
    dispatch(engine, "AMB-01", "INC-01")
    advance_until(engine, TriggerKind.PATIENTS_LOADED)
    return engine, sink


def _started(sink: MemoryEventSink) -> dict[str, tuple[float, str | None, str]]:
    """victim -> (time, doctor, detail) of TREATMENT_STARTED events."""
    starts = events_of(sink, SimEventType.TREATMENT_STARTED)
    return {r.victim_ids[0]: (r.t_min, r.staff_id, r.detail) for r in starts}


# --- reservations ------------------------------------------------------------


def test_reservation_holds_beds_and_converts_on_arrival() -> None:
    engine, sink = _loaded(_world(occupied=2))
    transport(engine, "AMB-01", "H1", reserve=True)

    board = _h(engine)
    assert (board.beds_occupied, board.beds_reserved, board.beds_free_unreserved) == (2, 2, 0)
    assert board.inbound_unit_ids == ("AMB-01",)
    assert events_of(sink, SimEventType.BEDS_RESERVED)[0].victim_ids == (RED_TRAUMA, GREEN_GENERAL)

    advance_until(engine, TriggerKind.UNIT_FREED)
    board = _h(engine)
    assert (board.beds_occupied, board.beds_reserved, board.inbound_unit_ids) == (4, 0, ())
    assert events_of(sink, SimEventType.HOSPITAL_FULL_ON_ARRIVAL) == []


def test_reservation_needs_enough_free_unreserved_beds() -> None:
    engine, sink = _loaded(_world(occupied=3))
    before = engine.view()

    result = engine.apply(AssignTransport(unit_id="AMB-01", hospital_id="H1", reserve_beds=True), ACTOR)

    assert (result.ok, result.reason) == (False, RejectReason.INSUFFICIENT_BEDS)
    assert engine.view() == before
    assert events_of(sink, SimEventType.BEDS_RESERVED) == []
    assert engine.check(AssignTransport(unit_id="AMB-01", hospital_id="H1", reserve_beds=False)).ok


# --- full on arrival ---------------------------------------------------------


def test_full_on_arrival_trigger_then_guard_queues() -> None:
    engine, sink = _loaded(_world(er_capacity=2, occupied=2))
    transport(engine, "AMB-01", "H1")

    point = advance_until(engine, TriggerKind.HOSPITAL_FULL_ON_ARRIVAL)
    trigger = next(t for t in point.triggers if t.kind is TriggerKind.HOSPITAL_FULL_ON_ARRIVAL)
    assert (trigger.unit_id, trigger.hospital_id) == ("AMB-01", "H1")
    assert trigger.victim_ids == (RED_TRAUMA, GREEN_GENERAL)
    assert unit_of(engine, "AMB-01").status is UnitStatus.AT_HOSPITAL_FULL

    engine.advance()  # controller did nothing: the guard hands over and queues
    queued = events_of(sink, SimEventType.PATIENT_QUEUED)
    assert [r.victim_ids for r in queued] == [(RED_TRAUMA,), (GREEN_GENERAL,)]
    assert [r.t_min for r in queued] == [point.t_min] * 2
    run_to_end(engine)
    from_queue = [r for r in events_of(sink, SimEventType.PATIENT_BEDDED) if r.detail == "from queue"]
    assert [(r.t_min, r.victim_ids) for r in from_queue] == [(15.0, (RED_TRAUMA,)), (30.0, (GREEN_GENERAL,))]
    outcome = {v.victim_id: v for v in engine.outcomes().victims}[RED_TRAUMA]
    assert (outcome.arrived_hospital_min, outcome.bedded_min, outcome.care_start_min) == (point.t_min, 15.0, 15.0)


def test_guard_beds_as_many_as_possible_and_queues_the_rest() -> None:
    engine, sink = _loaded(_world(er_capacity=2, occupied=1))
    transport(engine, "AMB-01", "H1")
    advance_until(engine, TriggerKind.HOSPITAL_FULL_ON_ARRIVAL)

    engine.advance()

    assert [r.victim_ids for r in events_of(sink, SimEventType.PATIENT_BEDDED)] == [(RED_TRAUMA,)]
    assert [r.victim_ids for r in events_of(sink, SimEventType.PATIENT_QUEUED)] == [(GREEN_GENERAL,)]


def test_full_on_arrival_redirect_to_another_hospital() -> None:
    other = f.hospital("H2", 5.0, 9.0, caps=("trauma", "general"))
    staff = [f.doctor("DR-1", "H1", ("emergency", "trauma")), f.nurse("RN-1", "H1"),
             f.doctor("DR-2", "H2", ("emergency", "trauma")), f.nurse("RN-2", "H2")]
    engine, sink = _loaded(_world(er_capacity=2, occupied=2, extra_hospitals=(other,), staff=staff))
    transport(engine, "AMB-01", "H1")
    advance_until(engine, TriggerKind.HOSPITAL_FULL_ON_ARRIVAL)

    same = engine.apply(AssignTransport(unit_id="AMB-01", hospital_id="H1", reserve_beds=False), ACTOR)
    assert (same.ok, same.reason) == (False, RejectReason.SAME_HOSPITAL)
    transport(engine, "AMB-01", "H2", reserve=True)
    advance_until(engine, TriggerKind.UNIT_FREED)
    run_to_end(engine)

    assert events_of(sink, SimEventType.PATIENT_QUEUED) == []
    arrivals = events_of(sink, SimEventType.ARRIVED_HOSPITAL)
    assert [r.hospital_id for r in arrivals] == ["H1", "H2"]
    outcome = {v.victim_id: v for v in engine.outcomes().victims}[RED_TRAUMA]
    assert (outcome.hospital_id, outcome.arrived_hospital_min) == ("H2", arrivals[1].t_min)


def test_queue_puts_red_before_earlier_green() -> None:
    scenario = _world(
        er_capacity=1, occupied=1,
        ambulances=[f.ambulance("AMB-01", "ALS"), f.ambulance("AMB-02", "ALS")],
        incidents=[
            f.incident("INC-01", 0, 7.0, 5.0, [f.victim("G1", "GREEN")]),
            f.incident("INC-02", 1, 7.0, 5.0, [f.victim("R1", "RED", "trauma")]),
        ],
    )
    engine, sink = make_engine(scenario)
    engine.advance()
    dispatch(engine, "AMB-01", "INC-01")
    engine.advance()
    dispatch(engine, "AMB-02", "INC-02")
    run_to_end(engine)  # the guard transports both to H1 and queues them

    queued = events_of(sink, SimEventType.PATIENT_QUEUED)
    assert [r.victim_ids for r in queued] == [("G1",), ("R1",)]
    from_queue = [r for r in events_of(sink, SimEventType.PATIENT_BEDDED) if r.detail == "from queue"]
    assert [(r.t_min, r.victim_ids) for r in from_queue][0] == (15.0, ("R1",))


# --- staff -------------------------------------------------------------------


def test_doctor_skill_matching_and_float_lending() -> None:
    staff = [f.doctor("DR-1", "H1", ("emergency",)), f.nurse("RN-1", "H1"),
             f.doctor("DR-F", "FLOAT", ("emergency", "trauma"))]
    engine, sink = _loaded(_world(staff=staff))
    transport(engine, "AMB-01", "H1", reserve=True)
    point = advance_until(engine, TriggerKind.UNIT_FREED)

    assert set(_started(sink)) == {GREEN_GENERAL}
    assert _started(sink)[GREEN_GENERAL][1] == "DR-1"
    assert victim_of(engine, RED_TRAUMA).status is VictimStatus.WAITING_FOR_STAFF
    assert [s.staff_id for s in engine.view().float_pool] == ["DR-F"]

    assert engine.apply(LendStaff(staff_id="DR-F", hospital_id="H1"), ACTOR).ok

    assert _started(sink)[RED_TRAUMA][:2] == (point.t_min, "DR-F")
    assert events_of(sink, SimEventType.STAFF_LENT)[0].hospital_id == "H1"
    assert engine.view().float_pool == ()
    assert "DR-F" in [s.staff_id for s in _h(engine).staff]


def test_holdover_only_without_an_on_shift_colleague_with_the_skill() -> None:
    staff = [
        f.doctor("DR-A", "H1", ("emergency", "trauma"), shift_end=5),
        f.doctor("DR-B", "H1", ("emergency",)),
        f.nurse("RN-1", "H1", shift_end=5),
        f.nurse("RN-2", "H1"),
    ]
    engine, sink = _loaded(_world(staff=staff))
    transport(engine, "AMB-01", "H1", reserve=True)
    run_to_end(engine)

    started = _started(sink)
    assert started[RED_TRAUMA][1] == "DR-A"  # held over: nobody on shift has trauma
    assert started[GREEN_GENERAL][1] == "DR-B"  # DR-B is on shift with "emergency"
    assert all("nurse RN-2" in s[2] for s in started.values())
    assert [r.staff_id for r in events_of(sink, SimEventType.SHIFT_ENDED)][:2] == ["DR-A", "RN-1"]


def test_holdover_blocked_by_a_saturated_on_shift_colleague() -> None:
    staff = [
        f.doctor("DR-A", "H1", ("emergency", "trauma"), shift_end=5),
        f.doctor("DR-C", "H1", ("emergency", "trauma"), load=4),
        f.nurse("RN-1", "H1"),
    ]
    engine, sink = _loaded(_world(staff=staff))
    transport(engine, "AMB-01", "H1", reserve=True)
    run_to_end(engine)

    started = _started(sink)
    assert started[RED_TRAUMA][:2] == (20.0, "DR-C")  # waits for DR-C's first t=0 release
    assert started[GREEN_GENERAL][:2] == (40.0, "DR-C")


def test_nurse_prefers_the_matching_specialty() -> None:
    staff = [f.doctor("DR-1", "H1", ("emergency", "trauma")), f.nurse("RN-1", "H1"),
             f.nurse("RN-2", "H1", ("emergency", "trauma"))]
    engine, sink = _loaded(_world(staff=staff))
    transport(engine, "AMB-01", "H1", reserve=True)
    run_to_end(engine)

    started = _started(sink)
    assert started[RED_TRAUMA][2].endswith("nurse RN-2")
    assert started[GREEN_GENERAL][2].endswith("nurse RN-1")


def test_nurse_skill_map_mirrors_datagen() -> None:
    assert dict(care.NURSE_SKILL_FOR) == datagen.NURSE_SKILL_FOR


def test_t0_bed_and_staff_releases() -> None:
    scenario = f.make_scenario(
        hospitals=[f.hospital("H1", er_capacity=4, occupied=3)],
        staff=[f.doctor("DR-1", "H1", load=2), f.nurse("RN-F", "FLOAT", load=1)],
    )
    engine, sink = make_engine(scenario)
    assert _h(engine).beds_occupied == 3

    run_to_end(engine)

    assert [r.t_min for r in events_of(sink, SimEventType.T0_BED_RELEASED)] == [15.0, 30.0, 45.0]
    staff_releases = [(r.t_min, r.staff_id) for r in events_of(sink, SimEventType.T0_STAFF_RELEASED)]
    assert staff_releases == [(20.0, "DR-1"), (20.0, "RN-F"), (40.0, "DR-1")]
    assert _h(engine).beds_occupied == 0
    assert [s.load for s in _h(engine).staff] == [0]


def test_treatment_end_frees_bed_and_staff() -> None:
    engine, sink = _loaded(_world())
    transport(engine, "AMB-01", "H1", reserve=True)
    run_to_end(engine)

    board = _h(engine)
    assert (board.beds_occupied, board.in_treatment, board.waiting_for_staff) == (0, 0, 0)
    assert all(s.load == 0 for s in board.staff)
    assert victim_of(engine, RED_TRAUMA).status is VictimStatus.TREATED
    assert len(events_of(sink, SimEventType.TREATMENT_ENDED)) == 2
