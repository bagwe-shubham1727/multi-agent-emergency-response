"""Ambulance and responder lifecycle on tiny hand-made worlds (plan sections 4.1 to 4.3, 4.5).

Also defines small driving helpers (``make_engine``, ``advance_until``,
``events_of``...) that the other engine test modules import.
"""

from collections.abc import Callable

import pytest

from emergency_sim import geo
from emergency_sim.domain.actions import AssignTransport, DispatchUnit, DivertUnit
from emergency_sim.domain.enums import (
    IncidentStatus,
    SimEventType,
    Triage,
    TriggerKind,
    UnitStatus,
    VictimStatus,
)
from emergency_sim.domain.log_records import ActionRecord, SimEventRecord, SnapshotRecord
from emergency_sim.domain.params import SimParams
from emergency_sim.domain.ports import EnginePort
from emergency_sim.domain.scenario import Scenario
from emergency_sim.domain.views import DecisionPoint, IncidentView, UnitView, VictimView
from emergency_sim.engine import GUARD_ACTOR, Engine, EngineError
from emergency_sim.eventlog import MemoryEventSink
from tests import factories as f

ACTOR = "test"


# --- helpers shared with the other engine test modules -----------------------


def make_engine(scenario: Scenario, params: SimParams | None = None) -> tuple[Engine, MemoryEventSink]:
    """Builds an engine with a fresh in-memory sink."""
    sink = MemoryEventSink()
    return Engine(scenario, params or SimParams(), sink), sink


def advance_until(
    engine: Engine, kind: TriggerKind, match: Callable[[DecisionPoint], bool] = lambda p: True
) -> DecisionPoint:
    """Advances (doing nothing at other points) until a point has a ``kind`` trigger."""
    while (point := engine.advance()) is not None:
        if any(t.kind is kind for t in point.triggers) and match(point):
            return point
    raise AssertionError(f"run ended without a {kind} trigger")


def run_to_end(engine: Engine) -> None:
    """Advances to the horizon without acting."""
    while engine.advance() is not None:
        pass


def events_of(sink: MemoryEventSink, kind: SimEventType) -> list[SimEventRecord]:
    """Sim events of one type, in log order."""
    return [r for r in sink.records if isinstance(r, SimEventRecord) and r.event is kind]


def unit_of(port: EnginePort, unit_id: str) -> UnitView:
    """The unit's current view."""
    return next(u for u in port.view().units if u.unit_id == unit_id)


def incident_of(port: EnginePort, incident_id: str) -> IncidentView:
    """The incident's current view."""
    return next(i for i in port.view().incidents if i.incident_id == incident_id)


def victim_of(port: EnginePort, victim_id: str) -> VictimView:
    """The victim's current view (its incident must be triaged)."""
    return next(v for i in port.view().incidents for v in i.victims if v.victim_id == victim_id)


def dispatch(engine: Engine, unit_id: str, incident_id: str) -> None:
    """Applies a dispatch that must be legal."""
    assert engine.apply(DispatchUnit(unit_id=unit_id, incident_id=incident_id), ACTOR).ok


def transport(engine: Engine, unit_id: str, hospital_id: str, reserve: bool = False) -> None:
    """Applies a transport that must be legal."""
    action = AssignTransport(unit_id=unit_id, hospital_id=hospital_id, reserve_beds=reserve)
    result = engine.apply(action, ACTOR)
    assert result.ok, result


def one_incident_world(
    victims: list, *, ambulances: list | None = None, responders: list | None = None,
    required: tuple[str, ...] = (), caps: tuple[str, ...] = ("trauma", "general", "cardiac"),
    t: int = 0, horizon: float = 180.0,
) -> Scenario:
    """H1 at (5, 5) with a capable doctor and nurse; INC-01 at (7, 5)."""
    return f.make_scenario(
        hospitals=[f.hospital("H1", caps=caps, er_capacity=4)],
        ambulances=ambulances if ambulances is not None else [f.ambulance("AMB-01", "ALS")],
        staff=[f.doctor("DR-1", "H1", ("emergency", "trauma", "cardiac")), f.nurse("RN-1", "H1")],
        responders=responders or [],
        incidents=[f.incident("INC-01", t, 7.0, 5.0, victims, responders=required)],
        horizon=horizon,
    )


SCENE = (7.0, 5.0)
BASE = (5.0, 5.0)


def amb_eta(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Ambulance drive time (no turnout)."""
    return geo.travel_time(a, b, "ambulance")


# --- dispatch and arrival ----------------------------------------------------


def test_first_decision_point_is_the_report() -> None:
    engine, sink = make_engine(f.tiny_scenario())

    point = engine.advance()

    assert point is not None and point.t_min == 0.0
    assert [t.kind for t in point.triggers] == [TriggerKind.NEW_INCIDENT]
    assert incident_of(engine, "INC-01").victims == ()
    assert events_of(sink, SimEventType.INCIDENT_REPORTED)[0].incident_id == "INC-01"


def test_ambulance_arrival_is_turnout_plus_travel() -> None:
    engine, _ = make_engine(f.tiny_scenario())
    engine.advance()
    dispatch(engine, "AMB-01", "INC-01")

    point = advance_until(engine, TriggerKind.SCENE_TRIAGED)

    assert point.t_min == round(1.0 + amb_eta(BASE, SCENE), 1)
    assert unit_of(engine, "AMB-01").status is UnitStatus.LOADING


def test_responder_arrival_uses_responder_turnout_and_speed() -> None:
    scenario = one_incident_world(
        [f.victim("V1", "GREEN")], ambulances=[], responders=[f.responder("FR-1", ("scene_control",), 1.0, 1.0)],
        required=("scene_control",),
    )
    engine, sink = make_engine(scenario)
    engine.advance()
    dispatch(engine, "FR-1", "INC-01")

    point = advance_until(engine, TriggerKind.SCENE_TRIAGED)

    assert point.t_min == round(2.0 + geo.travel_time((1.0, 1.0), SCENE, "responder"), 1)
    assert events_of(sink, SimEventType.SCENE_TRIAGED)[0].unit_id == "FR-1"


def test_triage_reveals_current_triage_to_the_view() -> None:
    engine, _ = make_engine(f.tiny_scenario())
    engine.advance()
    dispatch(engine, "AMB-02", "INC-01")

    point = advance_until(engine, TriggerKind.SCENE_TRIAGED)

    trigger = next(t for t in point.triggers if t.kind is TriggerKind.SCENE_TRIAGED)
    assert trigger.victim_ids == ("INC-01-V1", "INC-01-V2")
    view = incident_of(engine, "INC-01")
    assert view.triaged and [v.triage for v in view.victims] == [Triage.RED, Triage.GREEN]


# --- loading rule ------------------------------------------------------------


def test_als_takes_one_red_plus_one_non_red() -> None:
    engine, sink = make_engine(f.tiny_scenario())
    engine.advance()
    dispatch(engine, "AMB-01", "INC-01")

    point = advance_until(engine, TriggerKind.PATIENTS_LOADED)

    arrival = round(1.0 + amb_eta(BASE, SCENE), 1)
    assert point.t_min == round(arrival + 3.0, 1)
    assert unit_of(engine, "AMB-01").load_victim_ids == ("INC-01-V1", "INC-01-V2")
    assert unit_of(engine, "AMB-01").status is UnitStatus.AWAITING_DESTINATION
    assert incident_of(engine, "INC-01").status is IncidentStatus.CLEARED
    assert events_of(sink, SimEventType.INCIDENT_CLEARED)[0].t_min == point.t_min


def test_als_never_takes_two_reds() -> None:
    victims = [f.victim("V1", "RED", "trauma"), f.victim("V2", "RED", "trauma"), f.victim("V3", "YELLOW")]
    engine, _ = make_engine(one_incident_world(victims))
    engine.advance()
    dispatch(engine, "AMB-01", "INC-01")

    advance_until(engine, TriggerKind.PATIENTS_LOADED)

    assert unit_of(engine, "AMB-01").load_victim_ids == ("V1", "V3")
    assert victim_of(engine, "V2").status is VictimStatus.WAITING


def test_bls_refuses_red_and_clears_immediately() -> None:
    victims = [f.victim("V1", "RED", "trauma")]
    engine, sink = make_engine(one_incident_world(victims, ambulances=[f.ambulance("AMB-01", "BLS")]))
    engine.advance()
    dispatch(engine, "AMB-01", "INC-01")

    point = engine.advance()

    assert point is not None
    kinds = {t.kind for t in point.triggers}
    assert kinds == {TriggerKind.SCENE_TRIAGED, TriggerKind.UNIT_FREED}
    unit = unit_of(engine, "AMB-01")
    assert unit.status is UnitStatus.AVAILABLE and (unit.x, unit.y) == SCENE and unit.load_victim_ids == ()
    assert victim_of(engine, "V1").status is VictimStatus.WAITING
    assert events_of(sink, SimEventType.LOADING_STARTED) == []


def test_bls_takes_non_red_and_leaves_red() -> None:
    victims = [f.victim("V1", "RED", "trauma"), f.victim("V2", "YELLOW"), f.victim("V3", "GREEN")]
    engine, _ = make_engine(one_incident_world(victims, ambulances=[f.ambulance("AMB-01", "BLS")]))
    engine.advance()
    dispatch(engine, "AMB-01", "INC-01")

    advance_until(engine, TriggerKind.PATIENTS_LOADED)

    assert unit_of(engine, "AMB-01").load_victim_ids == ("V2", "V3")
    assert incident_of(engine, "INC-01").status is IncidentStatus.OPEN


def test_mixed_load_only_with_a_common_capable_hospital() -> None:
    victims = [f.victim("V1", "RED", "trauma"), f.victim("V2", "YELLOW", "pediatric")]
    split = f.make_scenario(
        hospitals=[f.hospital("H1", caps=("trauma",)), f.hospital("H3", 9.0, 9.0, caps=("pediatric",))],
        ambulances=[f.ambulance("AMB-01", "ALS")],
        incidents=[f.incident("INC-01", 0, 7.0, 5.0, victims)],
    )
    joint = split.model_copy(update={"hospitals": (f.hospital("H1", caps=("trauma", "pediatric")),)})
    loads = []
    for scenario in (split, joint):
        engine, _ = make_engine(scenario)
        engine.advance()
        dispatch(engine, "AMB-01", "INC-01")
        advance_until(engine, TriggerKind.PATIENTS_LOADED)
        loads.append(unit_of(engine, "AMB-01").load_victim_ids)

    assert loads == [("V1",), ("V1", "V2")]


def test_black_victim_is_expectant_and_never_loaded() -> None:
    victims = [f.victim("V1", "BLACK", "trauma"), f.victim("V2", "GREEN")]
    engine, _ = make_engine(one_incident_world(victims))
    engine.advance()
    dispatch(engine, "AMB-01", "INC-01")

    advance_until(engine, TriggerKind.PATIENTS_LOADED)

    assert unit_of(engine, "AMB-01").load_victim_ids == ("V2",)
    assert victim_of(engine, "V1").status is VictimStatus.EXPECTANT
    assert incident_of(engine, "INC-01").status is IncidentStatus.CLEARED


def test_all_black_incident_clears_at_triage_and_frees_the_unit() -> None:
    engine, _ = make_engine(one_incident_world([f.victim("V1", "BLACK", "trauma")]))
    engine.advance()
    dispatch(engine, "AMB-01", "INC-01")

    point = advance_until(engine, TriggerKind.SCENE_TRIAGED)

    assert TriggerKind.UNIT_FREED in {t.kind for t in point.triggers}
    assert incident_of(engine, "INC-01").status is IncidentStatus.CLEARED
    run_to_end(engine)
    outcome = engine.outcomes().victims[0]
    assert outcome.deadline_min is None and outcome.loaded_min is None


# --- field care --------------------------------------------------------------


@pytest.mark.parametrize(("unit_type", "field_care"), [("ALS", True), ("BLS", False)])
def test_cardiac_arrest_field_care_only_at_als_arrival(unit_type: str, field_care: bool) -> None:
    victims = [f.victim("V1", "RED", "cardiac", condition="cardiac arrest", window=10)]
    engine, sink = make_engine(one_incident_world(victims, ambulances=[f.ambulance("AMB-01", unit_type)]))
    engine.advance()
    dispatch(engine, "AMB-01", "INC-01")
    point = advance_until(engine, TriggerKind.SCENE_TRIAGED)
    run_to_end(engine)

    outcome = engine.outcomes().victims[0]
    if field_care:
        assert (outcome.care_start_min, outcome.care_mode) == (point.t_min, "field")
        assert outcome.als_on_scene_min == point.t_min
        assert events_of(sink, SimEventType.FIELD_CARE_STARTED)[0].victim_ids == ("V1",)
    else:
        assert outcome.care_start_min is None and outcome.als_on_scene_min is None


def test_field_care_is_not_overwritten_by_er_treatment() -> None:
    victims = [f.victim("V1", "RED", "cardiac", condition="cardiac arrest", window=10)]
    engine, _ = make_engine(one_incident_world(victims))
    engine.advance()
    dispatch(engine, "AMB-01", "INC-01")
    advance_until(engine, TriggerKind.PATIENTS_LOADED)
    transport(engine, "AMB-01", "H1")
    run_to_end(engine)

    outcome = engine.outcomes().victims[0]
    assert outcome.care_mode == "field"
    assert outcome.treatment_start_min is not None and outcome.care_start_min < outcome.treatment_start_min


# --- full trip ---------------------------------------------------------------


def test_full_trip_timeline() -> None:
    engine, sink = make_engine(f.tiny_scenario())
    engine.advance()
    dispatch(engine, "AMB-01", "INC-01")
    loaded = advance_until(engine, TriggerKind.PATIENTS_LOADED).t_min
    transport(engine, "AMB-01", "H1", reserve=True)

    freed = advance_until(engine, TriggerKind.UNIT_FREED)
    run_to_end(engine)

    arrival = round(loaded + amb_eta(SCENE, BASE), 1)
    assert freed.t_min == round(arrival + 5.0, 1)
    assert (unit_of(engine, "AMB-01").x, unit_of(engine, "AMB-01").y) == BASE
    red, green = engine.outcomes().victims
    assert (red.arrived_hospital_min, red.bedded_min, red.treatment_start_min) == (arrival, arrival, arrival)
    assert (red.care_start_min, red.care_mode, red.hospital_id) == (arrival, "hospital", "H1")
    assert green.treatment_start_min == arrival
    ends = {r.victim_ids[0]: r.t_min for r in events_of(sink, SimEventType.TREATMENT_ENDED)}
    assert ends == {"INC-01-V1": round(arrival + 60.0, 1), "INC-01-V2": round(arrival + 20.0, 1)}


def test_transport_position_moves_towards_hospital() -> None:
    scenario = f.make_scenario(
        hospitals=[f.hospital("H1", 1.0, 1.0, caps=("general",))],
        ambulances=[f.ambulance("AMB-01", "ALS", 9.0, 1.0)],
        staff=[f.doctor("DR-1", "H1"), f.nurse("RN-1", "H1")],
        incidents=[
            f.incident("INC-01", 0, 9.0, 1.0, [f.victim("V1", "GREEN")]),
            f.incident("INC-02", 8, 15.0, 15.0, [f.victim("V2", "GREEN")]),
        ],
    )
    engine, _ = make_engine(scenario)
    engine.advance()
    dispatch(engine, "AMB-01", "INC-01")
    loaded = advance_until(engine, TriggerKind.PATIENTS_LOADED).t_min
    transport(engine, "AMB-01", "H1")
    eta = amb_eta((9.0, 1.0), (1.0, 1.0))
    advance_until(engine, TriggerKind.NEW_INCIDENT)

    unit = unit_of(engine, "AMB-01")
    expected_x = 9.0 - 8.0 * (8.0 - loaded) / eta
    assert unit.status is UnitStatus.TRANSPORTING and unit.destination_id == "H1"
    assert unit.x == pytest.approx(expected_x, abs=1e-3) and unit.y == 1.0


# --- divert ------------------------------------------------------------------


def _divert_world() -> Scenario:
    return f.make_scenario(
        hospitals=[f.hospital("H1", 1.0, 1.0, caps=("general",))],
        ambulances=[f.ambulance("AMB-01", "ALS", 1.0, 1.0)],
        staff=[f.doctor("DR-1", "H1"), f.nurse("RN-1", "H1")],
        incidents=[
            f.incident("INC-01", 0, 15.0, 1.0, [f.victim("V1", "GREEN")]),
            f.incident("INC-02", 6, 7.0, 4.0, [f.victim("V2", "RED")]),
        ],
    )


def test_divert_interpolates_position_after_turnout() -> None:
    engine, sink = make_engine(_divert_world())
    engine.advance()
    dispatch(engine, "AMB-01", "INC-01")
    first_eta = round(1.0 + amb_eta((1.0, 1.0), (15.0, 1.0)), 1)
    point = engine.advance()
    assert point is not None and point.t_min == 6.0

    assert engine.apply(DivertUnit(unit_id="AMB-01", incident_id="INC-02"), ACTOR).ok
    x = 1.0 + 14.0 * (6.0 - 1.0) / (first_eta - 1.0)
    arrival = advance_until(engine, TriggerKind.SCENE_TRIAGED)

    assert arrival.t_min == round(6.0 + amb_eta((round(x, 3), 1.0), (7.0, 4.0)), 1)
    assert incident_of(engine, "INC-01").triaged is False
    assert "from INC-01" in events_of(sink, SimEventType.UNIT_DIVERTED)[0].detail
    assert len(events_of(sink, SimEventType.UNIT_ARRIVED_SCENE)) == 1


def test_divert_during_turnout_keeps_the_remaining_turnout() -> None:
    scenario = _divert_world().model_copy(
        update={
            "incidents": (
                f.incident("INC-01", 0, 15.0, 1.0, [f.victim("V1", "GREEN")]),
                f.incident("INC-02", 0, 7.0, 4.0, [f.victim("V2", "RED")]),
            )
        }
    )
    engine, _ = make_engine(scenario)
    engine.advance()
    dispatch(engine, "AMB-01", "INC-01")
    assert engine.apply(DivertUnit(unit_id="AMB-01", incident_id="INC-02"), ACTOR).ok

    point = advance_until(engine, TriggerKind.SCENE_TRIAGED)

    assert point.t_min == round(1.0 + amb_eta((1.0, 1.0), (7.0, 4.0)), 1)


# --- responders and access ---------------------------------------------------


def _access_world(required: tuple[str, ...], team_caps: tuple[str, ...]) -> Scenario:
    return one_incident_world(
        [f.victim("V1", "YELLOW")], required=required,
        responders=[f.responder("FR-1", team_caps, 7.0, 9.0)],
    )


def test_heavy_rescue_gates_access_for_ten_minutes() -> None:
    engine, sink = make_engine(_access_world(("extrication", "scene_control"), ("extrication", "scene_control")))
    engine.advance()
    dispatch(engine, "AMB-01", "INC-01")
    dispatch(engine, "FR-1", "INC-01")
    advance_until(engine, TriggerKind.SCENE_TRIAGED)
    assert unit_of(engine, "AMB-01").status is UnitStatus.ON_SCENE
    assert incident_of(engine, "INC-01").accessible is False

    loaded = advance_until(engine, TriggerKind.PATIENTS_LOADED)

    team_arrival = round(2.0 + geo.travel_time((7.0, 9.0), SCENE, "responder"), 1)
    assert events_of(sink, SimEventType.ACCESS_GRANTED)[0].t_min == round(team_arrival + 10.0, 1)
    assert loaded.t_min == round(team_arrival + 13.0, 1)
    assert incident_of(engine, "INC-01").responder_capabilities_present == ("extrication", "scene_control")


def test_scene_control_gates_access_without_delay() -> None:
    engine, sink = make_engine(_access_world(("scene_control",), ("scene_control",)))
    engine.advance()
    dispatch(engine, "AMB-01", "INC-01")
    dispatch(engine, "FR-1", "INC-01")

    loaded = advance_until(engine, TriggerKind.PATIENTS_LOADED)

    team_arrival = round(2.0 + geo.travel_time((7.0, 9.0), SCENE, "responder"), 1)
    assert events_of(sink, SimEventType.ACCESS_GRANTED)[0].t_min == team_arrival
    assert loaded.t_min == round(team_arrival + 3.0, 1)


def test_hazmat_covers_decontamination() -> None:
    engine, sink = make_engine(_access_world(("hazmat", "decontamination"), ("hazmat",)))
    engine.advance()
    dispatch(engine, "FR-1", "INC-01")
    advance_until(engine, TriggerKind.SCENE_TRIAGED)
    run_to_end(engine)

    assert len(events_of(sink, SimEventType.ACCESS_GRANTED)) == 1


def test_missing_capability_keeps_ambulance_staged() -> None:
    engine, sink = make_engine(_access_world(("extrication", "scene_control"), ("scene_control",)))
    engine.advance()
    dispatch(engine, "AMB-01", "INC-01")
    dispatch(engine, "FR-1", "INC-01")
    run_to_end(engine)

    assert events_of(sink, SimEventType.ACCESS_GRANTED) == []
    assert unit_of(engine, "AMB-01").status is UnitStatus.ON_SCENE
    assert set(incident_of(engine, "INC-01").assigned_unit_ids) == {"AMB-01", "FR-1"}


def test_responders_are_freed_when_the_incident_clears() -> None:
    engine, _ = make_engine(_access_world(("scene_control",), ("scene_control",)))
    engine.advance()
    dispatch(engine, "AMB-01", "INC-01")
    dispatch(engine, "FR-1", "INC-01")

    point = advance_until(engine, TriggerKind.PATIENTS_LOADED)

    assert TriggerKind.UNIT_FREED in {t.kind for t in point.triggers}
    team = unit_of(engine, "FR-1")
    assert team.status is UnitStatus.AVAILABLE and (team.x, team.y) == SCENE


def test_unit_arriving_at_cleared_incident_is_freed() -> None:
    victims = [f.victim("V1", "GREEN")]
    ambulances = [f.ambulance("AMB-01", "ALS"), f.ambulance("AMB-02", "BLS", 1.0, 1.0)]
    engine, sink = make_engine(one_incident_world(victims, ambulances=ambulances))
    engine.advance()
    dispatch(engine, "AMB-01", "INC-01")
    dispatch(engine, "AMB-02", "INC-01")

    advance_until(engine, TriggerKind.UNIT_FREED, lambda p: any(t.unit_id == "AMB-02" for t in p.triggers))

    assert unit_of(engine, "AMB-02").status is UnitStatus.AVAILABLE
    assert any("already cleared" in r.detail for r in events_of(sink, SimEventType.UNIT_AVAILABLE))


# --- t=0 status, guard, horizon ---------------------------------------------


def test_busy_t0_unit_is_freed_at_busy_until() -> None:
    scenario = one_incident_world(
        [f.victim("V1", "GREEN")], ambulances=[f.ambulance("AMB-01", "BLS", status="busy", busy_until=12)]
    )
    engine, _ = make_engine(scenario)
    engine.advance()
    assert unit_of(engine, "AMB-01").available_at_min == 12.0
    assert not engine.check(DispatchUnit(unit_id="AMB-01", incident_id="INC-01")).ok

    point = advance_until(engine, TriggerKind.UNIT_FREED)

    assert point.t_min == 12.0 and unit_of(engine, "AMB-01").status is UnitStatus.AVAILABLE


def test_liveness_guard_sends_unassigned_units_to_nearest_capable_hospital() -> None:
    scenario = one_incident_world([f.victim("V1", "RED", "trauma")]).model_copy(
        update={
            "hospitals": (
                f.hospital("H1", caps=("general",)),
                f.hospital("H2", 9.0, 5.0, caps=("trauma",)),
                f.hospital("H3", 12.0, 5.0, caps=("trauma",)),
            )
        }
    )
    engine, sink = make_engine(scenario)
    engine.advance()
    dispatch(engine, "AMB-01", "INC-01")
    advance_until(engine, TriggerKind.PATIENTS_LOADED)

    engine.advance()

    omission = events_of(sink, SimEventType.CONTROLLER_OMISSION)
    guard = [r for r in sink.records if isinstance(r, ActionRecord) and r.actor == GUARD_ACTOR]
    assert len(omission) == 1 and omission[0].unit_id == "AMB-01"
    assert len(guard) == 1 and guard[0].ok
    assert guard[0].action == AssignTransport(unit_id="AMB-01", hospital_id="H2", reserve_beds=False)


def test_horizon_accounting() -> None:
    scenario = f.make_scenario(
        hospitals=[f.hospital("H1", caps=("general",))],
        ambulances=[f.ambulance("AMB-01", "ALS")],
        staff=[f.doctor("DR-1", "H1"), f.nurse("RN-1", "H1")],
        incidents=[
            f.incident("INC-01", 0, 7.0, 5.0, [f.victim("V1", "GREEN")]),
            f.incident("INC-02", 25, 19.0, 19.0, [f.victim("V2", "YELLOW", deteriorates=10)]),
            f.incident("INC-03", 45, 7.0, 5.0, [f.victim("V3", "GREEN")]),
        ],
        horizon=30.0,
    )
    engine, sink = make_engine(scenario)
    engine.advance()
    dispatch(engine, "AMB-01", "INC-01")
    advance_until(engine, TriggerKind.NEW_INCIDENT)
    dispatch(engine, "AMB-01", "INC-02")  # arrives after the horizon
    run_to_end(engine)

    snapshots = [r.t_min for r in sink.records if isinstance(r, SnapshotRecord)]
    assert snapshots == [float(k) for k in range(31)]
    assert engine.now == 30.0 and engine.advance() is None
    assert max(r.t_min for r in sink.records) == 30.0
    outcomes = engine.outcomes()
    by_id = {v.victim_id: v for v in outcomes.victims}
    assert by_id["V1"].care_start_min is not None and by_id["V2"].care_start_min is None
    assert by_id["V2"].deteriorated_at_min is None
    assert [i.incident_id for i in engine.view().incidents] == ["INC-01", "INC-02"]
    busy = dict(outcomes.ambulance_busy_min)["AMB-01"]
    assert 0.0 < busy <= 30.0


def test_busy_minutes_count_busy_t0_and_trips() -> None:
    scenario = one_incident_world(
        [f.victim("V1", "GREEN")],
        ambulances=[f.ambulance("AMB-01", "ALS"), f.ambulance("AMB-02", "BLS", status="busy", busy_until=12),
                    f.ambulance("AMB-03", "BLS", status="out_of_service")],
    )
    engine, _ = make_engine(scenario)
    engine.advance()
    dispatch(engine, "AMB-01", "INC-01")
    advance_until(engine, TriggerKind.PATIENTS_LOADED)
    transport(engine, "AMB-01", "H1")
    freed = advance_until(engine, TriggerKind.UNIT_FREED, lambda p: any(t.unit_id == "AMB-01" for t in p.triggers))
    run_to_end(engine)

    outcomes = engine.outcomes()
    assert dict(outcomes.ambulance_busy_min) == {"AMB-01": freed.t_min, "AMB-02": 12.0, "AMB-03": 0.0}
    assert outcomes.ambulances_in_service == 2


def test_outcomes_before_the_end_and_apply_after_it_raise() -> None:
    engine, _ = make_engine(f.tiny_scenario(horizon=10.0))
    engine.advance()
    with pytest.raises(EngineError):
        engine.outcomes()
    run_to_end(engine)
    with pytest.raises(EngineError):
        engine.apply(DispatchUnit(unit_id="AMB-01", incident_id="INC-01"), ACTOR)
