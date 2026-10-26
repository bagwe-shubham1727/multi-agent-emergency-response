"""Tests of the B0 and B0-cues greedy baselines (``control/greedy.py``)."""

from emergency_sim.control.greedy import B0_ACTOR, B0_CUES_ACTOR, GreedyController
from emergency_sim.domain.actions import AssignTransport, DispatchUnit, DivertUnit, LendStaff
from emergency_sim.domain.enums import IncidentStatus, TriggerKind, UnitStatus, VictimStatus
from emergency_sim.domain.ports import Controller
from emergency_sim.domain.views import DecisionPoint, PublicView, Trigger
from emergency_sim.policies.params import PolicyParams
from tests.control_fakes import (
    FakePort,
    hospital_view,
    incident_view,
    public_view,
    unit_view,
    victim_view,
)

POINT = DecisionPoint(t_min=10.0, triggers=(Trigger(kind=TriggerKind.NEW_INCIDENT, incident_id="INC-01"),))
CPR = "He collapsed and is not breathing, we are doing CPR."


def _run(view: PublicView) -> FakePort:
    port = FakePort(view)
    GreedyController(PolicyParams()).handle(POINT, port)
    return port


def _dispatches(port: FakePort) -> list[tuple[str, str]]:
    return [(a.unit_id, a.incident_id) for a in port.applied_actions(DispatchUnit)]


def test_is_a_controller_and_acts_only_as_b0() -> None:
    assert isinstance(GreedyController(PolicyParams()), Controller)
    view = public_view(
        incidents=[incident_view(reported=2)],
        units=[unit_view("A1", x=6.0), unit_view("A2", x=9.0)],
    )
    port = _run(view)
    assert port.attempts and all(actor == B0_ACTOR for actor, _, _ in port.attempts)


def test_fcfs_order_ignores_severity_cues() -> None:
    view = public_view(
        incidents=[
            incident_view("INC-01", t=1.0, x=5.0, report="Twisted ankle."),
            incident_view("INC-02", t=2.0, x=5.5, report=CPR, incident_type="cardiac_event"),
        ],
        units=[unit_view("A1", x=5.4)],
    )
    assert _dispatches(_run(view)) == [("A1", "INC-01")]


def test_nearest_unit_and_pre_triage_count() -> None:
    units = [unit_view(f"A{i}", x=float(i), unit_type="BLS" if i % 2 else "ALS") for i in range(1, 6)]
    view = public_view(incidents=[incident_view(x=0.0, reported=7)], units=units)
    assert _dispatches(_run(view)) == [("A1", "INC-01"), ("A2", "INC-01"), ("A3", "INC-01")]
    view = public_view(incidents=[incident_view(x=0.0, reported=3)], units=units)
    assert _dispatches(_run(view)) == [("A1", "INC-01"), ("A2", "INC-01")]


def test_after_triage_als_per_red_then_seats() -> None:
    victims = [
        victim_view("V1", triage="RED", status=VictimStatus.WAITING),
        victim_view("V2", triage="GREEN", status=VictimStatus.WAITING),
        victim_view("V3", triage="GREEN", status=VictimStatus.WAITING),
    ]
    view = public_view(
        incidents=[incident_view(x=0.0, victims=victims)],
        units=[unit_view("B1", unit_type="BLS", x=1.0), unit_view("B2", unit_type="BLS", x=2.0),
               unit_view("A9", x=8.0)],
    )
    assert _dispatches(_run(view)) == [("A9", "INC-01"), ("B1", "INC-01")]


def test_red_only_scene_gets_no_bls() -> None:
    victims = [victim_view(f"V{i}", triage="RED", status=VictimStatus.WAITING) for i in (1, 2, 3)]
    view = public_view(
        incidents=[incident_view(x=0.0, victims=victims)],
        units=[unit_view("B1", unit_type="BLS", x=1.0), unit_view("A1", x=3.0)],
    )
    assert _dispatches(_run(view)) == [("A1", "INC-01")]


def test_responders_nearest_team_per_missing_capability() -> None:
    view = public_view(
        incidents=[incident_view(x=0.0, required=("extrication", "scene_control"), reported=0)],
        units=[
            unit_view("R1", caps=("extrication",), x=9.0),
            unit_view("R2", caps=("extrication", "scene_control"), x=3.0),
            unit_view("R3", caps=("scene_control",), x=1.0),
        ],
    )
    assert _dispatches(_run(view)) == [("R2", "INC-01")]


def _loaded_world(*hospitals: object, status: UnitStatus = UnitStatus.AWAITING_DESTINATION) -> PublicView:
    victims = [victim_view("V1", cap="trauma", unit_id="A1")]
    return public_view(
        incidents=[incident_view(victims=victims, status=IncidentStatus.CLEARED, x=0.0)],
        units=[unit_view("A1", x=0.0, status=status, load=("V1",), hospital_id="H1")],
        hospitals=hospitals,  # type: ignore[arg-type]
    )


def test_loaded_unit_goes_to_nearest_capable_with_free_bed_unreserved() -> None:
    view = _loaded_world(
        hospital_view("H1", x=1.0, caps=("general", "trauma"), free=0),
        hospital_view("H2", x=2.0, caps=("general",)),
        hospital_view("H3", x=6.0, caps=("trauma",)),
    )
    (transport,) = _run(view).applied_actions(AssignTransport)
    assert transport == AssignTransport(unit_id="A1", hospital_id="H3", reserve_beds=False)


def test_loaded_unit_falls_back_to_nearest_capable_when_all_full() -> None:
    view = _loaded_world(
        hospital_view("H1", x=1.0, caps=("trauma",), free=0),
        hospital_view("H3", x=6.0, caps=("trauma",), free=0),
    )
    (transport,) = _run(view).applied_actions(AssignTransport)
    assert transport.hospital_id == "H1" and not transport.reserve_beds


def test_full_arrival_redirects_to_next_hospital_with_beds_or_stays() -> None:
    hospitals = (hospital_view("H1", x=0.0, caps=("trauma",), free=0),
                 hospital_view("H2", x=4.0, caps=("trauma",)), hospital_view("H3", x=2.0, caps=("trauma",)))
    view = _loaded_world(*hospitals, status=UnitStatus.AT_HOSPITAL_FULL)
    (transport,) = _run(view).applied_actions(AssignTransport)
    assert transport == AssignTransport(unit_id="A1", hospital_id="H3", reserve_beds=False)
    full = tuple(h.model_copy(update={"beds_free_unreserved": 0}) for h in hospitals)
    assert _run(_loaded_world(*full, status=UnitStatus.AT_HOSPITAL_FULL)).attempts == []


def test_never_reserves_diverts_or_lends() -> None:
    victims = [victim_view("V1", triage="RED", status=VictimStatus.WAITING)]
    view = public_view(
        incidents=[incident_view("INC-01", x=0.0, victims=victims), incident_view("INC-02", x=9.0, reported=4)],
        units=[unit_view("A1", status=UnitStatus.EN_ROUTE_SCENE, incident_id="INC-02", x=1.0),
               unit_view("B1", unit_type="BLS", x=8.0)],
        hospitals=[hospital_view("H1")],
    )
    port = _run(view)
    assert not port.applied_actions(DivertUnit) and not port.applied_actions(LendStaff)
    assert all(not a.reserve_beds for a in port.applied_actions(AssignTransport))
    assert _dispatches(port) == [("B1", "INC-02")]


def _run_cues(view: PublicView) -> FakePort:
    port = FakePort(view)
    GreedyController(PolicyParams(), use_caller_cues=True).handle(POINT, port)
    return port


def test_b0_cues_serves_suspected_red_first_and_acts_as_b0_cues() -> None:
    view = public_view(
        incidents=[
            incident_view("INC-01", t=1.0, x=5.0, report="Twisted ankle."),
            incident_view("INC-02", t=2.0, x=5.5, report=CPR, incident_type="cardiac_event"),
        ],
        units=[unit_view("A1", x=5.4)],
    )
    port = _run_cues(view)
    assert _dispatches(port) == [("A1", "INC-02")]
    assert all(actor == B0_CUES_ACTOR for actor, _, _ in port.attempts)


def test_b0_cues_sends_nearest_als_first_to_suspected_red() -> None:
    units = [unit_view("B1", x=1.0, unit_type="BLS"), unit_view("A2", x=4.0, unit_type="ALS")]
    view = public_view(
        incidents=[incident_view(x=0.0, reported=1, report=CPR, incident_type="cardiac_event")],
        units=units,
    )
    assert _dispatches(_run_cues(view)) == [("A2", "INC-01")]
    assert _dispatches(_run(view)) == [("B1", "INC-01")]  # plain B0 ignores the cue


def test_b0_cues_never_reserves_diverts_or_lends() -> None:
    loaded = victim_view("V1", "INC-09", unit_id="A9")
    view = public_view(
        incidents=[
            incident_view(reported=2, report=CPR, incident_type="cardiac_event"),
            incident_view("INC-09", victims=[loaded], status=IncidentStatus.CLEARED),
        ],
        units=[unit_view("A1", x=6.0), unit_view("A2", x=9.0),
               unit_view("A9", status=UnitStatus.AWAITING_DESTINATION, load=("V1",))],
        hospitals=[hospital_view("H1")],
    )
    port = _run_cues(view)
    assert not port.applied_actions(DivertUnit) and not port.applied_actions(LendStaff)
    transports = port.applied_actions(AssignTransport)
    assert transports and all(not a.reserve_beds for a in transports)
