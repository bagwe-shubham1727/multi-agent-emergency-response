"""Tests of the shared need accounting (``control/needs.py``)."""

from emergency_sim import geo
from emergency_sim.control.needs import (
    capable_hospitals,
    has_unmet_need,
    incident_needs,
    seats_short,
    unit_victims,
)
from emergency_sim.domain.agent_io import PriorityReport
from emergency_sim.domain.enums import Capability, IncidentStatus, ResponderCapability, UnitStatus, VictimStatus
from emergency_sim.policies.params import PolicyParams
from tests.control_fakes import hospital_view, incident_view, public_view, unit_view, victim_view

PARAMS = PolicyParams()
WAITING = VictimStatus.WAITING


def _report(iid: str = "INC-01", **kw: object) -> PriorityReport:
    values: dict[str, object] = dict(
        incident_id=iid, urgency=85, suspected_red=True, suspected_cardiac_arrest=False,
        est_patients=4, needed_capabilities=[Capability.TRAUMA], als_units_needed=1,
        deadline_in_min=30.0, rationale="r",
    )
    values.update(kw)
    return PriorityReport.model_validate(values)


def test_seats_and_als_count_scene_bound_units_only() -> None:
    victims = [
        victim_view("V1", triage="RED", status=WAITING),
        victim_view("V2", triage="RED", status=WAITING),
        victim_view("V3", triage="GREEN", status=VictimStatus.LOADING, unit_id="A3"),
    ]
    view = public_view(
        incidents=[incident_view(victims=victims)],
        units=[
            unit_view("A1", status=UnitStatus.EN_ROUTE_SCENE, incident_id="INC-01"),
            unit_view("A2", unit_type="BLS", status=UnitStatus.ON_SCENE, incident_id="INC-01"),
            unit_view("A3", status=UnitStatus.LOADING, incident_id="INC-01"),
            unit_view("A4", status=UnitStatus.AVAILABLE),
        ],
    )
    (need,) = incident_needs(view, None, PARAMS)
    assert need.seats_en_route == 4  # A1 + A2; loading A3 excluded
    assert need.ambulances_assigned == 3
    assert need.als_en_route == 1
    assert need.red_waiting == 2 and need.als_needed == 2
    assert need.est_patients_remaining == 2
    assert seats_short(need, PARAMS) == 0
    assert has_unmet_need(need, PARAMS)  # one RED still lacks an ALS


def test_pre_triage_uses_board_and_b0_defaults() -> None:
    view = public_view(t=12.0, incidents=[incident_view(reported=5, t=2.0)])
    (b0,) = incident_needs(view, None, PARAMS)
    assert b0.urgency == 0 and b0.als_needed == 0 and not b0.suspected_red
    assert b0.est_patients_remaining == 5
    assert b0.deadline_in_min == PARAMS.nominal_yellow_deadline_min - 10.0
    (agent,) = incident_needs(view, {"INC-01": _report()}, PARAMS)
    assert agent.urgency == 85 and agent.suspected_red and agent.als_needed == 1
    assert agent.est_patients_remaining == 4 and agent.deadline_in_min == 30.0


def test_pre_triage_cap_stops_seat_demand() -> None:
    units = [unit_view(f"A{i}", status=UnitStatus.EN_ROUTE_SCENE, incident_id="INC-01") for i in range(3)]
    view = public_view(incidents=[incident_view(reported=20)], units=units)
    (need,) = incident_needs(view, None, PARAMS)
    assert need.est_patients_remaining - need.seats_en_route > 0
    assert seats_short(need, PARAMS) == 0
    assert not has_unmet_need(need, PARAMS)


def test_responder_caps_missing_counts_present_en_route_and_hazmat_decon() -> None:
    incident = incident_view(
        required=("extrication", "hazmat", "decontamination", "scene_control"), present=("scene_control",)
    )
    team = unit_view("R1", caps=("hazmat",), status=UnitStatus.EN_ROUTE_SCENE, incident_id="INC-01")
    (need,) = incident_needs(public_view(incidents=[incident], units=[team]), None, PARAMS)
    assert need.responder_caps_missing == [ResponderCapability.EXTRICATION]
    accessible = incident.model_copy(update={"accessible": True})
    (need,) = incident_needs(public_view(incidents=[accessible]), None, PARAMS)
    assert need.responder_caps_missing == []


def test_only_open_incidents_sorted_by_id() -> None:
    view = public_view(incidents=[
        incident_view("INC-02"), incident_view("INC-01"),
        incident_view("INC-03", status=IncidentStatus.CLEARED),
    ])
    assert [n.incident_id for n in incident_needs(view, None, PARAMS)] == ["INC-01", "INC-02"]


def test_unit_victims_and_capable_hospitals() -> None:
    victims = [victim_view("V1", cap="trauma", unit_id="A1"), victim_view("V2", unit_id="A1")]
    view = public_view(
        incidents=[incident_view(victims=victims, status=IncidentStatus.CLEARED)],
        units=[unit_view("A1", status=UnitStatus.AWAITING_DESTINATION, load=("V2", "V1"))],
        hospitals=[hospital_view("H1"), hospital_view("H2", caps=("general", "trauma"))],
    )
    loaded = unit_victims(view, view.units[0])
    assert [v.victim_id for v in loaded] == ["V2", "V1"]
    assert [h.hospital_id for h in capable_hospitals(view, loaded)] == ["H2"]


# --- Access estimate (regression: dispatch ignored scene access) -----------------


def test_access_wait_is_zero_when_accessible() -> None:
    (need,) = incident_needs(public_view(incidents=[incident_view()]), None, PARAMS)
    assert need.access_wait_min == 0.0


def test_access_wait_unknown_while_a_required_team_is_not_en_route() -> None:
    incident = incident_view(required=("extrication", "scene_control"), present=("scene_control",))
    (need,) = incident_needs(public_view(incidents=[incident]), None, PARAMS)
    assert need.access_wait_min is None


def test_access_wait_is_last_team_arrival_plus_heavy_delay() -> None:
    incident = incident_view(x=5.0, y=5.0, required=("extrication", "scene_control"))
    near = unit_view("R1", caps=("scene_control",), x=5.0, y=6.0,
                     status=UnitStatus.EN_ROUTE_SCENE, incident_id="INC-01")
    far = unit_view("R2", caps=("extrication",), x=5.0, y=13.0,
                    status=UnitStatus.EN_ROUTE_SCENE, incident_id="INC-01")
    (need,) = incident_needs(public_view(incidents=[incident], units=[near, far]), None, PARAMS)
    last = geo.travel_time((5.0, 13.0), (5.0, 5.0), "responder")
    assert need.access_wait_min == round(last + PARAMS.heavy_access_min, 1)


def test_access_wait_without_heavy_capability_has_no_delay_and_hazmat_covers_decon() -> None:
    incident = incident_view(required=("hazmat", "decontamination", "scene_control"), present=("scene_control",))
    team = unit_view("R1", caps=("hazmat",), x=5.0, y=7.0, status=UnitStatus.EN_ROUTE_SCENE, incident_id="INC-01")
    (need,) = incident_needs(public_view(incidents=[incident], units=[team]), None, PARAMS)
    assert need.access_wait_min == round(geo.travel_time((5.0, 7.0), (5.0, 5.0), "responder")
                                         + PARAMS.heavy_access_min, 1)
    police_only = incident_view(required=("scene_control",))
    car = unit_view("R2", caps=("scene_control",), x=5.0, y=7.0, status=UnitStatus.EN_ROUTE_SCENE,
                    incident_id="INC-01")
    (need,) = incident_needs(public_view(incidents=[police_only], units=[car]), None, PARAMS)
    assert need.access_wait_min == geo.travel_time((5.0, 7.0), (5.0, 5.0), "responder")


def test_access_wait_with_all_teams_on_scene_is_the_heavy_delay_bound() -> None:
    incident = incident_view(required=("extrication",), present=("extrication",), accessible=False)
    (need,) = incident_needs(public_view(incidents=[incident]), None, PARAMS)
    assert need.access_wait_min == PARAMS.heavy_access_min
