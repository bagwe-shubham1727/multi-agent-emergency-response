"""Tests of the shared pure estimates in ``policies/estimates.py``."""

import dataclasses
import math

import pytest

from emergency_sim import geo
from emergency_sim.domain.params import SimParams
from emergency_sim.domain.enums import (
    AmbulanceType,
    Capability,
    ResponderCapability,
    StaffRole,
    Triage,
    UnitClass,
    UnitStatus,
    VictimStatus,
)
from emergency_sim.policies import estimates
from emergency_sim.policies.params import PolicyParams
from tests.test_policies import (
    make_ambulance,
    make_board,
    make_dispatch_input,
    make_hospital_input,
    make_loaded,
    make_need,
    make_request,
    make_staff,
    make_team,
    make_victim,
)

PARAMS = PolicyParams()


def test_policy_params_are_the_fixed_a_priori_values() -> None:
    assert dataclasses.asdict(PARAMS) == {
        "red_weight": 3.0, "yellow_weight": 2.0, "green_weight": 1.0,
        "window_logistic_scale_min": 5.0, "travel_penalty_per_min": 0.01,
        "nominal_red_deadline_min": 40.0, "nominal_cardiac_arrest_deadline_min": 10.0,
        "nominal_yellow_deadline_min": 165.0, "nominal_green_deadline_min": 480.0,
        "deteriorated_red_deadline_min": 30.0, "scene_overhead_min": 5.0, "no_unit_wait_min": 20.0,
        "queue_wait_per_patient_min": 15.0, "max_ambulances_pre_triage": 3,
        "diversion_urgency_margin": 20, "diversion_max_eta_min": 15.0, "heavy_access_min": 10.0,
    }
    with pytest.raises(dataclasses.FrozenInstanceError):
        PARAMS.red_weight = 9.0  # type: ignore[misc]


def test_heavy_access_min_mirrors_the_public_engine_rule() -> None:
    assert PARAMS.heavy_access_min == SimParams().heavy_access_min


@pytest.mark.parametrize("unit_class", [UnitClass.AMBULANCE, UnitClass.RESPONDER])
@pytest.mark.parametrize("turnout", [True, False])
def test_eta_matches_engine_travel_model(unit_class: UnitClass, turnout: bool) -> None:
    a, b = (2.0, 3.0), (11.0, 9.5)
    assert estimates.eta_min(a, b, unit_class, turnout) == geo.travel_time(a, b, unit_class.value, turnout)
    assert estimates.eta_min(a, b, unit_class, True) > estimates.eta_min(a, b, unit_class, False)


def test_p_window_logistic_and_overflow_safe() -> None:
    assert estimates.p_window(0.0, PARAMS) == pytest.approx(0.5)
    assert estimates.p_window(5.0, PARAMS) == pytest.approx(1 / (1 + math.exp(-1)))
    assert estimates.p_window(1e6, PARAMS) == pytest.approx(1.0)
    assert estimates.p_window(-1e6, PARAMS) == pytest.approx(0.0)
    values = [estimates.p_window(s, PARAMS) for s in (-30, -5, 0, 5, 30)]
    assert values == sorted(values)


def test_utility_formula() -> None:
    expected = 3.0 * estimates.p_window(10.0, PARAMS) - 0.01 * 12.0
    assert estimates.utility(3.0, 10.0, 12.0, PARAMS) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("triage", "weight"),
    [(Triage.RED, 3.0), ("SUSPECTED_RED", 3.0), (Triage.YELLOW, 2.0), ("UNKNOWN", 2.0), (Triage.GREEN, 1.0), (Triage.BLACK, 0.0)],
)
def test_triage_weight(triage: estimates.TriageLike, weight: float) -> None:
    assert estimates.triage_weight(triage, PARAMS) == weight


def test_nominal_deadline() -> None:
    assert estimates.nominal_deadline(Triage.RED, "cardiac arrest", PARAMS) == 10.0
    assert estimates.nominal_deadline(Triage.RED, "stab wound", PARAMS) == 40.0
    assert estimates.nominal_deadline(Triage.RED, "hip fracture", PARAMS, deteriorated=True) == 30.0
    assert estimates.nominal_deadline(Triage.YELLOW, "hip fracture", PARAMS) == 165.0
    assert estimates.nominal_deadline(Triage.GREEN, "lacerations", PARAMS) == 480.0
    with pytest.raises(ValueError):
        estimates.nominal_deadline(Triage.BLACK, "lacerations", PARAMS)


def test_incident_type_profile() -> None:
    profile = estimates.incident_type_profile("cardiac_event")
    assert (profile.base_urgency, profile.needed_capabilities, profile.known) == (70, (Capability.CARDIAC,), True)
    unknown = estimates.incident_type_profile("alien_landing")
    assert (unknown.base_urgency, unknown.needed_capabilities, unknown.known) == (50, (Capability.GENERAL,), False)


def test_rank_units_filters_and_orders() -> None:
    units = [
        make_ambulance("AMB-03", "ALS", 6.0, 5.0), make_ambulance("AMB-01", "ALS", 4.0, 5.0),
        make_ambulance("AMB-02", "BLS", 5.0, 5.2), make_ambulance("AMB-04", "ALS", 5.0, 5.0, UnitStatus.EN_ROUTE_SCENE),
        make_team("FR-1", ("scene_control",), 5.0, 5.0),
    ]
    inp = make_dispatch_input([make_need("INC-01", 5.0, 5.0)], units)
    als = estimates.rank_units(inp, "INC-01", AmbulanceType.ALS)
    assert [r.unit_id for r in als] == ["AMB-01", "AMB-03"]  # equal ETA -> id order
    assert [r.unit_id for r in estimates.rank_units(inp, "INC-01", None)] == ["AMB-02", "AMB-01", "AMB-03"]
    assert als[0].eta_min == geo.travel_time((4.0, 5.0), (5.0, 5.0), "ambulance", True)
    with pytest.raises(KeyError):
        estimates.rank_units(inp, "INC-99", None)


def test_rank_responders() -> None:
    units = [make_team("FR-2", ("hazmat", "decontamination"), 9.0, 9.0), make_team("FR-1", ("hazmat",), 6.0, 5.0)]
    inp = make_dispatch_input([make_need("INC-01", 5.0, 5.0)], units)
    ranks = estimates.rank_responders(inp, "INC-01", ResponderCapability.HAZMAT)
    assert [r.unit_id for r in ranks] == ["FR-1", "FR-2"]
    assert estimates.rank_responders(inp, "INC-01", ResponderCapability.EXTRICATION) == ()


def test_rank_hospitals_free_bed_first_then_eta() -> None:
    unit = make_loaded("AMB-01", 5.0, 5.0, [make_victim("V1", "YELLOW", "neuro", "head injury (TBI)", VictimStatus.LOADED)])
    board = [make_board("H3", 5.5, 5.0, ("neuro", "general"), free=0), make_board("H2", 9.0, 5.0, ("neuro",), free=1),
             make_board("H1", 7.0, 5.0, ("neuro",), free=3), make_board("H4", 5.0, 5.0, ("general",), free=9)]
    ranks = estimates.rank_hospitals(board, unit)
    assert [r.hospital_id for r in ranks] == ["H1", "H2", "H3"]
    assert ranks[0].eta_min == geo.travel_time((5.0, 5.0), (7.0, 5.0), "ambulance", False)


def test_eligible_slots_holdover_rule() -> None:
    on = make_staff("DR-1", StaffRole.DOCTOR, ("emergency",), load=1)
    off_trauma = make_staff("DR-2", StaffRole.DOCTOR, ("emergency", "trauma"), load=0, on_shift=False)
    staff = [on, off_trauma]
    assert estimates.eligible_slots(staff, StaffRole.DOCTOR, "trauma") == 4  # holdover: no on-shift trauma colleague
    assert estimates.eligible_slots(staff, StaffRole.DOCTOR, "emergency") == 3  # DR-2 not needed
    assert estimates.eligible_slots(staff, StaffRole.NURSE, "emergency") == 0


def test_capacity_check_fields() -> None:
    request = make_request("P-1", [make_victim("V1", "YELLOW", "trauma", "hip fracture", VictimStatus.LOADED),
                                   make_victim("V2", "GREEN", "burns", "minor burns", VictimStatus.LOADED)])
    inp = make_hospital_input([request], free=3, reserve=2)
    check = estimates.capacity_check(inp, "P-1")
    assert check.patients == 2 and not check.has_red
    assert check.missing_capabilities == (Capability.BURNS,)
    assert (check.beds_free_after, check.reserve_breached) == (1, True)
    assert dict(check.doctor_slots) == {"burns": 0, "trauma": 4}
    assert check.missing_doctor_skills == ("burns",)
    assert (check.nurse_slots, check.nurse_short) == (3, False)
    assert estimates.capacity_check(inp, "P-1", beds_free=1).beds_free_after == -1
    with pytest.raises(KeyError):
        estimates.capacity_check(inp, "P-9")


def test_doctor_skill_for_general_is_emergency() -> None:
    assert estimates.doctor_skill_for(Capability.GENERAL) == "emergency"
    assert estimates.doctor_skill_for(Capability.PEDIATRIC) == "pediatric"
