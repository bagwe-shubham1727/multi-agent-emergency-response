"""Unit tests of the pure offline policies (plan 5.3, WP2 policy cases).

The ``make_*`` builders at the top create small LLM-facing payloads; the SDK
tests reuse them.
"""

from collections.abc import Sequence

import pytest

from emergency_sim.domain.agent_io import (
    AdmissionRequest,
    AllocationInput,
    AssessmentInput,
    CaseOption,
    DispatchInput,
    EscalationCase,
    HospitalBoardEntry,
    HospitalInput,
    IncidentNeed,
    IncidentReport,
    LoadedUnit,
    StaffMemberView,
    TriagedVictim,
    UnitStatusView,
)
from emergency_sim.domain.enums import (
    CaseKind,
    Capability,
    CounterCondition,
    ResponderCapability,
    StaffRole,
    Triage,
    UnitClass,
    UnitStatus,
    VictimStatus,
)
from emergency_sim.policies.allocation import DECLINE_OPTION_ID, allocate, grant_option_id, granted_staff
from emergency_sim.policies.assessment import assess, severity_cues
from emergency_sim.policies.dispatch import dispatch, priority_order, scene_ready, split_by_readiness
from emergency_sim.policies.hospital import hospital
from emergency_sim.policies.params import PolicyParams

PARAMS = PolicyParams()
FULL_RESPONSES = ["ACCEPT", "REJECT", "COUNTER"]
B1_RESPONSES = ["ACCEPT", "REJECT"]
ALL_CONDITIONS = [CounterCondition.FLOAT_DOCTOR, CounterCondition.FLOAT_NURSE, CounterCondition.RESERVE_RELEASE]


# --- Payload builders ---------------------------------------------------------


def make_victim(
    vid: str, triage: str, cap: str = "general", condition: str = "lacerations",
    status: VictimStatus = VictimStatus.WAITING, minutes_since_deterioration: float | None = None,
) -> TriagedVictim:
    """A triaged victim (not deteriorated unless ``minutes_since_deterioration`` is given)."""
    return TriagedVictim(
        victim_id=vid, triage=Triage(triage), condition=condition,
        required_capability=Capability(cap), status=status,
        minutes_since_deterioration=minutes_since_deterioration,
    )


def make_report(
    iid: str = "INC-01", incident_type: str = "fall", caller: str = "Caller reports someone hurt.",
    reported: int = 1, since: float = 0.0, victims: Sequence[TriagedVictim] = (),
) -> IncidentReport:
    """An incident report for the assessment agent."""
    return IncidentReport(
        incident_id=iid, incident_type=incident_type, t_report_min=0.0, minutes_since_report=since,
        x=5.0, y=5.0, intersection="Cedar St & 3rd Ave", reported_victims=reported, caller_report=caller,
        required_responder_capabilities=[], triaged_victims=list(victims),
    )


def make_assessment_input(*reports: IncidentReport) -> AssessmentInput:
    """An assessment payload."""
    return AssessmentInput(t_min=0.0, incidents=list(reports))


def make_need(iid: str, x: float, y: float, urgency: int = 50, **overrides: object) -> IncidentNeed:
    """An accessible incident need with no unmet ALS and one patient by default."""
    values: dict[str, object] = dict(
        incident_id=iid, x=x, y=y, urgency=urgency, suspected_red=False, suspected_cardiac_arrest=False,
        deadline_in_min=165.0, triaged=False, est_patients_remaining=1, red_waiting=0,
        ambulances_assigned=0, seats_en_route=0, als_en_route=0, als_needed=0, responder_caps_missing=[],
        access_wait_min=0.0,
    )
    values.update(overrides)
    return IncidentNeed.model_validate(values)


def make_ambulance(
    uid: str, kind: str, x: float, y: float, status: UnitStatus = UnitStatus.AVAILABLE,
    incident_id: str | None = None,
) -> UnitStatusView:
    """An ambulance (2 seats)."""
    return UnitStatusView(
        unit_id=uid, unit_class=UnitClass.AMBULANCE, unit_type=kind, capabilities=[], status=status,
        x=x, y=y, incident_id=incident_id, seats=2,
    )


def make_team(uid: str, caps: Sequence[str], x: float, y: float) -> UnitStatusView:
    """A free responder team."""
    return UnitStatusView(
        unit_id=uid, unit_class=UnitClass.RESPONDER, unit_type="test_team", capabilities=list(caps),
        status=UnitStatus.AVAILABLE, x=x, y=y, incident_id=None, seats=0,
    )


def make_board(hid: str, x: float, y: float, caps: Sequence[str], free: int, queue: int = 0) -> HospitalBoardEntry:
    """A bed board entry."""
    return HospitalBoardEntry(
        hospital_id=hid, x=x, y=y, capabilities=[Capability(c) for c in caps],
        beds_free_unreserved=free, queue_length=queue,
    )


def make_loaded(uid: str, x: float, y: float, victims: Sequence[TriagedVictim]) -> LoadedUnit:
    """A loaded unit awaiting a destination."""
    return LoadedUnit(unit_id=uid, incident_id="INC-01", x=x, y=y, victims=list(victims))


def make_dispatch_input(
    incidents: Sequence[IncidentNeed], units: Sequence[UnitStatusView],
    hospitals: Sequence[HospitalBoardEntry] = (), loaded: Sequence[LoadedUnit] = (),
    *, diversion: bool = False, escalation: bool = False,
) -> DispatchInput:
    """A dispatch payload."""
    return DispatchInput(
        t_min=10.0, incidents=list(incidents), units=list(units), hospitals=list(hospitals),
        loaded_units=list(loaded), allow_diversion=diversion, allow_escalation=escalation,
    )


def make_staff(
    sid: str, role: StaffRole, skills: Sequence[str], load: int = 0, max_patients: int = 4,
    on_shift: bool = True,
) -> StaffMemberView:
    """A staff member at the hospital."""
    return StaffMemberView(
        staff_id=sid, role=role, skills=list(skills), load=load, max_patients=max_patients,
        on_shift=on_shift, lent=False,
    )


def default_staff() -> list[StaffMemberView]:
    """One trauma doctor and one nurse, both idle."""
    return [
        make_staff("DR-1", StaffRole.DOCTOR, ("emergency", "trauma")),
        make_staff("RN-1", StaffRole.NURSE, ("emergency",), max_patients=3),
    ]


def make_request(pid: str, victims: Sequence[TriagedVictim]) -> AdmissionRequest:
    """An admission request."""
    return AdmissionRequest(proposal_id=pid, incident_id="INC-01", unit_id=f"AMB-{pid}", victims=list(victims), eta_min=6.0)


def make_hospital_input(
    requests: Sequence[AdmissionRequest], *, free: int = 4, reserve: int = 1,
    staff: Sequence[StaffMemberView] | None = None, caps: Sequence[str] = ("trauma", "general"),
    responses: Sequence[str] = FULL_RESPONSES, conditions: Sequence[CounterCondition] = ALL_CONDITIONS,
) -> HospitalInput:
    """A hospital payload (H1, 4 beds by default)."""
    return HospitalInput.model_validate(dict(
        t_min=10.0, hospital_id="H1", capabilities=list(caps), er_capacity=4, beds_occupied=4 - free,
        beds_reserved=0, beds_free_unreserved=free, critical_reserve=reserve, queue_length=0,
        waiting_for_staff=0, staff=list(default_staff() if staff is None else staff),
        requests=list(requests), allowed_responses=list(responses), allowed_counter_conditions=list(conditions),
    ))


def make_option(oid: str, utility: float, weight: float = 3.0) -> CaseOption:
    """A case option."""
    return CaseOption(option_id=oid, summary=oid, triage_weight=weight, slack_min=10.0, travel_min=5.0, utility=utility)


def make_case(cid: str, kind: CaseKind, options: Sequence[CaseOption]) -> EscalationCase:
    """An escalation case."""
    return EscalationCase(case_id=cid, kind=kind, incident_id="INC-01", hospital_id="H1", summary=cid, options=list(options))


def make_allocation_input(*cases: EscalationCase) -> AllocationInput:
    """An allocation payload."""
    return AllocationInput(t_min=10.0, cases=list(cases), float_pool=[])


# --- Incident Assessment ------------------------------------------------------


@pytest.mark.parametrize(
    ("caller", "cues"),
    [
        ("They are NOT RESPONDING and a bystander is starting CPR.", ("not responding", "cpr")),
        ("People are trapped under the rubble.", ("trapped", "rubble")),
        ("The child's lips look blue.", ("blue",)),
        ("They are awake but sweating and in pain.", ()),
    ],
)
def test_severity_cues_are_case_insensitive(caller: str, cues: tuple[str, ...]) -> None:
    assert severity_cues(caller) == cues


def test_pre_triage_cardiac_arrest_report() -> None:
    caller = "A person collapsed; they are not responding and a bystander is starting CPR."
    out = assess(make_assessment_input(make_report(incident_type="cardiac_event", caller=caller, since=2.0)), PARAMS)
    report = out.reports[0]
    assert report.suspected_red and report.suspected_cardiac_arrest
    assert report.urgency == 70 + 25 + 2
    assert report.als_units_needed == 1
    assert report.deadline_in_min == pytest.approx(10.0 - 2.0)
    assert report.needed_capabilities == [Capability.CARDIAC]


def test_pre_triage_mild_report_and_clipping() -> None:
    mild = make_report("INC-01", "fall", "They cannot get up.", reported=1)
    crowd = make_report("INC-02", "multi_vehicle_collision", "People are trapped.", reported=9)
    out = assess(make_assessment_input(crowd, mild), PARAMS)
    assert [r.incident_id for r in out.reports] == ["INC-01", "INC-02"]
    assert out.reports[0].urgency == 42 and not out.reports[0].suspected_red
    assert out.reports[0].als_units_needed == 0
    assert out.reports[0].deadline_in_min == pytest.approx(165.0)
    assert out.reports[1].urgency == 95  # 60 + 25 + min(10, 18)
    assert out.reports[1].est_patients == 9


@pytest.mark.parametrize(
    ("victims", "urgency", "red"),
    [
        ((("V1", "RED"), ("V2", "GREEN")), 100, 1),
        ((("V1", "YELLOW"), ("V2", "GREEN")), 60, 0),
        ((("V1", "GREEN"),), 30, 0),
    ],
)
def test_post_triage_urgency(victims: tuple[tuple[str, str], ...], urgency: int, red: int) -> None:
    triaged = [make_victim(v, t) for v, t in victims]
    report = assess(make_assessment_input(make_report(victims=triaged)), PARAMS).reports[0]
    assert report.urgency == urgency
    assert report.als_units_needed == red == int(report.suspected_red)
    assert report.est_patients == len(victims)


def test_post_triage_ignores_loaded_and_black_victims() -> None:
    victims = [
        make_victim("V1", "RED", status=VictimStatus.LOADED),
        make_victim("V2", "BLACK", status=VictimStatus.EXPECTANT),
        make_victim("V3", "YELLOW", "trauma", "hip fracture"),
    ]
    report = assess(make_assessment_input(make_report(victims=victims, since=5.0)), PARAMS).reports[0]
    assert report.urgency == 60 and report.est_patients == 1 and report.als_units_needed == 0
    assert report.needed_capabilities == [Capability.TRAUMA]
    assert report.deadline_in_min == pytest.approx(165.0 - 5.0)


def test_post_triage_cardiac_arrest_uses_short_window() -> None:
    victims = [make_victim("V1", "RED", "cardiac", "cardiac arrest")]
    report = assess(make_assessment_input(make_report(victims=victims, since=3.0)), PARAMS).reports[0]
    assert report.suspected_cardiac_arrest and report.deadline_in_min == pytest.approx(7.0)


def test_deteriorated_red_window_counts_from_the_deterioration() -> None:
    # Regression: a YELLOW that turned RED 5 min ago, 100 min after the report,
    # used to get 40 - 100 = -60 min (window counted from the report).
    victims = [make_victim("V1", "RED", "trauma", "hip fracture", minutes_since_deterioration=5.0),
               make_victim("V2", "YELLOW")]
    report = assess(make_assessment_input(make_report(victims=victims, since=100.0)), PARAMS).reports[0]
    assert report.urgency == 100
    assert report.deadline_in_min == pytest.approx(PARAMS.deteriorated_red_deadline_min - 5.0)


def test_initial_red_window_still_counts_from_the_report() -> None:
    victims = [make_victim("V1", "RED", "trauma", "crush injury")]
    report = assess(make_assessment_input(make_report(victims=victims, since=12.0)), PARAMS).reports[0]
    assert report.deadline_in_min == pytest.approx(PARAMS.nominal_red_deadline_min - 12.0)


def test_post_triage_nothing_waiting_is_zero_urgency() -> None:
    victims = [make_victim("V1", "GREEN", status=VictimStatus.LOADED)]
    report = assess(make_assessment_input(make_report(victims=victims)), PARAMS).reports[0]
    assert report.urgency == 0 and report.est_patients == 0


# --- Dispatch -------------------------------------------------------------------


def test_als_goes_to_red_and_bls_fills_seats() -> None:
    need = make_need("INC-01", 5.0, 5.0, 100, triaged=True, red_waiting=1, als_needed=1, est_patients_remaining=3)
    units = [make_ambulance("AMB-01", "BLS", 5.0, 5.5), make_ambulance("AMB-02", "ALS", 8.0, 5.0),
             make_ambulance("AMB-03", "BLS", 9.0, 9.0)]
    out = dispatch(make_dispatch_input([need], units), PARAMS)
    assert [(o.unit_id, o.incident_id) for o in out.orders] == [("AMB-02", "INC-01"), ("AMB-01", "INC-01")]
    assert out.contentions == [] and out.diversions == []


def test_incidents_served_by_urgency_then_deadline() -> None:
    low = make_need("INC-01", 5.0, 5.0, 40)
    high = make_need("INC-02", 15.0, 15.0, 90)
    units = [make_ambulance("AMB-01", "BLS", 5.0, 5.0)]
    out = dispatch(make_dispatch_input([low, high], units), PARAMS)
    assert [(o.unit_id, o.incident_id) for o in out.orders] == [("AMB-01", "INC-02")]


def _two_red_incidents_one_als() -> tuple[list[IncidentNeed], list[UnitStatusView]]:
    a = make_need("INC-01", 4.0, 5.0, 95, suspected_red=True, als_needed=1, est_patients_remaining=1)
    b = make_need("INC-02", 6.0, 5.0, 90, suspected_red=True, als_needed=1, est_patients_remaining=1)
    units = [make_ambulance("AMB-01", "ALS", 5.0, 5.0), make_ambulance("AMB-02", "ALS", 18.0, 18.0)]
    return [a, b], units


def test_contention_flag_only_with_escalation() -> None:
    needs, units = _two_red_incidents_one_als()
    full = dispatch(make_dispatch_input(needs, units, escalation=True), PARAMS)
    assert [(c.unit_id, c.incident_ids) for c in full.contentions] == [("AMB-01", ["INC-01", "INC-02"])]
    assert "AMB-01" not in {o.unit_id for o in full.orders}
    b1 = dispatch(make_dispatch_input(needs, units), PARAMS)
    assert b1.contentions == []
    assert [(o.unit_id, o.incident_id) for o in b1.orders] == [("AMB-01", "INC-01"), ("AMB-02", "INC-02")]


def _diversion_world(diversion: bool) -> DispatchInput:
    urgent = make_need("INC-01", 5.0, 5.0, 95, suspected_red=True, als_needed=1)
    minor = make_need("INC-02", 9.0, 5.0, 40, ambulances_assigned=1, seats_en_route=2, als_en_route=1)
    units = [make_ambulance("AMB-01", "ALS", 6.0, 5.0, UnitStatus.EN_ROUTE_SCENE, "INC-02")]
    return make_dispatch_input([urgent, minor], units, diversion=diversion)


def test_diversion_only_when_allowed() -> None:
    allowed = dispatch(_diversion_world(True), PARAMS)
    assert [(d.unit_id, d.to_incident_id) for d in allowed.diversions] == [("AMB-01", "INC-01")]
    assert dispatch(_diversion_world(False), PARAMS).diversions == []


def test_no_diversion_from_a_red_scene() -> None:
    inp = _diversion_world(True)
    minor = inp.incidents[1].model_copy(update={"red_waiting": 1, "urgency": 40})
    inp = inp.model_copy(update={"incidents": [inp.incidents[0], minor]})
    assert dispatch(inp, PARAMS).diversions == []


def test_pre_triage_seat_cap() -> None:
    units = [make_ambulance(f"AMB-0{i}", "BLS", 5.0, float(i)) for i in range(1, 6)]
    pre = dispatch(make_dispatch_input([make_need("INC-01", 5.0, 5.0, 60, est_patients_remaining=10)], units), PARAMS)
    assert len(pre.orders) == PARAMS.max_ambulances_pre_triage
    already = make_need("INC-01", 5.0, 5.0, 60, est_patients_remaining=10, ambulances_assigned=2, seats_en_route=4)
    assert len(dispatch(make_dispatch_input([already], units), PARAMS).orders) == 1
    post = make_need("INC-01", 5.0, 5.0, 60, triaged=True, est_patients_remaining=10)
    assert len(dispatch(make_dispatch_input([post], units), PARAMS).orders) == 5


def test_als_not_used_for_seats_while_an_als_need_is_unmet() -> None:
    red = make_need("INC-01", 15.0, 15.0, 100, triaged=True, red_waiting=1, als_needed=1)
    seats = make_need("INC-02", 5.0, 5.0, 60, triaged=True, est_patients_remaining=2)
    units = [make_ambulance("AMB-01", "ALS", 14.0, 15.0), make_ambulance("AMB-02", "ALS", 5.0, 6.0)]
    out = dispatch(make_dispatch_input([red, seats], units), PARAMS)
    assert [(o.unit_id, o.incident_id) for o in out.orders] == [("AMB-01", "INC-01"), ("AMB-02", "INC-02")]
    contested, contest_units = _two_red_incidents_one_als()
    far_seats = make_need("INC-03", 17.0, 18.0, 30, triaged=True, est_patients_remaining=2)
    out = dispatch(make_dispatch_input([*contested, far_seats], contest_units, escalation=True), PARAMS)
    assert out.orders == []  # AMB-02 stays free while the contested incidents lack an ALS


def test_ready_scene_seat_need_beats_blocked_scene_als_need() -> None:
    # Regression (constrained seed 3, t=97.4): the only free ALS went to an
    # untriaged suspected-RED scene still waiting for its police team (it then
    # idled there ~39 min) instead of the GREEN left at an accessible scene,
    # whose fire and police teams stayed tied up until the end of the run.
    leftover = make_need("INC-04", 5.0, 5.0, 30, triaged=True, est_patients_remaining=1)
    blocked = make_need("INC-07", 6.0, 5.0, 93, suspected_red=True, als_needed=1, est_patients_remaining=4,
                        responder_caps_missing=[ResponderCapability.SCENE_CONTROL], access_wait_min=None)
    units = [make_ambulance("AMB-01", "ALS", 5.5, 5.0)]
    out = dispatch(make_dispatch_input([leftover, blocked], units), PARAMS)
    assert [(o.unit_id, o.incident_id) for o in out.orders] == [("AMB-01", "INC-04")]


def test_blocked_scene_still_served_when_no_ready_scene_needs_the_unit() -> None:
    blocked = make_need("INC-07", 6.0, 5.0, 93, suspected_red=True, als_needed=1, access_wait_min=None)
    out = dispatch(make_dispatch_input([blocked], [make_ambulance("AMB-01", "ALS", 5.5, 5.0)]), PARAMS)
    assert [(o.unit_id, o.incident_id) for o in out.orders] == [("AMB-01", "INC-07")]


@pytest.mark.parametrize(("access_wait", "ready"), [(0.0, True), (None, False), (2.0, True), (30.0, False)])
def test_scene_ready_compares_access_wait_with_nearest_eta(access_wait: float | None, ready: bool) -> None:
    need = make_need("INC-01", 5.0, 9.0, 60, access_wait_min=access_wait)
    inp = make_dispatch_input([need], [make_ambulance("AMB-01", "BLS", 5.0, 5.0)])
    assert scene_ready(inp, need) is ready


def test_urgency_order_kept_within_ready_and_blocked_groups() -> None:
    ready_low = make_need("INC-01", 5.0, 5.0, 40)
    ready_high = make_need("INC-02", 6.0, 5.0, 60)
    blocked_top = make_need("INC-03", 5.5, 5.0, 95, access_wait_min=None)
    inp = make_dispatch_input([ready_low, ready_high, blocked_top], [make_ambulance("AMB-01", "BLS", 5.0, 5.0)])
    ready, blocked = split_by_readiness(inp, priority_order(inp.incidents))
    assert [n.incident_id for n in ready] == ["INC-02", "INC-01"]
    assert [n.incident_id for n in blocked] == ["INC-03"]


def test_als_pass_respects_the_pre_triage_cap() -> None:
    # Regression (surge seed 5, t=118.7): a suspected-RED incident with three
    # BLS already sent got a fourth ambulance (an ALS), which broke the
    # pre-triage cap shared with B0 and was flagged by the dispatch validator.
    capped = make_need("INC-11", 5.0, 5.0, 95, suspected_red=True, als_needed=1, est_patients_remaining=6,
                       ambulances_assigned=3, seats_en_route=6)
    out = dispatch(make_dispatch_input([capped], [make_ambulance("AMB-01", "ALS", 5.5, 5.0)], diversion=True,
                                       escalation=True), PARAMS)
    assert out.orders == [] and out.diversions == [] and out.contentions == []


def test_responder_pass_sends_one_team_per_missing_capability() -> None:
    need = make_need(
        "INC-01", 5.0, 5.0, 60, responder_caps_missing=[ResponderCapability.EXTRICATION, ResponderCapability.SCENE_CONTROL]
    )
    units = [make_team("FR-ENG1", ("fire_suppression", "extrication"), 5.0, 6.0),
             make_team("FR-RES1", ("heavy_rescue", "extrication"), 5.0, 5.5),
             make_team("FR-POL1", ("scene_control",), 9.0, 9.0)]
    out = dispatch(make_dispatch_input([need], units), PARAMS)
    assert sorted(o.unit_id for o in out.orders if o.unit_id.startswith("FR")) == ["FR-POL1", "FR-RES1"]


def test_responder_goes_where_it_completes_access() -> None:
    # The only free team is a police car. The more urgent scene also lacks
    # extrication (no free team has it), so police there would just wait; at
    # the other scene police is the last missing capability and opens it.
    stuck = make_need("INC-01", 5.0, 5.0, 95, access_wait_min=None,
                      responder_caps_missing=[ResponderCapability.EXTRICATION, ResponderCapability.SCENE_CONTROL])
    opens = make_need("INC-02", 9.0, 5.0, 60, access_wait_min=None,
                      responder_caps_missing=[ResponderCapability.SCENE_CONTROL])
    units = [make_team("FR-POL1", ("scene_control",), 5.0, 6.0)]
    out = dispatch(make_dispatch_input([stuck, opens], units), PARAMS)
    assert [(o.unit_id, o.incident_id) for o in out.orders] == [("FR-POL1", "INC-02")]


def test_partial_teams_still_go_by_priority_when_no_scene_can_be_completed() -> None:
    first = make_need("INC-01", 5.0, 5.0, 95, access_wait_min=None,
                      responder_caps_missing=[ResponderCapability.EXTRICATION, ResponderCapability.SCENE_CONTROL])
    second = make_need("INC-02", 9.0, 5.0, 60, access_wait_min=None,
                       responder_caps_missing=[ResponderCapability.HAZMAT, ResponderCapability.SCENE_CONTROL])
    units = [make_team("FR-POL1", ("scene_control",), 5.0, 6.0)]
    out = dispatch(make_dispatch_input([first, second], units), PARAMS)
    assert [(o.unit_id, o.incident_id) for o in out.orders] == [("FR-POL1", "INC-01")]


def test_transport_plans_rank_capable_hospitals() -> None:
    victims = [make_victim("V1", "RED", "trauma", "stab wound", VictimStatus.LOADED)]
    board = [make_board("H1", 6.0, 5.0, ("trauma", "general"), free=0),
             make_board("H2", 12.0, 5.0, ("trauma", "general"), free=2),
             make_board("H3", 5.0, 5.0, ("general",), free=5)]
    out = dispatch(make_dispatch_input([], [], board, [make_loaded("AMB-01", 5.0, 5.0, victims)]), PARAMS)
    assert [(t.unit_id, t.hospital_ranking) for t in out.transports] == [("AMB-01", ["H2", "H1"])]


def test_dispatch_is_deterministic() -> None:
    needs, units = _two_red_incidents_one_als()
    inp = make_dispatch_input(needs, units, escalation=True, diversion=True)
    assert dispatch(inp, PARAMS) == dispatch(inp, PARAMS)


# --- Hospital Coordination ------------------------------------------------------


def test_non_red_into_reserve_rejects_in_b1_and_counters_in_full() -> None:
    request = make_request("P-1", [make_victim("V1", "YELLOW", status=VictimStatus.LOADED)])
    b1 = hospital(make_hospital_input([request], free=1, responses=B1_RESPONSES, conditions=[]), PARAMS)
    assert (b1.responses[0].response, b1.responses[0].reason) == ("REJECT", "RESERVE_HELD")
    full = hospital(make_hospital_input([request], free=1), PARAMS).responses[0]
    assert (full.response, full.reason, full.counter_condition, full.counter_skill) == (
        "COUNTER", "RESERVE_HELD", CounterCondition.RESERVE_RELEASE, None,
    )


def test_red_may_use_the_reserve() -> None:
    request = make_request("P-1", [make_victim("V1", "RED", "trauma", "stab wound", VictimStatus.LOADED)])
    out = hospital(make_hospital_input([request], free=1), PARAMS).responses[0]
    assert (out.response, out.reason) == ("ACCEPT", "OK")


def test_counter_float_doctor_with_skill() -> None:
    staff = [make_staff("DR-1", StaffRole.DOCTOR, ("emergency",)), make_staff("RN-1", StaffRole.NURSE, ("emergency",))]
    request = make_request("P-1", [make_victim("V1", "RED", "trauma", "stab wound", VictimStatus.LOADED)])
    full = hospital(make_hospital_input([request], staff=staff), PARAMS).responses[0]
    assert (full.response, full.reason, full.counter_condition, full.counter_skill) == (
        "COUNTER", "NO_DOCTOR", CounterCondition.FLOAT_DOCTOR, "trauma",
    )
    b1 = hospital(make_hospital_input([request], staff=staff, responses=B1_RESPONSES, conditions=[]), PARAMS)
    assert (b1.responses[0].response, b1.responses[0].reason) == ("REJECT", "NO_DOCTOR")


def test_float_doctor_counter_needs_float_condition_allowed() -> None:
    staff = [make_staff("DR-1", StaffRole.DOCTOR, ("emergency",)), make_staff("RN-1", StaffRole.NURSE, ("emergency",))]
    request = make_request("P-1", [make_victim("V1", "RED", "trauma", "stab wound", VictimStatus.LOADED)])
    no_float = make_hospital_input([request], staff=staff, conditions=[CounterCondition.RESERVE_RELEASE])
    assert hospital(no_float, PARAMS).responses[0].response == "REJECT"


def test_off_shift_doctor_holds_over_without_on_shift_colleague() -> None:
    staff = [make_staff("DR-1", StaffRole.DOCTOR, ("emergency", "trauma"), on_shift=False),
             make_staff("RN-1", StaffRole.NURSE, ("emergency",))]
    request = make_request("P-1", [make_victim("V1", "RED", "trauma", "stab wound", VictimStatus.LOADED)])
    assert hospital(make_hospital_input([request], staff=staff), PARAMS).responses[0].response == "ACCEPT"


def test_counter_float_nurse() -> None:
    staff = [make_staff("DR-1", StaffRole.DOCTOR, ("emergency",)),
             make_staff("RN-1", StaffRole.NURSE, ("emergency",), load=3, max_patients=3)]
    request = make_request("P-1", [make_victim("V1", "RED", status=VictimStatus.LOADED)])
    out = hospital(make_hospital_input([request], staff=staff), PARAMS).responses[0]
    assert (out.response, out.reason, out.counter_condition, out.counter_skill) == (
        "COUNTER", "NO_NURSE", CounterCondition.FLOAT_NURSE, "emergency",
    )


def test_not_capable_and_no_beds() -> None:
    burns = make_request("P-1", [make_victim("V1", "RED", "burns", "major burns (>20% TBSA)", VictimStatus.LOADED)])
    two = make_request("P-2", [make_victim("V2", "RED", status=VictimStatus.LOADED),
                               make_victim("V3", "GREEN", status=VictimStatus.LOADED)])
    out = hospital(make_hospital_input([two, burns], free=1), PARAMS).responses
    assert [(r.proposal_id, r.response, r.reason) for r in out] == [
        ("P-1", "REJECT", "NOT_CAPABLE"), ("P-2", "REJECT", "NO_BEDS"),
    ]


def test_running_bed_count_across_two_proposals() -> None:
    first = make_request("P-1", [make_victim("V1", "RED", status=VictimStatus.LOADED)])
    second = make_request("P-2", [make_victim("V2", "RED", status=VictimStatus.LOADED)])
    out = hospital(make_hospital_input([second, first], free=1, reserve=0), PARAMS).responses
    assert [(r.proposal_id, r.response, r.reason) for r in out] == [
        ("P-1", "ACCEPT", "OK"), ("P-2", "REJECT", "NO_BEDS"),
    ]


def test_counter_fields_null_unless_counter() -> None:
    request = make_request("P-1", [make_victim("V1", "RED", status=VictimStatus.LOADED)])
    out = hospital(make_hospital_input([request]), PARAMS).responses[0]
    assert out.response == "ACCEPT" and out.counter_condition is None and out.counter_skill is None


# --- Resource Allocation ------------------------------------------------------


def test_ra_picks_max_utility_with_id_tie_break() -> None:
    case = make_case("CASE-0001", CaseKind.UNIT_CONTENTION,
                     [make_option("OPT-2", 2.5), make_option("OPT-1", 2.5), make_option("OPT-3", 1.0)])
    out = allocate(make_allocation_input(case), PARAMS)
    assert [(d.case_id, d.chosen_option_id) for d in out.decisions] == [("CASE-0001", "OPT-1")]


def test_ra_orders_cases_by_best_utility() -> None:
    low = make_case("CASE-0001", CaseKind.NO_ACCEPTING_HOSPITAL, [make_option("H1", 1.0)])
    high = make_case("CASE-0002", CaseKind.NO_ACCEPTING_HOSPITAL, [make_option("H2", 2.0)])
    empty = make_case("CASE-0003", CaseKind.NO_ACCEPTING_HOSPITAL, [])
    out = allocate(make_allocation_input(low, high, empty), PARAMS)
    assert [d.case_id for d in out.decisions] == ["CASE-0002", "CASE-0001"]


def test_ra_never_double_grants_float_staff() -> None:
    first = make_case("CASE-0001", CaseKind.FLOAT_STAFF,
                      [make_option(grant_option_id("DR-F1"), 2.8), make_option(DECLINE_OPTION_ID, 0.5)])
    second = make_case("CASE-0002", CaseKind.FLOAT_STAFF,
                       [make_option(grant_option_id("DR-F1"), 2.5), make_option(DECLINE_OPTION_ID, 0.4)])
    third = make_case("CASE-0003", CaseKind.FLOAT_STAFF,
                      [make_option(grant_option_id("DR-F1"), 2.0), make_option(grant_option_id("DR-F2"), 1.9),
                       make_option(DECLINE_OPTION_ID, 0.1)])
    out = allocate(make_allocation_input(third, second, first), PARAMS)
    chosen = {d.case_id: d.chosen_option_id for d in out.decisions}
    assert chosen == {"CASE-0001": "GRANT:DR-F1", "CASE-0002": DECLINE_OPTION_ID, "CASE-0003": "GRANT:DR-F2"}
    lent = [granted_staff(o) for o in chosen.values() if granted_staff(o)]
    assert len(lent) == len(set(lent))


def test_ra_declines_green_only_float_requests() -> None:
    case = make_case("CASE-0001", CaseKind.FLOAT_STAFF,
                     [make_option(grant_option_id("RN-F1"), 0.9, weight=1.0), make_option(DECLINE_OPTION_ID, 0.2, weight=1.0)])
    assert allocate(make_allocation_input(case), PARAMS).decisions[0].chosen_option_id == DECLINE_OPTION_ID


def test_ra_reserve_release_follows_precomputed_utility() -> None:
    case = make_case("CASE-0001", CaseKind.RESERVE_RELEASE, [make_option("APPROVE", 1.2), make_option("DECLINE", 2.9)])
    assert allocate(make_allocation_input(case), PARAMS).decisions[0].chosen_option_id == "DECLINE"
