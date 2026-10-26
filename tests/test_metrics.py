"""Exact-value tests of ``evaluation.metrics`` on hand-built outcomes and logs.

The fixture world (horizon 180) has six patients and one BLACK victim:

=====  =======  ======  ========  ======  ==================================
id     initial  final   deadline  care    notes
=====  =======  ======  ========  ======  ==================================
V1     RED      RED     40        20      in window; ALS +5; door-to-doctor 5
V2     RED      RED     50        60      late; ALS +2; door-to-doctor 30
V3     YELLOW   RED     80        75      deteriorated at 50 (deadline 50+30)
V4     RED      RED     40        -       unserved critical
V5     GREEN    GREEN   480       100     door-to-doctor 10
V6     YELLOW   YELLOW  165       -       unserved
V7     BLACK    BLACK   -         -       excluded from P
=====  =======  ======  ========  ======  ==================================
"""

from collections.abc import Sequence

import numpy as np
import pytest

from emergency_sim.domain.actions import AssignTransport
from emergency_sim.domain.agent_io import TokenUsage
from emergency_sim.domain.enums import (
    AgentRole,
    Capability,
    CaseKind,
    Condition,
    CounterCondition,
    MessageType,
    Resolution,
    SimEventType,
    StaffRole,
    Triage,
    TriggerKind,
)
from emergency_sim.domain.log_records import (
    AgentCallRecord,
    FallbackNote,
    LogRecord,
    MessageRecord,
    RunStartRecord,
    SimEventRecord,
    SnapshotCounts,
    SnapshotRecord,
)
from emergency_sim.domain.messages import (
    ALLOCATION_AGENT,
    DISPATCH_AGENT,
    AdmitAction,
    AssessAction,
    CommitAction,
    ConditionalAdmitAction,
    DeclineAction,
    ResolveAction,
    StaffRequestAction,
)
from emergency_sim.domain.outcomes import IncidentOutcome, RunOutcomes, VictimOutcome
from emergency_sim.domain.views import HospitalView, PublicView, StaffView
from emergency_sim.evaluation.metrics import (
    CONFLICT_KINDS,
    compute_metrics,
    kpis_at,
)
from tests import factories as f

H = 180.0


# --------------------------------------------------------------------------- builders


def _victim(
    vid: str, initial: Triage, final: Triage, *, t_report: float = 0.0, deadline: float | None,
    care: float | None = None, als: float | None = None, arrived: float | None = None,
    deteriorated_at: float | None = None,
) -> VictimOutcome:
    return VictimOutcome(
        victim_id=vid, incident_id="INC-01", t_report_min=t_report, condition="x",
        required_capability=Capability.GENERAL, initial_triage=initial, final_triage=final,
        deteriorated_at_min=deteriorated_at, deadline_min=deadline, als_on_scene_min=als,
        loaded_min=None, hospital_id="H1" if arrived is not None else None, arrived_hospital_min=arrived,
        bedded_min=arrived, treatment_start_min=care if arrived is not None else None,
        care_start_min=care, care_mode="hospital" if care is not None else None,
    )


def _incident(iid: str, t_report: float, first_ambulance: float | None) -> IncidentOutcome:
    return IncidentOutcome(
        incident_id=iid, t_report_min=t_report, mci_cluster=False,
        first_unit_on_scene_min=first_ambulance, first_ambulance_on_scene_min=first_ambulance,
        cleared_min=None,
    )


def make_outcomes() -> RunOutcomes:
    victims = (
        _victim("V1", Triage.RED, Triage.RED, deadline=40, care=20, als=5, arrived=15),
        _victim("V2", Triage.RED, Triage.RED, t_report=10, deadline=50, care=60, als=12, arrived=30),
        _victim("V3", Triage.YELLOW, Triage.RED, deadline=80, care=75, arrived=70, deteriorated_at=50),
        _victim("V4", Triage.RED, Triage.RED, deadline=40),
        _victim("V5", Triage.GREEN, Triage.GREEN, deadline=480, care=100, arrived=90),
        _victim("V6", Triage.YELLOW, Triage.YELLOW, deadline=165),
        _victim("V7", Triage.BLACK, Triage.BLACK, deadline=None),
    )
    incidents = (
        _incident("I1", 0, 2), _incident("I2", 10, 14), _incident("I3", 20, 26),
        _incident("I4", 30, 40), _incident("I5", 40, None),
    )
    return RunOutcomes(
        horizon_min=H, victims=victims, incidents=incidents,
        ambulance_busy_min=(("A1", 90.0), ("A2", 45.0), ("A3", 0.0)),
        responder_busy_min=(("R1", 36.0), ("R2", 0.0)),
        ambulances_in_service=3,
    )


def _staff(sid: str, role: StaffRole, skills: tuple[str, ...], load: int, max_p: int, on_shift: bool) -> StaffView:
    return StaffView(
        staff_id=sid, role=role, skills=skills, home="H1", hospital_id="H1",
        load=load, max_patients=max_p, on_shift=on_shift,
    )


def _hospital(hid: str, occupied: int, staff: tuple[StaffView, ...]) -> HospitalView:
    return HospitalView(
        hospital_id=hid, name=hid, x=0.0, y=0.0, capabilities=(Capability.GENERAL,), er_capacity=4,
        critical_reserve=1, beds_occupied=occupied, beds_reserved=0, beds_free_unreserved=4 - occupied,
        queue_length=0, waiting_for_staff=0, in_treatment=0, inbound_unit_ids=(), staff=staff,
    )


def _snapshot(t: float, occupied: tuple[int, int], d1_load: int, n1_load: int) -> SnapshotRecord:
    """H1 staff: D1 on shift; D2 off shift but sole trauma doctor (holdover);
    D3 off shift and covered by D1 (ineligible); N1 on shift."""
    staff = (
        _staff("D1", StaffRole.DOCTOR, ("emergency",), d1_load, 4, True),
        _staff("D2", StaffRole.DOCTOR, ("trauma",), 1, 4, False),
        _staff("D3", StaffRole.DOCTOR, ("emergency",), 3, 4, False),
        _staff("N1", StaffRole.NURSE, ("emergency",), n1_load, 3, True),
    )
    view = PublicView(
        t_min=t, horizon_min=H, incidents=(), units=(),
        hospitals=(_hospital("H1", occupied[0], staff), _hospital("H2", occupied[1], ())), float_pool=(),
    )
    counts = SnapshotCounts(
        incidents_reported=0, incidents_open=0, waiting_on_scene=0, in_transport=0, queued=0,
        waiting_for_staff=0, in_treatment=0, treated=0, care_started_cum=0, expectant=0,
    )
    return SnapshotRecord(t_min=t, view=view, counts=counts)


def _msg(mid: int, t: float, type_: MessageType, conv: str, action=None, *, rnd: int = 1,
         sender: str = DISPATCH_AGENT, recipient: str = "hospital_H1") -> MessageRecord:
    message = f.message(
        f"M-{mid:04d}", t=t, sender=sender, recipient=recipient, type=type_,
        proposed_action=action, conversation_id=conv, round=rnd,
    )
    return MessageRecord(t_min=t, message=message)


def _commit(resolution: Resolution, case_id: str | None = None) -> CommitAction:
    return CommitAction(
        case_id=case_id, option_id=None, resolution=resolution,
        actions=(AssignTransport(unit_id="AMB-01", hospital_id="H1", reserve_beds=True),),
    )


def _float_counter() -> ConditionalAdmitAction:
    return ConditionalAdmitAction(
        hospital_id="H1", unit_id="AMB-01", victim_ids=("V1",), condition=CounterCondition.FLOAT_DOCTOR,
        skill="trauma",
    )


def _trauma_request(hospital_id: str) -> StaffRequestAction:
    return StaffRequestAction(hospital_id=hospital_id, role=StaffRole.DOCTOR, skill="trauma")


def _start_record() -> RunStartRecord:
    scenario = f.make_scenario(
        hospitals=[f.hospital("H1")],
        ambulances=[f.ambulance("AMB-01")],
        staff=[f.doctor("DR-1", "H1", ("emergency",)), f.doctor("FL-D1", "FLOAT", ("emergency", "trauma"))],
        incidents=[f.incident("INC-01", 0, 4.0, 4.0, [f.victim("V1", "RED")])],
    )
    return RunStartRecord(
        t_min=0.0, run_id="r", scenario="tiny", seed=0, condition=Condition.FULL, backend="offline",
        repeat=0, sim_params={}, negotiation=None, scenario_data=scenario,
    )


def _message_records() -> list[LogRecord]:
    decline = DeclineAction(reason="NO_BEDS")
    reserve = ConditionalAdmitAction(
        hospital_id="H1", unit_id="AMB-02", victim_ids=("V5",), condition=CounterCondition.RESERVE_RELEASE,
        skill=None,
    )
    assess = AssessAction(
        urgency=50, suspected_red=False, suspected_cardiac_arrest=False, est_patients=1,
        needed_capabilities=(), als_units_needed=0, deadline_in_min=100.0,
    )
    m = _msg
    return [
        m(1, 10, MessageType.PRIORITY_REPORT, "PR-0001", assess, rnd=0),
        # NEG-0001: bed rejection, then accepted elsewhere.
        m(2, 10, MessageType.PROPOSE, "NEG-0001"),
        m(3, 10, MessageType.REJECT, "NEG-0001", decline),
        m(4, 10, MessageType.PROPOSE, "NEG-0001", rnd=2),
        m(5, 10, MessageType.ACCEPT, "NEG-0001", _commit(Resolution.ACCEPTED_AFTER_REJECT), rnd=2),
        # NEG-0002: plain acceptance (no conflict).
        m(6, 20, MessageType.PROPOSE, "NEG-0002"),
        m(7, 20, MessageType.ACCEPT, "NEG-0002", _commit(Resolution.ACCEPTED)),
        # NEG-0003 / NEG-0004: two FLOAT_STAFF requests competing for FL-D1 in round 1 at t=30.
        m(8, 30, MessageType.PROPOSE, "NEG-0003"),
        m(9, 30, MessageType.COUNTER, "NEG-0003", _float_counter()),
        m(10, 30, MessageType.REQUEST, "NEG-0003", _trauma_request("H1"), recipient=ALLOCATION_AGENT),
        m(11, 30, MessageType.PROPOSE, "NEG-0004", recipient="hospital_H2"),
        m(12, 30, MessageType.COUNTER, "NEG-0004", _float_counter()),
        m(13, 30, MessageType.REQUEST, "NEG-0004", _trauma_request("H2"), recipient=ALLOCATION_AGENT),
        SimEventRecord(t_min=30, event=SimEventType.STAFF_LENT, staff_id="FL-D1", hospital_id="H1"),
        m(14, 30, MessageType.COMMIT, "NEG-0003", _commit(Resolution.COUNTER_GRANTED, "CASE-0010")),
        m(15, 30, MessageType.REJECT, "NEG-0004", _commit(Resolution.COUNTER_DECLINED, "CASE-0011")),
        m(16, 30, MessageType.ESCALATE, "NEG-0004",
          ResolveAction(case_id="CASE-0003", case_kind=CaseKind.NO_ACCEPTING_HOSPITAL, option_ids=("H2",)), rnd=3),
        m(17, 30, MessageType.COMMIT, "NEG-0004", _commit(Resolution.RA_COMMIT, "CASE-0003"), rnd=3),
        # NEG-0005: unit contention resolved by keeping the assignment.
        m(18, 40, MessageType.ESCALATE, "NEG-0005",
          ResolveAction(case_id="CASE-0001", case_kind=CaseKind.UNIT_CONTENTION, option_ids=("O1", "O2")), rnd=0),
        m(19, 40, MessageType.COMMIT, "NEG-0005", _commit(Resolution.KEPT_ASSIGNMENT, "CASE-0001"), rnd=0),
        # NEG-0006: reserve-release counter, escalated and declined.
        m(20, 50, MessageType.PROPOSE, "NEG-0006"),
        m(21, 50, MessageType.COUNTER, "NEG-0006", reserve),
        m(22, 50, MessageType.ESCALATE, "NEG-0006",
          ResolveAction(case_id="CASE-0002", case_kind=CaseKind.RESERVE_RELEASE, option_ids=("A", "D"))),
        m(23, 50, MessageType.REJECT, "NEG-0006", _commit(Resolution.COUNTER_DECLINED, "CASE-0002")),
    ]


def _agent_call(source: str, notes: tuple[FallbackNote, ...] = ()) -> AgentCallRecord:
    return AgentCallRecord(
        t_min=10, agent="dispatch", role=AgentRole.DISPATCH, triggers=(TriggerKind.NEW_INCIDENT,),
        backend="offline", input={}, output={}, source=source, tool_calls=(), fallbacks=notes,
        usage=TokenUsage(requests=1, input_tokens=0, output_tokens=0), latency_ms=None,
    )


def make_records() -> list[LogRecord]:
    note = FallbackNote(scope="section", key="orders", rule="busy_unit", detail="")
    events = [
        SimEventRecord(t_min=5, event=SimEventType.HOSPITAL_FULL_ON_ARRIVAL, unit_id="AMB-01"),
        SimEventRecord(t_min=6, event=SimEventType.HOSPITAL_FULL_ON_ARRIVAL, unit_id="AMB-02"),
        SimEventRecord(t_min=7, event=SimEventType.CONTROLLER_OMISSION, unit_id="AMB-02"),
        SimEventRecord(t_min=8, event=SimEventType.UNIT_DIVERTED, unit_id="AMB-03"),
    ]
    return [
        _start_record(),
        _snapshot(0.0, (2, 1), d1_load=2, n1_load=1),
        _snapshot(1.0, (4, 3), d1_load=4, n1_load=3),
        *events,
        _agent_call("model"), _agent_call("policy_fallback"), _agent_call("model", (note,)),
        *_message_records(),
        _snapshot(H, (0, 0), d1_load=0, n1_load=0),
    ]


@pytest.fixture(scope="module")
def metrics():
    return compute_metrics(make_outcomes(), make_records())


# --------------------------------------------------------------------------- compute_metrics


def test_victim_counts_exclude_black(metrics) -> None:
    assert metrics.patients_total == 6
    assert metrics.black_victims == 1
    assert metrics.victims_served == 4
    assert metrics.served_rate == pytest.approx(4 / 6)


def test_red_metrics_use_final_triage_and_deteriorated_deadline(metrics) -> None:
    assert metrics.red_total == 4
    assert metrics.red_in_window == 2  # V1 and the deteriorated V3 (75 <= 50 + 30)
    assert metrics.red_in_window_rate == pytest.approx(0.5)
    assert metrics.unserved_critical == 1
    assert metrics.deteriorated == 1


def test_response_times_and_p90(metrics) -> None:
    assert metrics.incidents == 5
    assert metrics.incidents_reached == 4
    assert metrics.response_time_mean_min == pytest.approx(5.5)
    assert metrics.response_time_p90_min == pytest.approx(8.8)
    assert metrics.response_time_p90_min == pytest.approx(np.percentile([2, 4, 6, 10], 90, method="linear"))
    assert metrics.red_response_mean_min == pytest.approx(3.5)  # initially-RED V1 (5) and V2 (2)
    assert metrics.door_to_doctor_mean_min == pytest.approx(12.5)  # 5, 30, 5, 10


def test_survival_weighted_score(metrics) -> None:
    # Weights: V1 3 (in), V2 3, V3 3 (in), V4 3, V5 1 (in), V6 2 -> 7 / 15.
    assert metrics.survival_weighted_score == pytest.approx(7 / 15)


def test_utilisation(metrics) -> None:
    assert metrics.ambulance_utilisation == pytest.approx(135 / (3 * H))
    assert metrics.responder_utilisation == pytest.approx(36 / (2 * H))
    assert metrics.er_utilisation == pytest.approx((3 / 8 + 7 / 8) / 2)  # t=180 snapshot excluded
    assert metrics.doctor_utilisation == pytest.approx((3 / 8 + 5 / 8) / 2)  # D3 not eligible
    assert metrics.nurse_utilisation == pytest.approx((1 / 3 + 3 / 3) / 2)


def test_event_and_agent_counts(metrics) -> None:
    assert metrics.hospital_full_on_arrival == 2
    assert metrics.controller_omissions == 1
    assert metrics.diversions == 1
    assert metrics.float_staff_lent == 1
    assert metrics.agent_calls == 3
    assert metrics.fallbacks == 2


def test_messages_and_conflicts(metrics) -> None:
    assert metrics.messages_total == 23
    assert metrics.messages_by_type["PROPOSE"] == 6
    assert metrics.messages_by_type["INFORM"] == 0
    assert set(metrics.messages_by_type) == {t.value for t in MessageType}
    assert metrics.conflicts == 5
    assert metrics.conflicts_by_kind == {
        "BED_REJECTION": 1,
        "UNIT_CONTENTION": 1,
        "NO_ACCEPTING_HOSPITAL": 1,
        "FLOAT_STAFF": 2,
        "RESERVE_RELEASE": 1,
        "FLOAT_STAFF_COMPETITION": 2,
    }
    assert tuple(metrics.conflicts_by_kind) == CONFLICT_KINDS


def test_resolutions_count_each_threads_closing_commit(metrics) -> None:
    expected = dict.fromkeys((r.value for r in Resolution), 0) | {
        "ACCEPTED_AFTER_REJECT": 1, "ACCEPTED": 1, "COUNTER_GRANTED": 1,
        "RA_COMMIT": 1, "KEPT_ASSIGNMENT": 1, "COUNTER_DECLINED": 1,
    }
    assert metrics.resolutions == expected


def _competition(records: Sequence[LogRecord]) -> int:
    return compute_metrics(f.empty_outcomes(), records).conflicts_by_kind["FLOAT_STAFF_COMPETITION"]


def _requests(rounds: tuple[int, int]) -> list[LogRecord]:
    return [
        _msg(1, 30, MessageType.COUNTER, "NEG-A", _float_counter()),
        _msg(2, 30, MessageType.REQUEST, "NEG-A", _trauma_request("H1"), rnd=rounds[0]),
        _msg(3, 30, MessageType.COUNTER, "NEG-B", _float_counter()),
        _msg(4, 30, MessageType.REQUEST, "NEG-B", _trauma_request("H2"), rnd=rounds[1]),
    ]


def test_float_competition_requires_same_round() -> None:
    assert _competition([_start_record(), *_requests((1, 1))]) == 2
    assert _competition([_start_record(), *_requests((1, 2))]) == 0


def test_float_competition_ignores_already_lent_candidate() -> None:
    lent = SimEventRecord(t_min=20, event=SimEventType.STAFF_LENT, staff_id="FL-D1", hospital_id="H3")
    assert _competition([_start_record(), lent, *_requests((1, 1))]) == 0


def test_float_competition_without_scenario_matches_role_and_skill() -> None:
    assert _competition(_requests((1, 1))) == 2


def test_empty_run_gives_zeros_and_nones() -> None:
    metrics = compute_metrics(f.empty_outcomes(), [])
    assert metrics.patients_total == 0
    assert metrics.served_rate == 0.0
    assert metrics.red_in_window_rate is None
    assert metrics.response_time_mean_min is None
    assert metrics.response_time_p90_min is None
    assert metrics.survival_weighted_score == 0.0
    assert metrics.ambulance_utilisation == 0.0
    assert metrics.er_utilisation == 0.0
    assert metrics.conflicts == 0
    assert sum(metrics.resolutions.values()) == 0


def test_sample_log_records_are_accepted() -> None:
    records = f.sample_log_records()
    end = records[-1]
    metrics = compute_metrics(end.outcomes, records)
    assert metrics.agent_calls == 1
    assert metrics.fallbacks == 1  # the sample call carries a FallbackNote
    assert metrics.messages_by_type["PROPOSE"] == 1
    assert metrics.red_in_window == 1


# --------------------------------------------------------------------------- kpis_at


def test_kpis_at_exact_values() -> None:
    outcomes, records = make_outcomes(), make_records()
    early = kpis_at(outcomes, records, 50.0)
    assert (early.served, early.red_in_window, early.red_missed, early.unserved_critical_overdue) == (1, 1, 1, 1)
    assert early.messages == 23
    assert early.conflicts == 5
    final = kpis_at(outcomes, records, H)
    assert (final.served, final.red_in_window, final.red_missed, final.unserved_critical_overdue) == (4, 2, 2, 1)
    assert kpis_at(outcomes, records, 20.0).mean_response_min == pytest.approx(3.0)
    assert kpis_at(outcomes, records, 25.0).messages == 7
    assert kpis_at(outcomes, records, 25.0).conflicts == 1
    assert kpis_at(outcomes, records, 0.0).mean_response_min is None


def test_kpis_at_is_monotone_in_t() -> None:
    outcomes, records = make_outcomes(), make_records()
    snapshots = [kpis_at(outcomes, records, float(t)) for t in range(0, int(H) + 1)]
    for field in ("served", "red_in_window", "red_missed", "conflicts", "messages"):
        series = [getattr(s, field) for s in snapshots]
        assert series == sorted(series), field
    final = compute_metrics(outcomes, records)
    assert snapshots[-1].served == final.victims_served
    assert snapshots[-1].red_in_window == final.red_in_window
    assert snapshots[-1].conflicts == final.conflicts
