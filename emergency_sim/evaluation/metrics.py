"""Run metrics and time-sliced KPIs computed from outcomes and the run log.

Implements the formulas of section 6 of the implementation plan exactly.
Notation: ``P`` is the set of non-BLACK victims, ``H`` the horizon and
``R = {v in P : final_triage == RED}`` (deteriorated victims included).

Interpretations where the plan leaves room (documented in the report):

* ``conflicts_by_kind["BED_REJECTION"]`` counts conflict threads whose only
  conflict messages are REJECTs, i.e. threads that opened no case. A thread
  that opened cases is counted once per case kind it contains, so the kinds
  can sum to more than ``conflicts``.
* ``FLOAT_STAFF_COMPETITION`` counts threads whose float-staff REQUEST shared
  a candidate (an unlent, on-shift float member with the requested role and
  skill) with another REQUEST of the same decision time and round.
* ``resolutions`` counts, per thread, the resolution of its last message
  carrying a ``CommitAction``; threads without one are not counted.
* Staff and ER utilisation average over snapshots ``0 <= t < H`` whose
  denominator is positive; with none, the utilisation is 0.0.
"""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations
from types import MappingProxyType
from typing import Final

import numpy as np

from emergency_sim.domain.base import FrozenModel
from emergency_sim.domain.enums import (
    CaseKind,
    CounterCondition,
    MessageType,
    Resolution,
    SimEventType,
    StaffRole,
    Triage,
)
from emergency_sim.domain.log_records import (
    AgentCallRecord,
    LogRecord,
    MessageRecord,
    RunStartRecord,
    SimEventRecord,
    SnapshotRecord,
)
from emergency_sim.domain.messages import (
    CommitAction,
    ConditionalAdmitAction,
    Message,
    ResolveAction,
    StaffRequestAction,
)
from emergency_sim.domain.outcomes import (
    IncidentOutcome,
    RunMetrics,
    RunOutcomes,
    VictimOutcome,
)
from emergency_sim.domain.scenario import StaffSpec
from emergency_sim.domain.views import PublicView, StaffView

SCORE_WEIGHTS: Final = MappingProxyType({Triage.RED: 3.0, Triage.YELLOW: 2.0, Triage.GREEN: 1.0})
"""Survival-score weights by final triage (independent of ``PolicyParams``)."""

P90: Final = 90.0
CONFLICT_TYPES: Final = frozenset({MessageType.REJECT, MessageType.COUNTER, MessageType.ESCALATE})
BED_REJECTION: Final = "BED_REJECTION"
FLOAT_STAFF_COMPETITION: Final = "FLOAT_STAFF_COMPETITION"
CONFLICT_KINDS: Final = (BED_REJECTION, *(kind.value for kind in CaseKind), FLOAT_STAFF_COMPETITION)
"""Every key of ``RunMetrics.conflicts_by_kind``, in report order."""

FLOAT_HOME: Final = "FLOAT"
FLOAT_COUNTER_CONDITIONS: Final = frozenset({CounterCondition.FLOAT_DOCTOR, CounterCondition.FLOAT_NURSE})
FALLBACK_SOURCE: Final = "policy_fallback"


class KpiSnapshot(FrozenModel):
    """Cumulative KPIs as known at time ``t_min`` (used by the dashboard).

    Attributes:
        served: Patients whose care started at or before ``t_min``.
        red_in_window: RED patients whose care started by ``t_min`` and in window.
        red_missed: RED patients whose deadline passed before ``t_min``
            without care in window.
        unserved_critical_overdue: RED patients past their deadline and still
            without care at ``t_min`` (not monotone: they may still get care).
        conflicts: Conflict threads among messages sent by ``t_min``.
        messages: Messages sent by ``t_min``.
        mean_response_min: Mean response time of incidents reached by ``t_min``.
    """

    t_min: float
    served: int
    red_in_window: int
    red_missed: int
    unserved_critical_overdue: int
    conflicts: int
    messages: int
    mean_response_min: float | None


# --------------------------------------------------------------------------- helpers


def _mean(values: Sequence[float]) -> float | None:
    """Returns the arithmetic mean, or None for no values."""
    return float(np.mean(values)) if values else None


def _ratio(numerator: float, denominator: float) -> float:
    """Returns ``numerator / denominator``, or 0.0 when the denominator is 0."""
    return numerator / denominator if denominator > 0 else 0.0


def _cared_by(victim: VictimOutcome, limit: float | None) -> bool:
    """Whether care started and did so at or before ``limit``."""
    return victim.care_start_min is not None and limit is not None and victim.care_start_min <= limit


def _patients(outcomes: RunOutcomes) -> list[VictimOutcome]:
    """Returns ``P``: the non-BLACK victims."""
    return [v for v in outcomes.victims if v.final_triage != Triage.BLACK]


def _reds(patients: Iterable[VictimOutcome]) -> list[VictimOutcome]:
    """Returns ``R``: patients whose final triage is RED."""
    return [v for v in patients if v.final_triage == Triage.RED]


def _response_times(incidents: Iterable[IncidentOutcome], until: float | None = None) -> list[float]:
    """First-ambulance response times of reached incidents (optionally by ``until``)."""
    return [
        inc.first_ambulance_on_scene_min - inc.t_report_min
        for inc in incidents
        if inc.first_ambulance_on_scene_min is not None
        and (until is None or inc.first_ambulance_on_scene_min <= until)
    ]


# --------------------------------------------------------------------------- victims


def _victim_metrics(outcomes: RunOutcomes) -> dict[str, object]:
    """Served, RED-window, survival-score and timing metrics of the victims."""
    horizon = outcomes.horizon_min
    patients = _patients(outcomes)
    reds = _reds(patients)
    served = sum(_cared_by(v, horizon) for v in patients)
    red_in_window = sum(_cared_by(v, v.deadline_min) for v in reds)
    weights = [SCORE_WEIGHTS[v.final_triage] for v in patients]
    in_window_weight = sum(w for v, w in zip(patients, weights) if _cared_by(v, v.deadline_min))
    red_response = [
        v.als_on_scene_min - v.t_report_min
        for v in patients
        if v.initial_triage == Triage.RED and v.als_on_scene_min is not None
    ]
    door_to_doctor = [
        v.treatment_start_min - v.arrived_hospital_min
        for v in patients
        if v.treatment_start_min is not None and v.arrived_hospital_min is not None
    ]
    return {
        "patients_total": len(patients),
        "victims_served": served,
        "served_rate": _ratio(served, len(patients)),
        "red_total": len(reds),
        "red_in_window": red_in_window,
        "red_in_window_rate": red_in_window / len(reds) if reds else None,
        "unserved_critical": sum(not _cared_by(v, horizon) for v in reds),
        "deteriorated": sum(v.deteriorated_at_min is not None for v in patients),
        "black_victims": len(outcomes.victims) - len(patients),
        "survival_weighted_score": _ratio(in_window_weight, sum(weights)),
        "red_response_mean_min": _mean(red_response),
        "door_to_doctor_mean_min": _mean(door_to_doctor),
    }


def _response_metrics(outcomes: RunOutcomes) -> dict[str, object]:
    """Incident counts and first-ambulance response-time statistics."""
    times = _response_times(outcomes.incidents)
    return {
        "incidents": len(outcomes.incidents),
        "incidents_reached": len(times),
        "response_time_mean_min": _mean(times),
        "response_time_p90_min": float(np.percentile(times, P90, method="linear")) if times else None,
    }


# --------------------------------------------------------------------------- utilisation


def _is_eligible(member: StaffView, colleagues: Sequence[StaffView]) -> bool:
    """Whether a staff member at a hospital may take patients (on shift or holdover).

    Holdover: an off-shift member stays eligible if, for at least one of its
    skills, no on-shift colleague of the same role at the hospital has it.
    """
    if member.on_shift:
        return True
    covered = {
        skill
        for other in colleagues
        if other.on_shift and other.role == member.role and other.staff_id != member.staff_id
        for skill in other.skills
    }
    return any(skill not in covered for skill in member.skills)


def _staff_ratio(view: PublicView, role: StaffRole) -> tuple[int, int]:
    """Returns ``(sum load, sum max_patients)`` over eligible staff of ``role``."""
    load = capacity = 0
    for hospital in view.hospitals:
        for member in hospital.staff:
            if member.role == role and _is_eligible(member, hospital.staff):
                load += member.load
                capacity += member.max_patients
    return load, capacity


def _mean_ratio(pairs: Iterable[tuple[float, float]]) -> float:
    """Mean of ``num / den`` over pairs with a positive denominator (0.0 if none)."""
    ratios = [num / den for num, den in pairs if den > 0]
    return float(np.mean(ratios)) if ratios else 0.0


def _utilisation_metrics(outcomes: RunOutcomes, records: Sequence[LogRecord]) -> dict[str, object]:
    """Ambulance, responder, ER-bed and staff utilisation."""
    horizon = outcomes.horizon_min
    views = [r.view for r in records if isinstance(r, SnapshotRecord) and 0.0 <= r.t_min < horizon]
    beds = (
        (sum(h.beds_occupied for h in v.hospitals), sum(h.er_capacity for h in v.hospitals))
        for v in views
    )
    return {
        "ambulance_utilisation": _ratio(
            sum(busy for _, busy in outcomes.ambulance_busy_min), outcomes.ambulances_in_service * horizon
        ),
        "responder_utilisation": _ratio(
            sum(busy for _, busy in outcomes.responder_busy_min), len(outcomes.responder_busy_min) * horizon
        ),
        "er_utilisation": _mean_ratio(beds),
        "doctor_utilisation": _mean_ratio(_staff_ratio(v, StaffRole.DOCTOR) for v in views),
        "nurse_utilisation": _mean_ratio(_staff_ratio(v, StaffRole.NURSE) for v in views),
    }


# --------------------------------------------------------------------------- events and agents


def _event_metrics(records: Sequence[LogRecord]) -> dict[str, object]:
    """Counts of the engine events that are metrics."""
    counts = Counter(r.event for r in records if isinstance(r, SimEventRecord))
    return {
        "hospital_full_on_arrival": counts[SimEventType.HOSPITAL_FULL_ON_ARRIVAL],
        "controller_omissions": counts[SimEventType.CONTROLLER_OMISSION],
        "diversions": counts[SimEventType.UNIT_DIVERTED],
        "float_staff_lent": counts[SimEventType.STAFF_LENT],
    }


def _agent_metrics(records: Sequence[LogRecord]) -> dict[str, object]:
    """Agent-call and fallback counts."""
    calls = [r for r in records if isinstance(r, AgentCallRecord)]
    return {
        "agent_calls": len(calls),
        "fallbacks": sum(c.source == FALLBACK_SOURCE or bool(c.fallbacks) for c in calls),
    }


# --------------------------------------------------------------------------- messages and threads


def _threads(messages: Iterable[Message]) -> dict[str, list[Message]]:
    """Groups messages by ``conversation_id``, keeping log order inside each thread."""
    threads: dict[str, list[Message]] = {}
    for message in messages:
        threads.setdefault(message.conversation_id, []).append(message)
    return threads


def _is_conflict(thread: Sequence[Message]) -> bool:
    """Whether a thread contains a REJECT, COUNTER or ESCALATE."""
    return any(m.type in CONFLICT_TYPES for m in thread)


def _case_kind(message: Message, case_kinds: Mapping[str, CaseKind]) -> CaseKind | None:
    """The case kind a message opens or closes, if any."""
    match message.proposed_action:
        case ResolveAction(case_kind=kind):
            return kind
        case StaffRequestAction():
            return CaseKind.FLOAT_STAFF
        case ConditionalAdmitAction(condition=condition):
            return CaseKind.FLOAT_STAFF if condition in FLOAT_COUNTER_CONDITIONS else CaseKind.RESERVE_RELEASE
        case CommitAction(case_id=case_id) if case_id is not None:
            return case_kinds.get(case_id)
    return None


def _thread_kinds(threads: Mapping[str, Sequence[Message]]) -> dict[str, int]:
    """Counts conflict threads per kind (without FLOAT_STAFF_COMPETITION)."""
    case_kinds = {
        m.proposed_action.case_id: m.proposed_action.case_kind
        for thread in threads.values()
        for m in thread
        if isinstance(m.proposed_action, ResolveAction)
    }
    counts = Counter[str]()
    for thread in threads.values():
        if not _is_conflict(thread):
            continue
        kinds = {k for m in thread if (k := _case_kind(m, case_kinds)) is not None}
        counts.update(kind.value for kind in kinds)
        if not kinds:
            counts[BED_REJECTION] += 1
    return counts


def _float_staff(records: Sequence[LogRecord]) -> tuple[StaffSpec, ...] | None:
    """Float-pool staff from the ``RunStartRecord``, or None if the log has none."""
    start = next((r for r in records if isinstance(r, RunStartRecord)), None)
    if start is None:
        return None
    return tuple(s for s in start.scenario_data.staff if s.home == FLOAT_HOME)


def _request_candidates(
    action: StaffRequestAction, t: float, float_staff: Sequence[StaffSpec] | None, lent: frozenset[str]
) -> frozenset[str]:
    """Float staff who could satisfy a REQUEST at time ``t``.

    Without scenario data, a ``role:skill`` token stands in for the candidates.
    """
    if float_staff is None:
        return frozenset({f"{action.role}:{action.skill}"})
    return frozenset(
        s.id
        for s in float_staff
        if s.role == action.role and action.skill in s.skills
        and s.id not in lent and s.shift_end_min > t
    )


def _float_competition(records: Sequence[LogRecord]) -> int:
    """Threads whose float REQUEST shared a candidate with another in the same round."""
    float_staff = _float_staff(records)
    lent: set[str] = set()
    rounds: dict[tuple[float, int], list[tuple[str, frozenset[str]]]] = {}
    for record in records:
        if isinstance(record, SimEventRecord) and record.event == SimEventType.STAFF_LENT and record.staff_id:
            lent.add(record.staff_id)
        elif isinstance(record, MessageRecord):
            message = record.message
            if not isinstance(action := message.proposed_action, StaffRequestAction):
                continue
            candidates = _request_candidates(action, message.t_min, float_staff, frozenset(lent))
            rounds.setdefault((message.t_min, message.round), []).append((message.conversation_id, candidates))
    competing: set[str] = set()
    for requests in rounds.values():
        for (thread_a, cands_a), (thread_b, cands_b) in combinations(requests, 2):
            if thread_a != thread_b and cands_a & cands_b:
                competing.update((thread_a, thread_b))
    return len(competing)


def _resolutions(threads: Mapping[str, Sequence[Message]]) -> dict[str, int]:
    """Counts the resolution of each thread's last ``CommitAction`` message."""
    counts = dict.fromkeys((r.value for r in Resolution), 0)
    for thread in threads.values():
        commits = [m.proposed_action for m in thread if isinstance(m.proposed_action, CommitAction)]
        if commits:
            counts[commits[-1].resolution.value] += 1
    return counts


def _messages(records: Iterable[LogRecord], until: float | None = None) -> list[Message]:
    """Messages in log order, optionally only those sent at or before ``until``."""
    return [
        r.message for r in records
        if isinstance(r, MessageRecord) and (until is None or r.t_min <= until)
    ]


def _message_metrics(records: Sequence[LogRecord]) -> dict[str, object]:
    """Message counts, conflicts by kind and resolutions."""
    messages = _messages(records)
    threads = _threads(messages)
    type_counts = Counter(m.type for m in messages)
    kind_counts = _thread_kinds(threads)
    kind_counts[FLOAT_STAFF_COMPETITION] = _float_competition(records)
    return {
        "messages_total": len(messages),
        "messages_by_type": {t.value: type_counts[t] for t in MessageType},
        "conflicts": sum(_is_conflict(thread) for thread in threads.values()),
        "conflicts_by_kind": {kind: kind_counts[kind] for kind in CONFLICT_KINDS},
        "resolutions": _resolutions(threads),
    }


# --------------------------------------------------------------------------- public API


def compute_metrics(outcomes: RunOutcomes, records: Sequence[LogRecord]) -> RunMetrics:
    """Computes the headline metrics of one run (plan section 6).

    Args:
        outcomes: The engine's end-of-run outcomes.
        records: The run log in order (snapshots, events, messages, agent
            calls); a ``RunStartRecord`` enables exact float-staff candidates.

    Returns:
        The metrics. Dict-valued metrics always hold every key (zeros included).
    """
    values: dict[str, object] = {}
    values.update(_response_metrics(outcomes))
    values.update(_victim_metrics(outcomes))
    values.update(_utilisation_metrics(outcomes, records))
    values.update(_event_metrics(records))
    values.update(_message_metrics(records))
    values.update(_agent_metrics(records))
    return RunMetrics.model_validate(values)


def kpis_at(outcomes: RunOutcomes, records: Sequence[LogRecord], t: float) -> KpiSnapshot:
    """KPIs as they stood at time ``t``: every threshold is evaluated at ``t``.

    Args:
        outcomes: The engine's end-of-run outcomes.
        records: The run log in order.
        t: Query time in minutes.

    Returns:
        The KPI snapshot. All fields except ``unserved_critical_overdue`` and
        ``mean_response_min`` are non-decreasing in ``t``.
    """
    patients = _patients(outcomes)
    reds = _reds(patients)
    past_deadline = [v for v in reds if v.deadline_min is not None and v.deadline_min < t]
    messages = _messages(records, until=t)
    threads = _threads(messages)
    return KpiSnapshot(
        t_min=t,
        served=sum(_cared_by(v, t) for v in patients),
        red_in_window=sum(_cared_by(v, t) and _cared_by(v, v.deadline_min) for v in reds),
        red_missed=sum(not _cared_by(v, v.deadline_min) for v in past_deadline),
        unserved_critical_overdue=sum(not _cared_by(v, t) for v in past_deadline),
        conflicts=sum(_is_conflict(thread) for thread in threads.values()),
        messages=len(messages),
        mean_response_min=_mean(_response_times(outcomes.incidents, until=t)),
    )
