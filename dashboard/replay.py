"""Pure replay of JSONL run logs for the dashboard.

Everything here is a pure function of a ``RunLog`` (or of the results folder
for discovery): no Streamlit, no Plotly, no engine. The app caches the loaded
logs and calls these helpers to answer "what did the world and the agents look
like at simulation minute ``t``?".

Results layout (plan section 7)::

    {results_dir}/runs/{scenario}/{condition}/{backend}/seed-{seed}_r{repeat}/events.jsonl
    {results_dir}/summary.csv
"""

import json
import logging
import re
from bisect import bisect_right
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import pandas as pd

from emergency_sim.domain.actions import (
    Action,
    AssignTransport,
    DispatchUnit,
    DivertUnit,
    LendStaff,
)
from emergency_sim.domain.enums import (
    TRIAGE_RANK,
    AgentRole,
    CaseKind,
    Condition,
    CounterCondition,
    IncidentStatus,
    MessageType,
    Resolution,
    SimEventType,
    StaffRole,
    UnitClass,
    UnitStatus,
    VictimStatus,
)
from emergency_sim.domain.log_records import (
    ActionRecord,
    AgentCallRecord,
    LogRecord,
    MessageRecord,
    RunEndRecord,
    RunStartRecord,
    SimEventRecord,
    SnapshotRecord,
)
from emergency_sim.domain.messages import (
    ALLOCATION_AGENT,
    ASSESSMENT_AGENT,
    DISPATCH_AGENT,
    AdmitAction,
    AssessAction,
    CommitAction,
    ConditionalAdmitAction,
    DeclineAction,
    DispatchAction,
    Message,
    ProposedAction,
    ResolveAction,
    StaffRequestAction,
    hospital_agent_name,
)
from emergency_sim.domain.outcomes import RunMetrics
from emergency_sim.evaluation.metrics import kpis_at
from emergency_sim.domain.views import IncidentView, PublicView, StaffView, UnitView
from emergency_sim.eventlog import read_log

LOGGER = logging.getLogger(__name__)

LOG_FILE_NAME: Final = "events.jsonl"
RUN_DIR_PATTERN: Final = re.compile(r"^seed-(?P<seed>-?\d+)_r(?P<repeat>\d+)$")
CONDITION_ORDER: Final = tuple(c.value for c in Condition)
TIME_EPS: Final = 1e-6

NEGOTIATION_TYPES: Final = frozenset({
    MessageType.PROPOSE, MessageType.ACCEPT, MessageType.REJECT, MessageType.COUNTER,
    MessageType.REQUEST, MessageType.ESCALATE, MessageType.COMMIT,
})
CONFLICT_TYPES: Final = frozenset({MessageType.REJECT, MessageType.COUNTER, MessageType.ESCALATE})
REQUEST_TYPES: Final = frozenset({MessageType.PROPOSE, MessageType.REQUEST, MessageType.ESCALATE})
BED_REJECTION: Final = "BED_REJECTION"
TRANSPORT: Final = "TRANSPORT"
CONTROLLER_ROLE: Final = "controller"
RESOLUTION_TOKEN: Final = re.compile(r"\b(" + "|".join(r.value for r in Resolution) + r")\b")
UNIT_OPTION_VERBS: Final = frozenset({"ASSIGN", "DIVERT"})
# Scenario display order (lightest to hardest); unknown scenarios sort after, by name.
SCENARIO_ORDER: Final = ("normal", "surge", "constrained")
# Engine bookkeeping that drains the t=0 baseline load; hidden from "recent events".
BACKGROUND_EVENTS: Final = frozenset({SimEventType.T0_BED_RELEASED, SimEventType.T0_STAFF_RELEASED})

STATUS_GROUP_ORDER: Final = (
    "Available", "To scene", "On scene", "Transporting", "At hospital", "Unavailable",
)
STATUS_GROUPS: Final = MappingProxyType({
    UnitStatus.AVAILABLE: "Available",
    UnitStatus.EN_ROUTE_SCENE: "To scene",
    UnitStatus.ON_SCENE: "On scene",
    UnitStatus.LOADING: "On scene",
    UnitStatus.AWAITING_DESTINATION: "On scene",
    UnitStatus.TRANSPORTING: "Transporting",
    UnitStatus.AT_HOSPITAL_FULL: "At hospital",
    UnitStatus.HANDOVER: "At hospital",
    UnitStatus.BUSY_T0: "Unavailable",
    UnitStatus.OUT_OF_SERVICE: "Unavailable",
})
_IDLE_STATUSES: Final = frozenset({UnitStatus.AVAILABLE, UnitStatus.OUT_OF_SERVICE})
_ACTIVE_ASSIGNMENT: Final = frozenset({
    UnitStatus.EN_ROUTE_SCENE, UnitStatus.ON_SCENE, UnitStatus.LOADING,
})
UNTRIAGED: Final = "UNTRIAGED"
MAX_SUMMARY_CHARS: Final = 160


class RunLogError(ValueError):
    """Raised when a log lacks the records a replay needs (e.g. ``run_start``)."""


# --- Loading ----------------------------------------------------------------


@dataclass(frozen=True)
class RunRef:
    """Where one run's ``events.jsonl`` lives and what it is.

    Attributes:
        scenario: Scenario name, e.g. ``"surge"``.
        condition: Condition value, e.g. ``"full"``.
        backend: Backend label, e.g. ``"offline"``.
        seed: Scenario seed.
        repeat: Repeat index.
        path: Path of the ``events.jsonl`` file.
    """

    scenario: str
    condition: str
    backend: str
    seed: int
    repeat: int
    path: Path


@dataclass(frozen=True)
class RunLog:
    """A run log split by record kind (each tuple in log order).

    Attributes:
        start: The ``run_start`` record (scenario truth, condition, backend).
        snapshots: Per-minute public views.
        events: Simulation events.
        messages: Inter-agent messages.
        agent_calls: Agent decisions (inputs, outputs, tool calls, fallbacks).
        actions: Engine actions with validation results.
        end: The ``run_end`` record, or None for a truncated log.
    """

    start: RunStartRecord
    snapshots: tuple[SnapshotRecord, ...]
    events: tuple[SimEventRecord, ...]
    messages: tuple[MessageRecord, ...]
    agent_calls: tuple[AgentCallRecord, ...]
    actions: tuple[ActionRecord, ...]
    end: RunEndRecord | None


def condition_sort_key(condition: str) -> tuple[int, str]:
    """Orders conditions as B0, B1, Full, then ablations, then unknown labels.

    Args:
        condition: Condition value.

    Returns:
        A sort key.
    """
    rank = CONDITION_ORDER.index(condition) if condition in CONDITION_ORDER else len(CONDITION_ORDER)
    return rank, condition


def scenario_sort_key(scenario: str) -> tuple[int, str]:
    """Orders scenarios as normal, surge, constrained, then unknown names alphabetically.

    Args:
        scenario: Scenario name.

    Returns:
        A sort key.
    """
    rank = SCENARIO_ORDER.index(scenario) if scenario in SCENARIO_ORDER else len(SCENARIO_ORDER)
    return rank, scenario


def _ref_from_path(path: Path, runs_root: Path) -> RunRef | None:
    """Parses ``runs/{scenario}/{condition}/{backend}/seed-{s}_r{r}/events.jsonl``."""
    parts = path.relative_to(runs_root).parts
    if len(parts) != 5:
        return None
    scenario, condition, backend, run_dir, _ = parts
    match = RUN_DIR_PATTERN.match(run_dir)
    if match is None:
        LOGGER.warning("skipping run folder with unexpected name: %s", path.parent)
        return None
    return RunRef(scenario, condition, backend, int(match["seed"]), int(match["repeat"]), path)


def discover_runs(results_dir: Path) -> list[RunRef]:
    """Finds every run log under ``results_dir/runs``.

    Args:
        results_dir: The results folder (may not exist).

    Returns:
        Run references sorted by scenario, condition order, backend, seed and
        repeat; empty if there are no logs.
    """
    runs_root = results_dir / "runs"
    if not runs_root.is_dir():
        return []
    refs = (_ref_from_path(p, runs_root) for p in runs_root.glob(f"*/*/*/*/{LOG_FILE_NAME}"))
    return sorted(
        (ref for ref in refs if ref is not None),
        key=lambda r: (r.scenario, condition_sort_key(r.condition), r.backend, r.seed, r.repeat),
    )


def run_log_from_records(records: Sequence[LogRecord]) -> RunLog:
    """Splits log records by kind.

    Args:
        records: Records in log order.

    Returns:
        The run log.

    Raises:
        RunLogError: If there is no ``run_start`` record.
    """
    starts = [r for r in records if isinstance(r, RunStartRecord)]
    if not starts:
        raise RunLogError("log has no run_start record")
    ends = [r for r in records if isinstance(r, RunEndRecord)]
    return RunLog(
        start=starts[0],
        snapshots=tuple(r for r in records if isinstance(r, SnapshotRecord)),
        events=tuple(r for r in records if isinstance(r, SimEventRecord)),
        messages=tuple(r for r in records if isinstance(r, MessageRecord)),
        agent_calls=tuple(r for r in records if isinstance(r, AgentCallRecord)),
        actions=tuple(r for r in records if isinstance(r, ActionRecord)),
        end=ends[-1] if ends else None,
    )


def load_run(path: Path) -> RunLog:
    """Reads and splits one ``events.jsonl``.

    Args:
        path: The log file.

    Returns:
        The run log.

    Raises:
        RunLogError: If the log has no ``run_start`` record.
        pydantic.ValidationError: If a line violates the log contract.
    """
    return run_log_from_records(read_log(path))


def all_records(run: RunLog) -> list[LogRecord]:
    """Reassembles the full record sequence, ordered by ``seq`` (stable).

    Args:
        run: The run log.

    Returns:
        Every record, ``run_start`` first and ``run_end`` (if any) last.
    """
    middle: list[LogRecord] = [
        *run.snapshots, *run.events, *run.messages, *run.agent_calls, *run.actions,
    ]
    middle.sort(key=lambda r: (r.seq, r.t_min))
    tail: list[LogRecord] = [run.end] if run.end is not None else []
    return [run.start, *middle, *tail]


def horizon_of(run: RunLog) -> float:
    """Returns the run's horizon in minutes."""
    return run.start.scenario_data.horizon_min


# --- Time slicing -----------------------------------------------------------


def snapshot_at(run: RunLog, t: float) -> SnapshotRecord:
    """Returns the latest snapshot with ``t_min <= t`` (the first one if t is earlier).

    Args:
        run: The run log.
        t: Simulation minute.

    Returns:
        The snapshot.

    Raises:
        RunLogError: If the log has no snapshots.
    """
    if not run.snapshots:
        raise RunLogError("log has no snapshots")
    index = bisect_right([s.t_min for s in run.snapshots], t + TIME_EPS)
    return run.snapshots[max(index - 1, 0)]


def messages_until(run: RunLog, t: float) -> list[Message]:
    """Messages sent at or before ``t``, in log order.

    Args:
        run: The run log.
        t: Simulation minute.

    Returns:
        The messages.
    """
    return [r.message for r in run.messages if r.t_min <= t + TIME_EPS]


def events_until(run: RunLog, t: float) -> list[SimEventRecord]:
    """Simulation events at or before ``t``, in log order."""
    return [r for r in run.events if r.t_min <= t + TIME_EPS]


def foreground_events(events: Iterable[SimEventRecord]) -> list[SimEventRecord]:
    """Drops ``BACKGROUND_EVENTS`` (the t=0 baseline load draining away).

    Args:
        events: Simulation events.

    Returns:
        The remaining events, in order.
    """
    return [e for e in events if e.event not in BACKGROUND_EVENTS]


def agent_calls_until(run: RunLog, t: float) -> list[AgentCallRecord]:
    """Agent calls at or before ``t``, in log order."""
    return [r for r in run.agent_calls if r.t_min <= t + TIME_EPS]


def actions_until(run: RunLog, t: float) -> list[ActionRecord]:
    """Engine actions at or before ``t``, in log order."""
    return [r for r in run.actions if r.t_min <= t + TIME_EPS]


# --- Describing contents ----------------------------------------------------


def describe_action(action: Action) -> str:
    """One-line description of an engine action.

    Args:
        action: The action.

    Returns:
        E.g. ``"dispatch AMB-01 -> INC-01"``.
    """
    match action:
        case DispatchUnit(unit_id=unit, incident_id=incident):
            return f"dispatch {unit} -> {incident}"
        case DivertUnit(unit_id=unit, incident_id=incident):
            return f"divert {unit} -> {incident}"
        case AssignTransport(unit_id=unit, hospital_id=hospital, reserve_beds=reserve):
            return f"transport {unit} -> {hospital}" + (" (beds reserved)" if reserve else "")
        case LendStaff(staff_id=staff, hospital_id=hospital):
            return f"lend {staff} -> {hospital}"
    return str(action)


def describe_content(content: ProposedAction) -> str:
    """One-line description of a message's structured content.

    Args:
        content: ``Message.proposed_action``.

    Returns:
        A short human-readable summary.
    """
    match content:
        case AssessAction():
            red = ", suspected RED" if content.suspected_red else ""
            return (f"urgency {content.urgency}, ~{content.est_patients} patients{red}, "
                    f"{content.als_units_needed} ALS needed")
        case DispatchAction(unit_ids=units):
            return "dispatch " + ", ".join(units)
        case AdmitAction():
            return (f"admit {', '.join(content.victim_ids)} from {content.unit_id} at "
                    f"{content.hospital_id} (ETA {content.eta_min:.1f} min)")
        case ConditionalAdmitAction():
            skill = f" ({content.skill})" if content.skill else ""
            return f"admit {content.unit_id} at {content.hospital_id} if {content.condition}{skill}"
        case DeclineAction(reason=reason):
            return f"decline: {reason}"
        case StaffRequestAction():
            return f"request float {content.role} ({content.skill}) for {content.hospital_id}"
        case ResolveAction():
            return f"{content.case_kind} {content.case_id}: options {', '.join(content.option_ids)}"
        case CommitAction():
            acts = "; ".join(describe_action(a) for a in content.actions) or "no change"
            return f"{content.resolution}: {acts}"
    return str(content)


def message_row(message: Message) -> dict[str, Any]:
    """Flattens a message for a table.

    Args:
        message: The message.

    Returns:
        Column name to value.
    """
    return {
        "t (min)": message.t_min,
        "id": message.message_id,
        "from": message.sender,
        "to": message.recipient,
        "type": message.type.value,
        "incident": message.incident_id,
        "content": describe_content(message.proposed_action),
        "rationale": message.rationale,
        "thread": message.conversation_id,
        "round": message.round,
    }


def _truncate(text: str, limit: int = MAX_SUMMARY_CHARS) -> str:
    """Shortens ``text`` to ``limit`` characters with an ellipsis."""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _summarise_items(items: Iterable[Any], render: Callable[[Mapping[str, Any]], str]) -> str:
    """Renders the dict items of a list with ``render``, joined by '; '."""
    return "; ".join(render(item) for item in items if isinstance(item, Mapping))


def summarise_output(output: Mapping[str, Any]) -> str:
    """One-line summary of an agent's structured output (any of the four roles).

    Args:
        output: ``AgentCallRecord.output``.

    Returns:
        A short summary; falls back to compact JSON.
    """
    if "reports" in output:
        text = _summarise_items(output["reports"], lambda r: (
            f"{r.get('incident_id')} urgency {r.get('urgency')}"
            + (" (RED suspected)" if r.get("suspected_red") else "")))
    elif "orders" in output:
        orders = _summarise_items(output["orders"], lambda o: f"{o.get('unit_id')}->{o.get('incident_id')}")
        transports = _summarise_items(output.get("transports", []), lambda p: (
            f"{p.get('unit_id')}: {' > '.join(p.get('hospital_ranking', []))}"))
        extra = f"{len(output.get('diversions', []))} diversions, {len(output.get('contentions', []))} contentions"
        text = " | ".join(part for part in (orders, transports, extra) if part)
    elif "responses" in output:
        text = _summarise_items(output["responses"], lambda r: (
            f"{r.get('proposal_id')} {r.get('response')} ({r.get('reason')})"))
    elif "decisions" in output:
        text = _summarise_items(output["decisions"], lambda d: f"{d.get('case_id')} -> {d.get('chosen_option_id')}")
    else:
        text = json.dumps(output, sort_keys=True)
    return _truncate(text or "no items")


def call_rationales(call: AgentCallRecord) -> list[str]:
    """Collects every ``rationale`` string in a call's output (deduplicated, in order).

    Args:
        call: The agent call.

    Returns:
        The rationales.
    """
    found: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if key == "rationale" and isinstance(value, str) and value:
                    found.append(value)
                else:
                    walk(value)
        elif isinstance(node, list | tuple):
            for item in node:
                walk(item)

    walk(call.output)
    return list(dict.fromkeys(found))


# --- Incidents and resources ------------------------------------------------


def incident_severity(incident: IncidentView) -> str:
    """Worst known triage of an incident, or ``"UNTRIAGED"``.

    Args:
        incident: The incident view.

    Returns:
        ``"RED"``, ``"YELLOW"``, ``"GREEN"``, ``"BLACK"`` or ``"UNTRIAGED"``.
    """
    if not incident.triaged or not incident.victims:
        return UNTRIAGED
    worst = min((v.triage for v in incident.victims), key=TRIAGE_RANK.__getitem__)
    return worst.value


def triage_counts(incident: IncidentView) -> str:
    """Victim counts by triage, e.g. ``"1 RED, 2 GREEN"`` (``"untriaged"`` before triage)."""
    if not incident.triaged:
        return "untriaged"
    counts = Counter(v.triage for v in incident.victims)
    ordered = sorted(counts, key=TRIAGE_RANK.__getitem__)
    return ", ".join(f"{counts[t]} {t.value}" for t in ordered) or "none"


def incident_rows(view: PublicView) -> list[dict[str, Any]]:
    """One table row per reported incident at the view's time.

    Args:
        view: A public view.

    Returns:
        Rows with id, type, report time, status, severity, victims, waiting
        on scene, assigned units and location.
    """
    return [
        {
            "incident": inc.incident_id,
            "type": inc.incident_type,
            "reported (min)": inc.t_report_min,
            "status": inc.status.value,
            "severity": incident_severity(inc),
            "victims": triage_counts(inc),
            "waiting on scene": sum(v.status is VictimStatus.WAITING for v in inc.victims),
            "units": ", ".join(inc.assigned_unit_ids),
            "location": inc.intersection,
        }
        for inc in view.incidents
    ]


def event_rows(events: Sequence[SimEventRecord]) -> list[dict[str, Any]]:
    """Flattens simulation events for a table."""
    return [
        {
            "t (min)": e.t_min,
            "event": e.event.value,
            "incident": e.incident_id or "",
            "unit": e.unit_id or "",
            "hospital": e.hospital_id or "",
            "victims": ", ".join(e.victim_ids),
            "detail": e.detail,
        }
        for e in events
    ]


def _eligible(staff: StaffView) -> bool:
    """Staff counted as capacity: on shift, or still carrying patients (holdover)."""
    return staff.on_shift or staff.load > 0


@dataclass(frozen=True)
class ResourceSummary:
    """Availability and load of every resource class at one snapshot.

    Attributes:
        ambulances_available: Ambulances with status ``available``.
        ambulances_busy: Ambulances on a task (not available, not out of service).
        ambulances_in_service: Ambulances not out of service.
        responders_available: Responder teams available.
        responders_busy: Responder teams on a task.
        responders_total: All responder teams.
        er_capacity: Total ER beds.
        beds_occupied: Occupied ER beds.
        beds_reserved: Beds reserved for inbound units.
        beds_free: Free unreserved beds.
        queue: Patients queued for a bed.
        waiting_for_staff: Bedded patients waiting for a doctor or nurse.
        doctors_on_shift: Eligible doctors at hospitals.
        doctor_load: Patients carried by those doctors.
        doctor_capacity: Their combined ``max_patients``.
        nurses_on_shift: Eligible nurses at hospitals.
        nurse_load: Patients carried by those nurses.
        nurse_capacity: Their combined ``max_patients``.
        float_available: Float-pool staff not yet lent.
        float_lent: Float staff working at a hospital.
    """

    ambulances_available: int
    ambulances_busy: int
    ambulances_in_service: int
    responders_available: int
    responders_busy: int
    responders_total: int
    er_capacity: int
    beds_occupied: int
    beds_reserved: int
    beds_free: int
    queue: int
    waiting_for_staff: int
    doctors_on_shift: int
    doctor_load: int
    doctor_capacity: int
    nurses_on_shift: int
    nurse_load: int
    nurse_capacity: int
    float_available: int
    float_lent: int


def ratio(numerator: float, denominator: float) -> float:
    """``numerator / denominator``, or 0.0 when the denominator is 0."""
    return numerator / denominator if denominator else 0.0


def _units_of(view: PublicView, unit_class: UnitClass) -> list[UnitView]:
    """Units of one class."""
    return [u for u in view.units if u.unit_class is unit_class]


def _staff_totals(view: PublicView, role: StaffRole) -> tuple[int, int, int]:
    """(eligible count, load, capacity) of one role over all hospitals."""
    members = [s for h in view.hospitals for s in h.staff if s.role is role and _eligible(s)]
    return len(members), sum(s.load for s in members), sum(s.max_patients for s in members)


def resource_summary(view: PublicView) -> ResourceSummary:
    """Summarises resource availability and utilisation in a view.

    Args:
        view: A public view.

    Returns:
        The summary.
    """
    ambulances = _units_of(view, UnitClass.AMBULANCE)
    responders = _units_of(view, UnitClass.RESPONDER)
    doctors = _staff_totals(view, StaffRole.DOCTOR)
    nurses = _staff_totals(view, StaffRole.NURSE)
    lent = [s for h in view.hospitals for s in h.staff if s.home != h.hospital_id]
    return ResourceSummary(
        ambulances_available=sum(u.status is UnitStatus.AVAILABLE for u in ambulances),
        ambulances_busy=sum(u.status not in _IDLE_STATUSES for u in ambulances),
        ambulances_in_service=sum(u.status is not UnitStatus.OUT_OF_SERVICE for u in ambulances),
        responders_available=sum(u.status is UnitStatus.AVAILABLE for u in responders),
        responders_busy=sum(u.status not in _IDLE_STATUSES for u in responders),
        responders_total=len(responders),
        er_capacity=sum(h.er_capacity for h in view.hospitals),
        beds_occupied=sum(h.beds_occupied for h in view.hospitals),
        beds_reserved=sum(h.beds_reserved for h in view.hospitals),
        beds_free=sum(h.beds_free_unreserved for h in view.hospitals),
        queue=sum(h.queue_length for h in view.hospitals),
        waiting_for_staff=sum(h.waiting_for_staff for h in view.hospitals),
        doctors_on_shift=doctors[0], doctor_load=doctors[1], doctor_capacity=doctors[2],
        nurses_on_shift=nurses[0], nurse_load=nurses[1], nurse_capacity=nurses[2],
        float_available=sum(s.hospital_id is None for s in view.float_pool),
        float_lent=len(lent),
    )


UTILISATION_SERIES: Final = ("Ambulances", "Responders", "ER beds", "Doctors", "Nurses")


def utilisation_series(run: RunLog) -> list[dict[str, float]]:
    """Per-snapshot utilisation (percent) of each resource class.

    Args:
        run: The run log.

    Returns:
        One row per snapshot: ``t`` plus one key per ``UTILISATION_SERIES``.
    """
    rows = []
    for snap in run.snapshots:
        s = resource_summary(snap.view)
        rows.append({
            "t": snap.t_min,
            "Ambulances": 100 * ratio(s.ambulances_busy, s.ambulances_in_service),
            "Responders": 100 * ratio(s.responders_busy, s.responders_total),
            "ER beds": 100 * ratio(s.beds_occupied, s.er_capacity),
            "Doctors": 100 * ratio(s.doctor_load, s.doctor_capacity),
            "Nurses": 100 * ratio(s.nurse_load, s.nurse_capacity),
        })
    return rows


@dataclass(frozen=True)
class UnitInterval:
    """A maximal stretch of time a unit spent in one status group.

    Attributes:
        unit_id: Unit id.
        unit_label: Unit id with type, e.g. ``"AMB-01 (ALS)"``.
        group: One of ``STATUS_GROUP_ORDER``.
        status: The status at the start of the stretch.
        start: Start minute.
        end: End minute.
    """

    unit_id: str
    unit_label: str
    group: str
    status: str
    start: float
    end: float


def unit_intervals(run: RunLog) -> list[UnitInterval]:
    """Run-length encodes every unit's status group over the snapshots.

    A snapshot's status holds until the next snapshot (the last one until the
    horizon).

    Args:
        run: The run log.

    Returns:
        Intervals sorted by unit id, then start.
    """
    if not run.snapshots:
        return []
    times = [s.t_min for s in run.snapshots]
    ends = [*times[1:], max(horizon_of(run), times[-1])]
    open_: dict[str, UnitInterval] = {}
    done: list[UnitInterval] = []
    for snap, end in zip(run.snapshots, ends, strict=True):
        for unit in snap.view.units:
            group = STATUS_GROUPS[unit.status]
            current = open_.get(unit.unit_id)
            if current is not None and current.group == group:
                open_[unit.unit_id] = replace(current, end=end)
                continue
            if current is not None:
                done.append(current)
            label = f"{unit.unit_id} ({unit.unit_type})"
            open_[unit.unit_id] = UnitInterval(unit.unit_id, label, group, unit.status.value, snap.t_min, end)
    done.extend(open_.values())
    return sorted(done, key=lambda i: (i.unit_id, i.start))


# --- Negotiation threads ----------------------------------------------------


@dataclass(frozen=True)
class ThreadSummary:
    """One negotiation thread (all messages sharing a ``conversation_id``).

    Attributes:
        conversation_id: Thread id, e.g. ``"NEG-0003"``.
        incident_id: Incident of the first message.
        kind: Case kind(s) raised in the thread, or ``BED_REJECTION`` /
            ``TRANSPORT`` / ``OTHER``.
        contested_resource: What was contested, e.g. an ER bed or a float doctor.
        messages: The thread's messages in log order.
        resolution: ``Resolution`` value of the closing message, or None if open.
        t_min: Time of the first message.
    """

    conversation_id: str
    incident_id: str
    kind: str
    contested_resource: str
    messages: tuple[Message, ...]
    resolution: str | None
    t_min: float


def _thread_kind(messages: Sequence[Message]) -> str:
    """Case kinds raised in a thread, else a coarse label."""
    kinds: list[str] = []
    for m in messages:
        content = m.proposed_action
        if isinstance(content, ResolveAction):
            kinds.append(content.case_kind.value)
        elif isinstance(content, StaffRequestAction):
            kinds.append(CaseKind.FLOAT_STAFF.value)
    if kinds:
        return " -> ".join(dict.fromkeys(kinds))
    if any(m.type is MessageType.REJECT for m in messages):
        return BED_REJECTION
    if any(m.type is MessageType.PROPOSE for m in messages):
        return TRANSPORT
    return "OTHER"


def option_unit(option_id: str) -> str | None:
    """Unit named by a UNIT_CONTENTION option id (``ASSIGN:<unit>-><incident>`` / ``DIVERT:...``).

    Args:
        option_id: A case option id, e.g. ``"ASSIGN:AMB-01->INC-04"``.

    Returns:
        The unit id, or None for options that name no unit (``KEEP``, ``APPROVE``...).
    """
    verb, _, rest = option_id.partition(":")
    if verb not in UNIT_OPTION_VERBS or "->" not in rest:
        return None
    return rest.split("->", 1)[0] or None


def _contention_units(contents: Sequence[ProposedAction]) -> list[str]:
    """Units at stake in a UNIT_CONTENTION thread (from its options and COMMITs)."""
    units = {u for c in contents if isinstance(c, ResolveAction) for o in c.option_ids
             if (u := option_unit(o)) is not None}
    units |= {a.unit_id for c in contents if isinstance(c, CommitAction)
              for a in c.actions if isinstance(a, DispatchUnit | DivertUnit)}
    return sorted(units)


def plural(count: int, noun: str) -> str:
    """``"1 patient"`` / ``"2 patients"``."""
    return f"{count} {noun}" + ("" if count == 1 else "s")


def _contested_resource(messages: Sequence[Message]) -> str:
    """Names the resource the thread was about (most specific first)."""
    contents = [m.proposed_action for m in messages]
    for c in contents:
        if isinstance(c, StaffRequestAction):
            return f"float {c.role} ({c.skill}) for {c.hospital_id}"
    for c in contents:
        if isinstance(c, ConditionalAdmitAction) and c.condition is CounterCondition.RESERVE_RELEASE:
            return f"critical-reserve bed at {c.hospital_id} for {c.unit_id}"
    for c in contents:
        if isinstance(c, ResolveAction) and c.case_kind is CaseKind.UNIT_CONTENTION:
            units = _contention_units(contents)
            return "ambulance " + (" or ".join(units) if units else "assignment")
    for c in contents:
        if isinstance(c, AdmitAction | ConditionalAdmitAction):
            return f"ER bed for {c.unit_id} ({plural(len(c.victim_ids), 'patient')})"
    return describe_content(contents[0]) if contents else ""


def _commits(messages: Sequence[Message]) -> list[CommitAction]:
    """The ``CommitAction`` payloads of a thread, in log order."""
    return [m.proposed_action for m in messages if isinstance(m.proposed_action, CommitAction)]


def thread_resolution(messages: Sequence[Message]) -> str | None:
    """Resolution of a thread, read from its closing message.

    The coordinator puts every resolution in a ``CommitAction`` (ACCEPT,
    COMMIT, resource allocation's REJECT of a counter, INFORM), so in such
    logs the thread is resolved exactly when its latest message carries one;
    anything else (a hospital's REJECT, a pending PROPOSE / COUNTER /
    ESCALATE / REQUEST) means it is still open.

    Logs without any ``CommitAction`` (older or hand-made) fall back to
    heuristics: a ``Resolution`` token in the closing rationale; ACCEPT (after
    a REJECT -> ``ACCEPTED_AFTER_REJECT``); a dispatch INFORM carrying an
    admission (``FORCED_ADMISSION``); a REJECT from resource allocation
    (``COUNTER_DECLINED``).

    Args:
        messages: The thread's messages in log order.

    Returns:
        The ``Resolution`` value, or None if the thread is still open.
    """
    if not messages:
        return None
    last = messages[-1]
    if isinstance(last.proposed_action, CommitAction):
        return last.proposed_action.resolution.value
    if _commits(messages):
        return None
    return _legacy_resolution(messages)


def _legacy_resolution(messages: Sequence[Message]) -> str | None:
    """Heuristic resolution of a thread whose messages carry no ``CommitAction``."""
    last = messages[-1]
    if token := RESOLUTION_TOKEN.search(last.rationale):
        return token.group(1)
    if last.type is MessageType.ACCEPT:
        rejected = any(m.type in (MessageType.REJECT, MessageType.COUNTER) for m in messages[:-1])
        return (Resolution.ACCEPTED_AFTER_REJECT if rejected else Resolution.ACCEPTED).value
    if last.type is MessageType.INFORM and isinstance(last.proposed_action, AdmitAction):
        return Resolution.FORCED_ADMISSION.value
    if last.type is MessageType.REJECT and last.sender == ALLOCATION_AGENT:
        return Resolution.COUNTER_DECLINED.value
    return None


def resolution_path(messages: Sequence[Message]) -> tuple[str, ...]:
    """Every resolution a thread went through, in order (repeats collapsed).

    E.g. a reserve-release counter that resource allocation declined before
    another hospital accepted gives ``("COUNTER_DECLINED",
    "ACCEPTED_AFTER_REJECT")``; the COMMIT that RA sends to both parties of a
    granted counter counts once.

    Args:
        messages: The thread's messages in log order.

    Returns:
        ``Resolution`` values (empty if no message carries a ``CommitAction``).
    """
    path: list[str] = []
    for commit in _commits(messages):
        if not path or path[-1] != commit.resolution.value:
            path.append(commit.resolution.value)
    return tuple(path)


def threads_from_messages(messages: Sequence[Message]) -> list[ThreadSummary]:
    """Groups messages into negotiation threads.

    Conversations made only of PRIORITY_REPORT / INFORM messages are not
    negotiations and are skipped.

    Args:
        messages: Messages in log order.

    Returns:
        Threads ordered by first message time, then id.
    """
    grouped: dict[str, list[Message]] = {}
    for m in messages:
        grouped.setdefault(m.conversation_id, []).append(m)
    threads = [
        ThreadSummary(
            conversation_id=cid,
            incident_id=msgs[0].incident_id,
            kind=_thread_kind(msgs),
            contested_resource=_contested_resource(msgs),
            messages=tuple(msgs),
            resolution=thread_resolution(msgs),
            t_min=msgs[0].t_min,
        )
        for cid, msgs in grouped.items()
        if any(m.type in NEGOTIATION_TYPES for m in msgs)
    ]
    return sorted(threads, key=lambda th: (th.t_min, th.conversation_id))


def negotiation_threads(run: RunLog) -> list[ThreadSummary]:
    """All negotiation threads of a run.

    Args:
        run: The run log.

    Returns:
        Threads ordered by first message time, then id.
    """
    return threads_from_messages([r.message for r in run.messages])


def is_conflict(thread: ThreadSummary) -> bool:
    """True if the thread contains a REJECT, COUNTER or ESCALATE (plan section 6)."""
    return any(m.type in CONFLICT_TYPES for m in thread.messages)


def thread_row(thread: ThreadSummary) -> dict[str, Any]:
    """Flattens a thread for a table.

    Args:
        thread: The thread.

    Returns:
        Column name to value, including the performative sequence.
    """
    hospitals = sorted({m.recipient for m in thread.messages if m.recipient.startswith("hospital_")})
    path = resolution_path(thread.messages)
    return {
        "thread": thread.conversation_id,
        "t (min)": thread.t_min,
        "incident": thread.incident_id,
        "kind": thread.kind,
        "resolution": thread.resolution or "open",
        "conflict": is_conflict(thread),
        "contested resource": thread.contested_resource,
        "sequence": " > ".join(m.type.value for m in thread.messages),
        "resolution path": " > ".join(path) if len(path) > 1 else "",
        "rounds": max(m.round for m in thread.messages),
        "hospitals asked": ", ".join(h.removeprefix("hospital_") for h in hospitals),
    }


def thread_label(thread: ThreadSummary) -> str:
    """Short one-line label of a thread for pickers, e.g. ``"NEG-0011 · t=61.5 · UNIT_CONTENTION · RA_COMMIT"``.

    Args:
        thread: The thread.

    Returns:
        The label.
    """
    return f"{thread.conversation_id} · t={thread.t_min:g} · {thread.kind} · {thread.resolution or 'open'}"


# --- Agent activity ---------------------------------------------------------


@dataclass(frozen=True)
class AgentStatus:
    """What one agent (or the B0 controller) is doing at time ``t``.

    Attributes:
        agent: Agent name, e.g. ``"hospital_H1"``.
        role: ``AgentRole`` value, or ``"controller"`` for B0.
        calls: Calls made so far.
        last_call_t: Time of the last call (or last action for a controller).
        last_triggers: Trigger kinds of the last call.
        last_decision: Summary of the last output (or action).
        fallbacks: Calls so far that used a policy fallback.
        pending: Operational items waiting on this agent at ``t``.
    """

    agent: str
    role: str
    calls: int
    last_call_t: float | None
    last_triggers: tuple[str, ...]
    last_decision: str
    fallbacks: int
    pending: tuple[str, ...]


def used_fallback(call: AgentCallRecord) -> bool:
    """True if the call fell back to the policy wholly or in part."""
    return call.source == "policy_fallback" or bool(call.fallbacks)


def agent_roster(run: RunLog) -> list[tuple[str, str]]:
    """Agents to show as ``(name, role)``.

    With agent calls in the log: the four roles (one hospital agent per
    hospital) plus any other agent seen. Without (B0): the action actors.

    Args:
        run: The run log.

    Returns:
        The roster in display order.
    """
    if not run.agent_calls:
        return [(actor, CONTROLLER_ROLE) for actor in dict.fromkeys(a.actor for a in run.actions)]
    roster = {ASSESSMENT_AGENT: AgentRole.ASSESSMENT.value, DISPATCH_AGENT: AgentRole.DISPATCH.value}
    for spec in sorted(run.start.scenario_data.hospitals, key=lambda h: h.id):
        roster[hospital_agent_name(spec.id)] = AgentRole.HOSPITAL.value
    roster[ALLOCATION_AGENT] = AgentRole.ALLOCATION.value
    for call in run.agent_calls:
        roster.setdefault(call.agent, call.role.value)
    return list(roster.items())


def _incident_need(incident: IncidentView, units: Mapping[str, UnitView]) -> str:
    """Describes an open incident's unmet ambulance need, or '' if covered."""
    assigned = [units[u] for u in incident.assigned_unit_ids if u in units]
    ambulances = [u for u in assigned
                  if u.unit_class is UnitClass.AMBULANCE and u.status in _ACTIVE_ASSIGNMENT]
    if not incident.triaged:
        return "" if ambulances else "no ambulance assigned"
    waiting = sum(v.status is VictimStatus.WAITING for v in incident.victims)
    seats = sum(u.seats for u in ambulances)
    return f"{waiting} waiting, {seats} seats en route" if waiting > seats else ""


def dispatch_pending(view: PublicView) -> tuple[str, ...]:
    """Incidents with unmet needs and loaded units without a destination.

    Args:
        view: A public view.

    Returns:
        Pending items.
    """
    units = {u.unit_id: u for u in view.units}
    items = [
        f"{inc.incident_id}: {need}"
        for inc in view.incidents
        if inc.status is IncidentStatus.OPEN and (need := _incident_need(inc, units))
    ]
    items += [f"{u.unit_id}: loaded, awaiting destination"
              for u in view.units if u.status is UnitStatus.AWAITING_DESTINATION]
    return tuple(items)


def hospital_pending(view: PublicView, hospital_id: str) -> tuple[str, ...]:
    """Inbound reserved units and queued patients at one hospital.

    Args:
        view: A public view.
        hospital_id: Hospital id.

    Returns:
        Pending items (empty if the hospital is unknown).
    """
    for h in view.hospitals:
        if h.hospital_id == hospital_id:
            items = [f"{u} inbound" for u in h.inbound_unit_ids]
            if h.queue_length:
                items.append(f"{h.queue_length} queued for a bed")
            if h.waiting_for_staff:
                items.append(f"{h.waiting_for_staff} waiting for staff")
            return tuple(items)
    return ()


def unanswered_requests(messages: Sequence[Message]) -> list[Message]:
    """PROPOSE / REQUEST / ESCALATE messages that no message replies to.

    Args:
        messages: Messages (typically up to some time).

    Returns:
        The unanswered requests, in order.
    """
    answered = {m.in_reply_to for m in messages if m.in_reply_to}
    return [m for m in messages if m.type in REQUEST_TYPES and m.message_id not in answered]


def _pending_for(agent: str, view: PublicView, messages: Sequence[Message]) -> tuple[str, ...]:
    """Operational items pending for one agent."""
    if agent == ASSESSMENT_AGENT:
        return tuple(f"{i.incident_id}: awaiting triage" for i in view.incidents
                     if i.status is IncidentStatus.OPEN and not i.triaged)
    if agent == ALLOCATION_AGENT:
        return tuple(f"{m.conversation_id}: {describe_content(m.proposed_action)}"
                     for m in unanswered_requests(messages) if m.recipient == ALLOCATION_AGENT)
    if agent.startswith("hospital_"):
        return hospital_pending(view, agent.removeprefix("hospital_"))
    return dispatch_pending(view)


def _status_from_calls(agent: str, role: str, calls: Sequence[AgentCallRecord],
                       pending: tuple[str, ...]) -> AgentStatus:
    """Builds an agent's status from its calls so far."""
    last = calls[-1] if calls else None
    return AgentStatus(
        agent=agent,
        role=role,
        calls=len(calls),
        last_call_t=last.t_min if last else None,
        last_triggers=tuple(t.value for t in last.triggers) if last else (),
        last_decision=summarise_output(last.output) if last else "no decision yet",
        fallbacks=sum(used_fallback(c) for c in calls),
        pending=pending,
    )


def _status_from_actions(agent: str, actions: Sequence[ActionRecord],
                         pending: tuple[str, ...]) -> AgentStatus:
    """Builds a controller's (B0) status from its actions so far."""
    last = actions[-1] if actions else None
    decision = "no action yet"
    if last is not None:
        decision = describe_action(last.action) + ("" if last.ok else f" (rejected: {last.reason})")
    return AgentStatus(agent, CONTROLLER_ROLE, 0, last.t_min if last else None, (), decision, 0, pending)


def agent_status_at(run: RunLog, t: float) -> list[AgentStatus]:
    """Status of every agent at time ``t``.

    Args:
        run: The run log.
        t: Simulation minute.

    Returns:
        One status per agent in ``agent_roster`` order.
    """
    view = snapshot_at(run, t).view
    messages = messages_until(run, t)
    calls = agent_calls_until(run, t)
    actions = actions_until(run, t)
    statuses = []
    for agent, role in agent_roster(run):
        pending = _pending_for(agent, view, messages)
        if role == CONTROLLER_ROLE:
            mine = [a for a in actions if a.actor == agent]
            statuses.append(_status_from_actions(agent, mine, pending))
        else:
            statuses.append(_status_from_calls(agent, role, [c for c in calls if c.agent == agent], pending))
    return statuses


def last_call_of(run: RunLog, agent: str, t: float) -> AgentCallRecord | None:
    """The agent's latest call at or before ``t``, or None."""
    calls = [c for c in agent_calls_until(run, t) if c.agent == agent]
    return calls[-1] if calls else None


# --- Step through a decision point ------------------------------------------


@dataclass(frozen=True)
class TraceRow:
    """One record of a decision point, for the step-through view.

    Attributes:
        seq: Log sequence number (protocol order).
        kind: ``"message"``, ``"agent call"`` or ``"action"``.
        actor: Who acted (``sender -> recipient`` for messages).
        summary: What happened.
        detail: Rationale, tool calls / fallbacks, or validation result.
        open_request: True for a PROPOSE/REQUEST/ESCALATE not yet answered.
    """

    seq: int
    kind: str
    actor: str
    summary: str
    detail: str
    open_request: bool


def decision_times(run: RunLog) -> list[float]:
    """Distinct minutes at which agents or the controller decided or talked.

    Args:
        run: The run log.

    Returns:
        Sorted times.
    """
    records: Iterable[MessageRecord | AgentCallRecord | ActionRecord] = (
        *run.messages, *run.agent_calls, *run.actions)
    return sorted({r.t_min for r in records})


def _call_detail(call: AgentCallRecord) -> str:
    """Tool calls and fallbacks of an agent call, as one line."""
    tools = ", ".join(tc.name for tc in call.tool_calls) or "none"
    notes = "; ".join(f"{f.scope}:{f.rule}" + (f"[{f.key}]" if f.key else "") for f in call.fallbacks)
    return f"tools: {tools}; fallbacks: {notes or 'none'}"


def decision_trace(run: RunLog, t_point: float) -> list[TraceRow]:
    """Every message, agent call and action at one decision point, by ``seq``.

    Args:
        run: The run log.
        t_point: The decision point's minute.

    Returns:
        Rows in protocol order.
    """
    answered = {m.in_reply_to for m in messages_until(run, t_point) if m.in_reply_to}
    rows: list[TraceRow] = []
    for rec in run.messages:
        if abs(rec.t_min - t_point) <= TIME_EPS:
            m = rec.message
            rows.append(TraceRow(
                rec.seq, "message", f"{m.sender} -> {m.recipient}",
                f"{m.type.value}: {describe_content(m.proposed_action)}", m.rationale,
                m.type in REQUEST_TYPES and m.message_id not in answered))
    for call in run.agent_calls:
        if abs(call.t_min - t_point) <= TIME_EPS:
            rows.append(TraceRow(call.seq, "agent call", call.agent,
                                 f"[{call.source}] {summarise_output(call.output)}", _call_detail(call), False))
    for act in run.actions:
        if abs(act.t_min - t_point) <= TIME_EPS:
            result = "ok" if act.ok else f"rejected: {act.reason} {act.detail}".strip()
            rows.append(TraceRow(act.seq, "action", act.actor, describe_action(act.action), result, False))
    return sorted(rows, key=lambda r: r.seq)


# --- KPIs ---------------------------------------------------------------------


KpiValues = dict[str, float | int | None]


def _partial_kpis(run: RunLog, t: float) -> KpiValues:
    """KPIs at ``t`` for a truncated log (no ``run_end``): only what snapshots show.

    Args:
        run: The run log.
        t: Simulation minute.

    Returns:
        Keys of ``evaluation.metrics.KpiSnapshot``; outcome-based ones are None.
    """
    messages = messages_until(run, t)
    conflicts = sum(is_conflict(th) for th in threads_from_messages(messages))
    served = snapshot_at(run, t).counts.care_started_cum if run.snapshots else 0
    return {"t_min": t, "messages": len(messages), "conflicts": conflicts, "served": served,
            "red_in_window": None, "red_missed": None, "unserved_critical_overdue": None,
            "mean_response_min": None}


def kpis_at_time(run: RunLog, t: float) -> KpiValues:
    """KPIs at ``t``, computed by ``evaluation.metrics.kpis_at`` (single definition).

    A truncated log without a ``run_end`` record has no outcomes, so only the
    snapshot-based values are returned for it.

    Args:
        run: The run log.
        t: Simulation minute.

    Returns:
        KPI name to value.
    """
    if run.end is None:
        return _partial_kpis(run, t)
    return dict(kpis_at(run.end.outcomes, all_records(run), t).model_dump())


SCALAR_METRICS: Final = tuple(
    name for name, info in RunMetrics.model_fields.items() if "dict" not in str(info.annotation)
)


def final_metrics_rows(runs: Mapping[str, RunLog]) -> list[dict[str, Any]]:
    """Side-by-side final metrics, one row per scalar metric, one column per label.

    Args:
        runs: Label (e.g. condition) to run log; runs without ``run_end`` give None.

    Returns:
        Rows ``{"metric": name, label: value, ...}``.
    """
    metrics = {label: run.end.metrics if run.end else None for label, run in runs.items()}
    return [
        {"metric": name} | {label: getattr(m, name) if m else None for label, m in metrics.items()}
        for name in SCALAR_METRICS
    ]


# --- Cross-seed summary -----------------------------------------------------


SUMMARY_FILE_NAME: Final = "summary.csv"
SUMMARY_COLUMNS: Final = ("scenario", "condition", "backend", "metric", "n", "mean", "std", "min", "max")


def _summary_from_metrics(results_dir: Path) -> pd.DataFrame | None:
    """Builds the summary from ``metrics.json`` files via ``evaluation.report``, if available."""
    try:
        from emergency_sim.evaluation.report import load_results, summarise  # noqa: PLC0415 (optional)
    except ImportError:
        return None
    results = load_results(results_dir)
    return summarise(results) if results else None


def load_summary(results_dir: Path) -> pd.DataFrame:
    """Reads the cross-seed summary (``summary.csv``), or derives it from ``metrics.json`` files.

    Args:
        results_dir: The results folder (may not exist).

    Returns:
        A frame with ``SUMMARY_COLUMNS``; empty when nothing usable exists.
    """
    path = results_dir / SUMMARY_FILE_NAME
    frame = pd.read_csv(path) if path.is_file() else None
    if frame is None and (results_dir / "runs").is_dir():
        frame = _summary_from_metrics(results_dir)
    if frame is None or frame.empty:
        return pd.DataFrame(columns=list(SUMMARY_COLUMNS))
    missing = set(SUMMARY_COLUMNS) - set(frame.columns)
    if missing:
        LOGGER.warning("summary is missing columns %s; ignoring it", sorted(missing))
        return pd.DataFrame(columns=list(SUMMARY_COLUMNS))
    frame = frame.astype({"scenario": str, "condition": str, "backend": str, "metric": str})
    return frame.sort_values(["scenario", "metric", "condition", "backend"], ignore_index=True)
