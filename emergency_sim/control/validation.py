"""Semantic validation of agent outputs with policy fallback (plan 5.4).

Schema validity is checked by the backend (a malformed output becomes a
call-level ``policy_fallback``). This module checks *meaning* against the
agent's own input and replaces the smallest invalid part with the policy's
answer for that part:

* Assessment, Hospital, Allocation: per item (incident / proposal / case).
* Dispatch: per section (orders, diversions, contentions, transports), then
  a cross-section unit-conflict check that can replace the whole output.

Every replacement is recorded as a ``FallbackNote``. The engine remains the
final gate for anything that slips through (``ENGINE_REJECTED``).
"""

import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from emergency_sim.control.needs import pre_triage_slots
from emergency_sim.domain.agent_io import (
    AdmissionRequest,
    AdmissionResponse,
    AllocationInput,
    AllocationOutput,
    AssessmentInput,
    AssessmentOutput,
    CaseDecision,
    DispatchInput,
    DispatchOutput,
    HospitalInput,
    HospitalOutput,
    IncidentNeed,
    PriorityReport,
)
from emergency_sim.domain.base import AgentIOModel
from emergency_sim.domain.enums import (
    AmbulanceType,
    CounterCondition,
    UnitClass,
    UnitStatus,
)
from emergency_sim.domain.log_records import FallbackNote
from emergency_sim.policies.allocation import DECLINE_OPTION_ID, granted_staff
from emergency_sim.policies.estimates import NURSE_SKILL, doctor_skill_for
from emergency_sim.policies.params import PolicyParams

URGENCY_MIN = 0
URGENCY_MAX = 100
MAX_DETAIL_CHARS = 300

type NoteScope = Literal["call", "section", "item"]
"""Granularity of a fallback replacement."""

type Violation = tuple[str, str]
"""``(rule, detail)`` of one failed check."""


@dataclass(frozen=True)
class Validated[T: AgentIOModel]:
    """A validated output and the notes of every replacement made.

    Attributes:
        output: The output to act on.
        notes: Replacements made (empty if the agent output was used as is).
    """

    output: T
    notes: tuple[FallbackNote, ...]


def _note(scope: NoteScope, key: str | None, rule: str, detail: str) -> FallbackNote:
    """Builds a fallback note with a truncated detail.

    Args:
        scope: ``"call"``, ``"section"`` or ``"item"``.
        key: Section name or item id.
        rule: Failed rule.
        detail: Explanation.

    Returns:
        The note.
    """
    return FallbackNote(scope=scope, key=key, rule=rule, detail=detail[:MAX_DETAIL_CHARS])


def _merge_items[I](
    keys: Sequence[str],
    items: Sequence[I],
    policy_items: Sequence[I],
    key_of: Callable[[I], str],
    check: Callable[[I], Violation | None],
) -> tuple[list[I], list[FallbackNote]]:
    """Keeps one valid agent item per key, else the policy's item for it.

    Args:
        keys: Expected keys, in output order.
        items: Agent items.
        policy_items: Policy items (one per key, or none for keys the policy
            skips).
        key_of: Key of an item.
        check: Returns a violation for an invalid item, else None.

    Returns:
        The merged items (in ``keys`` order) and the notes.
    """
    notes: list[FallbackNote] = []
    counts = Counter(key_of(i) for i in items)
    chosen: dict[str, I] = {}
    duplicates_noted: set[str] = set()
    for item in items:
        key = key_of(item)
        if key not in keys:
            notes.append(_note("item", key, "unknown_item", item_json(item)))
        elif counts[key] > 1:
            if key not in duplicates_noted:
                duplicates_noted.add(key)
                notes.append(_note("item", key, "duplicate_item", f"{counts[key]} answers"))
        elif (violation := check(item)) is not None:
            notes.append(_note("item", key, violation[0], violation[1]))
        else:
            chosen[key] = item
    policy_by_key = {key_of(i): i for i in policy_items}
    merged: list[I] = []
    for key in keys:
        if key in chosen:
            merged.append(chosen[key])
        elif key in policy_by_key:
            if counts[key] == 0:
                notes.append(_note("item", key, "missing_item", "no answer for this item"))
            merged.append(policy_by_key[key])
    return merged, notes


def item_json(item: object) -> str:
    """Serialises an item for a note's detail.

    Args:
        item: A Pydantic model or anything printable.

    Returns:
        JSON (or ``str``) of the item.
    """
    dump = getattr(item, "model_dump_json", None)
    return dump() if callable(dump) else str(item)


# --- Assessment ---------------------------------------------------------------


def _check_report(report: PriorityReport) -> Violation | None:
    """Checks the ranges of one priority report.

    Args:
        report: The report.

    Returns:
        A violation or None.
    """
    if not URGENCY_MIN <= report.urgency <= URGENCY_MAX:
        return "urgency_out_of_range", f"urgency {report.urgency}"
    if report.est_patients < 0 or report.als_units_needed < 0:
        return "negative_count", f"est_patients {report.est_patients}, als {report.als_units_needed}"
    if not math.isfinite(report.deadline_in_min):
        return "bad_deadline", f"deadline {report.deadline_in_min}"
    return None


def validate_assessment(
    output: AssessmentOutput, inp: AssessmentInput, policy: AssessmentOutput
) -> Validated[AssessmentOutput]:
    """Validates an assessment output per incident.

    Args:
        output: The agent's output.
        inp: Its input.
        policy: The policy's output for the same input.

    Returns:
        One report per input incident, invalid ones replaced by the policy's.
    """
    keys = [i.incident_id for i in inp.incidents]
    reports, notes = _merge_items(
        keys, output.reports, policy.reports, lambda r: r.incident_id, _check_report
    )
    return Validated(AssessmentOutput(reports=reports), tuple(notes))


# --- Hospital -----------------------------------------------------------------


def _counter_skill_ok(response: AdmissionResponse, request: AdmissionRequest) -> bool:
    """Checks the consistency of a COUNTER's condition and skill.

    Args:
        response: A COUNTER response.
        request: The request it answers.

    Returns:
        True if FLOAT_DOCTOR names a doctor skill some patient needs,
        FLOAT_NURSE names no skill or the nurse skill, and RESERVE_RELEASE
        names no skill.
    """
    match response.counter_condition:
        case CounterCondition.FLOAT_DOCTOR:
            needed = {doctor_skill_for(v.required_capability) for v in request.victims}
            return response.counter_skill in needed
        case CounterCondition.FLOAT_NURSE:
            return response.counter_skill in (None, NURSE_SKILL)
        case CounterCondition.RESERVE_RELEASE:
            return response.counter_skill is None
        case _:
            return False


def _hospital_checker(inp: HospitalInput) -> Callable[[AdmissionResponse], Violation | None]:
    """Returns the per-response check for one hospital payload.

    Args:
        inp: The hospital payload.

    Returns:
        A check function.
    """
    requests = {r.proposal_id: r for r in inp.requests}

    def check(response: AdmissionResponse) -> Violation | None:
        """Checks one response to a known proposal."""
        request = requests[response.proposal_id]
        if response.response not in inp.allowed_responses:
            return "response_not_allowed", response.response
        needed = {v.required_capability for v in request.victims}
        if response.response == "ACCEPT" and not needed <= set(inp.capabilities):
            return "accept_not_capable", f"needs {sorted(needed)}"
        if response.response != "COUNTER":
            return None
        if response.counter_condition not in inp.allowed_counter_conditions:
            return "condition_not_allowed", str(response.counter_condition)
        if not _counter_skill_ok(response, request):
            return "counter_skill_inconsistent", f"{response.counter_condition}/{response.counter_skill}"
        return None

    return check


def validate_hospital(
    output: HospitalOutput, inp: HospitalInput, policy: HospitalOutput
) -> Validated[HospitalOutput]:
    """Validates a hospital output per proposal.

    Args:
        output: The agent's output.
        inp: Its input.
        policy: The policy's output for the same input.

    Returns:
        One response per request (``proposal_id`` order).
    """
    keys = sorted(r.proposal_id for r in inp.requests)
    responses, notes = _merge_items(
        keys, output.responses, policy.responses, lambda r: r.proposal_id, _hospital_checker(inp)
    )
    return Validated(HospitalOutput(responses=responses), tuple(notes))


# --- Allocation -----------------------------------------------------------------


def validate_allocation(
    output: AllocationOutput, inp: AllocationInput, policy: AllocationOutput
) -> Validated[AllocationOutput]:
    """Validates an allocation output per case, then forbids double grants.

    A decision granting a float staff member already granted by an earlier
    decision of the same output is replaced by the case's DECLINE option.

    Args:
        output: The agent's output.
        inp: Its input.
        policy: The policy's output for the same input.

    Returns:
        One decision per case (``case_id`` order).
    """
    options = {c.case_id: {o.option_id for o in c.options} for c in inp.cases}

    def check(decision: CaseDecision) -> Violation | None:
        """Checks that the chosen option belongs to its case."""
        if decision.chosen_option_id not in options[decision.case_id]:
            return "unknown_option", decision.chosen_option_id
        return None

    keys = sorted(options)
    decisions, notes = _merge_items(
        keys, output.decisions, policy.decisions, lambda d: d.case_id, check
    )
    granted: set[str] = set()
    final: list[CaseDecision] = []
    for decision in decisions:
        staff = granted_staff(decision.chosen_option_id)
        if staff is not None and staff in granted and DECLINE_OPTION_ID in options[decision.case_id]:
            notes.append(_note("item", decision.case_id, "double_grant", f"{staff} already granted"))
            decision = CaseDecision(
                case_id=decision.case_id,
                chosen_option_id=DECLINE_OPTION_ID,
                rationale=f"{staff} already granted to another case",
            )
        elif staff is not None:
            granted.add(staff)
        final.append(decision)
    return Validated(AllocationOutput(decisions=final), tuple(notes))


# --- Dispatch -------------------------------------------------------------------


class _DispatchChecks:
    """Section checks of one dispatch output against its input."""

    def __init__(self, inp: DispatchInput, params: PolicyParams) -> None:
        """Indexes the payload.

        Args:
            inp: The dispatch payload.
            params: Policy parameters (pre-triage cap).
        """
        self._inp = inp
        self._params = params
        self._units = {u.unit_id: u for u in inp.units}
        self._needs = {n.incident_id: n for n in inp.incidents}
        self._hospitals = {h.hospital_id: h for h in inp.hospitals}
        self._loaded = {u.unit_id: u for u in inp.loaded_units}

    def orders(self, out: DispatchOutput) -> Violation | None:
        """Checks the orders section.

        Args:
            out: The agent output.

        Returns:
            The first violation or None.
        """
        if (dup := _duplicate([o.unit_id for o in out.orders])) is not None:
            return "duplicate_unit", dup
        sent: Counter[str] = Counter()
        for order in out.orders:
            unit = self._units.get(order.unit_id)
            if unit is None:
                return "unknown_unit", order.unit_id
            if unit.status is not UnitStatus.AVAILABLE:
                return "unit_not_available", f"{order.unit_id} is {unit.status}"
            if order.incident_id not in self._needs:
                return "unknown_incident", order.incident_id
            if unit.unit_type == AmbulanceType.BLS and _red_only(self._needs[order.incident_id]):
                return "bls_for_red_only", f"{order.unit_id} to {order.incident_id}"
            if unit.unit_class is UnitClass.AMBULANCE:
                sent[order.incident_id] += 1
        for incident_id, count in sorted(sent.items()):
            if count > pre_triage_slots(self._needs[incident_id], self._params):
                return "pre_triage_cap", f"{count} ambulances to untriaged {incident_id}"
        return None

    def diversions(self, out: DispatchOutput) -> Violation | None:
        """Checks the diversions section.

        Args:
            out: The agent output.

        Returns:
            The first violation or None.
        """
        if out.diversions and not self._inp.allow_diversion:
            return "diversion_not_allowed", f"{len(out.diversions)} diversion(s)"
        if (dup := _duplicate([d.unit_id for d in out.diversions])) is not None:
            return "duplicate_unit", dup
        for request in out.diversions:
            unit = self._units.get(request.unit_id)
            if unit is None:
                return "unknown_unit", request.unit_id
            if unit.unit_type != AmbulanceType.ALS or unit.status is not UnitStatus.EN_ROUTE_SCENE:
                return "not_en_route_als", f"{request.unit_id} {unit.unit_type} {unit.status}"
            if request.to_incident_id not in self._needs:
                return "unknown_incident", request.to_incident_id
            if request.to_incident_id == unit.incident_id:
                return "same_target", request.to_incident_id
        return None

    def contentions(self, out: DispatchOutput) -> Violation | None:
        """Checks the contentions section.

        Args:
            out: The agent output.

        Returns:
            The first violation or None.
        """
        if out.contentions and not self._inp.allow_escalation:
            return "contention_not_allowed", f"{len(out.contentions)} flag(s)"
        if (dup := _duplicate([c.unit_id for c in out.contentions])) is not None:
            return "duplicate_unit", dup
        for flag in out.contentions:
            unit = self._units.get(flag.unit_id)
            if unit is None:
                return "unknown_unit", flag.unit_id
            if unit.unit_type != AmbulanceType.ALS or unit.status is not UnitStatus.AVAILABLE:
                return "not_free_als", f"{flag.unit_id} {unit.unit_type} {unit.status}"
            if len(set(flag.incident_ids)) < 2 or len(set(flag.incident_ids)) != len(flag.incident_ids):
                return "bad_contention", f"incidents {flag.incident_ids}"
            if unknown := sorted(set(flag.incident_ids) - self._needs.keys()):
                return "unknown_incident", ",".join(unknown)
        return None

    def transports(self, out: DispatchOutput) -> Violation | None:
        """Checks the transports section.

        Args:
            out: The agent output.

        Returns:
            The first violation or None.
        """
        planned = [p.unit_id for p in out.transports]
        if (dup := _duplicate(planned)) is not None:
            return "duplicate_unit", dup
        if set(planned) != self._loaded.keys():
            missing = sorted(self._loaded.keys() - set(planned))
            extra = sorted(set(planned) - self._loaded.keys())
            return "loaded_units_mismatch", f"missing {missing}, extra {extra}"
        for plan in out.transports:
            needed = {v.required_capability for v in self._loaded[plan.unit_id].victims}
            capable = {h for h, e in self._hospitals.items() if needed <= set(e.capabilities)}
            if (dup := _duplicate(plan.hospital_ranking)) is not None:
                return "duplicate_hospital", f"{plan.unit_id}: {dup}"
            if bad := [h for h in plan.hospital_ranking if h not in capable]:
                return "hospital_not_capable", f"{plan.unit_id}: {bad}"
            if not plan.hospital_ranking and capable:
                return "empty_ranking", plan.unit_id
        return None


def _red_only(need: IncidentNeed) -> bool:
    """Tells whether every victim still waiting at a triaged scene is RED.

    A BLS cannot carry RED, so sending one there is a wasted trip.

    Args:
        need: The incident's need.

    Returns:
        True if the scene is triaged and only RED victims wait.
    """
    return need.triaged and 0 < need.red_waiting == need.est_patients_remaining


def _duplicate(ids: Sequence[str]) -> str | None:
    """Returns the first duplicated id, if any.

    Args:
        ids: Ids.

    Returns:
        A duplicated id or None.
    """
    seen: set[str] = set()
    for item in ids:
        if item in seen:
            return item
        seen.add(item)
    return None


DISPATCH_SECTIONS = ("orders", "diversions", "contentions", "transports")
"""Dispatch output sections, validated (and replaced) independently."""


def _cross_section_conflict(out: DispatchOutput) -> str | None:
    """Finds a unit used by two sections (order, diversion, contention).

    Args:
        out: A dispatch output.

    Returns:
        A conflicting unit id or None.
    """
    used = (
        [o.unit_id for o in out.orders]
        + [d.unit_id for d in out.diversions]
        + [c.unit_id for c in out.contentions]
    )
    return _duplicate(used)


def validate_dispatch(
    output: DispatchOutput, inp: DispatchInput, policy: DispatchOutput, params: PolicyParams
) -> Validated[DispatchOutput]:
    """Validates a dispatch output per section, then across sections.

    Args:
        output: The agent's output.
        inp: Its input.
        policy: The policy's output for the same input.
        params: Policy parameters (pre-triage cap).

    Returns:
        The output with invalid sections replaced by the policy's sections,
        or the policy's whole output if a cross-section conflict remains.
    """
    checks = _DispatchChecks(inp, params)
    notes: list[FallbackNote] = []
    sections: dict[str, object] = {}
    for name in DISPATCH_SECTIONS:
        violation = getattr(checks, name)(output)
        if violation is None:
            sections[name] = getattr(output, name)
        else:
            notes.append(_note("section", name, violation[0], violation[1]))
            sections[name] = getattr(policy, name)
    merged = DispatchOutput.model_validate(sections)
    if (unit := _cross_section_conflict(merged)) is not None:
        notes.append(_note("call", None, "cross_section_conflict", f"{unit} used twice"))
        merged = policy
    return Validated(merged, tuple(notes))
