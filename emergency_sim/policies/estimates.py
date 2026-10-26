"""Pure estimates shared by the policies, the agents' tools and the coordinator.

Everything here is a deterministic function of an agent payload (or of plain
numbers), so the offline policies, the read-only ``mas`` tools and
``control/cases.py`` compute identical numbers. Rankings break ties by id.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

from emergency_sim import geo
from emergency_sim.domain.agent_io import (
    DispatchInput,
    HospitalBoardEntry,
    HospitalInput,
    LoadedUnit,
    StaffMemberView,
    UnitStatusView,
)
from emergency_sim.domain.enums import (
    AmbulanceType,
    Capability,
    ResponderCapability,
    StaffRole,
    Triage,
    UnitClass,
    UnitStatus,
)
from emergency_sim.policies.params import PolicyParams

CARDIAC_ARREST_CONDITION: Final = "cardiac arrest"
"""Condition label that gets the short cardiac-arrest window and field care."""

GENERAL_SKILL: Final = "emergency"
"""Skill every ER doctor and nurse has; it covers ``general`` patients."""

NURSE_SKILL: Final = GENERAL_SKILL
"""Skill a nurse needs to treat any patient (section 4.4)."""

type TriageLike = Triage | Literal["SUSPECTED_RED", "UNKNOWN"]
"""A known triage, a severity-cue suspicion, or no information."""


@dataclass(frozen=True)
class TypeProfile:
    """Public prior knowledge about an incident type.

    Attributes:
        incident_type: The type label.
        base_urgency: Urgency before severity cues and victim count (0..100).
        needed_capabilities: Hospital capabilities its patients typically need.
        known: False when the type is not in the public table.
    """

    incident_type: str
    base_urgency: int
    needed_capabilities: tuple[Capability, ...]
    known: bool


def _profile(incident_type: str, base: int, *caps: Capability) -> TypeProfile:
    return TypeProfile(incident_type, base, tuple(sorted(caps)), True)


_C = Capability
INCIDENT_TYPE_PROFILES: Final = MappingProxyType({
    "cardiac_event": _profile("cardiac_event", 70, _C.CARDIAC),
    "stroke": _profile("stroke", 70, _C.NEURO),
    "pediatric_emergency": _profile("pediatric_emergency", 65, _C.PEDIATRIC),
    "multi_vehicle_collision": _profile("multi_vehicle_collision", 60, _C.TRAUMA, _C.NEURO, _C.GENERAL),
    "building_fire": _profile("building_fire", 60, _C.BURNS, _C.GENERAL),
    "structural_collapse": _profile("structural_collapse", 60, _C.TRAUMA, _C.NEURO, _C.GENERAL),
    "assault": _profile("assault", 55, _C.TRAUMA, _C.NEURO, _C.GENERAL),
    "chemical_spill": _profile("chemical_spill", 50, _C.GENERAL),
    "fall": _profile("fall", 40, _C.TRAUMA, _C.NEURO, _C.GENERAL),
})
"""Public type table: base urgency (plan 5.3) and typical capabilities."""

UNKNOWN_TYPE_BASE_URGENCY: Final = 50
"""Base urgency of a type missing from the table (middle of the scale)."""


def incident_type_profile(incident_type: str) -> TypeProfile:
    """Looks up the public profile of an incident type.

    Args:
        incident_type: Type label, e.g. ``"cardiac_event"``.

    Returns:
        The profile; unknown types get base urgency 50, ``general`` care and
        ``known=False``.
    """
    profile = INCIDENT_TYPE_PROFILES.get(incident_type)
    if profile is not None:
        return profile
    return TypeProfile(incident_type, UNKNOWN_TYPE_BASE_URGENCY, (Capability.GENERAL,), False)


# --- Weights, deadlines, utility --------------------------------------------


def triage_weight(triage: TriageLike, params: PolicyParams) -> float:
    """Survival weight of a patient category.

    Args:
        triage: A triage, ``"SUSPECTED_RED"`` (severity cue before triage) or
            ``"UNKNOWN"`` (no information; weighted like YELLOW, the middle).
        params: Policy parameters.

    Returns:
        The weight; BLACK (expectant) is 0.
    """
    match triage:
        case Triage.RED | "SUSPECTED_RED":
            return params.red_weight
        case Triage.YELLOW | "UNKNOWN":
            return params.yellow_weight
        case Triage.GREEN:
            return params.green_weight
        case _:
            return 0.0


def nominal_deadline(
    triage: Triage, condition: str, params: PolicyParams, *, deteriorated: bool = False
) -> float:
    """Assumed care window in minutes (from report, or from deterioration).

    The true windows are hidden; these are public nominal values.

    Args:
        triage: Current triage.
        condition: Condition label; ``"cardiac arrest"`` gets the short window.
        params: Policy parameters.
        deteriorated: Whether the victim is a YELLOW that turned RED (its
            window counts from the deterioration).

    Returns:
        The window in minutes.

    Raises:
        ValueError: If ``triage`` is BLACK (expectant patients have no window).
    """
    if triage is Triage.BLACK:
        raise ValueError("BLACK (expectant) victims have no care window")
    if condition == CARDIAC_ARREST_CONDITION:
        return params.nominal_cardiac_arrest_deadline_min
    if triage is Triage.RED:
        return params.deteriorated_red_deadline_min if deteriorated else params.nominal_red_deadline_min
    if triage is Triage.YELLOW:
        return params.nominal_yellow_deadline_min
    return params.nominal_green_deadline_min


def p_window(slack: float, params: PolicyParams) -> float:
    """Probability that care starts within the window, given slack in minutes.

    Args:
        slack: Deadline minus expected care start (negative means late).
        params: Policy parameters (logistic scale).

    Returns:
        ``1 / (1 + exp(-slack / scale))``, computed without overflow.
    """
    z = slack / params.window_logistic_scale_min
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def utility(weight: float, slack: float, travel: float, params: PolicyParams) -> float:
    """Expected survival-weighted utility of an option.

    Args:
        weight: Triage weight of the patients concerned.
        slack: Window slack in minutes.
        travel: Travel minutes the option costs.
        params: Policy parameters.

    Returns:
        ``weight * p_window(slack) - travel_penalty_per_min * travel``.
    """
    return weight * p_window(slack, params) - params.travel_penalty_per_min * travel


# --- Travel and rankings -----------------------------------------------------


def eta_min(
    origin: tuple[float, float],
    dest: tuple[float, float],
    unit_class: UnitClass,
    include_turnout: bool,
) -> float:
    """Travel time of a unit, using the engine's own travel model.

    Args:
        origin: ``(x, y)`` in km.
        dest: ``(x, y)`` in km.
        unit_class: Ambulance or responder (decides speed and turnout).
        include_turnout: Add the alert-to-wheels-rolling time.

    Returns:
        Minutes, rounded to 0.1 like ``geo.travel_time``.
    """
    return geo.travel_time(origin, dest, unit_class.value, include_turnout)


@dataclass(frozen=True)
class UnitRank:
    """A free unit and its ETA to an incident (turnout included)."""

    unit_id: str
    unit_type: str
    eta_min: float


@dataclass(frozen=True)
class HospitalRank:
    """A capable hospital and its ETA from a loaded unit."""

    hospital_id: str
    eta_min: float
    beds_free_unreserved: int
    queue_length: int


def _incident_position(inp: DispatchInput, incident_id: str) -> tuple[float, float]:
    for need in inp.incidents:
        if need.incident_id == incident_id:
            return (need.x, need.y)
    raise KeyError(f"unknown incident {incident_id!r}")


def _rank(units: Sequence[UnitStatusView], dest: tuple[float, float]) -> tuple[UnitRank, ...]:
    ranks = [
        UnitRank(u.unit_id, u.unit_type, eta_min((u.x, u.y), dest, u.unit_class, True))
        for u in units
    ]
    return tuple(sorted(ranks, key=lambda r: (r.eta_min, r.unit_id)))


def rank_units(
    inp: DispatchInput, incident_id: str, ambulance_type: AmbulanceType | None
) -> tuple[UnitRank, ...]:
    """Ranks the free ambulances for an incident by ETA, then id.

    Args:
        inp: The dispatch payload.
        incident_id: Target incident (must be in ``inp.incidents``).
        ambulance_type: ``ALS``, ``BLS`` or None for any type.

    Returns:
        Free (``AVAILABLE``) ambulances, nearest first.

    Raises:
        KeyError: If the incident is not in the payload.
    """
    dest = _incident_position(inp, incident_id)
    units = [
        u for u in inp.units
        if u.unit_class is UnitClass.AMBULANCE and u.status is UnitStatus.AVAILABLE
        and (ambulance_type is None or u.unit_type == ambulance_type.value)
    ]
    return _rank(units, dest)


def rank_responders(
    inp: DispatchInput, incident_id: str, capability: ResponderCapability
) -> tuple[UnitRank, ...]:
    """Ranks the free responder teams that bring a capability, by ETA then id.

    Args:
        inp: The dispatch payload.
        incident_id: Target incident.
        capability: Required responder capability.

    Returns:
        Free teams with the capability, nearest first.

    Raises:
        KeyError: If the incident is not in the payload.
    """
    dest = _incident_position(inp, incident_id)
    units = [
        u for u in inp.units
        if u.unit_class is UnitClass.RESPONDER and u.status is UnitStatus.AVAILABLE
        and capability.value in u.capabilities
    ]
    return _rank(units, dest)


def rank_hospitals(board: Sequence[HospitalBoardEntry], unit: LoadedUnit) -> tuple[HospitalRank, ...]:
    """Ranks the hospitals capable of every patient on a loaded unit.

    Order: hospitals with a free unreserved bed first, then ETA, then id.

    Args:
        board: The public bed board.
        unit: The loaded ambulance.

    Returns:
        Capable hospitals, best first (empty if none is capable).
    """
    needed = {v.required_capability for v in unit.victims}
    ranks = [
        HospitalRank(
            h.hospital_id,
            eta_min((unit.x, unit.y), (h.x, h.y), UnitClass.AMBULANCE, False),
            h.beds_free_unreserved,
            h.queue_length,
        )
        for h in board
        if needed <= set(h.capabilities)
    ]
    return tuple(sorted(ranks, key=lambda r: (r.beds_free_unreserved == 0, r.eta_min, r.hospital_id)))


# --- Hospital capacity ---------------------------------------------------------


def doctor_skill_for(capability: Capability) -> str:
    """Doctor skill a patient needs: ``"emergency"`` for general, else the capability."""
    return GENERAL_SKILL if capability is Capability.GENERAL else capability.value


def eligible_slots(staff: Sequence[StaffMemberView], role: StaffRole, skill: str) -> int:
    """Free patient slots of the staff who may take a new patient needing ``skill``.

    On-shift staff with the skill are eligible. Off-shift staff are eligible
    only under the holdover rule: no on-shift colleague of the same role has
    the skill.

    Args:
        staff: Staff at the hospital (home and lent).
        role: Doctor or nurse.
        skill: Needed skill.

    Returns:
        Sum of ``max_patients - load`` over the eligible staff.
    """
    skilled = [s for s in staff if s.role is role and skill in s.skills]
    on_shift = [s for s in skilled if s.on_shift]
    eligible = on_shift or skilled
    return sum(max(0, s.max_patients - s.load) for s in eligible)


@dataclass(frozen=True)
class CapacityCheck:
    """Whether a hospital can take one admission request right now.

    Attributes:
        proposal_id: The request.
        patients: Number of patients in the request.
        has_red: Whether any patient is RED.
        missing_capabilities: Needed capabilities the hospital lacks.
        beds_free_unreserved: Free beds considered.
        beds_free_after: Free beds left if accepted.
        critical_reserve: Beds held back for RED patients.
        reserve_breached: Non-RED request that would cut into the reserve.
        doctor_slots: ``(skill, eligible slots)`` per needed doctor skill.
        missing_doctor_skills: Needed doctor skills with zero slots.
        nurse_slots: Eligible nurse slots.
        nurse_short: Fewer nurse slots than patients.
    """

    proposal_id: str
    patients: int
    has_red: bool
    missing_capabilities: tuple[Capability, ...]
    beds_free_unreserved: int
    beds_free_after: int
    critical_reserve: int
    reserve_breached: bool
    doctor_slots: tuple[tuple[str, int], ...]
    missing_doctor_skills: tuple[str, ...]
    nurse_slots: int
    nurse_short: bool


def capacity_check(
    inp: HospitalInput, proposal_id: str, *, beds_free: int | None = None
) -> CapacityCheck:
    """Checks capability, beds, reserve and staff for one admission request.

    Args:
        inp: The hospital payload.
        proposal_id: Request to check.
        beds_free: Free unreserved beds to assume (the policy's running count);
            defaults to ``inp.beds_free_unreserved``.

    Returns:
        The capacity check.

    Raises:
        KeyError: If the proposal is not in ``inp.requests``.
    """
    request = next((r for r in inp.requests if r.proposal_id == proposal_id), None)
    if request is None:
        raise KeyError(f"unknown proposal {proposal_id!r}")
    free = inp.beds_free_unreserved if beds_free is None else beds_free
    n = len(request.victims)
    has_red = any(v.triage is Triage.RED for v in request.victims)
    caps = sorted({v.required_capability for v in request.victims})
    skills = sorted({doctor_skill_for(c) for c in caps})
    doctor_slots = tuple((k, eligible_slots(inp.staff, StaffRole.DOCTOR, k)) for k in skills)
    nurse_slots = eligible_slots(inp.staff, StaffRole.NURSE, NURSE_SKILL)
    return CapacityCheck(
        proposal_id=proposal_id,
        patients=n,
        has_red=has_red,
        missing_capabilities=tuple(c for c in caps if c not in inp.capabilities),
        beds_free_unreserved=free,
        beds_free_after=free - n,
        critical_reserve=inp.critical_reserve,
        reserve_breached=not has_red and free - n < inp.critical_reserve,
        doctor_slots=doctor_slots,
        missing_doctor_skills=tuple(k for k, slots in doctor_slots if slots == 0),
        nurse_slots=nurse_slots,
        nurse_short=nurse_slots < n,
    )
