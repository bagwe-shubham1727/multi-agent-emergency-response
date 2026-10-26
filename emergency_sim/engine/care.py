"""ER beds, queue, staff selection and the holdover rule (plan section 4.4).

``assign_care`` runs on every resource change at a hospital. It mutates the
(engine-private) world state and returns what changed so that the engine can
log events and schedule treatment ends. Selection is deterministic: patients
by ``(TRIAGE_RANK, arrival time, victim_id)``, staff by load then id.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from emergency_sim.domain.enums import TRIAGE_RANK, Capability, StaffRole, VictimStatus
from emergency_sim.engine.state import EngineError, StaffState, VictimState, WorldState

EMERGENCY_SKILL: Final = "emergency"
"""Skill every ER doctor and nurse has; doctors use it for ``general`` patients."""

NURSE_SKILL_FOR: Final = MappingProxyType(
    {"trauma": "trauma", "burns": "burns", "pediatric": "pediatric", "cardiac": "icu", "neuro": "icu"}
)
"""Preferred nurse skill per capability.

Mirrors ``emergency_sim.datagen.NURSE_SKILL_FOR`` (the engine may not import
``datagen``); ``tests/test_engine_hospital.py`` asserts the two stay equal.
"""


@dataclass(frozen=True)
class Bedded:
    """A queued patient got a bed."""

    victim_id: str


@dataclass(frozen=True)
class TreatmentStarted:
    """A bedded patient started treatment."""

    victim_id: str
    doctor_id: str
    nurse_id: str


type CareChange = Bedded | TreatmentStarted


def care_order_key(victim: VictimState) -> tuple[int, float, str]:
    """Sort key of patients at a hospital: triage, then arrival, then id.

    Args:
        victim: A victim at a hospital.

    Returns:
        ``(TRIAGE_RANK, arrival time, victim_id)``.
    """
    arrived = victim.arrived_hospital if victim.arrived_hospital is not None else 0.0
    return (TRIAGE_RANK[victim.triage], arrived, victim.victim_id)


def doctor_skill(capability: Capability) -> str:
    """Doctor skill needed for a capability (``emergency`` for general).

    Args:
        capability: Required capability.

    Returns:
        The skill name.
    """
    return EMERGENCY_SKILL if capability is Capability.GENERAL else str(capability)


def is_eligible(state: WorldState, member: StaffState, skill: str, hospital_id: str, now: float) -> bool:
    """Whether a staff member may take a new patient needing ``skill`` at a hospital.

    A member past ``shift_end_min`` stays eligible only if no colleague of the
    same role at that hospital with the same skill is still on shift (holdover).

    Args:
        state: World state.
        member: Candidate staff member.
        skill: Needed skill.
        hospital_id: Hospital id.
        now: Current time.

    Returns:
        True if the member can take the patient now.
    """
    if member.hospital_id != hospital_id or skill not in member.spec.skills:
        return False
    if member.load >= member.spec.max_patients:
        return False
    if member.on_shift(now):
        return True
    return not any(
        c.staff_id != member.staff_id
        and c.spec.role is member.spec.role
        and skill in c.spec.skills
        and c.on_shift(now)
        for c in state.staff_at(hospital_id)
    )


def select_doctor(state: WorldState, hospital_id: str, capability: Capability, now: float) -> StaffState | None:
    """Chooses the doctor for a patient: eligible, lowest load, then id.

    Args:
        state: World state.
        hospital_id: Hospital id.
        capability: Patient's required capability.
        now: Current time.

    Returns:
        The doctor, or None if nobody is eligible.
    """
    skill = doctor_skill(capability)
    eligible = [
        s
        for s in state.staff_at(hospital_id)
        if s.spec.role is StaffRole.DOCTOR and is_eligible(state, s, skill, hospital_id, now)
    ]
    return min(eligible, key=lambda s: (s.load, s.staff_id), default=None)


def select_nurse(state: WorldState, hospital_id: str, capability: Capability, now: float) -> StaffState | None:
    """Chooses the nurse: eligible ``emergency`` nurse, matching specialty first.

    Args:
        state: World state.
        hospital_id: Hospital id.
        capability: Patient's required capability.
        now: Current time.

    Returns:
        The nurse, or None if nobody is eligible.
    """
    preferred = NURSE_SKILL_FOR.get(str(capability))
    eligible = [
        s
        for s in state.staff_at(hospital_id)
        if s.spec.role is StaffRole.NURSE and is_eligible(state, s, EMERGENCY_SKILL, hospital_id, now)
    ]
    return min(
        eligible,
        key=lambda s: (preferred is None or preferred not in s.spec.skills, s.load, s.staff_id),
        default=None,
    )


def bed_victim(victim: VictimState, hospital_id: str, now: float) -> None:
    """Puts a victim in an ER bed (waiting for staff).

    Args:
        victim: The victim.
        hospital_id: Hospital id.
        now: Current time.
    """
    victim.status = VictimStatus.WAITING_FOR_STAFF
    victim.hospital_id = hospital_id
    victim.unit_id = None
    victim.bedded_at = now


def queue_victim(victim: VictimState, hospital_id: str) -> None:
    """Puts a victim in the ER queue (no bed yet).

    Args:
        victim: The victim.
        hospital_id: Hospital id.
    """
    victim.status = VictimStatus.QUEUED
    victim.hospital_id = hospital_id
    victim.unit_id = None


def _bed_from_queue(state: WorldState, hospital_id: str, now: float) -> list[CareChange]:
    """Beds queued patients while free unreserved beds remain."""
    changes: list[CareChange] = []
    queue = sorted(state.hospital_victims(hospital_id, frozenset({VictimStatus.QUEUED})), key=care_order_key)
    for victim in queue:
        if state.beds_free_unreserved(hospital_id) <= 0:
            break
        bed_victim(victim, hospital_id, now)
        changes.append(Bedded(victim.victim_id))
    return changes


def _start_treatment(victim: VictimState, doctor: StaffState, nurse: StaffState, now: float) -> TreatmentStarted:
    """Starts ER treatment of a bedded patient."""
    victim.status = VictimStatus.IN_TREATMENT
    victim.treatment_start = now
    victim.doctor_id = doctor.staff_id
    victim.nurse_id = nurse.staff_id
    if victim.care_start is None:
        victim.care_start = now
        victim.care_mode = "hospital"
    doctor.patients.add(victim.victim_id)
    nurse.patients.add(victim.victim_id)
    return TreatmentStarted(victim.victim_id, doctor.staff_id, nurse.staff_id)


def assign_care(state: WorldState, hospital_id: str, now: float) -> list[CareChange]:
    """Beds queued patients, then starts treatment where a doctor and nurse are free.

    Args:
        state: World state (mutated).
        hospital_id: Hospital whose resources changed.
        now: Current time.

    Returns:
        The changes, in the order they happened.
    """
    changes = _bed_from_queue(state, hospital_id, now)
    waiting = state.hospital_victims(hospital_id, frozenset({VictimStatus.WAITING_FOR_STAFF}))
    for victim in sorted(waiting, key=care_order_key):
        cap = victim.spec.required_capability
        doctor = select_doctor(state, hospital_id, cap, now)
        nurse = select_nurse(state, hospital_id, cap, now)
        if doctor is not None and nurse is not None:
            changes.append(_start_treatment(victim, doctor, nurse, now))
    return changes


def end_treatment(state: WorldState, victim_id: str) -> str:
    """Ends a treatment: frees the bed and both staff slots.

    Args:
        state: World state (mutated).
        victim_id: Treated victim.

    Returns:
        The hospital id.

    Raises:
        EngineError: If the victim is not in treatment.
    """
    victim = state.victims[victim_id]
    if victim.status is not VictimStatus.IN_TREATMENT or victim.hospital_id is None:
        raise EngineError(f"{victim_id} is not in treatment")
    for staff_id in _treating_staff(victim):
        state.staff[staff_id].patients.discard(victim_id)
    victim.status = VictimStatus.TREATED
    return victim.hospital_id


def _treating_staff(victim: VictimState) -> Iterable[str]:
    """Ids of the doctor and nurse treating a victim."""
    return [s for s in (victim.doctor_id, victim.nurse_id) if s is not None]
