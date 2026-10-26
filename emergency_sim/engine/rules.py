"""Action legality, the loading rule and capability helpers (plan sections 4.2, 4.6).

Everything here is a pure function of the world state: nothing is mutated and
nothing is logged. ``check_action`` is the single validation point used by both
``Engine.check`` and ``Engine.apply``.
"""

from collections.abc import Iterable
from typing import Final

from emergency_sim import geo
from emergency_sim.domain.actions import (
    Action,
    ActionCheck,
    AssignTransport,
    DispatchUnit,
    DivertUnit,
    LendStaff,
)
from emergency_sim.domain.enums import (
    TRIAGE_RANK,
    Capability,
    IncidentStatus,
    RejectReason,
    Triage,
    UnitStatus,
    VictimStatus,
)
from emergency_sim.engine.state import (
    EngineError,
    HospitalState,
    UnitState,
    VictimState,
    WorldState,
)

_OK: Final = ActionCheck(ok=True)
_TRANSPORT_READY: Final = frozenset({UnitStatus.AWAITING_DESTINATION, UnitStatus.AT_HOSPITAL_FULL})


def _reject(reason: RejectReason, detail: str) -> ActionCheck:
    """Builds a failed check."""
    return ActionCheck(ok=False, reason=reason, detail=detail)


def hospital_is_capable(hospital: HospitalState, capabilities: Iterable[Capability]) -> bool:
    """Whether a hospital offers every capability.

    Args:
        hospital: Hospital state.
        capabilities: Required capabilities.

    Returns:
        True if all are offered.
    """
    return set(capabilities) <= set(hospital.spec.capabilities)


def capable_hospital_ids(state: WorldState, capabilities: Iterable[Capability]) -> list[str]:
    """Lists hospitals capable of every given capability.

    Args:
        state: World state.
        capabilities: Required capabilities.

    Returns:
        Hospital ids, sorted.
    """
    needed = set(capabilities)
    return [hid for hid, h in state.hospitals.items() if hospital_is_capable(h, needed)]


def load_capabilities(state: WorldState, unit: UnitState) -> set[Capability]:
    """Capabilities required by the victims a unit carries.

    Args:
        state: World state.
        unit: Unit.

    Returns:
        The set of required capabilities.
    """
    return {state.victims[v].spec.required_capability for v in unit.load}


def nearest_capable_hospital(state: WorldState, unit: UnitState, now: float) -> str:
    """Finds the hospital capable of the unit's load with the shortest drive.

    Args:
        state: World state.
        unit: A loaded ambulance.
        now: Current time (for the unit's position).

    Returns:
        Hospital id (ties broken by id).

    Raises:
        EngineError: If no hospital is capable of the load.
    """
    pos = unit.position_at(now)
    candidates = capable_hospital_ids(state, load_capabilities(state, unit))
    if not candidates:
        raise EngineError(f"no hospital capable of the load of {unit.unit_id}")
    return min(
        candidates,
        key=lambda hid: (geo.travel_time(pos, state.hospitals[hid].position, "ambulance"), hid),
    )


def _can_add(state: WorldState, unit: UnitState, chosen: list[VictimState], cand: VictimState) -> bool:
    """Whether ``cand`` may join ``chosen`` in ``unit`` under the loading rule."""
    if cand.triage is Triage.RED:
        if not unit.is_als:
            return False
        if sum(v.triage is Triage.RED for v in chosen) >= unit.max_red:
            return False
    caps = {v.spec.required_capability for v in chosen} | {cand.spec.required_capability}
    return bool(capable_hospital_ids(state, caps))


def choose_load(state: WorldState, unit: UnitState) -> tuple[str, ...]:
    """Applies the loading rule for an ambulance at its incident.

    Candidates are ``WAITING`` victims sorted by ``(TRIAGE_RANK, victim_id)``.
    The unit takes the first candidate it may carry (BLS never carries RED),
    then fills its seats with the next candidates that keep the RED count within
    ``max_red_per_trip`` and leave some hospital capable of the whole load.

    Args:
        state: World state.
        unit: An ambulance on scene.

    Returns:
        Victim ids to load, in pick order; empty if nothing can be carried.
    """
    if unit.incident_id is None:
        raise EngineError(f"{unit.unit_id} has no incident")
    waiting = [v for v in state.incident_victims(unit.incident_id) if v.status is VictimStatus.WAITING]
    waiting.sort(key=lambda v: (TRIAGE_RANK[v.triage], v.victim_id))
    chosen: list[VictimState] = []
    for cand in waiting:
        if len(chosen) >= unit.seats:
            break
        if _can_add(state, unit, chosen, cand):
            chosen.append(cand)
    return tuple(v.victim_id for v in chosen)


def check_action(state: WorldState, action: Action, now: float) -> ActionCheck:
    """Validates an action against the current state (plan section 4.6).

    Args:
        state: World state (not mutated).
        action: The action.
        now: Current simulation time.

    Returns:
        ``ActionCheck(ok=True)`` or a rejection with reason and detail.

    Raises:
        EngineError: If the action type is unknown.
    """
    match action:
        case DispatchUnit():
            return _check_dispatch(state, action)
        case DivertUnit():
            return _check_divert(state, action)
        case AssignTransport():
            return _check_transport(state, action)
        case LendStaff():
            return _check_lend(state, action, now)
        case _:
            raise EngineError(f"unknown action {action!r}")


def _check_open_incident(state: WorldState, incident_id: str) -> ActionCheck:
    """Checks that an incident is reported and open."""
    incident = state.incidents.get(incident_id)
    if incident is None or not incident.reported:
        return _reject(RejectReason.UNKNOWN_INCIDENT, f"no reported incident {incident_id}")
    if incident.status is not IncidentStatus.OPEN:
        return _reject(RejectReason.INCIDENT_NOT_OPEN, f"{incident_id} is {incident.status}")
    return _OK


def _check_dispatch(state: WorldState, action: DispatchUnit) -> ActionCheck:
    """Validates ``DispatchUnit``."""
    unit = state.units.get(action.unit_id)
    if unit is None:
        return _reject(RejectReason.UNKNOWN_UNIT, f"no unit {action.unit_id}")
    if unit.status is not UnitStatus.AVAILABLE:
        return _reject(RejectReason.UNIT_NOT_AVAILABLE, f"{unit.unit_id} is {unit.status}")
    return _check_open_incident(state, action.incident_id)


def _check_divert(state: WorldState, action: DivertUnit) -> ActionCheck:
    """Validates ``DivertUnit``."""
    unit = state.units.get(action.unit_id)
    if unit is None:
        return _reject(RejectReason.UNKNOWN_UNIT, f"no unit {action.unit_id}")
    if not unit.is_ambulance:
        return _reject(RejectReason.NOT_AN_AMBULANCE, f"{unit.unit_id} is a responder team")
    if unit.status is not UnitStatus.EN_ROUTE_SCENE:
        return _reject(RejectReason.UNIT_NOT_EN_ROUTE, f"{unit.unit_id} is {unit.status}")
    target = _check_open_incident(state, action.incident_id)
    if not target.ok:
        return target
    if unit.incident_id == action.incident_id:
        return _reject(RejectReason.SAME_TARGET, f"{unit.unit_id} already heads to {action.incident_id}")
    return _OK


def _check_transport(state: WorldState, action: AssignTransport) -> ActionCheck:
    """Validates ``AssignTransport``."""
    unit = state.units.get(action.unit_id)
    if unit is None:
        return _reject(RejectReason.UNKNOWN_UNIT, f"no unit {action.unit_id}")
    if not unit.is_ambulance:
        return _reject(RejectReason.NOT_AN_AMBULANCE, f"{unit.unit_id} is a responder team")
    if unit.status not in _TRANSPORT_READY:
        return _reject(RejectReason.NOT_AWAITING_DESTINATION, f"{unit.unit_id} is {unit.status}")
    hospital = state.hospitals.get(action.hospital_id)
    if hospital is None:
        return _reject(RejectReason.UNKNOWN_HOSPITAL, f"no hospital {action.hospital_id}")
    missing = load_capabilities(state, unit) - set(hospital.spec.capabilities)
    if missing:
        names = ", ".join(sorted(missing))
        return _reject(RejectReason.HOSPITAL_NOT_CAPABLE, f"{hospital.hospital_id} lacks {names}")
    if unit.status is UnitStatus.AT_HOSPITAL_FULL and unit.hospital_id == action.hospital_id:
        return _reject(RejectReason.SAME_HOSPITAL, f"{unit.unit_id} is already at {action.hospital_id}")
    if action.reserve_beds:
        free = state.beds_free_unreserved(hospital.hospital_id)
        if free < len(unit.load):
            return _reject(
                RejectReason.INSUFFICIENT_BEDS,
                f"{hospital.hospital_id} has {free} free unreserved beds, needs {len(unit.load)}",
            )
    return _OK


def _check_lend(state: WorldState, action: LendStaff, now: float) -> ActionCheck:
    """Validates ``LendStaff``."""
    member = state.staff.get(action.staff_id)
    if member is None:
        return _reject(RejectReason.UNKNOWN_STAFF, f"no staff {action.staff_id}")
    if not member.is_float:
        return _reject(RejectReason.STAFF_NOT_FLOAT, f"{member.staff_id} belongs to {member.spec.home}")
    if member.lent:
        return _reject(RejectReason.STAFF_ALREADY_LENT, f"{member.staff_id} is lent to {member.hospital_id}")
    if not member.on_shift(now):
        detail = f"{member.staff_id} shift ended at {member.spec.shift_end_min}"
        return _reject(RejectReason.STAFF_OFF_SHIFT, detail)
    if action.hospital_id not in state.hospitals:
        return _reject(RejectReason.UNKNOWN_HOSPITAL, f"no hospital {action.hospital_id}")
    return _OK


