"""Read-only projections of the world state: view, counts, outcomes, invariants.

``build_public_view`` is the information boundary (plan section 3.7). It only
reads observable fields: reported incidents, victims of triaged incidents with
their *current* triage, units, the bed board and staff. It never copies golden
windows, deterioration times, true victim counts, worst triage or time
sensitivity, and an untriaged incident always has ``victims == ()``.

``check_world`` lists consistency violations (beds, staff loads, unit/victim
bookkeeping, incident clearing); tests assert it is empty after every step.
"""

from types import MappingProxyType
from typing import Final

from emergency_sim.domain.enums import (
    IncidentStatus,
    Triage,
    UnitStatus,
    VictimStatus,
)
from emergency_sim.domain.log_records import SnapshotCounts
from emergency_sim.domain.outcomes import IncidentOutcome, RunOutcomes, VictimOutcome
from emergency_sim.domain.params import SimParams
from emergency_sim.domain.views import (
    HospitalView,
    IncidentView,
    PublicView,
    StaffView,
    UnitView,
    VictimView,
)
from emergency_sim.engine.state import (
    CARRYING_STATUSES,
    ON_SCENE_VICTIM_STATUSES,
    SCENE_COMMITTED_STATUSES,
    HospitalState,
    IncidentState,
    StaffState,
    UnitState,
    VictimState,
    WorldState,
)

GREEN_DEADLINE_MIN: Final = 480.0
"""Care deadline of a GREEN victim, in minutes after the report."""

_IN_TRANSPORT: Final = frozenset({VictimStatus.LOADED, VictimStatus.TRANSPORTING})

_UNIT_STATUS_FOR_VICTIM: Final = MappingProxyType(
    {
        VictimStatus.LOADING: frozenset({UnitStatus.LOADING}),
        VictimStatus.LOADED: frozenset({UnitStatus.AWAITING_DESTINATION}),
        VictimStatus.TRANSPORTING: frozenset({UnitStatus.TRANSPORTING, UnitStatus.AT_HOSPITAL_FULL}),
    }
)
"""Unit statuses compatible with a victim that is on board."""

_HOSPITAL_VICTIM_STATUSES: Final = frozenset(
    {VictimStatus.QUEUED, VictimStatus.WAITING_FOR_STAFF, VictimStatus.IN_TREATMENT, VictimStatus.TREATED}
)


def _victim_view(victim: VictimState) -> VictimView:
    """Projects a victim of a triaged incident."""
    return VictimView(
        victim_id=victim.victim_id,
        incident_id=victim.incident_id,
        triage=victim.triage,
        condition=victim.spec.condition,
        required_capability=victim.spec.required_capability,
        needs_als=victim.needs_als,
        status=victim.status,
        unit_id=victim.unit_id,
        hospital_id=victim.hospital_id,
        deteriorated=victim.deteriorated_at is not None,
    )


def _incident_view(state: WorldState, incident: IncidentState) -> IncidentView:
    """Projects a reported incident (victims hidden until triage)."""
    spec = incident.spec
    victims = (
        tuple(_victim_view(v) for v in state.incident_victims(incident.incident_id))
        if incident.triaged
        else ()
    )
    assigned = tuple(
        u.unit_id
        for u in state.units.values()
        if u.incident_id == incident.incident_id and u.status in SCENE_COMMITTED_STATUSES
    )
    return IncidentView(
        incident_id=incident.incident_id,
        incident_type=spec.type,
        t_report_min=float(spec.t_report_min),
        x=spec.x,
        y=spec.y,
        intersection=spec.intersection,
        reported_victims=spec.reported_victims,
        caller_report=spec.caller_report,
        required_responder_capabilities=spec.required_responder_capabilities,
        status=incident.status,
        triaged=incident.triaged,
        accessible=incident.accessible,
        victims=victims,
        assigned_unit_ids=assigned,
        responder_capabilities_present=tuple(sorted(incident.caps_present, key=str)),
    )


def _destination(unit: UnitState) -> str | None:
    """Incident or hospital the unit is driving to."""
    if unit.status is UnitStatus.EN_ROUTE_SCENE:
        return unit.incident_id
    if unit.status is UnitStatus.TRANSPORTING:
        return unit.hospital_id
    return None


def _unit_view(unit: UnitState, t: float) -> UnitView:
    """Projects a unit at time ``t`` (interpolated position while driving)."""
    x, y = unit.position_at(t)
    return UnitView(
        unit_id=unit.unit_id,
        unit_class=unit.unit_class,
        unit_type=unit.unit_type,
        capabilities=unit.capabilities,
        status=unit.status,
        x=x,
        y=y,
        incident_id=unit.incident_id,
        hospital_id=unit.hospital_id,
        destination_id=_destination(unit),
        load_victim_ids=tuple(unit.load),
        seats=unit.seats,
        available_at_min=unit.available_at if unit.status is UnitStatus.BUSY_T0 else None,
    )


def _staff_view(member: StaffState, t: float) -> StaffView:
    """Projects a staff member."""
    return StaffView(
        staff_id=member.staff_id,
        role=member.spec.role,
        skills=member.spec.skills,
        home=member.spec.home,
        hospital_id=member.hospital_id,
        load=member.load,
        max_patients=member.spec.max_patients,
        on_shift=member.on_shift(t),
    )


def _hospital_view(state: WorldState, hospital: HospitalState, t: float) -> HospitalView:
    """Projects one hospital's bed board and staff."""
    hid = hospital.hospital_id
    inbound = tuple(
        u.unit_id for u in state.units.values() if u.status is UnitStatus.TRANSPORTING and u.hospital_id == hid
    )
    return HospitalView(
        hospital_id=hid,
        name=hospital.spec.name,
        x=hospital.spec.x,
        y=hospital.spec.y,
        capabilities=hospital.spec.capabilities,
        er_capacity=hospital.spec.er_capacity,
        critical_reserve=hospital.spec.critical_reserve,
        beds_occupied=state.beds_occupied(hid),
        beds_reserved=state.beds_reserved(hid),
        beds_free_unreserved=state.beds_free_unreserved(hid),
        queue_length=len(state.hospital_victims(hid, frozenset({VictimStatus.QUEUED}))),
        waiting_for_staff=len(state.hospital_victims(hid, frozenset({VictimStatus.WAITING_FOR_STAFF}))),
        in_treatment=len(state.hospital_victims(hid, frozenset({VictimStatus.IN_TREATMENT}))),
        inbound_unit_ids=inbound,
        staff=tuple(_staff_view(s, t) for s in state.staff_at(hid)),
    )


def build_public_view(state: WorldState, t: float) -> PublicView:
    """Builds the public view at time ``t``.

    Args:
        state: World state (read only).
        t: View time (current time, or a snapshot minute with no event since).

    Returns:
        The view; every collection sorted by id, only reported incidents, and
        only unlent float-pool staff in ``float_pool``.
    """
    incidents = tuple(_incident_view(state, i) for _, i in sorted(state.incidents.items()) if i.reported)
    pool = tuple(_staff_view(s, t) for s in state.staff.values() if s.is_float and not s.lent)
    return PublicView(
        t_min=t,
        horizon_min=state.horizon,
        incidents=incidents,
        units=tuple(_unit_view(u, t) for u in state.units.values()),
        hospitals=tuple(_hospital_view(state, h, t) for h in state.hospitals.values()),
        float_pool=pool,
    )


def build_snapshot_counts(state: WorldState) -> SnapshotCounts:
    """Counts victims by stage (truth, for the log; only reported incidents).

    Args:
        state: World state.

    Returns:
        The counts. ``waiting_on_scene`` includes untriaged, waiting and loading
        victims; ``in_transport`` loaded and transporting ones.
    """
    reported = [i for i in state.incidents.values() if i.reported]
    victims = [v for i in reported for v in state.incident_victims(i.incident_id)]

    def count(statuses: frozenset[VictimStatus]) -> int:
        """Number of reported victims with one of ``statuses``."""
        return sum(v.status in statuses for v in victims)

    return SnapshotCounts(
        incidents_reported=len(reported),
        incidents_open=sum(i.status is IncidentStatus.OPEN for i in reported),
        waiting_on_scene=count(ON_SCENE_VICTIM_STATUSES),
        in_transport=count(_IN_TRANSPORT),
        queued=count(frozenset({VictimStatus.QUEUED})),
        waiting_for_staff=count(frozenset({VictimStatus.WAITING_FOR_STAFF})),
        in_treatment=count(frozenset({VictimStatus.IN_TREATMENT})),
        treated=count(frozenset({VictimStatus.TREATED})),
        care_started_cum=sum(v.care_start is not None for v in victims),
        expectant=count(frozenset({VictimStatus.EXPECTANT})),
    )


def victim_deadline(victim: VictimState, params: SimParams) -> float | None:
    """Absolute care deadline of a victim.

    Args:
        victim: Victim state.
        params: Engine parameters.

    Returns:
        None for BLACK; ``deteriorated_at + 30`` for deteriorated victims;
        ``t_report + 480`` for GREEN; ``t_report + golden window`` otherwise.
    """
    if victim.triage is Triage.BLACK:
        return None
    if victim.deteriorated_at is not None:
        return round(victim.deteriorated_at + params.deteriorated_red_window_min, 1)
    if victim.triage is Triage.GREEN:
        return round(victim.t_report + GREEN_DEADLINE_MIN, 1)
    return round(victim.t_report + victim.spec.golden_window_min, 1)


def _victim_outcome(victim: VictimState, params: SimParams) -> VictimOutcome:
    """Builds one victim's outcome."""
    return VictimOutcome(
        victim_id=victim.victim_id,
        incident_id=victim.incident_id,
        t_report_min=victim.t_report,
        condition=victim.spec.condition,
        required_capability=victim.spec.required_capability,
        initial_triage=victim.spec.triage,
        final_triage=victim.triage,
        deteriorated_at_min=victim.deteriorated_at,
        deadline_min=victim_deadline(victim, params),
        als_on_scene_min=victim.als_on_scene,
        loaded_min=victim.loaded_at,
        hospital_id=victim.hospital_id if victim.arrived_hospital is not None else None,
        arrived_hospital_min=victim.arrived_hospital,
        bedded_min=victim.bedded_at,
        treatment_start_min=victim.treatment_start,
        care_start_min=victim.care_start,
        care_mode=victim.care_mode,
    )


def _incident_outcome(incident: IncidentState) -> IncidentOutcome:
    """Builds one incident's outcome."""
    return IncidentOutcome(
        incident_id=incident.incident_id,
        t_report_min=float(incident.spec.t_report_min),
        mci_cluster=incident.spec.mci_cluster,
        first_unit_on_scene_min=incident.first_unit_on_scene,
        first_ambulance_on_scene_min=incident.first_ambulance_on_scene,
        cleared_min=incident.cleared_at,
    )


def build_outcomes(state: WorldState, params: SimParams) -> RunOutcomes:
    """Builds the run outcomes at the horizon.

    Args:
        state: World state at the end of the run.
        params: Engine parameters.

    Returns:
        Victim and incident timelines (all, sorted by id) and busy minutes per
        unit over ``[0, horizon]``. Ambulances out of service report 0 minutes
        and are excluded from ``ambulances_in_service``.
    """
    ambulances = [u for u in state.units.values() if u.is_ambulance]
    responders = [u for u in state.units.values() if not u.is_ambulance]
    return RunOutcomes(
        horizon_min=state.horizon,
        victims=tuple(_victim_outcome(v, params) for v in state.victims.values()),
        incidents=tuple(_incident_outcome(i) for _, i in sorted(state.incidents.items())),
        ambulance_busy_min=tuple((u.unit_id, u.busy_minutes(state.horizon)) for u in ambulances),
        responder_busy_min=tuple((u.unit_id, u.busy_minutes(state.horizon)) for u in responders),
        ambulances_in_service=sum(u.status is not UnitStatus.OUT_OF_SERVICE for u in ambulances),
    )


def _hospital_problems(state: WorldState) -> list[str]:
    """Bed accounting and reservations."""
    problems = []
    for hid, hospital in state.hospitals.items():
        occupied, reserved = state.beds_occupied(hid), state.beds_reserved(hid)
        if occupied + reserved > hospital.spec.er_capacity or hospital.t0_beds < 0:
            problems.append(f"{hid}: occupied {occupied} + reserved {reserved} > {hospital.spec.er_capacity}")
        for unit_id, beds in hospital.reservations.items():
            unit = state.units.get(unit_id)
            ok = unit is not None and unit.status is UnitStatus.TRANSPORTING and unit.hospital_id == hid
            if not ok or unit is None or beds != len(unit.load):
                problems.append(f"{hid}: stale reservation of {beds} for {unit_id}")
    return problems


def _staff_problems(state: WorldState) -> list[str]:
    """Staff loads and treatment bookkeeping."""
    problems = []
    for sid, member in state.staff.items():
        if not 0 <= member.load <= member.spec.max_patients or member.t0_load < 0:
            problems.append(f"{sid}: load {member.load} outside 0..{member.spec.max_patients}")
        if member.is_float and not member.lent and (member.patients or member.hospital_id is not None):
            problems.append(f"{sid}: unlent float staff is working")
        for vid in member.patients:
            victim = state.victims[vid]
            if victim.status is not VictimStatus.IN_TREATMENT or sid not in (victim.doctor_id, victim.nurse_id):
                problems.append(f"{sid}: treats {vid} which is {victim.status}")
            if victim.hospital_id != member.hospital_id:
                problems.append(f"{sid}: treats {vid} at another hospital")
    return problems


def _unit_problems(state: WorldState) -> list[str]:
    """Unit load consistency."""
    problems = []
    for uid, unit in state.units.items():
        carrying = unit.status in CARRYING_STATUSES
        if carrying != bool(unit.load) or len(unit.load) > unit.seats:
            problems.append(f"{uid}: status {unit.status} with load {unit.load}")
        for vid in unit.load:
            victim = state.victims[vid]
            allowed = _UNIT_STATUS_FOR_VICTIM.get(victim.status, frozenset())
            if victim.unit_id != uid or unit.status not in allowed:
                problems.append(f"{uid}: carries {vid} which is {victim.status} on {victim.unit_id}")
        if unit.status in SCENE_COMMITTED_STATUSES and unit.incident_id is None:
            problems.append(f"{uid}: {unit.status} without an incident")
    return problems


def _victim_problems(state: WorldState) -> list[str]:
    """Each victim is in exactly one place."""
    problems = []
    carried = [vid for unit in state.units.values() for vid in unit.load]
    if len(carried) != len(set(carried)):
        problems.append("a victim is on board two units")
    for vid, victim in state.victims.items():
        incident = state.incidents[victim.incident_id]
        on_board = victim.status in _UNIT_STATUS_FOR_VICTIM
        if on_board != (vid in carried) or on_board != (victim.unit_id is not None):
            problems.append(f"{vid}: status {victim.status} but unit {victim.unit_id}")
        if (victim.status is VictimStatus.UNTRIAGED) == incident.triaged:
            problems.append(f"{vid}: status {victim.status} but incident triaged={incident.triaged}")
        if victim.status in _HOSPITAL_VICTIM_STATUSES and victim.hospital_id is None:
            problems.append(f"{vid}: {victim.status} without a hospital")
        treated = victim.status in {VictimStatus.IN_TREATMENT, VictimStatus.TREATED}
        if treated and (victim.care_start is None or victim.treatment_start is None):
            problems.append(f"{vid}: {victim.status} without a care start")
        if (victim.status is VictimStatus.EXPECTANT) != (incident.triaged and victim.triage is Triage.BLACK):
            problems.append(f"{vid}: expectant status does not match BLACK triage")
    return problems


def _incident_problems(state: WorldState) -> list[str]:
    """Clearing and reporting consistency."""
    problems = []
    for iid, incident in state.incidents.items():
        on_scene = any(v.status in ON_SCENE_VICTIM_STATUSES for v in state.incident_victims(iid))
        if incident.status is IncidentStatus.CLEARED and on_scene:
            problems.append(f"{iid}: cleared with victims on scene")
        if not incident.reported and (incident.triaged or any(u.incident_id == iid for u in state.units.values())):
            problems.append(f"{iid}: unreported but already attended")
    return problems


def check_world(state: WorldState) -> list[str]:
    """Lists every invariant violation of the world state.

    Args:
        state: World state.

    Returns:
        Human-readable problems; empty when the state is consistent.
    """
    return [
        *_hospital_problems(state),
        *_staff_problems(state),
        *_unit_problems(state),
        *_victim_problems(state),
        *_incident_problems(state),
    ]
