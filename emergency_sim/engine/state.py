"""Mutable world state of the simulation engine (private to ``engine``).

The engine is the single source of truth. Its state lives in plain mutable
dataclasses built once from a ``Scenario`` by ``from_scenario``. Nothing outside
``emergency_sim.engine`` may import this module: controllers only ever see the
immutable ``PublicView`` built by ``engine.observe``.

Derived quantities (beds occupied, staff load, the ER queue) are computed from
victim statuses rather than stored twice, so they cannot drift apart.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Final, Literal

from emergency_sim.domain.enums import (
    AmbulanceType,
    IncidentStatus,
    ResponderCapability,
    Triage,
    UnitClass,
    UnitStatus,
    VictimStatus,
)
from emergency_sim.domain.scenario import (
    AmbulanceSpec,
    HospitalSpec,
    IncidentSpec,
    ResponderSpec,
    Scenario,
    StaffSpec,
    VictimSpec,
)

FLOAT_HOME: Final = "FLOAT"
"""``StaffSpec.home`` value of float-pool staff."""

CARDIAC_ARREST: Final = "cardiac arrest"
"""Victim condition that receives field care when an ALS unit reaches the scene."""

TIME_DECIMALS: Final = 1
"""Every scheduled time is rounded to this many decimals (0.1 min)."""

POSITION_DECIMALS: Final = 3
"""Interpolated positions are rounded to this many decimals (1 m)."""

IDLE_STATUSES: Final = frozenset({UnitStatus.AVAILABLE, UnitStatus.OUT_OF_SERVICE})
"""Unit statuses that do not count as busy."""

MOVING_STATUSES: Final = frozenset({UnitStatus.EN_ROUTE_SCENE, UnitStatus.TRANSPORTING})
"""Unit statuses in which the unit is driving."""

CARRYING_STATUSES: Final = frozenset(
    {
        UnitStatus.LOADING,
        UnitStatus.AWAITING_DESTINATION,
        UnitStatus.TRANSPORTING,
        UnitStatus.AT_HOSPITAL_FULL,
    }
)
"""Unit statuses in which the unit holds patients."""

SCENE_COMMITTED_STATUSES: Final = frozenset(
    {UnitStatus.EN_ROUTE_SCENE, UnitStatus.ON_SCENE, UnitStatus.LOADING}
)
"""Unit statuses that make a unit 'assigned' to its incident."""

ON_SCENE_VICTIM_STATUSES: Final = frozenset(
    {VictimStatus.UNTRIAGED, VictimStatus.WAITING, VictimStatus.LOADING}
)
"""Victim statuses that keep an incident open."""

BEDDED_STATUSES: Final = frozenset({VictimStatus.WAITING_FOR_STAFF, VictimStatus.IN_TREATMENT})
"""Victim statuses that occupy an ER bed."""

Point = tuple[float, float]


class EngineError(Exception):
    """A programming error inside the engine or a misuse of its API."""


def round_time(t: float) -> float:
    """Rounds a simulation time to the engine's 0.1-minute resolution.

    Args:
        t: Time in minutes.

    Returns:
        ``round(t, 1)``.
    """
    return round(t, TIME_DECIMALS)


@dataclass
class Movement:
    """A straight drive (in schedule terms) from ``origin`` to ``target``.

    Attributes:
        origin: Start position.
        target: End position.
        depart_t: Time the wheels start rolling (after turnout).
        arrive_t: Scheduled arrival time.
    """

    origin: Point
    target: Point
    depart_t: float
    arrive_t: float

    def position_at(self, t: float) -> Point:
        """Linearly interpolates the position by elapsed fraction after turnout.

        Args:
            t: Time in minutes.

        Returns:
            The (x, y) position, rounded to metres.
        """
        span = self.arrive_t - self.depart_t
        if t <= self.depart_t or span <= 0:
            frac = 0.0 if t < self.arrive_t else 1.0
        else:
            frac = min(1.0, (t - self.depart_t) / span)
        x = self.origin[0] + (self.target[0] - self.origin[0]) * frac
        y = self.origin[1] + (self.target[1] - self.origin[1]) * frac
        return (round(x, POSITION_DECIMALS), round(y, POSITION_DECIMALS))


@dataclass
class UnitState:
    """An ambulance or responder team.

    Attributes:
        unit_id: Unit id.
        unit_class: Ambulance or responder.
        unit_type: ``"ALS"``/``"BLS"`` or the responder team type.
        capabilities: Responder capabilities (empty for ambulances).
        seats: Patient capacity (0 for responders).
        max_red: RED patients allowed per trip.
        status: Lifecycle status.
        x: Current (or last fixed) position.
        y: Current (or last fixed) position.
        available_at: End of a ``BUSY_T0`` period, else None.
        incident_id: Incident the unit serves (kept until it is available again).
        hospital_id: Destination or current hospital.
        load: Victim ids on board (or being loaded).
        movement: Current drive, if moving.
        trip: Token that invalidates stale arrival events after a divert.
        busy_since: Start of the current busy period, if busy.
        busy_total: Busy minutes of closed busy periods.
    """

    unit_id: str
    unit_class: UnitClass
    unit_type: str
    capabilities: tuple[str, ...]
    seats: int
    max_red: int
    status: UnitStatus
    x: float
    y: float
    available_at: float | None = None
    incident_id: str | None = None
    hospital_id: str | None = None
    load: list[str] = field(default_factory=list)
    movement: Movement | None = None
    trip: int = 0
    busy_since: float | None = None
    busy_total: float = 0.0

    @property
    def is_ambulance(self) -> bool:
        """Whether the unit is an ambulance."""
        return self.unit_class is UnitClass.AMBULANCE

    @property
    def is_als(self) -> bool:
        """Whether the unit is an ALS ambulance."""
        return self.is_ambulance and self.unit_type == AmbulanceType.ALS

    def set_status(self, status: UnitStatus, now: float) -> None:
        """Changes status and keeps busy-minute accounting up to date.

        Args:
            status: New status.
            now: Current simulation time.
        """
        was_busy = self.status not in IDLE_STATUSES
        will_be_busy = status not in IDLE_STATUSES
        if will_be_busy and not was_busy:
            self.busy_since = now
        elif was_busy and not will_be_busy and self.busy_since is not None:
            self.busy_total += now - self.busy_since
            self.busy_since = None
        self.status = status

    def position_at(self, t: float) -> Point:
        """Returns the unit's position at time ``t`` (interpolated while driving).

        Args:
            t: Time in minutes.

        Returns:
            The (x, y) position.
        """
        if self.status in MOVING_STATUSES and self.movement is not None:
            return self.movement.position_at(t)
        return (self.x, self.y)

    def busy_minutes(self, until: float) -> float:
        """Total busy minutes up to ``until``, including an open busy period.

        Args:
            until: End of the accounting window (the horizon).

        Returns:
            Busy minutes, rounded to 0.1.
        """
        open_part = 0.0 if self.busy_since is None else max(0.0, until - self.busy_since)
        return round_time(self.busy_total + open_part)


@dataclass
class VictimState:
    """A victim: ground truth plus its timeline.

    Attributes:
        spec: Generated ground truth.
        incident_id: Incident the victim belongs to.
        t_report: Report time of the incident.
        triage: Current true triage (changes on deterioration).
        needs_als: Current ALS need.
        status: Lifecycle status.
        unit_id: Ambulance carrying (or loading) the victim.
        hospital_id: Destination or current hospital.
        deteriorated_at: Time the victim turned RED, if it did.
        als_on_scene: First ALS arrival while the victim was on scene.
        loaded_at: End of loading.
        arrived_hospital: Arrival at the (final) hospital door.
        bedded_at: Time the victim got an ER bed.
        treatment_start: Start of ER treatment.
        care_start: Start of care (field care or ER treatment).
        care_mode: Where care started.
        doctor_id: Treating doctor.
        nurse_id: Treating nurse.
    """

    spec: VictimSpec
    incident_id: str
    t_report: float
    triage: Triage
    needs_als: bool
    status: VictimStatus = VictimStatus.UNTRIAGED
    unit_id: str | None = None
    hospital_id: str | None = None
    deteriorated_at: float | None = None
    als_on_scene: float | None = None
    loaded_at: float | None = None
    arrived_hospital: float | None = None
    bedded_at: float | None = None
    treatment_start: float | None = None
    care_start: float | None = None
    care_mode: Literal["field", "hospital"] | None = None
    doctor_id: str | None = None
    nurse_id: str | None = None

    @property
    def victim_id(self) -> str:
        """The victim id."""
        return self.spec.victim_id


@dataclass
class IncidentState:
    """An incident and its scene state.

    Attributes:
        spec: Generated incident (report plus hidden truth).
        victim_ids: Victims at the scene, sorted.
        reported: Whether the report event has happened.
        status: Open or cleared.
        triaged: Whether START triage has happened.
        accessible: Whether ambulances may load.
        access_pending: Whether an ``ACCESS_GRANTED`` event is scheduled.
        caps_present: Responder capabilities that have arrived on scene.
        first_unit_on_scene: First arrival of any unit.
        first_ambulance_on_scene: First ambulance arrival.
        cleared_at: Time the incident was cleared.
    """

    spec: IncidentSpec
    victim_ids: tuple[str, ...]
    reported: bool = False
    status: IncidentStatus = IncidentStatus.OPEN
    triaged: bool = False
    accessible: bool = True
    access_pending: bool = False
    caps_present: set[ResponderCapability] = field(default_factory=set)
    first_unit_on_scene: float | None = None
    first_ambulance_on_scene: float | None = None
    cleared_at: float | None = None

    @property
    def incident_id(self) -> str:
        """The incident id."""
        return self.spec.incident_id

    @property
    def position(self) -> Point:
        """The scene position."""
        return (self.spec.x, self.spec.y)


@dataclass
class HospitalState:
    """An ER.

    Attributes:
        spec: Generated hospital.
        t0_beds: Beds still held by patients present at t=0.
        reservations: Beds reserved per inbound unit id.
    """

    spec: HospitalSpec
    t0_beds: int
    reservations: dict[str, int] = field(default_factory=dict)

    @property
    def hospital_id(self) -> str:
        """The hospital id."""
        return self.spec.id

    @property
    def position(self) -> Point:
        """The hospital position."""
        return (self.spec.x, self.spec.y)


@dataclass
class StaffState:
    """A doctor or nurse.

    Attributes:
        spec: Generated staff member.
        hospital_id: Hospital the member works at (None for unlent float staff).
        lent: Whether a float member has been lent.
        t0_load: Patients still carried from t=0.
        patients: Victim ids currently treated.
    """

    spec: StaffSpec
    hospital_id: str | None
    lent: bool = False
    t0_load: int = 0
    patients: set[str] = field(default_factory=set)

    @property
    def staff_id(self) -> str:
        """The staff id."""
        return self.spec.id

    @property
    def load(self) -> int:
        """Current number of patients (t=0 patients plus ER patients)."""
        return self.t0_load + len(self.patients)

    @property
    def is_float(self) -> bool:
        """Whether the member belongs to the float pool."""
        return self.spec.home == FLOAT_HOME

    def on_shift(self, t: float) -> bool:
        """Whether the member is on shift at ``t``.

        Args:
            t: Time in minutes.

        Returns:
            ``t < shift_end_min``.
        """
        return t < self.spec.shift_end_min


@dataclass
class WorldState:
    """Everything the engine knows, keyed by id.

    Attributes:
        horizon: Simulation horizon in minutes.
        units: Ambulances and responder teams.
        victims: All victims (including those of unreported incidents).
        incidents: All incidents.
        hospitals: All hospitals.
        staff: All doctors and nurses.
    """

    horizon: float
    units: dict[str, UnitState]
    victims: dict[str, VictimState]
    incidents: dict[str, IncidentState]
    hospitals: dict[str, HospitalState]
    staff: dict[str, StaffState]

    def incident_victims(self, incident_id: str) -> Iterator[VictimState]:
        """Yields the victims of an incident in id order.

        Args:
            incident_id: Incident id.

        Yields:
            Victim states.
        """
        for victim_id in self.incidents[incident_id].victim_ids:
            yield self.victims[victim_id]

    def hospital_victims(self, hospital_id: str, statuses: frozenset[VictimStatus]) -> list[VictimState]:
        """Lists victims at a hospital with one of the given statuses.

        Args:
            hospital_id: Hospital id.
            statuses: Statuses to include.

        Returns:
            Matching victims sorted by id.
        """
        return [
            v
            for _, v in sorted(self.victims.items())
            if v.hospital_id == hospital_id and v.status in statuses
        ]

    def beds_occupied(self, hospital_id: str) -> int:
        """Beds held by t=0 patients plus bedded ER patients.

        Args:
            hospital_id: Hospital id.

        Returns:
            Occupied beds.
        """
        bedded = len(self.hospital_victims(hospital_id, BEDDED_STATUSES))
        return self.hospitals[hospital_id].t0_beds + bedded

    def beds_reserved(self, hospital_id: str) -> int:
        """Beds reserved for inbound units.

        Args:
            hospital_id: Hospital id.

        Returns:
            Reserved beds.
        """
        return sum(self.hospitals[hospital_id].reservations.values())

    def beds_free_unreserved(self, hospital_id: str) -> int:
        """Beds neither occupied nor reserved.

        Args:
            hospital_id: Hospital id.

        Returns:
            Free unreserved beds (never negative while invariants hold).
        """
        capacity = self.hospitals[hospital_id].spec.er_capacity
        return capacity - self.beds_occupied(hospital_id) - self.beds_reserved(hospital_id)

    def staff_at(self, hospital_id: str) -> list[StaffState]:
        """Lists staff working at a hospital (home staff and lent float staff).

        Args:
            hospital_id: Hospital id.

        Returns:
            Staff sorted by id.
        """
        return [s for _, s in sorted(self.staff.items()) if s.hospital_id == hospital_id]


def _ambulance_state(spec: AmbulanceSpec) -> UnitState:
    """Builds the initial state of an ambulance."""
    status = {
        "available": UnitStatus.AVAILABLE,
        "busy": UnitStatus.BUSY_T0,
        "out_of_service": UnitStatus.OUT_OF_SERVICE,
    }[spec.status]
    busy = status is UnitStatus.BUSY_T0
    return UnitState(
        unit_id=spec.id,
        unit_class=UnitClass.AMBULANCE,
        unit_type=str(spec.type),
        capabilities=(),
        seats=spec.capacity,
        max_red=spec.max_red_per_trip,
        status=status,
        x=spec.x,
        y=spec.y,
        available_at=float(spec.busy_until_min) if busy else None,
        busy_since=0.0 if busy else None,
    )


def _responder_state(spec: ResponderSpec) -> UnitState:
    """Builds the initial state of a responder team (always available at t=0)."""
    return UnitState(
        unit_id=spec.id,
        unit_class=UnitClass.RESPONDER,
        unit_type=spec.type,
        capabilities=tuple(str(c) for c in spec.capabilities),
        seats=0,
        max_red=0,
        status=UnitStatus.AVAILABLE,
        x=spec.x,
        y=spec.y,
    )


def _incident_states(scenario: Scenario) -> tuple[dict[str, IncidentState], dict[str, VictimState]]:
    """Builds incident and victim states."""
    incidents: dict[str, IncidentState] = {}
    victims: dict[str, VictimState] = {}
    for spec in sorted(scenario.incidents, key=lambda i: i.incident_id):
        ids = tuple(sorted(v.victim_id for v in spec.victims))
        incidents[spec.incident_id] = IncidentState(
            spec=spec, victim_ids=ids, accessible=not spec.required_responder_capabilities
        )
        for v in spec.victims:
            if v.victim_id in victims:
                raise EngineError(f"duplicate victim id {v.victim_id}")
            victims[v.victim_id] = VictimState(
                spec=v,
                incident_id=spec.incident_id,
                t_report=float(spec.t_report_min),
                triage=v.triage,
                needs_als=v.needs_als,
            )
    return incidents, victims


def from_scenario(scenario: Scenario) -> WorldState:
    """Builds the initial world state of a scenario.

    Args:
        scenario: The scenario (ground truth).

    Returns:
        The world at t=0.

    Raises:
        EngineError: If ids collide.
    """
    units = {a.id: _ambulance_state(a) for a in scenario.ambulances}
    for r in scenario.responders:
        if r.id in units:
            raise EngineError(f"duplicate unit id {r.id}")
        units[r.id] = _responder_state(r)
    incidents, victims = _incident_states(scenario)
    hospitals = {h.id: HospitalState(spec=h, t0_beds=h.er_occupied_t0) for h in scenario.hospitals}
    staff = {
        s.id: StaffState(spec=s, hospital_id=None if s.home == FLOAT_HOME else s.home, t0_load=s.load_t0)
        for s in scenario.staff
    }
    return WorldState(
        horizon=scenario.horizon_min,
        units=dict(sorted(units.items())),
        victims=dict(sorted(victims.items())),
        incidents=incidents,
        hospitals=dict(sorted(hospitals.items())),
        staff=dict(sorted(staff.items())),
    )
