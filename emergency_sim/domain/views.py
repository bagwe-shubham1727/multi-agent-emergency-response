"""The information boundary: what controllers and agents may observe.

These views contain no hidden truth: no ``golden_window_min``,
``deteriorates_to_red_at_min``, ``true_victims``, ``worst_triage`` or
``time_sensitivity_min``, and no victims of untriaged incidents. The engine
builds them; everything else reads them.
"""

from emergency_sim.domain.base import FrozenModel
from emergency_sim.domain.enums import (
    Capability,
    IncidentStatus,
    ResponderCapability,
    StaffRole,
    Triage,
    TriggerKind,
    UnitClass,
    UnitStatus,
    VictimStatus,
)


class VictimView(FrozenModel):
    """A triaged victim as seen by controllers (current triage, not ground truth)."""

    victim_id: str
    incident_id: str
    triage: Triage
    condition: str
    required_capability: Capability
    needs_als: bool
    status: VictimStatus
    unit_id: str | None
    hospital_id: str | None
    deteriorated: bool


class IncidentView(FrozenModel):
    """A reported incident; ``victims`` is empty until the scene is triaged."""

    incident_id: str
    incident_type: str
    t_report_min: float
    x: float
    y: float
    intersection: str
    reported_victims: int
    caller_report: str
    required_responder_capabilities: tuple[ResponderCapability, ...]
    status: IncidentStatus
    triaged: bool
    accessible: bool
    victims: tuple[VictimView, ...]
    assigned_unit_ids: tuple[str, ...]
    responder_capabilities_present: tuple[ResponderCapability, ...]


class UnitView(FrozenModel):
    """An ambulance or responder team.

    Attributes:
        unit_type: ``"ALS"``/``"BLS"`` for ambulances, the team type otherwise.
        capabilities: Responder capabilities (empty for ambulances).
        destination_id: Incident or hospital the unit is heading to, if any.
        seats: Patient capacity (0 for responders).
        available_at_min: When a ``BUSY_T0`` unit becomes available, else None.
    """

    unit_id: str
    unit_class: UnitClass
    unit_type: str
    capabilities: tuple[str, ...]
    status: UnitStatus
    x: float
    y: float
    incident_id: str | None
    hospital_id: str | None
    destination_id: str | None
    load_victim_ids: tuple[str, ...]
    seats: int
    available_at_min: float | None


class StaffView(FrozenModel):
    """A doctor or nurse; ``hospital_id`` is None for unlent float staff."""

    staff_id: str
    role: StaffRole
    skills: tuple[str, ...]
    home: str
    hospital_id: str | None
    load: int
    max_patients: int
    on_shift: bool


class HospitalView(FrozenModel):
    """Bed board and staff of one hospital."""

    hospital_id: str
    name: str
    x: float
    y: float
    capabilities: tuple[Capability, ...]
    er_capacity: int
    critical_reserve: int
    beds_occupied: int
    beds_reserved: int
    beds_free_unreserved: int
    queue_length: int
    waiting_for_staff: int
    in_treatment: int
    inbound_unit_ids: tuple[str, ...]
    staff: tuple[StaffView, ...]


class PublicView(FrozenModel):
    """Everything observable at time ``t_min``.

    All collections are sorted by id; ``incidents`` holds reported incidents only.
    """

    t_min: float
    horizon_min: float
    incidents: tuple[IncidentView, ...]
    units: tuple[UnitView, ...]
    hospitals: tuple[HospitalView, ...]
    float_pool: tuple[StaffView, ...]


class Trigger(FrozenModel):
    """One reason the engine stopped for a decision."""

    kind: TriggerKind
    incident_id: str | None = None
    unit_id: str | None = None
    hospital_id: str | None = None
    victim_ids: tuple[str, ...] = ()


class DecisionPoint(FrozenModel):
    """A moment at which the controller is asked to act."""

    t_min: float
    triggers: tuple[Trigger, ...]
