"""Typed, immutable view of one generated scenario (the ground truth).

Field names match the JSON files written by ``emergency_sim.datagen`` exactly,
and ``extra="forbid"`` makes any drift in the data fail loudly at load time.
Only the engine and the evaluation layer see these objects; controllers and
agents only ever see ``domain.views.PublicView``.
"""

from typing import Literal

from emergency_sim.domain.base import FrozenModel
from emergency_sim.domain.enums import (
    AmbulanceType,
    Capability,
    ResponderCapability,
    StaffRole,
    Triage,
)


class HospitalSpec(FrozenModel):
    """A hospital ER as generated (``hospitals.json``)."""

    id: str
    name: str
    x: float
    y: float
    trauma_level: int
    capabilities: tuple[Capability, ...]
    er_capacity: int
    icu_beds: int
    critical_reserve: int
    er_occupied_t0: int
    icu_occupied_t0: int
    er_free_t0: int
    intersection: str


class StationSpec(FrozenModel):
    """An ambulance / responder station (``stations.json``)."""

    id: str
    name: str
    x: float
    y: float


class AmbulanceSpec(FrozenModel):
    """An ambulance and its state at t=0 (``ambulances.json``)."""

    id: str
    type: AmbulanceType
    station: str
    x: float
    y: float
    capacity: int
    max_red_per_trip: int
    status: Literal["available", "busy", "out_of_service"]
    busy_until_min: int


class StaffSpec(FrozenModel):
    """An ER doctor or nurse (``staff.json``); ``home == "FLOAT"`` for the float pool."""

    id: str
    role: StaffRole
    home: str
    skills: tuple[str, ...]
    max_patients: int
    load_t0: int
    shift_end_min: int


class ResponderSpec(FrozenModel):
    """A non-medical responder team (``responders.json``)."""

    id: str
    type: str
    base: str
    capabilities: tuple[ResponderCapability, ...]
    x: float
    y: float
    crew: int
    status: Literal["available"]
    busy_until_min: int


class VictimSpec(FrozenModel):
    """A victim's true state, embedded in its incident."""

    victim_id: str
    condition: str
    triage: Triage
    required_capability: Capability
    needs_als: bool
    needs_decon: bool
    golden_window_min: int
    deteriorates_to_red_at_min: int | None


class IncidentSpec(FrozenModel):
    """An incident: the caller-visible report plus hidden ground truth (``incidents.json``)."""

    incident_id: str
    type: str
    t_report_min: int
    x: float
    y: float
    intersection: str
    congestion: float
    reported_victims: int
    caller_report: str
    required_responder_capabilities: tuple[ResponderCapability, ...]
    true_victims: int
    worst_triage: Triage
    time_sensitivity_min: int
    mci_cluster: bool
    victims: tuple[VictimSpec, ...]


class Scenario(FrozenModel):
    """A complete scenario: all six data files plus seed and horizon from the manifest."""

    name: str
    seed: int
    horizon_min: float
    hospitals: tuple[HospitalSpec, ...]
    stations: tuple[StationSpec, ...]
    ambulances: tuple[AmbulanceSpec, ...]
    staff: tuple[StaffSpec, ...]
    responders: tuple[ResponderSpec, ...]
    incidents: tuple[IncidentSpec, ...]
