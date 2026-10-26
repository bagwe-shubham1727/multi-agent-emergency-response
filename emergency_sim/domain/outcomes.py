"""Per-run outcomes (from the engine) and metrics (from evaluation)."""

from typing import Literal

from emergency_sim.domain.base import FrozenModel
from emergency_sim.domain.enums import Capability, Triage


class VictimOutcome(FrozenModel):
    """Timeline and result of one victim.

    Attributes:
        deadline_min: Absolute care deadline; None for BLACK. RED:
            ``t_report + window``; deteriorated: ``det_at + 30``; YELLOW:
            ``t_report + window``; GREEN: ``t_report + 480``.
        care_start_min: Start of ER treatment at a capable hospital, or ALS
            arrival on scene for a cardiac arrest (field care).
        care_mode: Where care started.
    """

    victim_id: str
    incident_id: str
    t_report_min: float
    condition: str
    required_capability: Capability
    initial_triage: Triage
    final_triage: Triage
    deteriorated_at_min: float | None
    deadline_min: float | None
    als_on_scene_min: float | None
    loaded_min: float | None
    hospital_id: str | None
    arrived_hospital_min: float | None
    bedded_min: float | None
    treatment_start_min: float | None
    care_start_min: float | None
    care_mode: Literal["field", "hospital"] | None


class IncidentOutcome(FrozenModel):
    """Response timeline of one incident."""

    incident_id: str
    t_report_min: float
    mci_cluster: bool
    first_unit_on_scene_min: float | None
    first_ambulance_on_scene_min: float | None
    cleared_min: float | None


class RunOutcomes(FrozenModel):
    """Everything metrics need from the engine at the end of a run.

    Attributes:
        ambulance_busy_min: ``(unit_id, busy minutes)`` pairs sorted by id.
        responder_busy_min: ``(unit_id, busy minutes)`` pairs sorted by id.
        ambulances_in_service: Ambulances not out of service.
    """

    horizon_min: float
    victims: tuple[VictimOutcome, ...]
    incidents: tuple[IncidentOutcome, ...]
    ambulance_busy_min: tuple[tuple[str, float], ...]
    responder_busy_min: tuple[tuple[str, float], ...]
    ambulances_in_service: int


class RunMetrics(FrozenModel):
    """Headline metrics of one run (formulas in section 6 of the plan)."""

    incidents: int
    incidents_reached: int
    response_time_mean_min: float | None
    response_time_p90_min: float | None
    red_response_mean_min: float | None
    patients_total: int
    victims_served: int
    served_rate: float
    red_total: int
    red_in_window: int
    red_in_window_rate: float | None
    unserved_critical: int
    deteriorated: int
    black_victims: int
    survival_weighted_score: float
    door_to_doctor_mean_min: float | None
    ambulance_utilisation: float
    responder_utilisation: float
    er_utilisation: float
    doctor_utilisation: float
    nurse_utilisation: float
    hospital_full_on_arrival: int
    controller_omissions: int
    messages_total: int
    messages_by_type: dict[str, int]
    conflicts: int
    conflicts_by_kind: dict[str, int]
    resolutions: dict[str, int]
    agent_calls: int
    fallbacks: int
    float_staff_lent: int
    diversions: int
