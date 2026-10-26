"""Decision parameters shared by the offline policies, the tools and the coordinator.

Every value is fixed a priori from first principles (public clinical ranges,
the documented triage weights, the shared B0 dispatch cap). They must never be
tuned on evaluation seeds; see the honesty protocol in the implementation plan
(section 8.3).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyParams:
    """Constants of the rule-based agent policies.

    Attributes:
        red_weight: Survival weight of a RED patient.
        yellow_weight: Survival weight of a YELLOW patient.
        green_weight: Survival weight of a GREEN patient.
        window_logistic_scale_min: Scale of the logistic window model
            ``p_window(slack) = 1 / (1 + exp(-slack / scale))``.
        travel_penalty_per_min: Utility penalty (lambda) per travel minute.
        nominal_red_deadline_min: Assumed RED care window (midpoint of the
            public 20-60 minute range).
        nominal_cardiac_arrest_deadline_min: Assumed cardiac arrest window.
        nominal_yellow_deadline_min: Assumed YELLOW care window.
        nominal_green_deadline_min: Assumed GREEN care window.
        deteriorated_red_deadline_min: Window of a YELLOW that turned RED.
        scene_overhead_min: Load and handling estimate on scene.
        no_unit_wait_min: Assumed extra wait when an incident gets no unit.
        queue_wait_per_patient_min: Assumed ER wait per queued patient.
        max_ambulances_pre_triage: Ambulance cap per incident before triage
            (shared with B0 for fairness).
        diversion_urgency_margin: Minimum urgency gap for diverting an ALS.
        diversion_max_eta_min: Maximum ETA of a diverted ALS to the new scene.
        heavy_access_min: Public scene-safety rule (plan 4.3): access is
            granted this long after the last required heavy team
            (extrication, heavy rescue, fire suppression, hazmat) arrives.
    """

    red_weight: float = 3.0
    yellow_weight: float = 2.0
    green_weight: float = 1.0
    window_logistic_scale_min: float = 5.0
    travel_penalty_per_min: float = 0.01
    nominal_red_deadline_min: float = 40.0
    nominal_cardiac_arrest_deadline_min: float = 10.0
    nominal_yellow_deadline_min: float = 165.0
    nominal_green_deadline_min: float = 480.0
    deteriorated_red_deadline_min: float = 30.0
    scene_overhead_min: float = 5.0
    no_unit_wait_min: float = 20.0
    queue_wait_per_patient_min: float = 15.0
    max_ambulances_pre_triage: int = 3
    diversion_urgency_margin: int = 20
    diversion_max_eta_min: float = 15.0
    heavy_access_min: float = 10.0
