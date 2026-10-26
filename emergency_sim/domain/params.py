"""Engine constants (documented modelling assumptions)."""

from dataclasses import dataclass

from emergency_sim import geo
from emergency_sim.domain.enums import Triage


@dataclass(frozen=True)
class SimParams:
    """Engine constants (documented assumptions). Horizon comes from Scenario.

    Attributes:
        snapshot_interval_min: Minutes between two ``SnapshotRecord``s.
        ambulance_turnout_min: Alert-to-wheels-rolling time of an ambulance.
        responder_turnout_min: Alert-to-wheels-rolling time of a responder team.
        scene_load_min: Minutes to load patients into an ambulance on scene.
        handover_min: Minutes an ambulance is busy handing over at the ER.
        heavy_access_min: Access delay after the last required heavy team
            (extrication / heavy_rescue / fire_suppression / hazmat) arrives.
        treatment_red_min: ER treatment duration of a RED patient.
        treatment_yellow_min: ER treatment duration of a YELLOW patient.
        treatment_green_min: ER treatment duration of a GREEN patient.
        deteriorated_red_window_min: Care window of a YELLOW that turned RED.
        t0_bed_release_interval_min: A hospital frees one t=0 bed every this
            many minutes.
        t0_staff_release_interval_min: A staff member sheds one t=0 patient
            every this many minutes.
    """

    snapshot_interval_min: float = 1.0
    ambulance_turnout_min: float = geo.TURNOUT_MIN["ambulance"]
    responder_turnout_min: float = geo.TURNOUT_MIN["responder"]
    scene_load_min: float = 3.0
    handover_min: float = 5.0
    heavy_access_min: float = 10.0
    treatment_red_min: float = 60.0
    treatment_yellow_min: float = 40.0
    treatment_green_min: float = 20.0
    deteriorated_red_window_min: float = 30.0
    t0_bed_release_interval_min: float = 15.0
    t0_staff_release_interval_min: float = 20.0

    def treatment_min(self, triage: Triage) -> float:
        """Returns the ER treatment duration for a patient's current triage.

        Args:
            triage: Current triage of the patient.

        Returns:
            Treatment duration in minutes.

        Raises:
            ValueError: If ``triage`` is BLACK (expectant victims are never treated).
        """
        match triage:
            case Triage.RED:
                return self.treatment_red_min
            case Triage.YELLOW:
                return self.treatment_yellow_min
            case Triage.GREEN:
                return self.treatment_green_min
            case _:
                raise ValueError(f"no treatment duration for triage {triage}")
