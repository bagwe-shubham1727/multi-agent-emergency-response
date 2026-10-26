"""Incident Assessment policy: turn caller reports and triage results into priorities.

Before triage only the caller's words are known, so severity is inferred from a
keyword lexicon. The lexicon was written against the severe phrasings that the
synthetic data generator uses, so it is only as informative as those reports;
the generator adds wording noise (see ``datagen``), and the B0-cues baseline
uses the same rule without any communication. After triage the visible victims decide (plan 5.3).
"""

from typing import Final

from emergency_sim.domain.agent_io import (
    AssessmentInput,
    AssessmentOutput,
    IncidentReport,
    PriorityReport,
    TriagedVictim,
)
from emergency_sim.domain.enums import TRIAGE_RANK, Triage, VictimStatus
from emergency_sim.policies.estimates import (
    CARDIAC_ARREST_CONDITION,
    incident_type_profile,
    nominal_deadline,
)
from emergency_sim.policies.params import PolicyParams

SEVERITY_CUES: Final = (
    "not responding", "not breathing", "cpr", "trapped", "badly burned", "collapsed",
    "blue", "a lot of blood", "not making sense", "drooping", "rubble",
)
"""Severity keywords (lower case) matching the generator's severe phrasings; used pre-triage."""

CARDIAC_ARREST_CUES: Final = ("cpr", "not responding")
"""Cues that, in a ``cardiac_event``, suggest a cardiac arrest."""

CARDIAC_EVENT_TYPE: Final = "cardiac_event"
SUSPECTED_RED_BONUS: Final = 25
VICTIM_BONUS_PER_VICTIM: Final = 2
VICTIM_BONUS_MAX: Final = 10
URGENCY_RED_WAITING: Final = 100
URGENCY_YELLOW_ONLY: Final = 60
URGENCY_GREEN_ONLY: Final = 30
MAX_RATIONALE_CHARS: Final = 200


def severity_cues(caller_report: str) -> tuple[str, ...]:
    """Lists the severity cues found in a caller report (case-insensitive).

    Args:
        caller_report: Free-text report.

    Returns:
        Matching cues in lexicon order.
    """
    text = caller_report.lower()
    return tuple(cue for cue in SEVERITY_CUES if cue in text)


def suspects_cardiac_arrest(incident_type: str, caller_report: str) -> bool:
    """Whether a pre-triage report suggests a cardiac arrest.

    Args:
        incident_type: Incident type label.
        caller_report: Free-text report.

    Returns:
        True for a ``cardiac_event`` mentioning CPR or "not responding".
    """
    text = caller_report.lower()
    return incident_type == CARDIAC_EVENT_TYPE and any(cue in text for cue in CARDIAC_ARREST_CUES)


def _clip(value: int, low: int = 0, high: int = 100) -> int:
    return max(low, min(high, value))


def _assess_untriaged(report: IncidentReport, params: PolicyParams) -> PriorityReport:
    profile = incident_type_profile(report.incident_type)
    cues = severity_cues(report.caller_report)
    suspected_red = bool(cues)
    arrest = suspects_cardiac_arrest(report.incident_type, report.caller_report)
    victim_bonus = min(VICTIM_BONUS_MAX, VICTIM_BONUS_PER_VICTIM * report.reported_victims)
    urgency = _clip(profile.base_urgency + (SUSPECTED_RED_BONUS if suspected_red else 0) + victim_bonus)
    if arrest:
        nominal = params.nominal_cardiac_arrest_deadline_min
    elif suspected_red:
        nominal = params.nominal_red_deadline_min
    else:
        nominal = params.nominal_yellow_deadline_min
    cue_text = ", ".join(cues) if cues else "no severity cue"
    return PriorityReport(
        incident_id=report.incident_id,
        urgency=urgency,
        suspected_red=suspected_red,
        suspected_cardiac_arrest=arrest,
        est_patients=report.reported_victims,
        needed_capabilities=list(profile.needed_capabilities),
        als_units_needed=1 if suspected_red else 0,
        deadline_in_min=round(nominal - report.minutes_since_report, 1),
        rationale=f"pre-triage {report.incident_type}: {cue_text}"[:MAX_RATIONALE_CHARS],
    )


def _waiting(victims: list[TriagedVictim]) -> list[TriagedVictim]:
    return sorted(
        (v for v in victims if v.status is VictimStatus.WAITING and v.triage is not Triage.BLACK),
        key=lambda v: (TRIAGE_RANK[v.triage], v.victim_id),
    )


def _triaged_urgency(waiting: list[TriagedVictim]) -> int:
    triages = {v.triage for v in waiting}
    if Triage.RED in triages:
        return URGENCY_RED_WAITING
    if Triage.YELLOW in triages:
        return URGENCY_YELLOW_ONLY
    if Triage.GREEN in triages:
        return URGENCY_GREEN_ONLY
    return 0


def victim_deadline_in(victim: TriagedVictim, minutes_since_report: float, params: PolicyParams) -> float:
    """Minutes left in a waiting victim's nominal care window.

    A YELLOW that turned RED gets the deteriorated RED window counted from
    its deterioration (plan 5.3: "a deteriorated victim gets 30"); every
    other window counts from the report.

    Args:
        victim: A waiting, triaged victim (not BLACK).
        minutes_since_report: Minutes since the incident was reported.
        params: Policy parameters.

    Returns:
        Minutes left (negative once the window has passed).
    """
    if victim.triage is Triage.RED and victim.minutes_since_deterioration is not None:
        window = nominal_deadline(victim.triage, victim.condition, params, deteriorated=True)
        return window - victim.minutes_since_deterioration
    return nominal_deadline(victim.triage, victim.condition, params) - minutes_since_report


def _assess_triaged(report: IncidentReport, params: PolicyParams) -> PriorityReport:
    waiting = _waiting(report.triaged_victims)
    red = [v for v in waiting if v.triage is Triage.RED]
    since = report.minutes_since_report
    if waiting:
        deadline_in = min(victim_deadline_in(v, since, params) for v in waiting)
    else:
        deadline_in = params.nominal_green_deadline_min - since
    counts = {t: sum(v.triage is t for v in waiting) for t in (Triage.RED, Triage.YELLOW, Triage.GREEN)}
    return PriorityReport(
        incident_id=report.incident_id,
        urgency=_triaged_urgency(waiting),
        suspected_red=bool(red),
        suspected_cardiac_arrest=any(v.condition == CARDIAC_ARREST_CONDITION for v in red),
        est_patients=len(waiting),
        needed_capabilities=sorted({v.required_capability for v in waiting}),
        als_units_needed=len(red),
        deadline_in_min=round(deadline_in, 1),
        rationale=(
            f"triaged: waiting RED {counts[Triage.RED]}, YELLOW {counts[Triage.YELLOW]}, "
            f"GREEN {counts[Triage.GREEN]}"
        )[:MAX_RATIONALE_CHARS],
    )


def assess_incident(report: IncidentReport, params: PolicyParams) -> PriorityReport:
    """Assesses one incident (pre-triage cues or post-triage victims).

    Args:
        report: The incident report.
        params: Policy parameters.

    Returns:
        Its priority report.
    """
    if report.triaged_victims:
        return _assess_triaged(report, params)
    return _assess_untriaged(report, params)


def assess(inp: AssessmentInput, params: PolicyParams) -> AssessmentOutput:
    """Incident Assessment policy (plan 5.3).

    Args:
        inp: Incidents to (re)assess.
        params: Policy parameters.

    Returns:
        One priority report per incident, sorted by incident id.
    """
    reports = sorted(inp.incidents, key=lambda r: r.incident_id)
    return AssessmentOutput(reports=[assess_incident(r, params) for r in reports])
