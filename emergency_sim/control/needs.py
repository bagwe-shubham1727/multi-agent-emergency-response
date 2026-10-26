"""Incident need accounting shared by B0 and the agent observations.

Both the greedy baseline and the dispatch agent's input are built from
``incident_needs`` so that seat, ALS and responder accounting never diverges
between conditions (plan section 5.4, ``control/needs.py``).
"""

from collections.abc import Mapping, Sequence
from typing import Final

from emergency_sim import geo
from emergency_sim.domain.public_rules import HEAVY_ACCESS_CAPABILITIES, effective_capabilities
from emergency_sim.domain.agent_io import IncidentNeed, PriorityReport
from emergency_sim.domain.enums import (
    AmbulanceType,
    IncidentStatus,
    ResponderCapability,
    Triage,
    UnitClass,
    UnitStatus,
    VictimStatus,
)
from emergency_sim.domain.views import (
    HospitalView,
    IncidentView,
    PublicView,
    UnitView,
    VictimView,
)
from emergency_sim.policies.params import PolicyParams

SCENE_BOUND_STATUSES: Final = frozenset({UnitStatus.EN_ROUTE_SCENE, UnitStatus.ON_SCENE})
"""Statuses of a unit that will still pick up patients at its incident."""

AT_INCIDENT_STATUSES: Final = frozenset(
    {UnitStatus.EN_ROUTE_SCENE, UnitStatus.ON_SCENE, UnitStatus.LOADING}
)
"""Statuses of an ambulance that counts as assigned to its incident."""



def incident_needs(
    view: PublicView,
    board: Mapping[str, PriorityReport] | None,
    params: PolicyParams,
) -> list[IncidentNeed]:
    """Computes the needs of every open incident, sorted by incident id.

    Args:
        view: Current public view.
        board: Latest priority report per incident (deadline already aged
            to ``view.t_min``), or None for B0, which has no assessments.
        params: Policy parameters (nominal deadlines when no report exists).

    Returns:
        One ``IncidentNeed`` per open incident.
    """
    reports = board or {}
    return [
        incident_need(incident, view.units, reports.get(incident.incident_id), view.t_min, params)
        for incident in sorted(view.incidents, key=lambda i: i.incident_id)
        if incident.status is IncidentStatus.OPEN
    ]


def incident_need(
    incident: IncidentView,
    units: Sequence[UnitView],
    report: PriorityReport | None,
    t_min: float,
    params: PolicyParams,
) -> IncidentNeed:
    """Computes the need of one incident.

    Args:
        incident: The incident.
        units: All units of the view.
        report: Its priority report, if any.
        t_min: Current time.
        params: Policy parameters.

    Returns:
        The need; flags and urgency come from ``report`` (defaults without).
    """
    assigned = assigned_ambulances(incident, units)
    scene_bound = [u for u in assigned if u.status in SCENE_BOUND_STATUSES]
    als_bound = [u for u in scene_bound if u.unit_type == AmbulanceType.ALS]
    red_waiting = sum(
        1 for v in incident.victims if v.triage is Triage.RED and v.status is VictimStatus.WAITING
    )
    if incident.triaged:
        remaining = sum(1 for v in incident.victims if v.status is VictimStatus.WAITING)
        als_needed = red_waiting
    else:
        remaining = report.est_patients if report else incident.reported_victims
        als_needed = report.als_units_needed if report else 0
    return IncidentNeed(
        incident_id=incident.incident_id,
        x=incident.x,
        y=incident.y,
        urgency=report.urgency if report else 0,
        suspected_red=report.suspected_red if report else False,
        suspected_cardiac_arrest=report.suspected_cardiac_arrest if report else False,
        deadline_in_min=(
            report.deadline_in_min
            if report
            else params.nominal_yellow_deadline_min - (t_min - incident.t_report_min)
        ),
        triaged=incident.triaged,
        est_patients_remaining=max(0, remaining),
        red_waiting=red_waiting,
        ambulances_assigned=len(assigned),
        seats_en_route=sum(u.seats for u in scene_bound),
        als_en_route=len(als_bound),
        als_needed=max(0, als_needed),
        responder_caps_missing=missing_responder_caps(incident, units),
        access_wait_min=access_wait(incident, units, params),
    )


def assigned_ambulances(incident: IncidentView, units: Sequence[UnitView]) -> list[UnitView]:
    """Returns the ambulances heading to, waiting at or loading at an incident.

    Args:
        incident: The incident.
        units: All units of the view.

    Returns:
        The ambulances, sorted by unit id.
    """
    return sorted(
        (
            u
            for u in units
            if u.unit_class is UnitClass.AMBULANCE
            and u.status in AT_INCIDENT_STATUSES
            and (u.incident_id == incident.incident_id or u.unit_id in incident.assigned_unit_ids)
        ),
        key=lambda u: u.unit_id,
    )


def missing_responder_caps(
    incident: IncidentView, units: Sequence[UnitView]
) -> list[ResponderCapability]:
    """Returns required responder capabilities neither present nor en route.

    Args:
        incident: The incident.
        units: All units of the view.

    Returns:
        Missing capabilities in the incident's required order; empty once the
        incident is accessible.
    """
    if incident.accessible:
        return []
    covered: set[str] = set(incident.responder_capabilities_present)
    for unit in units:
        if (
            unit.unit_class is UnitClass.RESPONDER
            and unit.incident_id == incident.incident_id
            and unit.status in SCENE_BOUND_STATUSES
        ):
            covered.update(unit.capabilities)
    covered = effective_capabilities(covered)
    return [c for c in incident.required_responder_capabilities if c not in covered]


def _team_capabilities(unit: UnitView) -> set[str]:
    """Returns what a responder team covers under the public access rule."""
    return effective_capabilities(unit.capabilities)


def access_wait(
    incident: IncidentView, units: Sequence[UnitView], params: PolicyParams
) -> float | None:
    """Estimates the minutes until ambulances may load at an incident.

    Access needs every required responder capability on scene (plan 4.3), so
    the estimate is the latest arrival among the teams still en route for
    the capabilities not yet present (travel from their current position),
    plus ``heavy_access_min`` when any heavy capability is required. When
    every team is already on scene but access is pending, the heavy delay is
    an upper bound (the arrival time of the last team is not public).

    Args:
        incident: The incident.
        units: All units of the view.
        params: Policy parameters (``heavy_access_min``).

    Returns:
        0.0 if accessible; None if some required capability is neither on
        scene nor en route (unknown); otherwise the estimate in minutes.
    """
    if incident.accessible:
        return 0.0
    present = effective_capabilities(incident.responder_capabilities_present)
    en_route = [
        u for u in units
        if u.unit_class is UnitClass.RESPONDER and u.incident_id == incident.incident_id
        and u.status is UnitStatus.EN_ROUTE_SCENE
    ]
    arrival = 0.0
    for cap in incident.required_responder_capabilities:
        if cap.value in present:
            continue
        etas = [
            geo.travel_time((u.x, u.y), (incident.x, incident.y), UnitClass.RESPONDER.value)
            for u in en_route if cap.value in _team_capabilities(u)
        ]
        if not etas:
            return None
        arrival = max(arrival, min(etas))
    heavy = any(c in HEAVY_ACCESS_CAPABILITIES for c in incident.required_responder_capabilities)
    return round(arrival + (params.heavy_access_min if heavy else 0.0), 1)


def pre_triage_slots(need: IncidentNeed, params: PolicyParams) -> int:
    """Returns how many more ambulances the pre-triage cap still allows.

    Args:
        need: The incident need.
        params: Policy parameters (``max_ambulances_pre_triage``).

    Returns:
        A large number after triage (no cap), else the remaining cap.
    """
    if need.triaged:
        return 1_000_000
    return max(0, params.max_ambulances_pre_triage - need.ambulances_assigned)


def seats_short(need: IncidentNeed, params: PolicyParams) -> int:
    """Returns the unmet seat demand, respecting the pre-triage cap.

    Args:
        need: The incident need.
        params: Policy parameters.

    Returns:
        ``est_patients_remaining - seats_en_route`` (at least 0), or 0 when
        the pre-triage ambulance cap is already reached.
    """
    if pre_triage_slots(need, params) == 0:
        return 0
    return max(0, need.est_patients_remaining - need.seats_en_route)


def has_unmet_need(need: IncidentNeed, params: PolicyParams) -> bool:
    """Tells whether an incident still needs seats, an ALS or a responder.

    Args:
        need: The incident need.
        params: Policy parameters.

    Returns:
        True if any need is unmet.
    """
    return (
        seats_short(need, params) > 0
        or need.als_needed > need.als_en_route
        or bool(need.responder_caps_missing)
    )


def unit_victims(view: PublicView, unit: UnitView) -> list[VictimView]:
    """Returns the victims on board a unit, in ``load_victim_ids`` order.

    Args:
        view: Current public view.
        unit: The unit.

    Returns:
        The visible victims whose id is in the unit's load.
    """
    by_id = {v.victim_id: v for i in view.incidents for v in i.victims}
    return [by_id[vid] for vid in unit.load_victim_ids if vid in by_id]


def capable_hospitals(view: PublicView, victims: Sequence[VictimView]) -> list[HospitalView]:
    """Returns the hospitals capable of every victim, sorted by id.

    Args:
        view: Current public view.
        victims: The victims to treat together.

    Returns:
        Hospitals whose capabilities cover all required capabilities.
    """
    needed = {v.required_capability for v in victims}
    return [h for h in view.hospitals if needed <= set(h.capabilities)]
