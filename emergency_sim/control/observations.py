"""Builds each agent's input payload from the public view (the information boundary).

Every payload is derived from ``PublicView`` only (plus the coordinator's own
assessments), so the offline and LLM backends see exactly the same data.
Staff loads are included only in a hospital's own payload and in the float
pool given to Resource Allocation.
"""

from collections.abc import Mapping, Sequence

from emergency_sim.control import ProtocolError
from emergency_sim.control.conditions import NegotiationConfig
from emergency_sim.control.needs import unit_victims
from emergency_sim.domain.agent_io import (
    AdmissionRequest,
    AllocationInput,
    AssessmentInput,
    DispatchInput,
    EscalationCase,
    HospitalBoardEntry,
    HospitalInput,
    IncidentNeed,
    IncidentReport,
    LoadedUnit,
    StaffMemberView,
    TriagedVictim,
    UnitStatusView,
)
from emergency_sim.domain.enums import UnitStatus
from emergency_sim.domain.views import (
    HospitalView,
    IncidentView,
    PublicView,
    StaffView,
    UnitView,
    VictimView,
)


class DeteriorationLog:
    """Remembers when the coordinator first saw each victim as deteriorated.

    The view only flags a deteriorated victim; its shorter RED window counts
    from the deterioration, so the coordinator keeps the time it first saw
    the flag (the DETERIORATION trigger, or the triage that revealed it).
    """

    def __init__(self) -> None:
        """Creates an empty log."""
        self._first_seen: dict[str, float] = {}

    def observe(self, view: PublicView) -> dict[str, float]:
        """Records newly deteriorated victims and returns every victim's age.

        Args:
            view: Current public view.

        Returns:
            Minutes since each deteriorated victim was first seen, by id.
        """
        for incident in view.incidents:
            for victim in incident.victims:
                if victim.deteriorated:
                    self._first_seen.setdefault(victim.victim_id, view.t_min)
        return {vid: round(view.t_min - t, 1) for vid, t in self._first_seen.items()}


def triaged_victim(
    victim: VictimView, deterioration_age: Mapping[str, float] | None = None
) -> TriagedVictim:
    """Converts a victim view to the agent-facing DTO.

    Args:
        victim: The victim.
        deterioration_age: Minutes since each deteriorated victim was first
            seen (``DeteriorationLog.observe``); a deteriorated victim missing
            from it counts as just seen.

    Returns:
        The DTO.
    """
    since = None
    if victim.deteriorated:
        since = (deterioration_age or {}).get(victim.victim_id, 0.0)
    return TriagedVictim(
        victim_id=victim.victim_id,
        triage=victim.triage,
        condition=victim.condition,
        required_capability=victim.required_capability,
        status=victim.status,
        minutes_since_deterioration=since,
    )


def incident_report(
    incident: IncidentView, t_min: float, deterioration_age: Mapping[str, float] | None = None
) -> IncidentReport:
    """Builds the assessment DTO of one incident.

    Args:
        incident: The incident.
        t_min: Current time.
        deterioration_age: See ``triaged_victim``.

    Returns:
        The report (triaged victims empty before triage).
    """
    return IncidentReport(
        incident_id=incident.incident_id,
        incident_type=incident.incident_type,
        t_report_min=incident.t_report_min,
        minutes_since_report=round(t_min - incident.t_report_min, 1),
        x=incident.x,
        y=incident.y,
        intersection=incident.intersection,
        reported_victims=incident.reported_victims,
        caller_report=incident.caller_report,
        required_responder_capabilities=list(incident.required_responder_capabilities),
        triaged_victims=[triaged_victim(v, deterioration_age) for v in incident.victims],
    )


def assessment_input(
    view: PublicView,
    incident_ids: Sequence[str],
    deterioration_age: Mapping[str, float] | None = None,
) -> AssessmentInput:
    """Builds the Incident Assessment payload.

    Args:
        view: Current public view.
        incident_ids: Incidents to (re)assess.
        deterioration_age: See ``triaged_victim``.

    Returns:
        The payload, incidents sorted by id.

    Raises:
        ProtocolError: If an id is not a reported incident.
    """
    by_id = {i.incident_id: i for i in view.incidents}
    missing = sorted(set(incident_ids) - by_id.keys())
    if missing:
        raise ProtocolError(f"cannot assess unknown incidents {missing}")
    return AssessmentInput(
        t_min=view.t_min,
        incidents=[
            incident_report(by_id[i], view.t_min, deterioration_age) for i in sorted(set(incident_ids))
        ],
    )


def unit_status_view(unit: UnitView) -> UnitStatusView:
    """Converts a unit view to the dispatch DTO.

    Args:
        unit: The unit.

    Returns:
        The DTO.
    """
    return UnitStatusView(
        unit_id=unit.unit_id,
        unit_class=unit.unit_class,
        unit_type=unit.unit_type,
        capabilities=list(unit.capabilities),
        status=unit.status,
        x=unit.x,
        y=unit.y,
        incident_id=unit.incident_id,
        seats=unit.seats,
    )


def board_entry(hospital: HospitalView) -> HospitalBoardEntry:
    """Converts a hospital view to its public bed-board entry.

    Args:
        hospital: The hospital.

    Returns:
        The entry (no staff information).
    """
    return HospitalBoardEntry(
        hospital_id=hospital.hospital_id,
        x=hospital.x,
        y=hospital.y,
        capabilities=list(hospital.capabilities),
        beds_free_unreserved=hospital.beds_free_unreserved,
        queue_length=hospital.queue_length,
    )


def loaded_unit(
    view: PublicView, unit: UnitView, deterioration_age: Mapping[str, float] | None = None
) -> LoadedUnit:
    """Builds the DTO of an ambulance awaiting a destination.

    Args:
        view: Current public view.
        unit: The loaded unit.
        deterioration_age: See ``triaged_victim``.

    Returns:
        The DTO with its victims.
    """
    victims = unit_victims(view, unit)
    incident_id = unit.incident_id or (victims[0].incident_id if victims else "")
    return LoadedUnit(
        unit_id=unit.unit_id,
        incident_id=incident_id,
        x=unit.x,
        y=unit.y,
        victims=[triaged_victim(v, deterioration_age) for v in victims],
    )


def awaiting_units(view: PublicView) -> list[UnitView]:
    """Returns the units awaiting a destination, sorted by id.

    Args:
        view: Current public view.

    Returns:
        The units.
    """
    return sorted(
        (u for u in view.units if u.status is UnitStatus.AWAITING_DESTINATION),
        key=lambda u: u.unit_id,
    )


def dispatch_input(
    view: PublicView,
    needs: Sequence[IncidentNeed],
    config: NegotiationConfig,
    deterioration_age: Mapping[str, float] | None = None,
) -> DispatchInput:
    """Builds the Dispatch payload.

    Args:
        view: Current public view.
        needs: Needs of the open incidents (``needs.incident_needs``).
        config: Negotiation config (diversion / escalation flags).
        deterioration_age: See ``triaged_victim``.

    Returns:
        The payload.
    """
    return DispatchInput(
        t_min=view.t_min,
        incidents=list(needs),
        units=[unit_status_view(u) for u in sorted(view.units, key=lambda u: u.unit_id)],
        hospitals=[board_entry(h) for h in sorted(view.hospitals, key=lambda h: h.hospital_id)],
        loaded_units=[loaded_unit(view, u, deterioration_age) for u in awaiting_units(view)],
        allow_diversion=config.allow_diversion,
        allow_escalation=config.allow_escalation,
    )


def staff_member(staff: StaffView) -> StaffMemberView:
    """Converts a staff view to the agent DTO.

    Args:
        staff: The staff member.

    Returns:
        The DTO; ``lent`` is True for float staff working at a hospital.
    """
    return StaffMemberView(
        staff_id=staff.staff_id,
        role=staff.role,
        skills=list(staff.skills),
        load=staff.load,
        max_patients=staff.max_patients,
        on_shift=staff.on_shift,
        lent=staff.hospital_id is not None and staff.home != staff.hospital_id,
    )


def find_hospital(view: PublicView, hospital_id: str) -> HospitalView:
    """Returns a hospital of the view.

    Args:
        view: Current public view.
        hospital_id: Hospital id.

    Returns:
        The hospital.

    Raises:
        ProtocolError: If the hospital does not exist.
    """
    for hospital in view.hospitals:
        if hospital.hospital_id == hospital_id:
            return hospital
    raise ProtocolError(f"unknown hospital {hospital_id}")


def hospital_input(
    view: PublicView,
    hospital_id: str,
    requests: Sequence[AdmissionRequest],
    config: NegotiationConfig,
) -> HospitalInput:
    """Builds one Hospital Coordination payload (private state + requests).

    Args:
        view: Current public view.
        hospital_id: The hospital.
        requests: Admission requests to answer.
        config: Negotiation config (allowed responses and counter conditions).

    Returns:
        The payload.
    """
    h = find_hospital(view, hospital_id)
    return HospitalInput(
        t_min=view.t_min,
        hospital_id=h.hospital_id,
        capabilities=list(h.capabilities),
        er_capacity=h.er_capacity,
        beds_occupied=h.beds_occupied,
        beds_reserved=h.beds_reserved,
        beds_free_unreserved=h.beds_free_unreserved,
        critical_reserve=h.critical_reserve,
        queue_length=h.queue_length,
        waiting_for_staff=h.waiting_for_staff,
        staff=[staff_member(s) for s in sorted(h.staff, key=lambda s: s.staff_id)],
        requests=sorted(requests, key=lambda r: r.proposal_id),
        allowed_responses=config.allowed_responses(),
        allowed_counter_conditions=config.allowed_counter_conditions(),
    )


def free_float_pool(view: PublicView) -> list[StaffMemberView]:
    """Returns the float staff not yet lent, as agent DTOs.

    Args:
        view: Current public view.

    Returns:
        Unlent float staff sorted by id.
    """
    return [
        staff_member(s)
        for s in sorted(view.float_pool, key=lambda s: s.staff_id)
        if s.hospital_id is None
    ]


def allocation_input(view: PublicView, cases: Sequence[EscalationCase]) -> AllocationInput:
    """Builds the Resource Allocation payload.

    Args:
        view: Current public view.
        cases: Open cases (each with at least two options).

    Returns:
        The payload, cases sorted by id.
    """
    return AllocationInput(
        t_min=view.t_min,
        cases=sorted(cases, key=lambda c: c.case_id),
        float_pool=free_float_pool(view),
    )
