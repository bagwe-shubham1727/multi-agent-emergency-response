"""Escalation cases for Resource Allocation: options, utilities and their actions.

Each builder returns a ``CasePlan``: the LLM-facing ``EscalationCase`` (with
precomputed option utilities, ``policies.estimates.utility``) plus, for each
option, the engine actions and the resolution a COMMIT of that option means.
Builders are pure functions of the public view.

Option-id conventions (shared with ``policies.allocation``):

* UNIT_CONTENTION: ``"ASSIGN:<unit>-><incident>"``, ``"DIVERT:<unit>-><incident>"``, ``"KEEP"``.
* FLOAT_STAFF: ``"GRANT:<staff_id>"`` (``allocation.grant_option_id``) and ``"DECLINE"``.
* RESERVE_RELEASE: ``"APPROVE"`` and ``"DECLINE"``.
* NO_ACCEPTING_HOSPITAL: ``"HOSPITAL:<hospital_id>"``.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from emergency_sim.control import ProtocolError
from emergency_sim.control.needs import capable_hospitals
from emergency_sim.control.observations import find_hospital, staff_member
from emergency_sim.domain.actions import (
    Action,
    AssignTransport,
    DispatchUnit,
    DivertUnit,
    LendStaff,
)
from emergency_sim.domain.agent_io import (
    CaseOption,
    ContentionFlag,
    DiversionRequest,
    EscalationCase,
    IncidentNeed,
)
from emergency_sim.domain.enums import (
    TRIAGE_RANK,
    AmbulanceType,
    CaseKind,
    CounterCondition,
    Resolution,
    StaffRole,
    Triage,
    UnitClass,
    UnitStatus,
    VictimStatus,
)
from emergency_sim.domain.views import HospitalView, PublicView, UnitView, VictimView
from emergency_sim.policies.allocation import DECLINE_OPTION_ID, grant_option_id
from emergency_sim.policies.estimates import (
    NURSE_SKILL,
    doctor_skill_for,
    eligible_slots,
    eta_min,
    nominal_deadline,
    p_window,
    triage_weight,
    utility,
)
from emergency_sim.policies.params import PolicyParams

APPROVE_OPTION_ID: Final = "APPROVE"
KEEP_OPTION_ID: Final = "KEEP"
NO_FEASIBLE_ALTERNATIVE: Final = "no feasible alternative"


@dataclass(frozen=True)
class OptionPlan:
    """What choosing one option means.

    Attributes:
        option: The option shown to Resource Allocation.
        actions: Engine actions to apply, in order (may be empty).
        resolution: Resolution of the COMMIT (or decline) for this option.
    """

    option: CaseOption
    actions: tuple[Action, ...]
    resolution: Resolution

    @property
    def grants(self) -> bool:
        """Whether the option grants a counter (vs declining it)."""
        return self.resolution is not Resolution.COUNTER_DECLINED


@dataclass(frozen=True)
class CasePlan:
    """An escalation case and the meaning of each of its options.

    Attributes:
        case: The case as shown to Resource Allocation.
        plans: One plan per option, in option order.
        unit_id: The loaded unit concerned (transport cases), else None.
    """

    case: EscalationCase
    plans: tuple[OptionPlan, ...]
    unit_id: str | None = None

    def plan(self, option_id: str) -> OptionPlan:
        """Returns the plan of an option.

        Args:
            option_id: The option.

        Returns:
            Its plan.

        Raises:
            ProtocolError: If the option does not belong to the case.
        """
        for plan in self.plans:
            if plan.option.option_id == option_id:
                return plan
        raise ProtocolError(f"{self.case.case_id} has no option {option_id}")

    @property
    def needs_arbitration(self) -> bool:
        """Whether there is a real choice (two or more options)."""
        return len(self.plans) > 1


@dataclass(frozen=True)
class LoadUrgency:
    """Urgency summary of the patients on one unit.

    Attributes:
        weight: Triage weight of the most severe patient.
        deadline_in_min: Earliest nominal remaining care window.
    """

    weight: float
    deadline_in_min: float


def _option(
    option_id: str, summary: str, weight: float, slack: float, travel: float, value: float
) -> CaseOption:
    """Builds a case option with rounded numbers (stable logs).

    Args:
        option_id: Option id.
        summary: Human-readable summary.
        weight: Triage weight.
        slack: Window slack in minutes.
        travel: Travel minutes.
        value: Utility.

    Returns:
        The option.
    """
    return CaseOption(
        option_id=option_id, summary=summary, triage_weight=weight,
        slack_min=round(slack, 2), travel_min=round(travel, 2), utility=round(value, 6),
    )


# --- Patient and hospital estimates ---------------------------------------------


def load_urgency(view: PublicView, victims: Sequence[VictimView], params: PolicyParams) -> LoadUrgency:
    """Summarises the urgency of a set of victims.

    The remaining window of a victim is its nominal window minus the minutes
    since the report (a deteriorated victim gets the full deteriorated window,
    since the deterioration time is not public).

    Args:
        view: Current public view.
        victims: The victims (non-empty).
        params: Policy parameters.

    Returns:
        Weight of the most severe victim and the earliest remaining window.
    """
    reported = {i.incident_id: i.t_report_min for i in view.incidents}
    deadlines = []
    for victim in victims:
        window = nominal_deadline(
            victim.triage, victim.condition, params, deteriorated=victim.deteriorated
        )
        elapsed = 0.0 if victim.deteriorated else view.t_min - reported.get(victim.incident_id, view.t_min)
        deadlines.append(window - elapsed)
    worst = min((v.triage for v in victims), key=lambda t: TRIAGE_RANK[t])
    return LoadUrgency(triage_weight(worst, params), min(deadlines))


def expected_wait(hospital: HospitalView, victims: Sequence[VictimView], params: PolicyParams) -> float:
    """Estimates the wait before treatment at a hospital.

    Args:
        hospital: The hospital.
        victims: Patients to admit.
        params: Policy parameters.

    Returns:
        0 if beds and the needed staff are free now, else
        ``queue_wait_per_patient_min * (queue_length + 1)``.
    """
    staff = [staff_member(s) for s in hospital.staff]
    skills = {doctor_skill_for(v.required_capability) for v in victims}
    staff_free = all(eligible_slots(staff, StaffRole.DOCTOR, k) > 0 for k in skills) and (
        eligible_slots(staff, StaffRole.NURSE, NURSE_SKILL) >= len(victims)
    )
    if hospital.beds_free_unreserved >= len(victims) and staff_free:
        return 0.0
    return params.queue_wait_per_patient_min * (hospital.queue_length + 1)


def hospital_eta(unit: UnitView, hospital: HospitalView) -> float:
    """Driving time of a loaded unit to a hospital.

    Args:
        unit: The unit.
        hospital: The hospital.

    Returns:
        Minutes (no turnout).
    """
    return eta_min((unit.x, unit.y), (hospital.x, hospital.y), UnitClass.AMBULANCE, False)


def best_alternative(
    view: PublicView, unit: UnitView, victims: Sequence[VictimView], exclude: str, params: PolicyParams
) -> tuple[float, float]:
    """Expected ``(eta, wait)`` of the best capable hospital other than ``exclude``.

    Args:
        view: Current public view.
        unit: The loaded unit.
        victims: Its patients.
        exclude: Hospital being negotiated with.
        params: Policy parameters.

    Returns:
        ETA and wait of the alternative with the earliest expected care
        start; if there is none, queueing at ``exclude`` itself.
    """
    candidates = [
        (hospital_eta(unit, h), expected_wait(h, victims, params), h.hospital_id)
        for h in capable_hospitals(view, victims)
        if h.hospital_id != exclude
    ]
    if candidates:
        eta, wait, _ = min(candidates, key=lambda c: (c[0] + c[1], c[2]))
        return eta, wait
    here = find_hospital(view, exclude)
    return hospital_eta(unit, here), params.queue_wait_per_patient_min * (here.queue_length + 1)


def pending_red_demand(view: PublicView, hospital_id: str, exclude_unit: str) -> int:
    """Counts RED patients not yet placed for whom a hospital is the nearest capable one.

    Counted: RED victims waiting or loading on scene and RED victims on units
    still awaiting a destination (except ``exclude_unit``). Untriaged
    incidents are not counted (their severity is unknown in the view).

    Args:
        view: Current public view.
        hospital_id: The hospital holding a reserve.
        exclude_unit: The unit whose admission is being decided.

    Returns:
        Expected RED demand for the hospital's reserve.
    """
    awaiting = {
        vid
        for u in view.units
        if u.status is UnitStatus.AWAITING_DESTINATION and u.unit_id != exclude_unit
        for vid in u.load_victim_ids
    }
    demand = 0
    for incident in view.incidents:
        for victim in incident.victims:
            unplaced = victim.status in (VictimStatus.WAITING, VictimStatus.LOADING) or (
                victim.victim_id in awaiting
            )
            if victim.triage is Triage.RED and unplaced and (
                _nearest_capable(view, incident.x, incident.y, victim) == hospital_id
            ):
                demand += 1
    return demand


def _nearest_capable(view: PublicView, x: float, y: float, victim: VictimView) -> str | None:
    """Returns the nearest hospital capable of one victim.

    Args:
        view: Current public view.
        x: Origin x.
        y: Origin y.
        victim: The victim.

    Returns:
        Hospital id or None.
    """
    hospitals = capable_hospitals(view, [victim])
    if not hospitals:
        return None
    return min(
        hospitals,
        key=lambda h: (eta_min((x, y), (h.x, h.y), UnitClass.AMBULANCE, False), h.hospital_id),
    ).hospital_id


# --- Transport cases ------------------------------------------------------------


def no_accepting_case(
    case_id: str, view: PublicView, unit: UnitView, victims: Sequence[VictimView],
    incident_id: str, params: PolicyParams,
) -> CasePlan:
    """Builds a NO_ACCEPTING_HOSPITAL case: one option per capable hospital.

    Utility uses ``slack = deadline - (ETA + expected wait)``; the chosen
    hospital gets ``AssignTransport(reserve_beds = beds available)``.

    Args:
        case_id: Case id.
        view: Current public view.
        unit: The loaded unit nobody accepted.
        victims: Its patients.
        incident_id: Incident the patients come from.
        params: Policy parameters.

    Returns:
        The case plan (no options if no hospital is capable).
    """
    urgency = load_urgency(view, victims, params)
    plans = []
    for h in capable_hospitals(view, victims):
        eta, wait = hospital_eta(unit, h), expected_wait(h, victims, params)
        slack = urgency.deadline_in_min - (eta + wait)
        reserve = h.beds_free_unreserved >= len(victims)
        option = _option(
            f"HOSPITAL:{h.hospital_id}",
            f"{unit.unit_id} to {h.hospital_id}: ETA {eta} min, expected wait {wait:.0f} min",
            urgency.weight, slack, eta, utility(urgency.weight, slack, eta, params),
        )
        action = AssignTransport(unit_id=unit.unit_id, hospital_id=h.hospital_id, reserve_beds=reserve)
        plans.append(OptionPlan(option, (action,), Resolution.RA_COMMIT))
    case = EscalationCase(
        case_id=case_id, kind=CaseKind.NO_ACCEPTING_HOSPITAL, incident_id=incident_id,
        hospital_id=None, summary=f"no hospital accepted {unit.unit_id} ({len(victims)} patient(s))",
        options=[p.option for p in plans],
    )
    return CasePlan(case, tuple(plans), unit.unit_id)


@dataclass(frozen=True)
class CounterContext:
    """A hospital's COUNTER on one loaded unit.

    Attributes:
        case_id: Case id to create.
        unit: The loaded unit.
        victims: Its patients.
        incident_id: Incident the patients come from.
        hospital_id: Hospital that countered.
        condition: The counter condition.
        skill: Needed staff skill (FLOAT_* only).
    """

    case_id: str
    unit: UnitView
    victims: tuple[VictimView, ...]
    incident_id: str
    hospital_id: str
    condition: CounterCondition
    skill: str | None


def counter_case(view: PublicView, ctx: CounterContext, params: PolicyParams) -> CasePlan:
    """Builds the FLOAT_STAFF or RESERVE_RELEASE case of a COUNTER.

    Args:
        view: Current public view.
        ctx: The counter.
        params: Policy parameters.

    Returns:
        The case plan.
    """
    if ctx.condition is CounterCondition.RESERVE_RELEASE:
        return _reserve_release_case(view, ctx, params)
    return _float_staff_case(view, ctx, params)


def _transport(ctx: CounterContext) -> AssignTransport:
    """Returns the reserved transport a granted counter commits.

    Args:
        ctx: The counter.

    Returns:
        ``AssignTransport`` to the countering hospital with reservation.
    """
    return AssignTransport(unit_id=ctx.unit.unit_id, hospital_id=ctx.hospital_id, reserve_beds=True)


def _float_staff_case(view: PublicView, ctx: CounterContext, params: PolicyParams) -> CasePlan:
    """Builds a FLOAT_STAFF case: GRANT per eligible float staff member, or DECLINE.

    Args:
        view: Current public view.
        ctx: The counter.
        params: Policy parameters.

    Returns:
        The case plan (only DECLINE when no float staff qualifies).
    """
    role = StaffRole.DOCTOR if ctx.condition is CounterCondition.FLOAT_DOCTOR else StaffRole.NURSE
    skill = ctx.skill or NURSE_SKILL
    hospital = find_hospital(view, ctx.hospital_id)
    urgency = load_urgency(view, ctx.victims, params)
    eta_here = hospital_eta(ctx.unit, hospital)
    slack_here = urgency.deadline_in_min - eta_here
    plans = [
        OptionPlan(
            _option(
                grant_option_id(s.staff_id), f"lend {role.value} {s.staff_id} ({skill}) to {ctx.hospital_id}",
                urgency.weight, slack_here, eta_here, utility(urgency.weight, slack_here, eta_here, params),
            ),
            (LendStaff(staff_id=s.staff_id, hospital_id=ctx.hospital_id), _transport(ctx)),
            Resolution.COUNTER_GRANTED,
        )
        for s in sorted(view.float_pool, key=lambda s: s.staff_id)
        if s.hospital_id is None and s.on_shift and s.role is role and skill in s.skills
    ]
    eta_alt, wait_alt = best_alternative(view, ctx.unit, ctx.victims, ctx.hospital_id, params)
    slack_alt = urgency.deadline_in_min - (eta_alt + wait_alt)
    plans.append(OptionPlan(
        _option(DECLINE_OPTION_ID, f"keep float staff; {ctx.unit.unit_id} tries another hospital",
                urgency.weight, slack_alt, eta_alt, utility(urgency.weight, slack_alt, eta_alt, params)),
        (), Resolution.COUNTER_DECLINED,
    ))
    case = EscalationCase(
        case_id=ctx.case_id, kind=CaseKind.FLOAT_STAFF, incident_id=ctx.incident_id,
        hospital_id=ctx.hospital_id,
        summary=f"{ctx.hospital_id} accepts {ctx.unit.unit_id} if lent a float {role.value} ({skill})",
        options=[p.option for p in plans],
    )
    return CasePlan(case, tuple(plans), ctx.unit.unit_id)


def _reserve_release_case(view: PublicView, ctx: CounterContext, params: PolicyParams) -> CasePlan:
    """Builds a RESERVE_RELEASE case: APPROVE (use reserve) or DECLINE.

    Approve utility is ``w * p_window(slack_here)``; decline utility is
    ``w * p_window(slack_alternative) + red_weight * min(1, shortfall)``
    where ``shortfall = max(0, pending_red_demand - beds_free_after)``. So
    approval wins when the beds left after admission still cover the RED
    demand expected at this hospital (plan 5.3: approve iff
    ``beds_free_after - pending_red_demand >= 0``) unless the alternative is
    clearly better for the patients themselves.

    Args:
        view: Current public view.
        ctx: The counter.
        params: Policy parameters.

    Returns:
        The case plan.
    """
    hospital = find_hospital(view, ctx.hospital_id)
    urgency = load_urgency(view, ctx.victims, params)
    eta_here = hospital_eta(ctx.unit, hospital)
    slack_here = urgency.deadline_in_min - eta_here
    eta_alt, wait_alt = best_alternative(view, ctx.unit, ctx.victims, ctx.hospital_id, params)
    slack_alt = urgency.deadline_in_min - (eta_alt + wait_alt)
    beds_after = hospital.beds_free_unreserved - len(ctx.victims)
    demand = pending_red_demand(view, ctx.hospital_id, ctx.unit.unit_id)
    shortfall = max(0, demand - beds_after)
    approve = urgency.weight * p_window(slack_here, params)
    decline = urgency.weight * p_window(slack_alt, params) + params.red_weight * min(1, shortfall)
    plans = (
        OptionPlan(
            _option(APPROVE_OPTION_ID, f"release reserve at {ctx.hospital_id}; {beds_after} bed(s) left, "
                    f"RED demand {demand}", urgency.weight, slack_here, eta_here, approve),
            (_transport(ctx),), Resolution.COUNTER_GRANTED,
        ),
        OptionPlan(
            _option(DECLINE_OPTION_ID, f"hold reserve for RED; {ctx.unit.unit_id} tries another hospital",
                    urgency.weight, slack_alt, eta_alt, decline),
            (), Resolution.COUNTER_DECLINED,
        ),
    )
    case = EscalationCase(
        case_id=ctx.case_id, kind=CaseKind.RESERVE_RELEASE, incident_id=ctx.incident_id,
        hospital_id=ctx.hospital_id,
        summary=f"{ctx.hospital_id} accepts non-RED {ctx.unit.unit_id} only from its critical reserve",
        options=[p.option for p in plans],
    )
    return CasePlan(case, plans, ctx.unit.unit_id)


# --- Unit contention (dispatch) ---------------------------------------------------


def incident_weight(need: IncidentNeed, params: PolicyParams) -> float:
    """Triage weight of an incident's most urgent known or suspected patient.

    Args:
        need: The incident need.
        params: Policy parameters.

    Returns:
        RED weight if a RED waits or RED is suspected, else the UNKNOWN weight.
    """
    if need.red_waiting > 0:
        return triage_weight(Triage.RED, params)
    if need.suspected_red:
        return triage_weight("SUSPECTED_RED", params)
    return triage_weight("UNKNOWN", params)


def incident_utility(need: IncidentNeed, eta: float, params: PolicyParams) -> tuple[float, float, float]:
    """Expected utility of serving an incident with a unit arriving after ``eta``.

    Args:
        need: The incident need.
        eta: Minutes until the unit is on scene.
        params: Policy parameters.

    Returns:
        ``(weight, slack, utility)``. An ambulance cannot load before the
        scene is accessible, and it drives while the responders work, so
        loading starts after ``max(eta, access wait)``; slack is the deadline
        minus that and the scene overhead (none for a suspected cardiac
        arrest, whose care starts on ALS arrival). The access wait is the
        need's estimate, or ``no_unit_wait_min`` when it is unknown.
    """
    weight = incident_weight(need, params)
    overhead = 0.0 if need.suspected_cardiac_arrest else params.scene_overhead_min
    access = params.no_unit_wait_min if need.access_wait_min is None else need.access_wait_min
    slack = need.deadline_in_min - max(eta, access) - overhead
    return weight, slack, utility(weight, slack, eta, params)


def _scene_eta(unit: UnitView, need: IncidentNeed, *, turnout: bool) -> float:
    """ETA of a unit to an incident.

    Args:
        unit: The unit.
        need: The incident.
        turnout: Include the turnout time (free units) or not (diversions).

    Returns:
        Minutes.
    """
    return eta_min((unit.x, unit.y), (need.x, need.y), UnitClass.AMBULANCE, turnout)


def _free_als(view: PublicView, excluded: set[str]) -> list[UnitView]:
    """Returns free ALS ambulances not excluded, sorted by id.

    Args:
        view: Current public view.
        excluded: Unit ids not available as alternatives.

    Returns:
        The units.
    """
    return sorted(
        (
            u for u in view.units
            if u.unit_class is UnitClass.AMBULANCE and u.unit_type == AmbulanceType.ALS
            and u.status is UnitStatus.AVAILABLE and u.unit_id not in excluded
        ),
        key=lambda u: u.unit_id,
    )


def contention_case(
    case_id: str, view: PublicView, flag: ContentionFlag,
    needs: Mapping[str, IncidentNeed], excluded: set[str], params: PolicyParams,
) -> CasePlan:
    """Builds a UNIT_CONTENTION case from a contention flag.

    Each option gives the contested ALS to one incident and the next nearest
    free ALS (if any) to each other incident; utility is the sum over the
    affected incidents (an incident left without a unit waits
    ``no_unit_wait_min``).

    Args:
        case_id: Case id.
        view: Current public view (after the dispatch orders were applied).
        flag: The flag.
        needs: Needs by incident id.
        excluded: Units that may not serve as alternatives (contested units).
        params: Policy parameters.

    Returns:
        The case plan.

    Raises:
        ProtocolError: If the flagged unit or an incident is unknown.
    """
    unit = _unit(view, flag.unit_id)
    incidents = sorted(set(flag.incident_ids))
    if unknown := [i for i in incidents if i not in needs]:
        raise ProtocolError(f"contention on unknown incidents {unknown}")
    pool = _free_als(view, excluded | {unit.unit_id})
    plans = tuple(_contention_option(unit, target, incidents, needs, pool, params) for target in incidents)
    case = EscalationCase(
        case_id=case_id, kind=CaseKind.UNIT_CONTENTION, incident_id=incidents[0], hospital_id=None,
        summary=f"{unit.unit_id} is the nearest free ALS for {', '.join(incidents)}",
        options=[p.option for p in plans],
    )
    return CasePlan(case, plans)


def _contention_option(
    unit: UnitView, target: str, incidents: Sequence[str],
    needs: Mapping[str, IncidentNeed], pool: Sequence[UnitView], params: PolicyParams,
) -> OptionPlan:
    """Builds the option "contested unit to ``target``, alternatives to the others".

    Args:
        unit: The contested ALS.
        target: Incident that gets it.
        incidents: All contending incidents.
        needs: Needs by incident id.
        pool: Free ALS usable as alternatives.
        params: Policy parameters.

    Returns:
        The option plan.
    """
    eta = _scene_eta(unit, needs[target], turnout=True)
    weight, slack, total = incident_utility(needs[target], eta, params)
    actions: list[Action] = [DispatchUnit(unit_id=unit.unit_id, incident_id=target)]
    notes = [f"{unit.unit_id} -> {target}"]
    remaining = list(pool)
    for other in (i for i in incidents if i != target):
        alt = min(remaining, key=lambda u: (_scene_eta(u, needs[other], turnout=True), u.unit_id), default=None)
        alt_eta = params.no_unit_wait_min if alt is None else _scene_eta(alt, needs[other], turnout=True)
        total += incident_utility(needs[other], alt_eta, params)[2]
        if alt is None:
            notes.append(f"{other} keeps none")
            continue
        remaining.remove(alt)
        actions.append(DispatchUnit(unit_id=alt.unit_id, incident_id=other))
        notes.append(f"{other} gets {alt.unit_id}")
    option = _option(f"ASSIGN:{unit.unit_id}->{target}", "; ".join(notes), weight, slack, eta, total)
    return OptionPlan(option, tuple(actions), Resolution.RA_COMMIT)


def diversion_case(
    case_id: str, view: PublicView, request: DiversionRequest,
    needs: Mapping[str, IncidentNeed], params: PolicyParams,
) -> CasePlan:
    """Builds a UNIT_CONTENTION case from a diversion request: DIVERT or KEEP.

    Args:
        case_id: Case id.
        view: Current public view.
        request: The diversion request.
        needs: Needs by incident id.
        params: Policy parameters.

    Returns:
        The case plan.

    Raises:
        ProtocolError: If the unit or target incident is unknown.
    """
    unit = _unit(view, request.unit_id)
    target = needs.get(request.to_incident_id)
    if target is None:
        raise ProtocolError(f"diversion to unknown incident {request.to_incident_id}")
    current = needs.get(unit.incident_id or "")
    eta_new = _scene_eta(unit, target, turnout=False)
    weight, slack, gain = incident_utility(target, eta_new, params)
    _, slack_keep, target_unserved = incident_utility(target, params.no_unit_wait_min, params)
    loss = keep = 0.0
    if current is not None:
        loss = incident_utility(current, params.no_unit_wait_min, params)[2]
        keep = incident_utility(current, _scene_eta(unit, current, turnout=False), params)[2]
    plans = (
        OptionPlan(
            _option(f"DIVERT:{unit.unit_id}->{target.incident_id}",
                    f"divert {unit.unit_id} from {unit.incident_id} to {target.incident_id}, ETA {eta_new} min",
                    weight, slack, eta_new, gain + loss),
            (DivertUnit(unit_id=unit.unit_id, incident_id=target.incident_id),),
            Resolution.RA_COMMIT,
        ),
        OptionPlan(
            _option(KEEP_OPTION_ID, f"{unit.unit_id} keeps {unit.incident_id}; {target.incident_id} waits",
                    weight, slack_keep, params.no_unit_wait_min, target_unserved + keep),
            (), Resolution.KEPT_ASSIGNMENT,
        ),
    )
    case = EscalationCase(
        case_id=case_id, kind=CaseKind.UNIT_CONTENTION, incident_id=target.incident_id, hospital_id=None,
        summary=f"divert en-route ALS {unit.unit_id} from {unit.incident_id} to {target.incident_id}?",
        options=[p.option for p in plans],
    )
    return CasePlan(case, plans)


def _unit(view: PublicView, unit_id: str) -> UnitView:
    """Returns a unit of the view.

    Args:
        view: Current public view.
        unit_id: Unit id.

    Returns:
        The unit.

    Raises:
        ProtocolError: If unknown.
    """
    for unit in view.units:
        if unit.unit_id == unit_id:
            return unit
    raise ProtocolError(f"unknown unit {unit_id}")
