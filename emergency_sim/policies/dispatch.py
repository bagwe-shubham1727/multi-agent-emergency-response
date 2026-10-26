"""Dispatch policy: send help now and minimise time to care (plan 5.3).

Passes, in order: transport rankings for loaded units, ALS to suspected/known
RED (with contention flags and diversions when the condition allows them),
seats for the remaining patients, and responder teams for scene access. The
ALS and seat passes run first for scenes where an ambulance can load on
arrival, then for scenes still waiting for access (``split_by_readiness``);
within each group, and for responders, incidents go by urgency. B1 calls the same function with both
flags false, which gives greedy dispatch without arbitration.
"""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Final

from emergency_sim.domain.agent_io import (
    ContentionFlag,
    DispatchInput,
    DispatchOutput,
    DiversionRequest,
    IncidentNeed,
    TransportPlan,
    UnitOrder,
    UnitStatusView,
)
from emergency_sim.domain.enums import AmbulanceType, UnitClass, UnitStatus
from emergency_sim.policies.estimates import (
    UnitRank,
    eta_min,
    rank_hospitals,
    rank_responders,
    rank_units,
)
from emergency_sim.policies.params import PolicyParams

MAX_RATIONALE_CHARS: Final = 200


@dataclass
class _Round:
    """Working state of one dispatch call (local to the call, never shared)."""

    free_ids: set[str]
    units: dict[str, UnitStatusView]
    orders: list[UnitOrder] = field(default_factory=list)
    diversions: list[DiversionRequest] = field(default_factory=list)
    contentions: list[ContentionFlag] = field(default_factory=list)
    contested: set[str] = field(default_factory=set)
    diverted: set[str] = field(default_factory=set)
    seats_added: defaultdict[str, int] = field(default_factory=lambda: defaultdict(int))
    als_added: defaultdict[str, int] = field(default_factory=lambda: defaultdict(int))
    ambulances_added: defaultdict[str, int] = field(default_factory=lambda: defaultdict(int))

    def order(self, unit_id: str, incident_id: str, why: str) -> None:
        """Records an order and books the unit's seats for the incident."""
        unit = self.units[unit_id]
        self.free_ids.discard(unit_id)
        self.orders.append(
            UnitOrder(unit_id=unit_id, incident_id=incident_id, rationale=why[:MAX_RATIONALE_CHARS])
        )
        if unit.unit_class is UnitClass.AMBULANCE:
            self.seats_added[incident_id] += unit.seats
            self.ambulances_added[incident_id] += 1
            if unit.unit_type == AmbulanceType.ALS.value:
                self.als_added[incident_id] += 1

    def free(self, ranks: Iterable[UnitRank]) -> list[UnitRank]:
        """Keeps the ranked units that are still free in this round."""
        return [r for r in ranks if r.unit_id in self.free_ids]


def priority_order(incidents: Iterable[IncidentNeed]) -> list[IncidentNeed]:
    """Sorts incidents by ``(-urgency, deadline_in_min, incident_id)``.

    Args:
        incidents: Incident needs.

    Returns:
        Most urgent first.
    """
    return sorted(incidents, key=lambda n: (-n.urgency, n.deadline_in_min, n.incident_id))


def scene_ready(inp: DispatchInput, need: IncidentNeed) -> bool:
    """Tells whether the nearest free ambulance could start loading on arrival.

    An ambulance that reaches a scene before access is granted waits idle
    there, so its patients get care no earlier than with one sent later,
    while patients at a scene that is ready wait for it.

    Args:
        inp: The dispatch payload.
        need: The incident need.

    Returns:
        True if the scene is accessible, or the estimated access wait is no
        longer than the nearest free ambulance's ETA; False if access is
        unknown (a required responder is not even en route) or later.
    """
    if need.access_wait_min is None:
        return False
    if need.access_wait_min <= 0:
        return True
    nearest = rank_units(inp, need.incident_id, None)
    return bool(nearest) and nearest[0].eta_min >= need.access_wait_min


def split_by_readiness(
    inp: DispatchInput, needs: Iterable[IncidentNeed]
) -> tuple[list[IncidentNeed], list[IncidentNeed]]:
    """Splits incidents into scenes ready for an ambulance and scenes still blocked.

    Args:
        inp: The dispatch payload.
        needs: Incident needs in ``priority_order``.

    Returns:
        ``(ready, blocked)`` by ``scene_ready``, each keeping priority order.
    """
    ready: list[IncidentNeed] = []
    blocked: list[IncidentNeed] = []
    for need in needs:
        (ready if scene_ready(inp, need) else blocked).append(need)
    return ready, blocked


def transport_plans(inp: DispatchInput) -> list[TransportPlan]:
    """Ranks capable hospitals for every loaded unit (free bed first, then ETA, id).

    Args:
        inp: The dispatch payload.

    Returns:
        One plan per loaded unit, sorted by unit id (empty ranking if no
        hospital is capable).
    """
    plans = []
    for unit in sorted(inp.loaded_units, key=lambda u: u.unit_id):
        ranks = rank_hospitals(inp.hospitals, unit)
        if ranks:
            why = f"capable, free bed first then ETA; best {ranks[0].hospital_id} at {ranks[0].eta_min} min"
        else:
            why = "no capable hospital on the board"
        plans.append(
            TransportPlan(
                unit_id=unit.unit_id,
                hospital_ranking=[r.hospital_id for r in ranks],
                rationale=why[:MAX_RATIONALE_CHARS],
            )
        )
    return plans


def _pre_triage_capped(need: IncidentNeed, rnd: _Round, params: PolicyParams) -> bool:
    """Tells whether an untriaged incident already has the shared ambulance cap."""
    sent = need.ambulances_assigned + rnd.ambulances_added[need.incident_id]
    return not need.triaged and sent >= params.max_ambulances_pre_triage


def _als_gap(need: IncidentNeed, rnd: _Round, params: PolicyParams) -> int:
    """Missing ALS units of an incident; none once the pre-triage cap is reached."""
    if _pre_triage_capped(need, rnd, params):
        return 0
    return max(0, need.als_needed - need.als_en_route - rnd.als_added[need.incident_id])


def _nearest_free_als(inp: DispatchInput, need: IncidentNeed, rnd: _Round) -> UnitRank | None:
    ranks = rnd.free(rank_units(inp, need.incident_id, AmbulanceType.ALS))
    return ranks[0] if ranks else None


def _flag_contentions(
    inp: DispatchInput, needs: list[IncidentNeed], rnd: _Round, params: PolicyParams
) -> None:
    """Withholds every free ALS that is the nearest one for two or more incidents."""
    claims: defaultdict[str, list[str]] = defaultdict(list)
    for need in needs:
        if _als_gap(need, rnd, params) > 0 and (nearest := _nearest_free_als(inp, need, rnd)) is not None:
            claims[nearest.unit_id].append(need.incident_id)
    for unit_id in sorted(claims):
        incident_ids = sorted(claims[unit_id])
        if len(incident_ids) < 2:
            continue
        rnd.free_ids.discard(unit_id)
        rnd.contested.update(incident_ids)
        why = f"nearest free ALS for {len(incident_ids)} incidents needing ALS; arbitration needed"
        rnd.contentions.append(
            ContentionFlag(unit_id=unit_id, incident_ids=incident_ids, rationale=why)
        )


def _diversion_candidate(
    inp: DispatchInput, need: IncidentNeed, rnd: _Round, params: PolicyParams
) -> tuple[float, str] | None:
    """Finds the nearest en-route ALS serving a clearly lower-acuity incident."""
    by_id = {n.incident_id: n for n in inp.incidents}
    best: tuple[float, str] | None = None
    for unit in inp.units:
        target = by_id.get(unit.incident_id or "")
        if (
            unit.unit_class is not UnitClass.AMBULANCE or unit.unit_type != AmbulanceType.ALS.value
            or unit.status is not UnitStatus.EN_ROUTE_SCENE or unit.unit_id in rnd.diverted
            or target is None or target.incident_id == need.incident_id
            or target.urgency > need.urgency - params.diversion_urgency_margin
            or target.suspected_red or target.red_waiting > 0
        ):
            continue
        eta = eta_min((unit.x, unit.y), (need.x, need.y), UnitClass.AMBULANCE, False)
        if eta <= params.diversion_max_eta_min and (best is None or (eta, unit.unit_id) < best):
            best = (eta, unit.unit_id)
    return best


def _divert(inp: DispatchInput, need: IncidentNeed, rnd: _Round, params: PolicyParams) -> None:
    candidate = _diversion_candidate(inp, need, rnd, params)
    if candidate is None:
        return
    eta, unit_id = candidate
    unit = rnd.units[unit_id]
    rnd.diverted.add(unit_id)
    rnd.seats_added[need.incident_id] += unit.seats
    rnd.als_added[need.incident_id] += 1
    why = f"no free ALS; divert from lower-acuity {unit.incident_id}, ETA {eta} min"
    rnd.diversions.append(
        DiversionRequest(unit_id=unit_id, to_incident_id=need.incident_id, rationale=why)
    )


def _als_pass(inp: DispatchInput, needs: list[IncidentNeed], rnd: _Round, params: PolicyParams) -> None:
    """Sends ALS to incidents with an ALS need, flagging contention or diverting if allowed."""
    if inp.allow_escalation:
        _flag_contentions(inp, needs, rnd, params)
    for need in needs:
        if need.incident_id in rnd.contested:
            continue
        while _als_gap(need, rnd, params) > 0 and (nearest := _nearest_free_als(inp, need, rnd)) is not None:
            why = f"ALS for suspected/known RED, urgency {need.urgency}, ETA {nearest.eta_min} min"
            rnd.order(nearest.unit_id, need.incident_id, why)
        if _als_gap(need, rnd, params) > 0 and inp.allow_diversion:
            _divert(inp, need, rnd, params)


def _seat_pool(inp: DispatchInput, need: IncidentNeed, rnd: _Round, als_allowed: bool) -> list[UnitRank]:
    """Free ambulances usable for seats: nearest BLS first, then ALS if allowed."""
    pool: list[UnitRank] = []
    bls_useful = not need.triaged or need.est_patients_remaining > need.red_waiting
    if bls_useful:
        pool += rnd.free(rank_units(inp, need.incident_id, AmbulanceType.BLS))
    if als_allowed:
        pool += rnd.free(rank_units(inp, need.incident_id, AmbulanceType.ALS))
    return pool


def _seat_pass(
    inp: DispatchInput, needs: list[IncidentNeed], als_scope: list[IncidentNeed],
    rnd: _Round, params: PolicyParams,
) -> None:
    """Fills the remaining seat shortfall of each incident in priority order.

    An ALS fills seats only if no incident of ``als_scope`` (the incidents
    served so far this call) still lacks an ALS.
    """
    als_allowed = not any(n.incident_id in rnd.contested or _als_gap(n, rnd, params) > 0 for n in als_scope)
    for need in needs:
        iid = need.incident_id
        for rank in _seat_pool(inp, need, rnd, als_allowed):
            short = need.est_patients_remaining - need.seats_en_route - rnd.seats_added[iid]
            if short <= 0 or _pre_triage_capped(need, rnd, params):
                break
            why = f"{rank.unit_type} for {short} unserved seat(s), ETA {rank.eta_min} min"
            rnd.order(rank.unit_id, iid, why)


def _team_plan(
    inp: DispatchInput, need: IncidentNeed, rnd: _Round
) -> tuple[list[tuple[str, UnitRank]], bool]:
    """Picks the nearest free team for each missing responder capability.

    Args:
        inp: The dispatch payload.
        need: The incident need.
        rnd: Working state of the call (free units).

    Returns:
        ``(capability, team)`` pairs (one team may cover several
        capabilities) and whether they cover every missing capability, i.e.
        whether sending them completes the scene's access.
    """
    covered: set[str] = set()
    plan: list[tuple[str, UnitRank]] = []
    complete = True
    for cap in sorted(need.responder_caps_missing):
        if cap.value in covered:
            continue
        taken = {team.unit_id for _, team in plan}
        ranks = [r for r in rnd.free(rank_responders(inp, need.incident_id, cap)) if r.unit_id not in taken]
        if not ranks:
            complete = False
            continue
        plan.append((cap.value, ranks[0]))
        covered.update(rnd.units[ranks[0].unit_id].capabilities)
    return plan, complete


def _send_teams(need: IncidentNeed, plan: list[tuple[str, UnitRank]], rnd: _Round) -> None:
    for cap, team in plan:
        why = f"{cap} needed for scene access, ETA {team.eta_min} min"
        rnd.order(team.unit_id, need.incident_id, why)


def _responder_pass(inp: DispatchInput, needs: list[IncidentNeed], rnd: _Round) -> None:
    """Sends responder teams, first where they complete a scene's access.

    A team on a scene that still lacks another required capability cannot
    open it, so incidents whose every missing capability can be covered by
    free teams now are served first (in priority order); the free teams
    left then go, nearest per missing capability, to the other incidents.
    """
    served: set[str] = set()
    for need in needs:
        plan, complete = _team_plan(inp, need, rnd)
        if plan and complete:
            _send_teams(need, plan, rnd)
            served.add(need.incident_id)
    for need in needs:
        if need.incident_id not in served:
            _send_teams(need, _team_plan(inp, need, rnd)[0], rnd)


def dispatch(inp: DispatchInput, params: PolicyParams) -> DispatchOutput:
    """Dispatch policy (plan 5.3).

    Args:
        inp: The dispatch payload (contention flags only if
            ``allow_escalation``, diversions only if ``allow_diversion``).
        params: Policy parameters.

    Returns:
        Orders, diversion requests, contention flags and transport plans.
    """
    units = {u.unit_id: u for u in inp.units}
    free_ids = {u.unit_id for u in inp.units if u.status is UnitStatus.AVAILABLE}
    rnd = _Round(free_ids=free_ids, units=units)
    needs = priority_order(inp.incidents)
    served: list[IncidentNeed] = []
    for group in split_by_readiness(inp, needs):
        served += group
        _als_pass(inp, group, rnd, params)
        _seat_pass(inp, group, served, rnd, params)
    _responder_pass(inp, needs, rnd)
    return DispatchOutput(
        orders=rnd.orders,
        diversions=rnd.diversions,
        contentions=rnd.contentions,
        transports=transport_plans(inp),
    )
