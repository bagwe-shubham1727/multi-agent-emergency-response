"""B0 and B0-cues: competent greedy dispatchers without communication.

Plan section 5.5. B0 sees the same ``PublicView`` (including the public bed
board) and shares the ambulance cap and crew protocol with the agent
conditions, but it does not communicate: no messages, no agent calls, no
diversion, no bed reservation, no staff lending, and it ignores severity cues
in caller reports.

Competence refinements over the literal plan text (documented, still
communication-free):

* After triage, when only RED victims remain unserved, seat filling uses ALS
  units only (a BLS cannot carry RED, so sending one would be a wasted trip).
* "Free beds" for a destination subtracts beds already claimed by B0 for
  units sent there and still transporting (B0 never reserves, so every
  inbound load will need an unreserved bed). This uses only the public board.

B0-cues (``use_caller_cues=True``) is the same controller, but it reads the
caller reports with the Incident Assessment policy (``policies.assessment``):
incidents are served in assessed priority order instead of first come, first
served, and an ALS goes first to a scene with a suspected RED. It still sends
no messages and has no hospital agents, reservations, staff information or
negotiation. It separates the value of the caller-report information
(B0 -> B0-cues) from the value of coordination (B0-cues -> B1) and of
negotiation (B1 -> Full).
"""

import logging
import math
from collections.abc import Callable, Iterable
from typing import Final

from emergency_sim import geo
from emergency_sim.control.needs import (
    capable_hospitals,
    incident_needs,
    unit_victims,
)
from emergency_sim.control.observations import DeteriorationLog, assessment_input
from emergency_sim.domain.public_rules import effective_capabilities
from emergency_sim.domain.actions import Action, AssignTransport, DispatchUnit
from emergency_sim.domain.agent_io import IncidentNeed, PriorityReport
from emergency_sim.domain.enums import (
    AmbulanceType,
    IncidentStatus,
    Triage,
    UnitClass,
    UnitStatus,
    VictimStatus,
)
from emergency_sim.domain.ports import EnginePort
from emergency_sim.domain.views import (
    DecisionPoint,
    HospitalView,
    IncidentView,
    PublicView,
    UnitView,
)
from emergency_sim.policies.assessment import assess
from emergency_sim.policies.dispatch import priority_order
from emergency_sim.policies.params import PolicyParams

LOGGER = logging.getLogger(__name__)

B0_ACTOR: Final = "b0_greedy"
"""Actor name of every action B0 applies."""

B0_CUES_ACTOR: Final = "b0_cues_greedy"
"""Actor name of every action B0-cues applies."""


def travel_min(unit: UnitView, x: float, y: float) -> float:
    """Returns the driving time of a unit to a point (no turnout).

    Args:
        unit: The unit.
        x: Target x in km.
        y: Target y in km.

    Returns:
        Minutes, rounded to 0.1 as ``geo.travel_time``.
    """
    vehicle = "ambulance" if unit.unit_class is UnitClass.AMBULANCE else "responder"
    return geo.travel_time((unit.x, unit.y), (x, y), vehicle)


def nearest_unit(
    free: dict[str, UnitView], x: float, y: float, accept: Callable[[UnitView], bool]
) -> UnitView | None:
    """Returns the nearest free unit satisfying a predicate.

    Args:
        free: Free units by id.
        x: Target x in km.
        y: Target y in km.
        accept: Predicate a candidate must satisfy.

    Returns:
        The nearest candidate (ties by unit id), or None.
    """
    candidates = [u for u in free.values() if accept(u)]
    if not candidates:
        return None
    return min(candidates, key=lambda u: (travel_min(u, x, y), u.unit_id))


def is_ambulance(unit: UnitView) -> bool:
    """Tells whether a unit is an ambulance.

    Args:
        unit: The unit.

    Returns:
        True for ALS and BLS ambulances.
    """
    return unit.unit_class is UnitClass.AMBULANCE


def is_als(unit: UnitView) -> bool:
    """Tells whether a unit is an ALS ambulance.

    Args:
        unit: The unit.

    Returns:
        True for ALS ambulances.
    """
    return is_ambulance(unit) and unit.unit_type == AmbulanceType.ALS


class GreedyController:
    """B0 / B0-cues greedy baseline; implements ``domain.ports.Controller``.

    Attributes:
        params: Policy parameters; B0 uses only ``max_ambulances_pre_triage``
            (the cap shared with the agents), B0-cues also the assessment
            parameters.
        use_caller_cues: Whether caller reports are assessed (B0-cues).
        actor: Actor name logged on every applied action.
    """

    def __init__(self, params: PolicyParams, *, use_caller_cues: bool = False) -> None:
        """Creates the controller.

        Args:
            params: Policy parameters.
            use_caller_cues: True for B0-cues, False for plain B0.
        """
        self.params = params
        self.use_caller_cues = use_caller_cues
        self.actor = B0_CUES_ACTOR if use_caller_cues else B0_ACTOR
        self._deteriorations = DeteriorationLog()

    def handle(self, point: DecisionPoint, port: EnginePort) -> None:
        """Reacts to a decision point; the triggers themselves are not needed.

        Args:
            point: The decision point (the controller rescans the whole view).
            port: Engine access.
        """
        LOGGER.debug("%s decision point at t=%.1f", self.actor, point.t_min)
        view = port.view()
        board = self._assess(view) if self.use_caller_cues else None
        _IncidentDispatcher(port, self.params, self.actor, board).run()
        self._assign_transports(port)
        self._redirect_full_arrivals(port)

    def _assess(self, view: PublicView) -> dict[str, PriorityReport]:
        """Assesses every open incident with the Incident Assessment policy.

        The assessment is recomputed from the current view at each decision
        point, so its deadlines are already current and no message is sent.

        Args:
            view: Current public view.

        Returns:
            Priority report per open incident id.
        """
        open_ids = [i.incident_id for i in view.incidents if i.status is IncidentStatus.OPEN]
        if not open_ids:
            return {}
        payload = assessment_input(view, open_ids, self._deteriorations.observe(view))
        return {r.incident_id: r for r in assess(payload, self.params).reports}

    def _assign_transports(self, port: EnginePort) -> None:
        """Sends every loaded unit to the nearest capable hospital with beds.

        Args:
            port: Engine access.
        """
        view = port.view()
        claimed = _inbound_loads(view)
        for unit in _units_with_status(view, UnitStatus.AWAITING_DESTINATION):
            victims = unit_victims(view, unit)
            hospitals = capable_hospitals(view, victims)
            if not hospitals:
                LOGGER.warning("B0: no hospital can take all victims of %s", unit.unit_id)
                continue
            with_beds = [h for h in hospitals if _free_beds(h, claimed) >= len(victims)]
            target = _nearest_hospital(unit, with_beds or hospitals)
            if _apply(port, self.actor, AssignTransport(
                unit_id=unit.unit_id, hospital_id=target.hospital_id, reserve_beds=False
            )):
                claimed[target.hospital_id] = claimed.get(target.hospital_id, 0) + len(victims)

    def _redirect_full_arrivals(self, port: EnginePort) -> None:
        """Redirects units that found a full ER to the next capable hospital with beds.

        If no other capable hospital has free beds, the unit stays and its
        patients queue (the engine guard performs the handover).

        Args:
            port: Engine access.
        """
        view = port.view()
        claimed = _inbound_loads(view)
        for unit in _units_with_status(view, UnitStatus.AT_HOSPITAL_FULL):
            victims = unit_victims(view, unit)
            options = [
                h
                for h in capable_hospitals(view, victims)
                if h.hospital_id != unit.hospital_id and _free_beds(h, claimed) >= len(victims)
            ]
            if not options:
                continue
            target = _nearest_hospital(unit, options)
            if _apply(port, self.actor, AssignTransport(
                unit_id=unit.unit_id, hospital_id=target.hospital_id, reserve_beds=False
            )):
                claimed[target.hospital_id] = claimed.get(target.hospital_id, 0) + len(victims)


class _IncidentDispatcher:
    """One greedy dispatch pass over open incidents.

    B0 goes first come, first served; B0-cues goes in assessed priority order.
    """

    def __init__(
        self,
        port: EnginePort,
        params: PolicyParams,
        actor: str,
        board: dict[str, PriorityReport] | None,
    ) -> None:
        """Snapshots the view and the free units.

        Args:
            port: Engine access.
            params: Policy parameters.
            actor: Actor name for applied actions.
            board: Assessments by incident id (B0-cues), or None (B0).
        """
        self._port = port
        self._params = params
        self._actor = actor
        self._view = port.view()
        self._free = {
            u.unit_id: u for u in self._view.units if u.status is UnitStatus.AVAILABLE
        }
        self._needs = {n.incident_id: n for n in incident_needs(self._view, board, params)}
        self._prioritised = board is not None

    def _ordered_incidents(self) -> list[IncidentView]:
        """Returns open incidents in service order.

        Returns:
            Assessed priority order for B0-cues, else report order (FCFS).
        """
        open_incidents = {
            i.incident_id: i for i in self._view.incidents if i.status is IncidentStatus.OPEN
        }
        if self._prioritised:
            return [open_incidents[n.incident_id] for n in priority_order(self._needs.values())]
        return sorted(open_incidents.values(), key=lambda i: (i.t_report_min, i.incident_id))

    def run(self) -> None:
        """Dispatches responders and ambulances incident by incident."""
        for incident in self._ordered_incidents():
            need = self._needs[incident.incident_id]
            self._send_responders(incident, need)
            if incident.triaged:
                self._send_after_triage(incident, need)
            else:
                self._send_before_triage(incident, need)

    def _send_responders(self, incident: IncidentView, need: IncidentNeed) -> None:
        """Sends the nearest free team for each missing responder capability.

        Args:
            incident: The incident.
            need: Its needs.
        """
        covered: set[str] = set()
        for cap in need.responder_caps_missing:
            if cap in covered:
                continue
            team = nearest_unit(
                self._free, incident.x, incident.y,
                lambda u, c=cap: u.unit_class is UnitClass.RESPONDER and c in u.capabilities,
            )
            if team is not None and self._dispatch(team, incident):
                covered = effective_capabilities(covered | set(team.capabilities))

    def _send_before_triage(self, incident: IncidentView, need: IncidentNeed) -> None:
        """Sends ambulances of any type up to ``min(ceil(reported/2), cap)``.

        When the assessment suspects a RED (B0-cues only; B0's ``als_needed``
        is 0 before triage), the first ambulances sent are the nearest ALS.

        Args:
            incident: The incident.
            need: Its needs.
        """
        target = min(
            max(1, math.ceil(incident.reported_victims / 2)),
            self._params.max_ambulances_pre_triage,
        )
        assigned = need.ambulances_assigned
        for _ in range(min(need.als_needed - need.als_en_route, target - assigned)):
            unit = nearest_unit(self._free, incident.x, incident.y, is_als)
            if unit is None:
                break
            if self._dispatch(unit, incident):
                assigned += 1
        for _ in range(target - assigned):
            unit = nearest_unit(self._free, incident.x, incident.y, is_ambulance)
            if unit is None:
                return
            self._dispatch(unit, incident)

    def _send_after_triage(self, incident: IncidentView, need: IncidentNeed) -> None:
        """Sends an ALS per uncovered RED, then fills the remaining seats.

        Args:
            incident: The incident.
            need: Its needs.
        """
        seats = need.seats_en_route
        for _ in range(need.red_waiting - need.als_en_route):
            unit = nearest_unit(self._free, incident.x, incident.y, is_als)
            if unit is None:
                break
            if self._dispatch(unit, incident):
                seats += unit.seats
        non_red_waiting = any(
            v.status is VictimStatus.WAITING and v.triage is not Triage.RED
            for v in incident.victims
        )
        accept = is_ambulance if non_red_waiting else is_als
        while seats < need.est_patients_remaining:
            unit = nearest_unit(self._free, incident.x, incident.y, accept)
            if unit is None:
                return
            if self._dispatch(unit, incident):
                seats += unit.seats

    def _dispatch(self, unit: UnitView, incident: IncidentView) -> bool:
        """Applies a dispatch and removes the unit from the free set.

        Args:
            unit: The unit.
            incident: The target incident.

        Returns:
            Whether the engine accepted the dispatch.
        """
        self._free.pop(unit.unit_id, None)
        return _apply(
            self._port, self._actor,
            DispatchUnit(unit_id=unit.unit_id, incident_id=incident.incident_id),
        )


def _apply(port: EnginePort, actor: str, action: Action) -> bool:
    """Applies an action and logs engine rejections.

    Args:
        port: Engine access.
        actor: Actor name to log.
        action: The action.

    Returns:
        Whether the engine accepted it.
    """
    check = port.apply(action, actor)
    if not check.ok:
        LOGGER.warning("%s action rejected: %s (%s)", actor, action, check.reason)
    return check.ok


def _units_with_status(view: PublicView, status: UnitStatus) -> Iterable[UnitView]:
    """Returns the units with a status, sorted by id.

    Args:
        view: Current public view.
        status: Wanted status.

    Returns:
        The units.
    """
    return sorted((u for u in view.units if u.status is status), key=lambda u: u.unit_id)


def _inbound_loads(view: PublicView) -> dict[str, int]:
    """Returns, per hospital, the patients on units transporting there.

    Args:
        view: Current public view.

    Returns:
        Hospital id to inbound patient count (B0 never reserves, so all of
        them will need an unreserved bed).
    """
    loads: dict[str, int] = {}
    for unit in view.units:
        if unit.status is UnitStatus.TRANSPORTING and unit.destination_id is not None:
            loads[unit.destination_id] = loads.get(unit.destination_id, 0) + len(unit.load_victim_ids)
    return loads


def _free_beds(hospital: HospitalView, claimed: dict[str, int]) -> int:
    """Returns the free unreserved beds minus beds claimed by inbound units.

    Args:
        hospital: The hospital.
        claimed: Inbound patients per hospital.

    Returns:
        Effective free beds (may be negative).
    """
    return hospital.beds_free_unreserved - claimed.get(hospital.hospital_id, 0)


def _nearest_hospital(unit: UnitView, hospitals: list[HospitalView]) -> HospitalView:
    """Returns the hospital nearest to a unit (ties by id).

    Args:
        unit: The unit.
        hospitals: Non-empty candidate list.

    Returns:
        The nearest hospital.
    """
    return min(hospitals, key=lambda h: (travel_min(unit, h.x, h.y), h.hospital_id))
