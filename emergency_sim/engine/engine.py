"""The deterministic discrete-event simulation engine (plan section 4).

``Engine`` satisfies ``domain.ports.EnginePort``, but controllers receive the
narrow ``engine.port.EnginePortAdapter`` so they cannot reach engine
internals. A run is driven by::

    engine = Engine(scenario, SimParams(), sink)
    port = EnginePortAdapter(engine)
    while (point := engine.advance()) is not None:
        controller.handle(point, port)
    outcomes = engine.outcomes()

``advance`` first applies the liveness guard (loaded units without a
destination, units stuck at a full hospital), then pops events in
``(t, priority, seq)`` order, emitting one ``SnapshotRecord`` per snapshot
minute before time moves past it, and returns a ``DecisionPoint`` as soon as an
event batch produced triggers. ``apply`` validates an action, logs an
``ActionRecord`` and mutates state only if the action is legal.
"""

import logging
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final

from emergency_sim import geo
from emergency_sim.domain import public_rules
from emergency_sim.domain.actions import (
    Action,
    ActionCheck,
    AssignTransport,
    DispatchUnit,
    DivertUnit,
    LendStaff,
)
from emergency_sim.domain.enums import (
    IncidentStatus,
    ResponderCapability,
    SimEventType,
    Triage,
    TriggerKind,
    UnitStatus,
    VictimStatus,
)
from emergency_sim.domain.log_records import ActionRecord, SimEventRecord, SnapshotRecord
from emergency_sim.domain.outcomes import RunOutcomes
from emergency_sim.domain.params import SimParams
from emergency_sim.domain.ports import EventSink
from emergency_sim.domain.scenario import Scenario
from emergency_sim.domain.views import DecisionPoint, PublicView, Trigger
from emergency_sim.engine import care, observe, rules
from emergency_sim.engine.events import Event, EventKind, EventQueue
from emergency_sim.engine.observe import check_world
from emergency_sim.engine.state import (
    CARDIAC_ARREST,
    ON_SCENE_VICTIM_STATUSES,
    EngineError,
    IncidentState,
    Movement,
    UnitState,
    VictimState,
    WorldState,
    from_scenario,
    round_time,
)

logger = logging.getLogger(__name__)

GUARD_ACTOR: Final = "engine_guard"
"""Actor name of actions the liveness guard applies."""

_FIELD_CARE_STATUSES: Final = frozenset({VictimStatus.WAITING, VictimStatus.LOADING})


class Engine:
    """Deterministic discrete-event simulation; the single source of truth.

    Satisfies ``domain.ports.EnginePort``. Not thread-safe.
    """

    def __init__(self, scenario: Scenario, params: SimParams, sink: EventSink) -> None:
        """Builds the world at t=0 and schedules the initial events.

        Args:
            scenario: Ground truth of the run.
            params: Engine constants.
            sink: Destination of sim events, action records and snapshots.

        Raises:
            EngineError: If ``params.snapshot_interval_min`` is not positive or
                the scenario has colliding ids.
        """
        if params.snapshot_interval_min <= 0:
            raise EngineError("snapshot_interval_min must be positive")
        self._params = params
        self._sink = sink
        self._state: WorldState = from_scenario(scenario)
        self._queue = EventQueue()
        self._now = 0.0
        self._next_snapshot = 0
        self._finished = False
        self._triggers: list[Trigger] = []
        self._handlers: Mapping[EventKind, Callable[[Event], None]] = MappingProxyType(
            {
                EventKind.TREATMENT_END: self._on_treatment_end,
                EventKind.T0_BED_RELEASE: self._on_t0_bed_release,
                EventKind.T0_STAFF_RELEASE: self._on_t0_staff_release,
                EventKind.SHIFT_END: self._on_shift_end,
                EventKind.BUSY_T0_END: self._on_busy_t0_end,
                EventKind.ARRIVE_HOSPITAL: self._on_arrive_hospital,
                EventKind.HANDOVER_DONE: self._on_handover_done,
                EventKind.ARRIVE_SCENE: self._on_arrive_scene,
                EventKind.ACCESS_GRANTED: self._on_access_granted,
                EventKind.LOAD_DONE: self._on_load_done,
                EventKind.INCIDENT_REPORTED: self._on_incident_reported,
                EventKind.DETERIORATION: self._on_deterioration,
            }
        )
        self._schedule_initial_events()

    # ------------------------------------------------------------------ API

    @property
    def now(self) -> float:
        """Current simulation time in minutes."""
        return self._now

    @property
    def finished(self) -> bool:
        """Whether ``advance`` has returned None (the horizon was reached)."""
        return self._finished

    def advance(self) -> DecisionPoint | None:
        """Runs the simulation to the next decision point.

        Returns:
            The next decision point, or None once the horizon is reached (and
            on every later call).

        Raises:
            EngineError: If the event queue went back in time.
        """
        if self._finished:
            return None
        self._run_liveness_guard()
        while True:
            t_next = self._queue.peek_time()
            if t_next is None or t_next > self._state.horizon:
                self._finish()
                return None
            if t_next < self._now:
                raise EngineError(f"event at {t_next} is before now={self._now}")
            self._emit_snapshots(until=t_next, inclusive=False)
            self._process_events_at(t_next)
            if self._triggers:
                return self._take_decision_point()

    def view(self) -> PublicView:
        """Returns the public view at the current time."""
        return observe.build_public_view(self._state, self._now)

    def check(self, action: Action) -> ActionCheck:
        """Validates an action without applying or logging it.

        Args:
            action: The action.

        Returns:
            Whether it is legal now, and why not.
        """
        return rules.check_action(self._state, action, self._now)

    def apply(self, action: Action, actor: str) -> ActionCheck:
        """Validates an action, logs an ``ActionRecord`` and applies it if legal.

        Args:
            action: The action.
            actor: Name of the controller or agent submitting it.

        Returns:
            The validation result; state is unchanged when ``ok`` is False.

        Raises:
            EngineError: If the run has already finished.
        """
        if self._finished:
            raise EngineError("cannot apply actions after the horizon")
        result = self.check(action)
        self._sink.emit(
            ActionRecord(
                t_min=self._now, actor=actor, action=action,
                ok=result.ok, reason=result.reason, detail=result.detail,
            )
        )
        if result.ok:
            self._mutate(action)
        return result

    def outcomes(self) -> RunOutcomes:
        """Returns the run outcomes.

        Returns:
            Victim/incident timelines and busy minutes over ``[0, horizon]``.

        Raises:
            EngineError: If called before ``advance`` returned None.
        """
        if not self._finished:
            raise EngineError("outcomes are only available after advance() returned None")
        return observe.build_outcomes(self._state, self._params)

    def check_invariants(self) -> tuple[str, ...]:
        """Checks internal consistency (used by tests).

        Returns:
            Human-readable violations; an empty tuple when consistent.
        """
        problems = list(check_world(self._state))
        if self._now > self._state.horizon:
            problems.append(f"now={self._now} is past the horizon")
        return tuple(problems)

    # ------------------------------------------------------- loop internals

    def _schedule_initial_events(self) -> None:
        """Schedules reports, busy-t0 ends, t=0 releases, shift ends, deterioration."""
        s, p = self._state, self._params
        for inc in sorted(s.incidents.values(), key=lambda i: i.incident_id):
            self._queue.push(inc.spec.t_report_min, Event(EventKind.INCIDENT_REPORTED, inc.incident_id))
        for unit in s.units.values():
            if unit.status is UnitStatus.BUSY_T0 and unit.available_at is not None:
                self._schedule_unit(unit, unit.available_at, EventKind.BUSY_T0_END)
        for hid, hospital in s.hospitals.items():
            for k in range(1, hospital.t0_beds + 1):
                self._queue.push(k * p.t0_bed_release_interval_min, Event(EventKind.T0_BED_RELEASE, hid))
        for sid, member in s.staff.items():
            for k in range(1, member.t0_load + 1):
                self._queue.push(k * p.t0_staff_release_interval_min, Event(EventKind.T0_STAFF_RELEASE, sid))
        for sid, member in s.staff.items():
            self._queue.push(member.spec.shift_end_min, Event(EventKind.SHIFT_END, sid))
        for vid, victim in s.victims.items():
            delay = victim.spec.deteriorates_to_red_at_min
            if victim.spec.triage is Triage.YELLOW and delay is not None:
                self._queue.push(victim.t_report + delay, Event(EventKind.DETERIORATION, vid))

    def _schedule_unit(self, unit: UnitState, t: float, kind: EventKind) -> float:
        """Schedules a unit event, invalidating any pending event of that unit."""
        unit.trip += 1
        return self._queue.push(t, Event(kind, unit.unit_id, unit.trip))

    def _process_events_at(self, t: float) -> None:
        """Pops and handles every event at time ``t`` (including same-time follow-ups)."""
        self._now = t
        while self._queue.peek_time() == t:
            _, event = self._queue.pop()
            self._handlers[event.kind](event)

    def _take_decision_point(self) -> DecisionPoint:
        """Builds the decision point from collected triggers (deduplicated, sorted)."""
        unique = {
            (t.kind.value, t.incident_id or "", t.unit_id or "", t.hospital_id or "", t.victim_ids): t
            for t in self._triggers
        }
        self._triggers = []
        point = DecisionPoint(t_min=self._now, triggers=tuple(unique[k] for k in sorted(unique)))
        logger.debug("decision point at %.1f with %d triggers", self._now, len(point.triggers))
        return point

    def _emit_snapshots(self, until: float, inclusive: bool) -> None:
        """Emits snapshots for minutes ``k < until`` (or ``<=``), never past the horizon."""
        while True:
            k = round(self._next_snapshot * self._params.snapshot_interval_min, 6)
            if k > self._state.horizon or k > until or (k == until and not inclusive):
                return
            self._sink.emit(
                SnapshotRecord(
                    t_min=k,
                    view=observe.build_public_view(self._state, k),
                    counts=observe.build_snapshot_counts(self._state),
                )
            )
            self._next_snapshot += 1

    def _finish(self) -> None:
        """Emits the remaining snapshots up to the horizon and closes the run."""
        self._emit_snapshots(until=self._state.horizon, inclusive=True)
        self._now = max(self._now, self._state.horizon)
        self._finished = True

    def _emit(self, event: SimEventType, **fields: object) -> None:
        """Logs a ``SimEventRecord`` at the current time."""
        self._sink.emit(SimEventRecord.model_validate({"t_min": self._now, "event": event, **fields}))

    def _trigger(self, kind: TriggerKind, **ids: object) -> None:
        """Adds a trigger to the pending decision point."""
        self._triggers.append(Trigger.model_validate({"kind": kind, **ids}))

    def _unit_for(self, event: Event, status: UnitStatus) -> UnitState | None:
        """Returns the event's unit if the event is current and the unit is in ``status``."""
        unit = self._state.units[event.subject_id]
        if unit.trip != event.token or unit.status is not status:
            logger.debug("ignoring stale %s for %s", event.kind, unit.unit_id)
            return None
        return unit

    # ------------------------------------------------------- liveness guard

    def _run_liveness_guard(self) -> None:
        """Resolves units the controller left without a destination or at a full ER."""
        for unit in list(self._state.units.values()):
            if unit.status is UnitStatus.AWAITING_DESTINATION:
                self._guard_transport(unit)
            elif unit.status is UnitStatus.AT_HOSPITAL_FULL and unit.hospital_id is not None:
                free = max(0, self._state.beds_free_unreserved(unit.hospital_id))
                self._handover(unit, beds=min(free, len(unit.load)))

    def _guard_transport(self, unit: UnitState) -> None:
        """Sends a loaded unit to the nearest capable hospital (controller omission)."""
        hospital_id = rules.nearest_capable_hospital(self._state, unit, self._now)
        self._emit(
            SimEventType.CONTROLLER_OMISSION,
            unit_id=unit.unit_id,
            incident_id=unit.incident_id,
            hospital_id=hospital_id,
            victim_ids=tuple(unit.load),
            detail="loaded unit had no destination; guard sends it to the nearest capable hospital",
        )
        action = AssignTransport(unit_id=unit.unit_id, hospital_id=hospital_id, reserve_beds=False)
        if not self.apply(action, GUARD_ACTOR).ok:
            raise EngineError(f"guard transport of {unit.unit_id} rejected")

    # -------------------------------------------------------------- actions

    def _mutate(self, action: Action) -> None:
        """Applies a validated action."""
        match action:
            case DispatchUnit():
                self._do_dispatch(action)
            case DivertUnit():
                self._do_divert(action)
            case AssignTransport():
                self._do_transport(action)
            case LendStaff():
                self._do_lend(action)
            case _:
                raise EngineError(f"unknown action {action!r}")

    def _do_dispatch(self, action: DispatchUnit) -> None:
        """Sends an available unit to an incident (turnout + travel)."""
        unit = self._state.units[action.unit_id]
        incident = self._state.incidents[action.incident_id]
        vehicle = "ambulance" if unit.is_ambulance else "responder"
        p = self._params
        turnout = p.ambulance_turnout_min if unit.is_ambulance else p.responder_turnout_min
        origin = (unit.x, unit.y)
        depart = round_time(self._now + turnout)
        arrive = round_time(depart + geo.travel_time(origin, incident.position, vehicle))
        unit.movement = Movement(origin, incident.position, depart, arrive)
        unit.incident_id, unit.hospital_id = incident.incident_id, None
        unit.set_status(UnitStatus.EN_ROUTE_SCENE, self._now)
        self._schedule_unit(unit, arrive, EventKind.ARRIVE_SCENE)
        self._emit(
            SimEventType.UNIT_DISPATCHED, unit_id=unit.unit_id,
            incident_id=incident.incident_id, detail=f"eta {arrive}",
        )

    def _do_divert(self, action: DivertUnit) -> None:
        """Redirects an en-route ambulance from its interpolated position (no new turnout)."""
        unit = self._state.units[action.unit_id]
        incident = self._state.incidents[action.incident_id]
        if unit.movement is None:
            raise EngineError(f"{unit.unit_id} is en route without a movement")
        origin = unit.position_at(self._now)
        depart = max(self._now, unit.movement.depart_t)
        arrive = round_time(depart + geo.travel_time(origin, incident.position, "ambulance"))
        previous = unit.incident_id
        unit.movement = Movement(origin, incident.position, depart, arrive)
        unit.incident_id = incident.incident_id
        self._schedule_unit(unit, arrive, EventKind.ARRIVE_SCENE)
        self._emit(
            SimEventType.UNIT_DIVERTED, unit_id=unit.unit_id,
            incident_id=incident.incident_id, detail=f"from {previous}; eta {arrive}",
        )

    def _do_transport(self, action: AssignTransport) -> None:
        """Starts the drive of a loaded unit to a hospital, reserving beds if asked."""
        unit = self._state.units[action.unit_id]
        hospital = self._state.hospitals[action.hospital_id]
        load = tuple(unit.load)
        if action.reserve_beds:
            hospital.reservations[unit.unit_id] = len(load)
            self._emit(
                SimEventType.BEDS_RESERVED, unit_id=unit.unit_id, hospital_id=hospital.hospital_id,
                victim_ids=load, detail=f"{len(load)} bed(s)",
            )
        origin = unit.position_at(self._now)
        arrive = round_time(self._now + geo.travel_time(origin, hospital.position, "ambulance"))
        unit.movement = Movement(origin, hospital.position, self._now, arrive)
        unit.hospital_id = hospital.hospital_id
        unit.set_status(UnitStatus.TRANSPORTING, self._now)
        for victim in self._victims(load):
            victim.status, victim.hospital_id = VictimStatus.TRANSPORTING, hospital.hospital_id
            victim.arrived_hospital = None  # a redirected victim's door time is the final hospital's
        self._schedule_unit(unit, arrive, EventKind.ARRIVE_HOSPITAL)
        self._emit(
            SimEventType.TRANSPORT_STARTED, unit_id=unit.unit_id, incident_id=unit.incident_id,
            hospital_id=hospital.hospital_id, victim_ids=load, detail=f"eta {arrive}",
        )

    def _do_lend(self, action: LendStaff) -> None:
        """Lends a float staff member to a hospital for the rest of the run."""
        member = self._state.staff[action.staff_id]
        member.hospital_id, member.lent = action.hospital_id, True
        self._emit(SimEventType.STAFF_LENT, staff_id=member.staff_id, hospital_id=action.hospital_id)
        self._run_care(action.hospital_id)

    def _victims(self, victim_ids: tuple[str, ...] | list[str]) -> list[VictimState]:
        """Looks up victims by id."""
        return [self._state.victims[v] for v in victim_ids]

    # ------------------------------------------------------- scene handlers

    def _on_incident_reported(self, event: Event) -> None:
        """An incident is reported: it becomes visible."""
        incident = self._state.incidents[event.subject_id]
        incident.reported = True
        self._emit(SimEventType.INCIDENT_REPORTED, incident_id=incident.incident_id)
        self._trigger(TriggerKind.NEW_INCIDENT, incident_id=incident.incident_id)

    def _on_busy_t0_end(self, event: Event) -> None:
        """A unit busy at t=0 becomes available at its station."""
        unit = self._unit_for(event, UnitStatus.BUSY_T0)
        if unit is None:
            return
        unit.available_at = None
        unit.set_status(UnitStatus.AVAILABLE, self._now)
        self._emit(SimEventType.UNIT_AVAILABLE, unit_id=unit.unit_id, detail="end of t=0 assignment")
        self._trigger(TriggerKind.UNIT_FREED, unit_id=unit.unit_id)

    def _on_arrive_scene(self, event: Event) -> None:
        """A unit reaches its incident: triage, field care, access, loading."""
        unit = self._unit_for(event, UnitStatus.EN_ROUTE_SCENE)
        if unit is None or unit.incident_id is None:
            return
        incident = self._state.incidents[unit.incident_id]
        unit.x, unit.y = incident.position
        unit.movement = None
        unit.set_status(UnitStatus.ON_SCENE, self._now)
        self._emit(SimEventType.UNIT_ARRIVED_SCENE, unit_id=unit.unit_id, incident_id=incident.incident_id)
        self._record_first_arrival(incident, unit)
        if incident.status is not IncidentStatus.OPEN:
            self._free_unit_at_scene(unit, "incident already cleared")
            return
        if not incident.triaged:
            self._triage(incident, unit)
        if unit.status is not UnitStatus.ON_SCENE:
            return
        if unit.is_ambulance:
            self._ambulance_on_scene(incident, unit)
        else:
            incident.caps_present.update(ResponderCapability(c) for c in unit.capabilities)
            self._update_access(incident)

    def _record_first_arrival(self, incident: IncidentState, unit: UnitState) -> None:
        """Records the first unit / first ambulance arrival times."""
        if incident.first_unit_on_scene is None:
            incident.first_unit_on_scene = self._now
        if unit.is_ambulance and incident.first_ambulance_on_scene is None:
            incident.first_ambulance_on_scene = self._now

    def _triage(self, incident: IncidentState, unit: UnitState) -> None:
        """START triage by the first unit on scene: reveals every victim."""
        incident.triaged = True
        for victim in self._state.incident_victims(incident.incident_id):
            if victim.status is VictimStatus.UNTRIAGED:
                black = victim.triage is Triage.BLACK
                victim.status = VictimStatus.EXPECTANT if black else VictimStatus.WAITING
        self._emit(
            SimEventType.SCENE_TRIAGED, incident_id=incident.incident_id, unit_id=unit.unit_id,
            victim_ids=incident.victim_ids,
        )
        self._trigger(TriggerKind.SCENE_TRIAGED, incident_id=incident.incident_id, victim_ids=incident.victim_ids)
        self._check_cleared(incident)

    def _ambulance_on_scene(self, incident: IncidentState, unit: UnitState) -> None:
        """ALS field care, then loading if the scene is accessible (else staged)."""
        if unit.is_als:
            self._als_arrival(incident, unit)
        if incident.accessible:
            self._start_loading(unit)

    def _als_arrival(self, incident: IncidentState, unit: UnitState) -> None:
        """Records ALS arrival and starts field care for cardiac arrests."""
        on_scene = [
            v for v in self._state.incident_victims(incident.incident_id) if v.status in _FIELD_CARE_STATUSES
        ]
        for victim in on_scene:
            if victim.als_on_scene is None:
                victim.als_on_scene = self._now
        arrests = [v for v in on_scene if v.spec.condition == CARDIAC_ARREST and v.care_start is None]
        for victim in arrests:
            victim.care_start, victim.care_mode = self._now, "field"
        if arrests:
            self._emit(
                SimEventType.FIELD_CARE_STARTED, incident_id=incident.incident_id, unit_id=unit.unit_id,
                victim_ids=tuple(v.victim_id for v in arrests),
            )

    def _update_access(self, incident: IncidentState) -> None:
        """Grants (or schedules) access once every required capability is on scene."""
        if incident.accessible or incident.access_pending:
            return
        required = set(incident.spec.required_responder_capabilities)
        if not required <= public_rules.effective_capabilities(incident.caps_present):
            return
        if required & public_rules.HEAVY_ACCESS_CAPABILITIES:
            incident.access_pending = True
            when = self._now + self._params.heavy_access_min
            self._queue.push(when, Event(EventKind.ACCESS_GRANTED, incident.incident_id))
        else:
            self._grant_access(incident)

    def _on_access_granted(self, event: Event) -> None:
        """Delayed access after heavy work on scene."""
        incident = self._state.incidents[event.subject_id]
        incident.access_pending = False
        self._grant_access(incident)

    def _grant_access(self, incident: IncidentState) -> None:
        """Makes the scene accessible; staged ambulances start loading in id order."""
        incident.accessible = True
        self._emit(SimEventType.ACCESS_GRANTED, incident_id=incident.incident_id)
        if incident.status is not IncidentStatus.OPEN:
            return
        for unit in list(self._state.units.values()):
            staged = unit.is_ambulance and unit.status is UnitStatus.ON_SCENE
            if staged and unit.incident_id == incident.incident_id:
                self._start_loading(unit)

    def _start_loading(self, unit: UnitState) -> None:
        """Chooses the load (loading rule) or clears the unit if it can carry nobody."""
        picks = rules.choose_load(self._state, unit)
        if not picks:
            self._free_unit_at_scene(unit, "no waiting victim this unit may carry")
            return
        for victim in self._victims(picks):
            victim.status, victim.unit_id = VictimStatus.LOADING, unit.unit_id
        unit.load = list(picks)
        unit.set_status(UnitStatus.LOADING, self._now)
        self._schedule_unit(unit, self._now + self._params.scene_load_min, EventKind.LOAD_DONE)
        self._emit(
            SimEventType.LOADING_STARTED, unit_id=unit.unit_id, incident_id=unit.incident_id, victim_ids=picks
        )

    def _on_load_done(self, event: Event) -> None:
        """Loading finished: the unit awaits a destination."""
        unit = self._unit_for(event, UnitStatus.LOADING)
        if unit is None or unit.incident_id is None:
            return
        for victim in self._victims(unit.load):
            victim.status, victim.loaded_at = VictimStatus.LOADED, self._now
        unit.set_status(UnitStatus.AWAITING_DESTINATION, self._now)
        load = tuple(unit.load)
        self._emit(
            SimEventType.PATIENTS_LOADED, unit_id=unit.unit_id, incident_id=unit.incident_id, victim_ids=load
        )
        self._trigger(
            TriggerKind.PATIENTS_LOADED, unit_id=unit.unit_id, incident_id=unit.incident_id, victim_ids=load
        )
        self._check_cleared(self._state.incidents[unit.incident_id])

    def _check_cleared(self, incident: IncidentState) -> None:
        """Clears a triaged incident with nobody left on scene; frees units there."""
        if incident.status is not IncidentStatus.OPEN or not incident.triaged:
            return
        if any(v.status in ON_SCENE_VICTIM_STATUSES for v in self._state.incident_victims(incident.incident_id)):
            return
        incident.status, incident.cleared_at = IncidentStatus.CLEARED, self._now
        self._emit(SimEventType.INCIDENT_CLEARED, incident_id=incident.incident_id)
        for unit in list(self._state.units.values()):
            if unit.incident_id == incident.incident_id and unit.status is UnitStatus.ON_SCENE:
                self._free_unit_at_scene(unit, "incident cleared")

    def _free_unit_at_scene(self, unit: UnitState, reason: str) -> None:
        """Makes an on-scene unit available at the incident location."""
        incident_id = unit.incident_id
        unit.set_status(UnitStatus.AVAILABLE, self._now)
        unit.incident_id, unit.hospital_id, unit.movement = None, None, None
        self._emit(SimEventType.UNIT_AVAILABLE, unit_id=unit.unit_id, incident_id=incident_id, detail=reason)
        self._trigger(TriggerKind.UNIT_FREED, unit_id=unit.unit_id)

    def _on_deterioration(self, event: Event) -> None:
        """A YELLOW without care turns RED (visible only once triaged)."""
        victim = self._state.victims[event.subject_id]
        if victim.care_start is not None or victim.triage is not Triage.YELLOW:
            return
        victim.triage, victim.needs_als, victim.deteriorated_at = Triage.RED, True, self._now
        visible = victim.status is not VictimStatus.UNTRIAGED
        ids = {"incident_id": victim.incident_id, "unit_id": victim.unit_id, "hospital_id": victim.hospital_id}
        self._emit(
            SimEventType.VICTIM_DETERIORATED, victim_ids=(victim.victim_id,),
            detail="visible" if visible else "hidden: scene not yet triaged", **ids,
        )
        if visible:
            self._trigger(TriggerKind.DETERIORATION, victim_ids=(victim.victim_id,), **ids)

    # ---------------------------------------------------- hospital handlers

    def _on_arrive_hospital(self, event: Event) -> None:
        """A unit reaches its hospital: reserved/free beds, else full on arrival."""
        unit = self._unit_for(event, UnitStatus.TRANSPORTING)
        if unit is None or unit.hospital_id is None:
            return
        hospital = self._state.hospitals[unit.hospital_id]
        unit.x, unit.y = hospital.position
        unit.movement = None
        load = tuple(unit.load)
        for victim in self._victims(load):
            victim.arrived_hospital = self._now
        self._emit(
            SimEventType.ARRIVED_HOSPITAL, unit_id=unit.unit_id, incident_id=unit.incident_id,
            hospital_id=hospital.hospital_id, victim_ids=load,
        )
        hospital.reservations.pop(unit.unit_id, None)
        if self._state.beds_free_unreserved(hospital.hospital_id) >= len(load):
            self._handover(unit, beds=len(load))
            return
        unit.set_status(UnitStatus.AT_HOSPITAL_FULL, self._now)
        ids = {"unit_id": unit.unit_id, "incident_id": unit.incident_id, "hospital_id": hospital.hospital_id}
        self._emit(SimEventType.HOSPITAL_FULL_ON_ARRIVAL, victim_ids=load, **ids)
        self._trigger(TriggerKind.HOSPITAL_FULL_ON_ARRIVAL, victim_ids=load, **ids)

    def _handover(self, unit: UnitState, beds: int) -> None:
        """Hands patients over: the first ``beds`` (by care order) are bedded, the rest queue."""
        if unit.hospital_id is None:
            raise EngineError(f"{unit.unit_id} hands over without a hospital")
        hospital_id = unit.hospital_id
        for i, victim in enumerate(sorted(self._victims(unit.load), key=care.care_order_key)):
            if i < beds:
                care.bed_victim(victim, hospital_id, self._now)
                event = SimEventType.PATIENT_BEDDED
            else:
                care.queue_victim(victim, hospital_id)
                event = SimEventType.PATIENT_QUEUED
            self._emit(
                event, unit_id=unit.unit_id, incident_id=victim.incident_id,
                hospital_id=hospital_id, victim_ids=(victim.victim_id,),
            )
        unit.load = []
        unit.set_status(UnitStatus.HANDOVER, self._now)
        self._schedule_unit(unit, self._now + self._params.handover_min, EventKind.HANDOVER_DONE)
        self._run_care(hospital_id)

    def _on_handover_done(self, event: Event) -> None:
        """Handover finished: the unit is available at the hospital."""
        unit = self._unit_for(event, UnitStatus.HANDOVER)
        if unit is None:
            return
        hospital_id = unit.hospital_id
        unit.set_status(UnitStatus.AVAILABLE, self._now)
        unit.incident_id, unit.hospital_id = None, None
        self._emit(
            SimEventType.UNIT_AVAILABLE, unit_id=unit.unit_id, hospital_id=hospital_id, detail="handover done"
        )
        self._trigger(TriggerKind.UNIT_FREED, unit_id=unit.unit_id)

    def _run_care(self, hospital_id: str) -> None:
        """Runs ``care.assign_care`` and logs/schedules its consequences."""
        for change in care.assign_care(self._state, hospital_id, self._now):
            victim = self._state.victims[change.victim_id]
            match change:
                case care.Bedded():
                    self._emit(
                        SimEventType.PATIENT_BEDDED, incident_id=victim.incident_id,
                        hospital_id=hospital_id, victim_ids=(victim.victim_id,), detail="from queue",
                    )
                case care.TreatmentStarted(doctor_id=doctor, nurse_id=nurse):
                    end = self._now + self._params.treatment_min(victim.triage)
                    self._queue.push(end, Event(EventKind.TREATMENT_END, victim.victim_id))
                    self._emit(
                        SimEventType.TREATMENT_STARTED, incident_id=victim.incident_id, hospital_id=hospital_id,
                        staff_id=doctor, victim_ids=(victim.victim_id,), detail=f"doctor {doctor}, nurse {nurse}",
                    )

    def _on_treatment_end(self, event: Event) -> None:
        """Treatment finished: bed and staff slots are freed."""
        hospital_id = care.end_treatment(self._state, event.subject_id)
        victim = self._state.victims[event.subject_id]
        self._emit(
            SimEventType.TREATMENT_ENDED, incident_id=victim.incident_id, hospital_id=hospital_id,
            victim_ids=(victim.victim_id,),
        )
        self._run_care(hospital_id)

    def _on_t0_bed_release(self, event: Event) -> None:
        """A patient present at t=0 leaves the ER."""
        hospital = self._state.hospitals[event.subject_id]
        if hospital.t0_beds <= 0:
            raise EngineError(f"{hospital.hospital_id} has no t=0 bed to release")
        hospital.t0_beds -= 1
        self._emit(SimEventType.T0_BED_RELEASED, hospital_id=hospital.hospital_id)
        self._run_care(hospital.hospital_id)

    def _on_t0_staff_release(self, event: Event) -> None:
        """A staff member sheds one patient carried from t=0."""
        member = self._state.staff[event.subject_id]
        if member.t0_load <= 0:
            raise EngineError(f"{member.staff_id} has no t=0 patient to release")
        member.t0_load -= 1
        self._emit(SimEventType.T0_STAFF_RELEASED, staff_id=member.staff_id, hospital_id=member.hospital_id)
        if member.hospital_id is not None:
            self._run_care(member.hospital_id)

    def _on_shift_end(self, event: Event) -> None:
        """A shift ends: the member stops taking patients unless held over."""
        member = self._state.staff[event.subject_id]
        self._emit(SimEventType.SHIFT_ENDED, staff_id=member.staff_id, hospital_id=member.hospital_id)
        if member.hospital_id is not None:
            self._run_care(member.hospital_id)
