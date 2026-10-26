"""Full runs of the real scenarios: invariants, log ordering and determinism.

Defines ``NearestUnitController``, a trivial controller that only reads the
``PublicView`` (nearest free unit per need, nearest capable hospital with a
free bed), and ``run_engine``, which drives a run and optionally checks the
engine invariants after every ``advance()``. Other engine tests import both.
"""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from emergency_sim import geo
from emergency_sim.domain.actions import Action, AssignTransport, DispatchUnit
from emergency_sim.domain.enums import (
    IncidentStatus,
    ResponderCapability,
    SimEventType,
    Triage,
    UnitClass,
    UnitStatus,
    VictimStatus,
)
from emergency_sim.domain.log_records import LogRecord, SimEventRecord, SnapshotRecord
from emergency_sim.domain.outcomes import RunOutcomes
from emergency_sim.domain.params import SimParams
from emergency_sim.domain.ports import Controller, EnginePort
from emergency_sim.domain.scenario import Scenario
from emergency_sim.domain.views import DecisionPoint, IncidentView, PublicView, UnitView
from emergency_sim.engine import Engine
from emergency_sim.eventlog import MemoryEventSink, read_log, write_jsonl

STUB_ACTOR = "stub_nearest"
MAX_PRE_TRIAGE_AMBULANCES = 2


UnitFilter = Callable[[UnitView], bool]


def _eta(unit: UnitView, x: float, y: float) -> float:
    """Drive time of a unit to a point (no turnout)."""
    vehicle = "ambulance" if unit.unit_class is UnitClass.AMBULANCE else "responder"
    return geo.travel_time((unit.x, unit.y), (x, y), vehicle)


def _is_ambulance(unit: UnitView) -> bool:
    return unit.unit_class is UnitClass.AMBULANCE


def _is_als(unit: UnitView) -> bool:
    return _is_ambulance(unit) and unit.unit_type == "ALS"


class _Picker:
    """Takes the nearest matching free units for one incident."""

    def __init__(self, free: Mapping[str, UnitView], incident: IncidentView) -> None:
        self._pool = dict(free)
        self._incident = incident
        self.chosen: list[UnitView] = []

    def take(self, ok: UnitFilter) -> UnitView | None:
        options = [u for u in self._pool.values() if ok(u)]
        key = lambda u: (_eta(u, self._incident.x, self._incident.y), u.unit_id)  # noqa: E731
        best = min(options, key=key, default=None)
        if best is not None:
            self.chosen.append(best)
            del self._pool[best.unit_id]
        return best


@dataclass
class NearestUnitController:
    """Trivial controller: nearest free unit per need, nearest capable hospital.

    Attributes:
        reserve: Whether transports reserve beds when the hospital has room.
        transport: Whether to assign destinations at all (False leaves loaded
            units to the engine's liveness guard).
        actions: Every action applied, for inspection.
    """

    reserve: bool = True
    transport: bool = True
    actions: list[Action] = field(default_factory=list)

    def handle(self, point: DecisionPoint, port: EnginePort) -> None:
        """Assigns destinations, then dispatches free units to open incidents.

        Args:
            point: The decision point.
            port: The engine.
        """
        if self.transport:
            self._assign_destinations(port)
        self._dispatch_all(port)

    def _apply(self, port: EnginePort, action: Action) -> None:
        result = port.apply(action, STUB_ACTOR)
        assert result.ok, (action, result)
        self.actions.append(action)

    def _assign_destinations(self, port: EnginePort) -> None:
        view = port.view()
        victims = {v.victim_id: v for i in view.incidents for v in i.victims}
        for unit in view.units:
            if unit.status is not UnitStatus.AWAITING_DESTINATION:
                continue
            caps = {victims[v].required_capability for v in unit.load_victim_ids}
            capable = [h for h in port.view().hospitals if caps <= set(h.capabilities)]
            roomy = [h for h in capable if h.beds_free_unreserved >= len(unit.load_victim_ids)]
            best = min(roomy or capable, key=lambda h: (_eta(unit, h.x, h.y), h.hospital_id))
            reserve = self.reserve and bool(roomy)
            self._apply(port, AssignTransport(unit_id=unit.unit_id, hospital_id=best.hospital_id, reserve_beds=reserve))

    def _dispatch_all(self, port: EnginePort) -> None:
        view = port.view()
        free = {u.unit_id: u for u in view.units if u.status is UnitStatus.AVAILABLE}
        for incident in sorted(view.incidents, key=lambda i: (i.t_report_min, i.incident_id)):
            if incident.status is IncidentStatus.OPEN:
                for unit in self._needs(view, incident, free):
                    del free[unit.unit_id]
                    self._apply(port, DispatchUnit(unit_id=unit.unit_id, incident_id=incident.incident_id))

    def _needs(self, view: PublicView, incident: IncidentView, free: Mapping[str, UnitView]) -> list[UnitView]:
        units = {u.unit_id: u for u in view.units}
        assigned = [units[u] for u in incident.assigned_unit_ids]
        picker = _Picker(free, incident)
        _pick_responders(incident, assigned, picker)
        ambulances = [u for u in assigned if _is_ambulance(u)]
        if incident.triaged:
            _pick_ambulances_after_triage(incident, ambulances, picker)
        else:
            want = min(MAX_PRE_TRIAGE_AMBULANCES, max(1, math.ceil(incident.reported_victims / 2)))
            for _ in range(want - len(ambulances)):
                picker.take(_is_ambulance)
        return picker.chosen


def _pick_responders(incident: IncidentView, assigned: list[UnitView], picker: _Picker) -> None:
    """One nearest team per required capability not present or en route."""
    covered = set(incident.responder_capabilities_present)
    for unit in assigned:
        covered |= {ResponderCapability(c) for c in unit.capabilities}
    for cap in incident.required_responder_capabilities:
        if cap not in covered and (team := picker.take(lambda u, c=cap: c in u.capabilities)) is not None:
            covered |= {ResponderCapability(c) for c in team.capabilities}


def _pick_ambulances_after_triage(incident: IncidentView, ambulances: list[UnitView], picker: _Picker) -> None:
    """One ALS per waiting RED, then any ambulance until seats cover the waiting victims."""
    waiting = [v for v in incident.victims if v.status is VictimStatus.WAITING]
    reds = sum(v.triage is Triage.RED for v in waiting)
    for _ in range(reds - sum(_is_als(u) for u in ambulances)):
        picker.take(_is_als)
    seats = sum(u.seats for u in [*ambulances, *picker.chosen])
    useful: UnitFilter = lambda u: _is_ambulance(u) and (_is_als(u) or reds < len(waiting))  # noqa: E731
    while seats < len(waiting) and (unit := picker.take(useful)) is not None:
        seats += unit.seats


@dataclass(frozen=True)
class RunResult:
    """Result of ``run_engine``."""

    engine: Engine
    records: tuple[LogRecord, ...]
    outcomes: RunOutcomes
    points: int
    violations: tuple[tuple[float, str], ...]


def run_engine(
    scenario: Scenario,
    controller: Controller | None = None,
    *,
    params: SimParams | None = None,
    check_invariants: bool = True,
) -> RunResult:
    """Runs a scenario to the horizon.

    Args:
        scenario: Scenario.
        controller: Controller; defaults to ``NearestUnitController()``.
        params: Engine parameters; defaults to ``SimParams()``.
        check_invariants: Whether to collect invariant violations after every
            ``advance()`` and every ``handle()``.

    Returns:
        The engine, its log, outcomes, number of decision points and violations.
    """
    sink = MemoryEventSink()
    engine = Engine(scenario, params or SimParams(), sink)
    controller = controller or NearestUnitController()
    violations: list[tuple[float, str]] = []
    points = 0
    while (point := engine.advance()) is not None:
        points += 1
        if check_invariants:
            violations += [(engine.now, v) for v in engine.check_invariants()]
        controller.handle(point, engine)
        if check_invariants:
            violations += [(engine.now, v) for v in engine.check_invariants()]
    if check_invariants:
        violations += [(engine.now, v) for v in engine.check_invariants()]
    return RunResult(engine, sink.records, engine.outcomes(), points, tuple(violations))


def _events(records: tuple[LogRecord, ...], kind: SimEventType) -> list[SimEventRecord]:
    return [r for r in records if isinstance(r, SimEventRecord) and r.event is kind]


@pytest.mark.parametrize("name", ["normal", "surge", "constrained"])
def test_invariants_hold_after_every_advance(real_scenarios: Mapping[str, Scenario], name: str) -> None:
    result = run_engine(real_scenarios[name])

    assert result.violations == ()
    assert result.points > 0


@pytest.mark.parametrize("name", ["normal", "surge", "constrained"])
def test_full_run_serves_victims_and_reaches_incidents(real_scenarios: Mapping[str, Scenario], name: str) -> None:
    result = run_engine(real_scenarios[name])
    victims = [v for v in result.outcomes.victims if v.initial_triage is not Triage.BLACK]
    served = [v for v in victims if v.care_start_min is not None and v.care_start_min <= 180.0]

    assert len(served) >= len(victims) // 2
    assert _events(result.records, SimEventType.CONTROLLER_OMISSION) == []
    early = [i for i in result.outcomes.incidents if i.t_report_min < 120.0]
    assert all(i.first_unit_on_scene_min is not None for i in early)


@pytest.mark.parametrize("name", ["normal", "surge", "constrained"])
def test_log_ordering_and_snapshot_minutes(real_scenarios: Mapping[str, Scenario], name: str) -> None:
    records = run_engine(real_scenarios[name], check_invariants=False).records
    times = [r.t_min for r in records]
    snapshots = [r.t_min for r in records if isinstance(r, SnapshotRecord)]

    assert [r.seq for r in records] == list(range(len(records)))
    assert times == sorted(times)
    assert snapshots == [float(k) for k in range(181)]


@pytest.mark.parametrize("name", ["normal", "surge", "constrained"])
def test_two_runs_give_byte_identical_jsonl(
    real_scenarios: Mapping[str, Scenario], name: str, tmp_path: Path
) -> None:
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_jsonl(run_engine(real_scenarios[name], check_invariants=False).records, first)
    write_jsonl(run_engine(real_scenarios[name], check_invariants=False).records, second)

    assert first.read_bytes() == second.read_bytes()
    assert len(read_log(first)) > 181


def test_outcomes_are_identical_across_runs(real_scenarios: Mapping[str, Scenario]) -> None:
    a = run_engine(real_scenarios["surge"], check_invariants=False).outcomes
    b = run_engine(real_scenarios["surge"], check_invariants=False).outcomes

    assert a.model_dump_json() == b.model_dump_json()


def test_guard_only_run_keeps_invariants(real_scenarios: Mapping[str, Scenario]) -> None:
    result = run_engine(real_scenarios["surge"], NearestUnitController(reserve=False, transport=False))

    assert result.violations == ()
    assert _events(result.records, SimEventType.CONTROLLER_OMISSION)
