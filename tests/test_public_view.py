"""The information boundary (plan section 3.7): ``PublicView`` never leaks hidden truth.

Checks every snapshot and every decision-point view of full runs of the real
scenarios: no hidden keys anywhere in the serialized JSON, no victims of
untriaged incidents, no unreported incidents, and every collection sorted.
"""

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import pytest

from emergency_sim.domain.log_records import SnapshotRecord
from emergency_sim.domain.ports import EnginePort
from emergency_sim.domain.scenario import Scenario
from emergency_sim.domain.views import DecisionPoint, PublicView
from emergency_sim.engine import Engine
from tests import factories as f
from tests.test_engine_determinism import NearestUnitController, run_engine
from tests.test_engine_lifecycle import make_engine

HIDDEN_KEYS = frozenset(
    {"golden_window_min", "deteriorates_to_red_at_min", "true_victims", "worst_triage", "time_sensitivity_min"}
)


@dataclass
class RecordingController:
    """Wraps the stub controller and records the view at every decision point."""

    inner: NearestUnitController = field(default_factory=NearestUnitController)
    views: list[PublicView] = field(default_factory=list)

    def handle(self, point: DecisionPoint, port: EnginePort) -> None:
        self.views.append(port.view())
        self.inner.handle(point, port)


def _keys(obj: Any) -> Iterator[str]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key
            yield from _keys(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _keys(item)


def _all_views(scenario: Scenario) -> list[PublicView]:
    controller = RecordingController()
    result = run_engine(scenario, controller, check_invariants=False)
    snapshots = [r.view for r in result.records if isinstance(r, SnapshotRecord)]
    return [*controller.views, *snapshots]


@pytest.fixture(scope="module")
def views_by_scenario(real_scenarios: Mapping[str, Scenario]) -> Mapping[str, list[PublicView]]:
    return {name: _all_views(s) for name, s in real_scenarios.items()}


@pytest.mark.parametrize("name", ["normal", "surge", "constrained"])
def test_serialized_view_has_no_hidden_keys(views_by_scenario: Mapping[str, list[PublicView]], name: str) -> None:
    for view in views_by_scenario[name]:
        assert HIDDEN_KEYS.isdisjoint(_keys(json.loads(view.model_dump_json())))


@pytest.mark.parametrize("name", ["normal", "surge", "constrained"])
def test_untriaged_incidents_show_no_victims(views_by_scenario: Mapping[str, list[PublicView]], name: str) -> None:
    for view in views_by_scenario[name]:
        for incident in view.incidents:
            if not incident.triaged:
                assert incident.victims == ()
            else:
                assert incident.victims


@pytest.mark.parametrize("name", ["normal", "surge", "constrained"])
def test_only_reported_incidents_are_visible(
    views_by_scenario: Mapping[str, list[PublicView]], real_scenarios: Mapping[str, Scenario], name: str
) -> None:
    report = {i.incident_id: i.t_report_min for i in real_scenarios[name].incidents}
    for view in views_by_scenario[name]:
        visible = {i.incident_id for i in view.incidents}
        assert visible == {iid for iid, t in report.items() if t <= view.t_min}


@pytest.mark.parametrize("name", ["normal", "surge", "constrained"])
def test_every_collection_is_sorted_by_id(views_by_scenario: Mapping[str, list[PublicView]], name: str) -> None:
    def ids(items: Any, attr: str) -> list[str]:
        return [getattr(x, attr) for x in items]

    for view in views_by_scenario[name]:
        assert ids(view.incidents, "incident_id") == sorted(ids(view.incidents, "incident_id"))
        assert ids(view.units, "unit_id") == sorted(ids(view.units, "unit_id"))
        assert ids(view.hospitals, "hospital_id") == sorted(ids(view.hospitals, "hospital_id"))
        assert ids(view.float_pool, "staff_id") == sorted(ids(view.float_pool, "staff_id"))
        for hospital in view.hospitals:
            assert ids(hospital.staff, "staff_id") == sorted(ids(hospital.staff, "staff_id"))
        for incident in view.incidents:
            assert ids(incident.victims, "victim_id") == sorted(ids(incident.victims, "victim_id"))


def test_bed_board_is_consistent(views_by_scenario: Mapping[str, list[PublicView]]) -> None:
    for view in views_by_scenario["constrained"]:
        for h in view.hospitals:
            assert h.beds_occupied + h.beds_reserved + h.beds_free_unreserved == h.er_capacity
            assert h.beds_free_unreserved >= 0
            assert h.waiting_for_staff + h.in_treatment <= h.beds_occupied


def test_view_before_any_report_is_empty_of_incidents() -> None:
    scenario = f.make_scenario(
        hospitals=[f.hospital("H1")],
        ambulances=[f.ambulance("AMB-01")],
        incidents=[f.incident("INC-01", 5, 7.0, 5.0, [f.victim("V1", "RED", window=25)])],
    )
    engine, _ = make_engine(scenario)

    assert engine.view().incidents == ()
    point = engine.advance()
    assert point is not None and point.t_min == 5.0
    incident = engine.view().incidents[0]
    assert (incident.triaged, incident.victims, incident.reported_victims) == (False, (), 1)


def test_engine_satisfies_the_engine_port() -> None:
    engine, _ = make_engine(f.tiny_scenario())

    assert isinstance(engine, Engine) and isinstance(engine, EnginePort)
