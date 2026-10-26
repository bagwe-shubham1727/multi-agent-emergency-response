"""Tests for ``emergency_sim.scenarios`` (loading committed data and regenerating seeds)."""

import json
import shutil
from collections.abc import Mapping
from pathlib import Path

import pydantic
import pytest

from emergency_sim.domain.enums import Triage
from emergency_sim.domain.scenario import Scenario
from emergency_sim.scenarios import DEFAULT_DATA_ROOT, load_scenario, scenario_for
from tests.conftest import SCENARIO_NAMES

EXPECTED_COUNTS = {"normal": (11, 21), "surge": (14, 39), "constrained": (14, 39)}


def _summary(data_root: Path, name: str) -> dict:
    return json.loads((data_root / name / "summary.json").read_text())


def test_default_data_root_is_repo_data() -> None:
    assert DEFAULT_DATA_ROOT.name == "data"
    assert (DEFAULT_DATA_ROOT / "normal" / "manifest.json").is_file()


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_committed_scenarios_match_summary(
    name: str, real_scenarios: Mapping[str, Scenario], data_root: Path
) -> None:
    scenario = real_scenarios[name]
    summary = _summary(data_root, name)
    victims = [v for i in scenario.incidents for v in i.victims]

    assert (len(scenario.incidents), len(victims)) == EXPECTED_COUNTS[name]
    assert len(scenario.incidents) == summary["incidents"]
    assert len(victims) == summary["victims"]
    assert len(scenario.hospitals) == summary["hospitals"]
    assert len(scenario.ambulances) == summary["ambulances"]
    assert len(scenario.responders) == summary["responder_teams"]
    assert sum(s.role == "doctor" for s in scenario.staff) == summary["doctors"]
    assert sum(s.role == "nurse" for s in scenario.staff) == summary["nurses"]
    assert sum(s.home == "FLOAT" for s in scenario.staff) == summary["float_pool"]
    assert sum(h.er_capacity for h in scenario.hospitals) == summary["er_beds_total"]
    for triage in Triage:
        assert sum(v.triage == triage for v in victims) == summary["victims_by_triage"][triage]
    assert scenario.name == name
    assert scenario.seed == 7380
    assert scenario.horizon_min == 180.0


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_scenario_for_committed_seed_reads_data(
    name: str, real_scenarios: Mapping[str, Scenario]
) -> None:
    assert scenario_for(name, 7380) == real_scenarios[name]


def test_scenario_for_regenerates_committed_seed_identically(
    tmp_path: Path, real_scenarios: Mapping[str, Scenario]
) -> None:
    # tmp_path has no committed data, so this goes through datagen.
    assert scenario_for("normal", 7380, data_root=tmp_path) == real_scenarios["normal"]
    assert list(tmp_path.iterdir()) == []


def test_scenario_for_other_seed_is_deterministic(real_scenarios: Mapping[str, Scenario]) -> None:
    first = scenario_for("surge", 1)
    second = scenario_for("surge", 1)

    assert first == second
    assert first.seed == 1
    assert first.name == "surge"
    assert first != real_scenarios["surge"]


def test_scenario_for_unknown_name_raises_key_error() -> None:
    with pytest.raises(KeyError):
        scenario_for("hurricane", 7380)


def test_load_scenario_rejects_unknown_fields(tmp_path: Path, data_root: Path) -> None:
    target = tmp_path / "normal"
    shutil.copytree(data_root / "normal", target)
    hospitals = json.loads((target / "hospitals.json").read_text())
    hospitals[0]["helipad"] = True
    (target / "hospitals.json").write_text(json.dumps(hospitals))

    with pytest.raises(pydantic.ValidationError):
        load_scenario(target)


def test_load_scenario_missing_file_raises(tmp_path: Path, data_root: Path) -> None:
    target = tmp_path / "normal"
    shutil.copytree(data_root / "normal", target)
    (target / "staff.json").unlink()

    with pytest.raises(FileNotFoundError):
        load_scenario(target)


def test_scenario_is_immutable(real_scenarios: Mapping[str, Scenario]) -> None:
    with pytest.raises(pydantic.ValidationError):
        real_scenarios["normal"].hospitals[0].er_capacity = 99  # type: ignore[misc]
