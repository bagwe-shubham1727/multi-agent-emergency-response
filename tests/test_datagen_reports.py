"""Tests of the seeded caller-report wording noise in ``emergency_sim.datagen``.

Hermetic: scenarios are generated in memory or into ``tmp_path``; the committed
``data/`` folder is never written.
"""

import json
import random
from pathlib import Path

import pytest

from emergency_sim import datagen, validate
from tests.conftest import SCENARIO_NAMES

NOISE_SEEDS = range(1, 41)
RATE_TOLERANCE = 0.06
PARITY_SEEDS = (7380, 1, 2, 3)


def _incidents(seed: int, name: str) -> list[dict]:
    return datagen.make_incidents(datagen.make_streams(seed), datagen.SCENARIOS[name])


def _generate_all(seed: int, out: Path) -> None:
    for name in SCENARIO_NAMES:
        datagen.generate_scenario(name, seed, out)


def _files(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def _without_report(incidents: list[dict]) -> list[dict]:
    return [{k: v for k, v in i.items() if k != "caller_report"} for i in incidents]


def test_generation_is_deterministic(tmp_path: Path) -> None:
    _generate_all(7380, tmp_path / "a")
    _generate_all(7380, tmp_path / "b")

    first, second = _files(tmp_path / "a"), _files(tmp_path / "b")
    assert first == second
    assert len(first) == 10 * len(SCENARIO_NAMES)


def test_generation_matches_committed_data(tmp_path: Path, data_root: Path) -> None:
    _generate_all(7380, tmp_path)

    for name in SCENARIO_NAMES:
        assert ((tmp_path / name / "incidents.json").read_bytes()
                == (data_root / name / "incidents.json").read_bytes())


def test_empirical_noise_rates_are_near_the_constants() -> None:
    totals = dict.fromkeys(("severe_red", "severe_not_red", "mild_red", "mild_not_red"), 0)
    for seed in NOISE_SEEDS:
        for name in SCENARIO_NAMES:
            counts = validate.wording_confusion(_incidents(seed, name))
            for key in totals:
                totals[key] += counts[key]
    red = totals["severe_red"] + totals["mild_red"]
    not_red = totals["severe_not_red"] + totals["mild_not_red"]

    assert red > 300 and not_red > 300
    assert abs(totals["mild_red"] / red - datagen.REPORT_UNDERTRIAGE_P) < RATE_TOLERANCE
    assert abs(totals["severe_not_red"] / not_red - datagen.REPORT_OVERTRIAGE_P) < RATE_TOLERANCE


@pytest.mark.parametrize("seed", PARITY_SEEDS)
def test_noise_changes_only_the_caller_report(
    seed: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _generate_all(seed, tmp_path / "noisy")
    monkeypatch.setattr(datagen, "REPORT_UNDERTRIAGE_P", 0.0)
    monkeypatch.setattr(datagen, "REPORT_OVERTRIAGE_P", 0.0)
    _generate_all(seed, tmp_path / "clean")

    noisy, clean = _files(tmp_path / "noisy"), _files(tmp_path / "clean")
    assert noisy.keys() == clean.keys()
    for path in noisy:
        if path.endswith(("incidents.json", "manifest.json")):
            continue
        assert noisy[path] == clean[path], path
    for name in SCENARIO_NAMES:
        noisy_inc = json.loads(noisy[f"{name}/incidents.json"])
        clean_inc = json.loads(clean[f"{name}/incidents.json"])
        assert _without_report(noisy_inc) == _without_report(clean_inc)
        # without noise the wording is the old deterministic rule: severe iff RED
        counts = validate.wording_confusion(clean_inc)
        assert counts["mild_red"] == counts["severe_not_red"] == 0
        noisy_sha = json.loads(noisy[f"{name}/manifest.json"])["sha256"]
        clean_sha = json.loads(clean[f"{name}/manifest.json"])["sha256"]
        assert {k: v for k, v in noisy_sha.items() if k != "incidents.json"} == {
            k: v for k, v in clean_sha.items() if k != "incidents.json"}


@pytest.mark.parametrize("seed", NOISE_SEEDS)
def test_shared_incidents_get_the_same_report_in_every_scenario(seed: int) -> None:
    normal, surge, constrained = (_incidents(seed, n) for n in ("normal", "surge", "constrained"))

    assert surge == constrained
    assert ([i["caller_report"] for i in surge if not i["mci_cluster"]]
            == [i["caller_report"] for i in normal])


def test_caller_hazard_takes_one_draw_whatever_the_triage() -> None:
    spec = datagen.INCIDENT_TYPES["building_fire"]
    states = []
    for triage in datagen.TRIAGE_LEVELS:
        rng = random.Random("draws")
        datagen.caller_hazard(rng, spec, triage)
        states.append(rng.getstate())

    assert all(state == states[0] for state in states)


def test_caller_hazard_extremes(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = datagen.INCIDENT_TYPES["assault"]
    rng = random.Random(0)
    monkeypatch.setattr(datagen, "REPORT_UNDERTRIAGE_P", 1.0)
    monkeypatch.setattr(datagen, "REPORT_OVERTRIAGE_P", 1.0)

    assert all(datagen.caller_hazard(rng, spec, "RED") == spec["hazard_mild"] for _ in range(50))
    assert all(datagen.caller_hazard(rng, spec, "YELLOW") == spec["hazard"] for _ in range(50))


def test_validator_flags_a_shared_incident_with_a_different_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _generate_all(7380, tmp_path)
    monkeypatch.setattr(validate, "DATA", tmp_path)
    check_name = "shared incidents have identical caller reports"

    ok = {c["check"]: c["passed"] for c in validate.check_controlled({})}
    path = tmp_path / "constrained" / "incidents.json"
    incidents = json.loads(path.read_text())
    incidents[0]["caller_report"] += " Extra."
    path.write_text(json.dumps(incidents))
    tampered = {c["check"]: c["passed"] for c in validate.check_controlled({})}

    assert ok[check_name] is True
    assert tampered[check_name] is False


def test_wording_confusion_counts_every_incident() -> None:
    incidents = _incidents(7380, "surge")
    counts = validate.wording_confusion(incidents)

    assert sum(counts.values()) == len(incidents)
    assert counts["same_wording"] == sum(i["type"] == "stroke" for i in incidents)
