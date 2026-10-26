"""Tests of ``evaluation.report`` and the runner's result persistence on synthetic results."""

import math
from pathlib import Path

import pandas as pd
import pytest

from emergency_sim.domain.enums import Condition
from emergency_sim.evaluation import report, runner
from emergency_sim.evaluation.runner import RunResult, RunSpec
from tests import factories as f


def _result(
    condition: Condition, seed: int, *, scenario: str = "surge", backend: str = "offline",
    model: str | None = None, **metrics: object,
) -> RunResult:
    spec = RunSpec(scenario, seed, condition, backend, model)
    return RunResult(spec=spec, metrics=f.zero_metrics(**metrics), log_path=None)


def _row(frame: pd.DataFrame, **match: object) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column, value in match.items():
        mask &= frame[column] == value
    rows = frame[mask]
    assert len(rows) == 1, (match, rows)
    return rows.iloc[0]


# --------------------------------------------------------------------------- sign test


@pytest.mark.parametrize(
    ("wins", "losses", "expected"),
    [(5, 0, 0.0625), (0, 5, 0.0625), (0, 0, 1.0), (1, 1, 1.0), (10, 0, 2 / 1024), (8, 2, 2 * 56 / 1024)],
)
def test_sign_test_p(wins: int, losses: int, expected: float) -> None:
    assert report.sign_test_p(wins, losses) == pytest.approx(expected)


# --------------------------------------------------------------------------- summarise


def test_summarise_mean_std_min_max() -> None:
    results = [_result(Condition.FULL, seed, served_rate=v) for seed, v in zip((1, 2, 3), (0.5, 0.7, 0.9))]
    summary = report.summarise(results)
    assert list(summary.columns) == list(report.SUMMARY_COLUMNS)
    row = _row(summary, scenario="surge", condition="full", backend="offline", metric="served_rate")
    assert row["n"] == 3
    assert row["mean"] == pytest.approx(0.7)
    assert row["std"] == pytest.approx(0.2)  # ddof=1
    assert (row["min"], row["max"]) == (pytest.approx(0.5), pytest.approx(0.9))


def test_summarise_skips_none_and_flattens_dicts() -> None:
    results = [
        _result(Condition.B0, 1, red_in_window_rate=0.5, messages_by_type={"PROPOSE": 4}),
        _result(Condition.B0, 2, red_in_window_rate=None, messages_by_type={}),
    ]
    summary = report.summarise(results)
    rate = _row(summary, condition="b0", metric="red_in_window_rate")
    assert rate["n"] == 1
    assert math.isnan(rate["std"])
    propose = _row(summary, condition="b0", metric="messages_by_type.PROPOSE")
    assert propose["n"] == 2  # the missing key counts as 0
    assert propose["mean"] == pytest.approx(2.0)


def test_summarise_orders_scenarios_and_conditions() -> None:
    results = [
        _result(Condition.FULL, 1, scenario="surge"),
        _result(Condition.B0, 1, scenario="surge"),
        _result(Condition.B1, 1, scenario="normal"),
    ]
    summary = report.summarise(results)
    groups = list(dict.fromkeys(zip(summary["scenario"], summary["condition"])))
    assert groups == [("normal", "b1"), ("surge", "b0"), ("surge", "full")]


def test_summarise_empty() -> None:
    assert report.summarise([]).empty


# --------------------------------------------------------------------------- paired differences


def _paired_fixture() -> list[RunResult]:
    results = []
    for seed in range(1, 6):
        results.append(_result(Condition.B0, seed, served_rate=0.5, response_time_mean_min=10.0, conflicts=0))
        results.append(_result(
            Condition.FULL, seed, served_rate=0.5 + seed / 100, response_time_mean_min=9.0,
            conflicts=1 if seed > 3 else 0,
        ))
    results.append(_result(Condition.FULL, 99, served_rate=1.0))  # unpaired: ignored
    return results


def test_paired_wins_ties_losses_and_sign_test() -> None:
    paired = report.paired_differences(_paired_fixture(), pairs=((Condition.FULL, Condition.B0),))
    assert list(paired.columns) == list(report.PAIRED_COLUMNS)
    served = _row(paired, metric="served_rate")
    assert (served["treatment"], served["baseline"], served["n"]) == ("full", "b0", 5)
    assert (served["wins"], served["ties"], served["losses"]) == (5, 0, 0)
    assert served["sign_test_p"] == pytest.approx(0.0625)
    assert served["mean_diff"] == pytest.approx(0.03)
    assert served["std_diff"] == pytest.approx(pd.Series([0.01, 0.02, 0.03, 0.04, 0.05]).std(ddof=1))


def test_paired_direction_lower_is_better_for_other_metrics() -> None:
    paired = report.paired_differences(_paired_fixture(), pairs=((Condition.FULL, Condition.B0),))
    response = _row(paired, metric="response_time_mean_min")
    assert (response["wins"], response["losses"]) == (5, 0)
    assert response["mean_diff"] == pytest.approx(-1.0)
    conflicts = _row(paired, metric="conflicts")
    assert (conflicts["wins"], conflicts["ties"], conflicts["losses"]) == (0, 3, 2)
    assert conflicts["sign_test_p"] == pytest.approx(0.5)


def test_paired_default_pairs_and_missing_conditions() -> None:
    paired = report.paired_differences(_paired_fixture())
    assert set(zip(paired["treatment"], paired["baseline"])) == {("full", "b0")}  # no B1 runs


def test_paired_does_not_mix_backends() -> None:
    results = [
        _result(Condition.B0, 1, served_rate=0.2),
        _result(Condition.FULL, 1, backend="llm", model="m", served_rate=0.9),
    ]
    assert report.paired_differences(results).empty


# --------------------------------------------------------------------------- persistence and report files


def _save(results: list[RunResult], out: Path) -> None:
    for result in results:
        label = runner.backend_label(result.spec)
        runner.write_result(result, runner.run_dir(out, result.spec, label), label)


def test_metrics_json_round_trip_and_load_results(tmp_path: Path) -> None:
    results = _paired_fixture()
    _save(results, tmp_path)
    loaded = report.load_results(tmp_path)
    assert len(loaded) == len(results)
    key = lambda r: (r.spec.condition.value, r.spec.seed)  # noqa: E731
    for original, back in zip(sorted(results, key=key), sorted(loaded, key=key)):
        assert back.spec == original.spec
        assert back.metrics == original.metrics
        assert back.log_path is None


def test_load_results_finds_events_file(tmp_path: Path) -> None:
    result = _result(Condition.B1, 7380)
    directory = runner.run_dir(tmp_path, result.spec, "offline")
    runner.write_result(result, directory, "offline")
    (directory / runner.EVENTS_FILE).write_text("", encoding="utf-8")
    assert report.load_results(tmp_path)[0].log_path == directory / runner.EVENTS_FILE


def test_metrics_json_has_no_absolute_paths(tmp_path: Path) -> None:
    result = RunResult(RunSpec("surge", 1, Condition.FULL), f.zero_metrics(), tmp_path / "x" / "events.jsonl")
    path = runner.write_result(result, tmp_path / "x", "offline")
    text = path.read_text(encoding="utf-8")
    assert str(tmp_path) not in text
    assert '"events_file": "events.jsonl"' in text
    assert '"run_id": "surge-full-offline-s1-r0"' in text


def test_load_results_missing_dir(tmp_path: Path) -> None:
    assert report.load_results(tmp_path / "nope") == []


def test_write_report_files_and_determinism(tmp_path: Path) -> None:
    results = _paired_fixture()
    report.write_report(results, tmp_path / "a")
    report.write_report(list(reversed(results)), tmp_path / "b")
    for name in (report.SUMMARY_CSV, report.PAIRED_CSV, report.SUMMARY_MD):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    summary = pd.read_csv(tmp_path / "a" / report.SUMMARY_CSV)
    assert set(summary["condition"]) == {"b0", "full"}
    markdown = (tmp_path / "a" / report.SUMMARY_MD).read_text(encoding="utf-8")
    assert markdown.startswith("# Evaluation summary")
    assert "## surge (offline)" in markdown
    assert "| full | 6 |" in markdown
    assert "| full vs b0 | served_rate | 5 | +0.030 | 5/0/0 | 0.0625 |" in markdown
