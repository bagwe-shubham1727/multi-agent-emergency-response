"""Tests of the CLI and the runner's grid wiring, with ``run_simulation`` monkeypatched.

No engine, controller or SDK backend is needed: the fake ``run_simulation``
records its calls and writes ``metrics.json`` the way the real one does.
"""

import argparse
from pathlib import Path

import pytest

from emergency_sim import cli
from emergency_sim.domain.enums import Condition
from emergency_sim.evaluation import report, runner
from emergency_sim.evaluation.runner import RunResult, RunSpec
from tests import factories as f


class FakeRunSimulation:
    """Stands in for ``runner.run_simulation``; records calls, writes metrics.json."""

    def __init__(self) -> None:
        self.calls: list[tuple[RunSpec, Path | None, Path | None]] = []

    def __call__(
        self, spec: RunSpec, *, log_path: Path | None = None, result_dir: Path | None = None, **_: object
    ) -> RunResult:
        self.calls.append((spec, log_path, result_dir))
        served = 0.5 if spec.condition == Condition.B0 else 0.75
        result = RunResult(spec, f.zero_metrics(served_rate=served, patients_total=4), log_path)
        if result_dir is not None:
            runner.write_result(result, result_dir, runner.backend_label(spec))
        return result


@pytest.fixture
def fake_run(monkeypatch: pytest.MonkeyPatch) -> FakeRunSimulation:
    fake = FakeRunSimulation()
    monkeypatch.setattr(runner, "run_simulation", fake)
    return fake


# --------------------------------------------------------------------------- seeds and parsing


def test_parse_seeds_expands_ranges() -> None:
    assert cli.parse_seeds(["7380", "1001-1003"]) == [7380, 1001, 1002, 1003]


def test_parse_seeds_drops_duplicates_keeping_order() -> None:
    assert cli.parse_seeds(["3", "1-4"]) == [3, 1, 2, 4]


@pytest.mark.parametrize("token", ["abc", "5-2", "1-x", ""])
def test_parse_seed_token_rejects_bad_tokens(token: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        cli.parse_seed_token(token)


def test_grid_parser_defaults() -> None:
    args = cli.build_parser().parse_args(["grid"])
    assert args.scenarios == ["normal", "surge", "constrained"]
    assert args.conditions == ["b0", "b0-cues", "b1", "full"]
    assert args.backend == "offline"
    assert args.repeats == 1
    assert args.out == Path("results")


def test_grid_parser_seed_ranges() -> None:
    args = cli.build_parser().parse_args(["grid", "--seeds", "7380", "1001-1003"])
    assert cli.flatten_seeds(args.seeds) == [7380, 1001, 1002, 1003]


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["run", "--scenario", "surge"],
        ["run", "--scenario", "nowhere", "--condition", "full"],
        ["run", "--scenario", "surge", "--condition", "b2"],
        ["grid", "--seeds", "9-1"],
        ["grid", "--backend", "magic"],
    ],
)
def test_usage_errors_exit_2(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(argv)
    assert excinfo.value.code == 2


def test_grid_specs_order() -> None:
    specs = cli.grid_specs(["surge"], ["b0", "full"], [1, 2], 2, "offline", None)
    assert len(specs) == 8
    assert specs[0] == RunSpec("surge", 1, Condition.B0, "offline", None, 0)
    assert specs[1] == RunSpec("surge", 1, Condition.FULL, "offline", None, 0)
    assert specs[2].repeat == 1
    assert specs[-1] == RunSpec("surge", 2, Condition.FULL, "offline", None, 1)


# --------------------------------------------------------------------------- run / grid / report commands


def test_run_command(fake_run: FakeRunSimulation, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main([
        "run", "--scenario", "surge", "--condition", "full", "--seed", "11", "--out", str(tmp_path),
    ])
    assert code == 0
    (spec, log_path, result_dir), = fake_run.calls
    assert spec == RunSpec("surge", 11, Condition.FULL, "offline", None, 0)
    expected_dir = tmp_path / "runs" / "surge" / "full" / "offline" / "seed-11_r0"
    assert result_dir == expected_dir
    assert log_path == expected_dir / "events.jsonl"
    assert (expected_dir / "metrics.json").is_file()
    assert "surge/full/offline seed 11 r0" in capsys.readouterr().out


def test_run_command_llm_model_label(fake_run: FakeRunSimulation, tmp_path: Path) -> None:
    code = cli.main([
        "run", "--scenario", "normal", "--condition", "b1", "--backend", "llm", "--model", "gpt-test",
        "--repeat", "2", "--out", str(tmp_path),
    ])
    assert code == 0
    spec, _, result_dir = fake_run.calls[0]
    assert (spec.backend, spec.model, spec.repeat) == ("llm", "gpt-test", 2)
    assert result_dir == tmp_path / "runs" / "normal" / "b1" / "llm-gpt-test" / "seed-7380_r2"


def test_grid_command_runs_keeps_logs_and_reports(
    fake_run: FakeRunSimulation, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main([
        "grid", "--scenarios", "normal", "surge", "--conditions", "b0", "full",
        "--seeds", "7380", "1001-1002", "--keep-logs-for", "7380", "--out", str(tmp_path),
    ])
    assert code == 0
    assert len(fake_run.calls) == 2 * 2 * 3
    for spec, log_path, result_dir in fake_run.calls:
        assert result_dir == runner.run_dir(tmp_path, spec, "offline")
        assert (log_path is not None) == (spec.seed == 7380)
    out = capsys.readouterr().out
    assert "[12/12]" in out
    for name in (report.SUMMARY_CSV, report.PAIRED_CSV, report.SUMMARY_MD):
        assert (tmp_path / name).is_file()
    assert "| full vs b0 | served_rate | 3 | +0.250 | 3/0/0 | 0.2500 |" in (tmp_path / report.SUMMARY_MD).read_text()


def test_grid_default_keeps_logs_for_7380_only(fake_run: FakeRunSimulation, tmp_path: Path) -> None:
    assert cli.main(["grid", "--scenarios", "surge", "--conditions", "b0", "--seeds", "7380", "1",
                     "--out", str(tmp_path)]) == 0
    kept = {spec.seed for spec, log_path, _ in fake_run.calls if log_path is not None}
    assert kept == {7380}


def test_grid_without_kept_logs(fake_run: FakeRunSimulation, tmp_path: Path) -> None:
    assert cli.main(["grid", "--scenarios", "surge", "--conditions", "b0", "--seeds", "7380",
                     "--keep-logs-for", "--out", str(tmp_path)]) == 0
    assert all(log_path is None for _, log_path, _ in fake_run.calls)


def test_grid_rejects_zero_repeats(fake_run: FakeRunSimulation, tmp_path: Path) -> None:
    assert cli.main(["grid", "--repeats", "0", "--out", str(tmp_path)]) == 1
    assert fake_run.calls == []


def test_report_command_rebuilds_from_metrics_files(fake_run: FakeRunSimulation, tmp_path: Path) -> None:
    cli.main(["grid", "--scenarios", "surge", "--conditions", "b0", "full", "--seeds", "1-2",
              "--out", str(tmp_path)])
    (tmp_path / report.SUMMARY_MD).unlink()
    assert cli.main(["report", "--results", str(tmp_path)]) == 0
    assert (tmp_path / report.SUMMARY_MD).is_file()


def test_report_command_without_results_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["report", "--results", str(tmp_path)]) == 1
    assert "no metrics.json" in capsys.readouterr().err


def test_runtime_error_exits_1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                               capsys: pytest.CaptureFixture[str]) -> None:
    def boom(spec: RunSpec, **_: object) -> RunResult:
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(runner, "run_simulation", boom)
    code = cli.main(["run", "--scenario", "surge", "--condition", "b0", "--out", str(tmp_path)])
    assert code == 1
    assert "RuntimeError: engine exploded" in capsys.readouterr().err


# --------------------------------------------------------------------------- runner helpers


def test_run_dir_layout(tmp_path: Path) -> None:
    spec = RunSpec("constrained", 1005, Condition.FULL_NO_FLOAT, repeat=3)
    assert runner.run_dir(tmp_path, spec, "offline") == (
        tmp_path / "runs" / "constrained" / "full-no-float" / "offline" / "seed-1005_r3"
    )
    assert runner.run_dir(tmp_path, spec, "llm-org/model").parts[-2] == "llm-org_model"


def test_run_id_and_labels() -> None:
    spec = RunSpec("surge", 7380, Condition.B1, "llm", "gpt-x", 1)
    assert runner.backend_label(spec) == "llm-gpt-x"
    assert runner.run_id(spec, "llm-gpt-x") == "surge-b1-llm-gpt-x-s7380-r1"
    assert runner.backend_label(RunSpec("surge", 1, Condition.B0)) == "offline"
    assert runner.resolve_spec(spec) is spec


def test_build_controller_needs_backend_for_agent_conditions() -> None:
    with pytest.raises(ValueError, match="needs a decision backend"):
        runner.build_controller(Condition.FULL, None, _NullSink(), params=None)


def test_run_grid_calls_run_simulation_in_order(fake_run: FakeRunSimulation, tmp_path: Path) -> None:
    specs = [RunSpec("surge", s, c) for s in (1, 2) for c in (Condition.B0, Condition.FULL)]
    seen: list[int] = []
    results = runner.run_grid(
        specs, tmp_path, keep_logs_for_seeds=frozenset({2}), progress=lambda i, n, r: seen.append(i)
    )
    assert [r.spec for r in results] == specs
    assert seen == [1, 2, 3, 4]
    assert [log is not None for _, log, _ in fake_run.calls] == [False, False, True, True]
    assert len(report.load_results(tmp_path)) == 4


class _NullSink:
    """Minimal ``EventSink`` for wiring tests."""

    def emit(self, record: object) -> None:
        """Discards the record."""
