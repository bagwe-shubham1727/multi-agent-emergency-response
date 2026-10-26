"""Command-line interface: ``python -m emergency_sim.cli {run,grid,report}``.

Examples::

    python -m emergency_sim.cli run --scenario surge --condition full
    python -m emergency_sim.cli grid --conditions b0 b0-cues b1 full --seeds 7380 1001-1020
    python -m emergency_sim.cli report --results results

``run`` keeps the event log of its single run; ``grid`` keeps logs only for
``--keep-logs-for`` seeds and writes the report at the end; ``report``
rebuilds the report from the ``metrics.json`` files. Exit status is 0 on
success, 1 on a runtime error and 2 on a usage error.
"""

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Final

from emergency_sim.domain.enums import Condition
from emergency_sim.evaluation import report, runner
from emergency_sim.evaluation.runner import RunResult, RunSpec

SCENARIOS: Final = ("normal", "surge", "constrained")
CONDITIONS: Final = tuple(c.value for c in Condition)
DEFAULT_SEED: Final = 7380
DEFAULT_GRID_SEEDS: Final = ("7380", "1001-1020")
DEFAULT_GRID_CONDITIONS: Final = ("b0", "b0-cues", "b1", "full")
DEFAULT_OUT: Final = Path("results")
EXIT_OK: Final = 0
EXIT_ERROR: Final = 1


# --------------------------------------------------------------------------- argument parsing


def parse_seed_token(token: str) -> list[int]:
    """Parses one seed token: ``"7380"`` or an inclusive range ``"1001-1020"``.

    Args:
        token: The token.

    Returns:
        The seeds it denotes, ascending.

    Raises:
        argparse.ArgumentTypeError: If the token is malformed or the range is empty.
    """
    try:
        if "-" in token.strip("-"):
            low, high = (int(part) for part in token.split("-", 1))
            if low > high:
                raise argparse.ArgumentTypeError(f"empty seed range {token!r}")
            return list(range(low, high + 1))
        return [int(token)]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid seed {token!r}") from error


def flatten_seeds(groups: Sequence[Sequence[int]]) -> list[int]:
    """Flattens parsed seed tokens, dropping duplicates and keeping first-seen order.

    Args:
        groups: Seeds per token.

    Returns:
        The seeds.
    """
    return list(dict.fromkeys(seed for group in groups for seed in group))


def parse_seeds(tokens: Sequence[str]) -> list[int]:
    """Parses seed tokens such as ``["7380", "1001-1003"]``.

    Args:
        tokens: Seed tokens.

    Returns:
        ``[7380, 1001, 1002, 1003]`` for the example above.

    Raises:
        argparse.ArgumentTypeError: If a token is malformed.
    """
    return flatten_seeds([parse_seed_token(token) for token in tokens])


def _add_backend_args(parser: argparse.ArgumentParser) -> None:
    """Adds ``--backend`` and ``--model``."""
    parser.add_argument("--backend", choices=runner.BACKEND_KINDS, default="offline")
    parser.add_argument("--model", default=None, help="LLM model (default: EMS_LLM_MODEL or the built-in default)")


def build_parser() -> argparse.ArgumentParser:
    """Builds the argument parser with the ``run``, ``grid`` and ``report`` commands.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(prog="python -m emergency_sim.cli", description=__doc__.splitlines()[0])
    parser.add_argument("-v", "--verbose", action="store_true", help="log progress details")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run one simulation")
    run.add_argument("--scenario", choices=SCENARIOS, required=True)
    run.add_argument("--condition", choices=CONDITIONS, required=True)
    run.add_argument("--seed", type=int, default=DEFAULT_SEED)
    _add_backend_args(run)
    run.add_argument("--repeat", type=int, default=0)
    run.add_argument("--out", type=Path, default=DEFAULT_OUT)

    grid = commands.add_parser("grid", help="run scenarios x conditions x seeds x repeats, then report")
    grid.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    grid.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(DEFAULT_GRID_CONDITIONS))
    grid.add_argument("--seeds", nargs="+", type=parse_seed_token, default=None, help="e.g. 7380 1001-1020")
    _add_backend_args(grid)
    grid.add_argument("--repeats", type=int, default=1)
    grid.add_argument("--keep-logs-for", nargs="*", type=parse_seed_token, default=None, help="seeds whose events.jsonl is kept")
    grid.add_argument("--out", type=Path, default=DEFAULT_OUT)

    rep = commands.add_parser("report", help="rebuild summary.csv, paired.csv and summary.md")
    rep.add_argument("--results", type=Path, default=DEFAULT_OUT)
    return parser


# --------------------------------------------------------------------------- commands


def grid_specs(
    scenarios: Sequence[str], conditions: Sequence[str], seeds: Sequence[int],
    repeats: int, backend: str, model: str | None,
) -> list[RunSpec]:
    """Builds the grid's run specs in a deterministic order.

    Args:
        scenarios: Scenario names.
        conditions: Condition values.
        seeds: Seeds.
        repeats: Repeats per (scenario, condition, seed).
        backend: Backend kind.
        model: LLM model name or None.

    Returns:
        Specs ordered by scenario, seed, repeat, then condition.
    """
    return [
        RunSpec(scenario, seed, Condition(condition), backend, model, repeat)
        for scenario in scenarios
        for seed in seeds
        for repeat in range(repeats)
        for condition in conditions
    ]


def describe(result: RunResult) -> str:
    """One-line summary of a run's headline metrics.

    Args:
        result: The run result.

    Returns:
        The line.
    """
    s, m = result.spec, result.metrics
    rate = "-" if m.red_in_window_rate is None else f"{m.red_in_window_rate:.2f}"
    return (
        f"{s.scenario}/{s.condition.value}/{runner.backend_label(s)} seed {s.seed} r{s.repeat}: "
        f"served {m.victims_served}/{m.patients_total} ({m.served_rate:.2f}), RED in window {rate}, "
        f"score {m.survival_weighted_score:.3f}, messages {m.messages_total}, fallbacks {m.fallbacks}"
    )


def cmd_run(args: argparse.Namespace) -> int:
    """Runs one simulation and keeps its event log.

    Args:
        args: Parsed ``run`` arguments.

    Returns:
        Exit status.
    """
    spec = runner.resolve_spec(
        RunSpec(args.scenario, args.seed, Condition(args.condition), args.backend, args.model, args.repeat)
    )
    directory = runner.run_dir(args.out, spec, runner.backend_label(spec))
    result = runner.run_simulation(spec, log_path=directory / runner.EVENTS_FILE, result_dir=directory)
    print(describe(result))
    print(f"wrote {directory}")
    return EXIT_OK


def _print_progress(index: int, total: int, result: RunResult) -> None:
    """Prints one grid progress line."""
    print(f"[{index}/{total}] {describe(result)}", flush=True)


def cmd_grid(args: argparse.Namespace) -> int:
    """Runs the evaluation grid and writes the report.

    Args:
        args: Parsed ``grid`` arguments.

    Returns:
        Exit status.
    """
    seeds = flatten_seeds(args.seeds) if args.seeds else parse_seeds(DEFAULT_GRID_SEEDS)
    keep = flatten_seeds(args.keep_logs_for) if args.keep_logs_for is not None else [DEFAULT_SEED]
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    specs = grid_specs(args.scenarios, args.conditions, seeds, args.repeats, args.backend, args.model)
    print(f"grid: {len(specs)} runs ({len(args.scenarios)} scenarios x {len(args.conditions)} conditions "
          f"x {len(seeds)} seeds x {args.repeats} repeats), backend {args.backend}, out {args.out}")
    results = runner.run_grid(specs, args.out, keep_logs_for_seeds=frozenset(keep), progress=_print_progress)
    report.write_report(results, args.out)
    print(f"wrote report to {args.out}")
    return EXIT_OK


def cmd_report(args: argparse.Namespace) -> int:
    """Rebuilds the report from saved ``metrics.json`` files.

    Args:
        args: Parsed ``report`` arguments.

    Returns:
        Exit status (1 if no results were found).
    """
    results = report.load_results(args.results)
    if not results:
        print(f"error: no metrics.json found under {args.results}", file=sys.stderr)
        return EXIT_ERROR
    report.write_report(results, args.results)
    print(f"{len(results)} runs -> {args.results / report.SUMMARY_MD}")
    return EXIT_OK


COMMANDS: Final = MappingProxyType({"run": cmd_run, "grid": cmd_grid, "report": cmd_report})


def main(argv: Sequence[str] | None = None) -> int:
    """Parses arguments and runs a command.

    Args:
        argv: Arguments without the program name; None uses ``sys.argv``.

    Returns:
        Exit status: 0 on success, 1 on a runtime error (usage errors exit 2
        from argparse).
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        return COMMANDS[args.command](args)
    except Exception as error:  # top-level boundary: report and exit non-zero
        if args.verbose:
            logging.getLogger(__name__).exception("command failed")
        print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
