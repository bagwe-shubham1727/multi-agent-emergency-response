"""Cross-seed summaries, paired comparisons and the written evaluation report.

Every scalar metric of ``RunMetrics`` is reported; dict-valued metrics are
flattened to ``name.key`` columns (a missing key counts as 0). ``None``
values (e.g. no RED patient) are left out of that metric's sample, so ``n``
can differ between metrics.

Paired comparisons match runs on identical worlds: same scenario, backend
label, seed and repeat. A *win* is a difference in the better direction:
higher for ``HIGHER_IS_BETTER`` metrics, lower for every other metric. The
sign test is the exact two-sided binomial test on wins vs losses (ties
dropped).
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Final

import pandas as pd

from emergency_sim.domain.enums import Condition
from emergency_sim.domain.outcomes import RunMetrics
from emergency_sim.evaluation.runner import (
    METRICS_FILE,
    RunResult,
    backend_label,
    read_result,
)

HIGHER_IS_BETTER: Final = frozenset(
    {"served_rate", "victims_served", "red_in_window", "red_in_window_rate", "survival_weighted_score"}
)
DEFAULT_PAIRS: Final = (
    (Condition.FULL, Condition.B0),
    (Condition.FULL, Condition.B0_CUES),
    (Condition.FULL, Condition.B1),
    (Condition.B1, Condition.B0_CUES),
    (Condition.B0_CUES, Condition.B0),
    (Condition.B1, Condition.B0),
)
"""Paired comparisons in the report.

B0 -> B0-cues isolates the caller-report information, B0-cues -> B1 the
coordination between agents, and B1 -> Full the negotiation.
"""
SCENARIO_ORDER: Final = ("normal", "surge", "constrained")
CONDITION_ORDER: Final = tuple(c.value for c in Condition)
HEADLINE_METRICS: Final = (
    ("served_rate", "Served rate", "{:.3f}"),
    ("red_in_window_rate", "RED in window", "{:.3f}"),
    ("survival_weighted_score", "Survival score", "{:.3f}"),
    ("unserved_critical", "Unserved RED", "{:.2f}"),
    ("response_time_mean_min", "Response (min)", "{:.1f}"),
    ("door_to_doctor_mean_min", "Door-to-doctor (min)", "{:.1f}"),
    ("conflicts", "Conflicts", "{:.1f}"),
)
"""(metric, column title, number format) of the summary.md headline tables."""

TIE_TOLERANCE: Final = 1e-12
SUMMARY_COLUMNS: Final = ("scenario", "condition", "backend", "metric", "n", "mean", "std", "min", "max")
PAIRED_COLUMNS: Final = (
    "scenario", "backend", "treatment", "baseline", "metric", "n",
    "mean_diff", "std_diff", "wins", "ties", "losses", "sign_test_p",
)
SUMMARY_CSV: Final = "summary.csv"
PAIRED_CSV: Final = "paired.csv"
SUMMARY_MD: Final = "summary.md"


# --------------------------------------------------------------------------- flattening


def flatten_metrics(result: RunResult) -> dict[str, float | None]:
    """Returns every metric of a run as ``name -> number`` (dicts flattened).

    Args:
        result: A run result.

    Returns:
        Scalar metrics by field name; dict metrics as ``field.key``.
    """
    flat: dict[str, float | None] = {}
    for name, value in result.metrics.model_dump().items():
        if isinstance(value, Mapping):
            flat.update({f"{name}.{key}": float(count) for key, count in sorted(value.items())})
        else:
            flat[name] = None if value is None else float(value)
    return flat


def _long_frame(results: Iterable[RunResult]) -> pd.DataFrame:
    """One row per (run, metric) with the run's identity columns."""
    rows = []
    for result in results:
        spec = result.spec
        identity = {
            "scenario": spec.scenario, "condition": spec.condition.value,
            "backend": backend_label(spec), "seed": spec.seed, "repeat": spec.repeat,
        }
        rows.extend({**identity, "metric": k, "value": v} for k, v in flatten_metrics(result).items())
    frame = pd.DataFrame(rows, columns=["scenario", "condition", "backend", "seed", "repeat", "metric", "value"])
    frame = _fill_missing_dict_keys(frame)
    return frame.sort_values(["scenario", "condition", "backend", "seed", "repeat", "metric"], ignore_index=True)


def _fill_missing_dict_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Adds 0 rows for flattened dict keys that some runs lack."""
    dict_metrics = sorted({m for m in frame["metric"] if "." in m})
    if not dict_metrics:
        return frame
    runs = frame[["scenario", "condition", "backend", "seed", "repeat"]].drop_duplicates()
    full = runs.merge(pd.DataFrame({"metric": dict_metrics}), how="cross")
    keys = ["scenario", "condition", "backend", "seed", "repeat", "metric"]
    missing = full.merge(frame[keys], on=keys, how="left", indicator=True)
    missing = missing[missing["_merge"] == "left_only"].drop(columns="_merge").assign(value=0.0)
    return pd.concat([frame, missing], ignore_index=True) if len(missing) else frame


def _order_key(column: pd.Series) -> pd.Series:
    """Sort key: known scenarios/conditions first in report order, others alphabetically."""
    order = SCENARIO_ORDER if column.name == "scenario" else CONDITION_ORDER
    if column.name not in ("scenario", "condition"):
        return column
    return column.map(lambda v: f"{order.index(v):03d}" if v in order else f"999{v}")


def _metric_rank(frame: pd.DataFrame) -> pd.Series:
    """Canonical metric order: ``RunMetrics`` field order, then dict key."""
    fields = {name: f"{i:03d}" for i, name in enumerate(RunMetrics.model_fields)}
    return frame["metric"].map(lambda m: fields.get(m.split(".", 1)[0], "999") + m)


# --------------------------------------------------------------------------- statistics


def sign_test_p(wins: int, losses: int) -> float:
    """Exact two-sided sign test p-value (binomial, p = 0.5; ties excluded).

    Args:
        wins: Number of positive differences.
        losses: Number of negative differences.

    Returns:
        ``min(1, 2 * P(X <= min(wins, losses)))`` for ``X ~ Bin(wins + losses, 0.5)``;
        1.0 when there are no non-tied pairs.
    """
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / 2**n)


def summarise(results: Sequence[RunResult]) -> pd.DataFrame:
    """Per scenario, condition, backend and metric: n, mean, std (ddof=1), min, max.

    Args:
        results: Run results (any mix of scenarios, conditions and backends).

    Returns:
        One row per group and metric; ``std`` is NaN when ``n < 2``.
    """
    frame = _long_frame(results).dropna(subset=["value"])
    if frame.empty:
        return pd.DataFrame(columns=list(SUMMARY_COLUMNS))
    grouped = frame.groupby(["scenario", "condition", "backend", "metric"], sort=False)["value"]
    summary = grouped.agg(n="count", mean="mean", std=lambda s: s.std(ddof=1), min="min", max="max")
    summary = summary.reset_index()
    summary["_rank"] = _metric_rank(summary)
    summary = summary.sort_values(["scenario", "condition", "backend", "_rank"], key=_order_key)
    return summary.drop(columns="_rank").reset_index(drop=True)[list(SUMMARY_COLUMNS)]


def _pair_row(metric: str, diffs: pd.Series) -> dict[str, object]:
    """Statistics of one metric's paired differences (treatment - baseline)."""
    better = diffs if metric in HIGHER_IS_BETTER else -diffs
    wins = int((better > TIE_TOLERANCE).sum())
    losses = int((better < -TIE_TOLERANCE).sum())
    return {
        "metric": metric,
        "n": len(diffs),
        "mean_diff": float(diffs.mean()),
        "std_diff": float(diffs.std(ddof=1)) if len(diffs) > 1 else float("nan"),
        "wins": wins,
        "ties": len(diffs) - wins - losses,
        "losses": losses,
        "sign_test_p": sign_test_p(wins, losses),
    }


def _paired_for(frame: pd.DataFrame, treatment: str, baseline: str, pair_index: int) -> list[dict[str, object]]:
    """Paired rows of one (treatment, baseline) pair over all scenarios and backends."""
    if frame.empty:
        return []
    keys = ["scenario", "backend", "seed", "repeat", "metric"]
    left = frame[frame["condition"] == treatment][[*keys, "value"]]
    right = frame[frame["condition"] == baseline][[*keys, "value"]]
    merged = left.merge(right, on=keys, suffixes=("_t", "_b")).dropna(subset=["value_t", "value_b"])
    merged["diff"] = merged["value_t"] - merged["value_b"]
    rows = []
    for (scenario, backend, metric), group in merged.groupby(["scenario", "backend", "metric"], sort=False):
        row = _pair_row(metric, group["diff"])
        rows.append({
            "scenario": scenario, "backend": backend, "treatment": treatment, "baseline": baseline,
            "_pair": pair_index, **row,
        })
    return rows


def paired_differences(
    results: Sequence[RunResult], pairs: Sequence[tuple[Condition, Condition]] = DEFAULT_PAIRS
) -> pd.DataFrame:
    """Paired differences ``treatment - baseline`` on identical worlds.

    Args:
        results: Run results.
        pairs: ``(treatment, baseline)`` condition pairs.

    Returns:
        One row per scenario, backend, pair and metric with n, mean_diff,
        std_diff (ddof=1), wins, ties, losses and the sign-test p-value.
    """
    frame = _long_frame(results)
    rows = [row for i, (t, b) in enumerate(pairs) for row in _paired_for(frame, t.value, b.value, i)]
    if not rows:
        return pd.DataFrame(columns=list(PAIRED_COLUMNS))
    paired = pd.DataFrame(rows)
    paired["_rank"] = _metric_rank(paired)
    paired = paired.sort_values(["scenario", "backend", "_pair", "_rank"], key=_order_key)
    return paired.drop(columns=["_pair", "_rank"]).reset_index(drop=True)[list(PAIRED_COLUMNS)]


# --------------------------------------------------------------------------- markdown


def _cell(row: pd.Series | None, fmt: str) -> str:
    """Formats ``mean ± std`` of a summary row (``-`` when missing)."""
    if row is None or pd.isna(row["mean"]):
        return "-"
    mean = fmt.format(row["mean"])
    return mean if pd.isna(row["std"]) else f"{mean} ± {fmt.format(row['std'])}"


def _headline_table(summary: pd.DataFrame, scenario: str, backend: str) -> list[str]:
    """Markdown rows of one scenario's headline table (conditions x metrics)."""
    part = summary[(summary["scenario"] == scenario) & (summary["backend"] == backend)]
    lines = [
        "| Condition | n | " + " | ".join(title for _, title, _ in HEADLINE_METRICS) + " |",
        "|---|---:|" + "---:|" * len(HEADLINE_METRICS),
    ]
    for condition in dict.fromkeys(part["condition"]):
        rows = {r["metric"]: r for _, r in part[part["condition"] == condition].iterrows()}
        n = int(rows["served_rate"]["n"]) if "served_rate" in rows else 0
        cells = [_cell(rows.get(metric), fmt) for metric, _, fmt in HEADLINE_METRICS]
        lines.append(f"| {condition} | {n} | " + " | ".join(cells) + " |")
    return lines


def _paired_table(paired: pd.DataFrame, scenario: str, backend: str) -> list[str]:
    """Markdown rows of one scenario's paired comparisons on headline metrics."""
    headline = [metric for metric, _, _ in HEADLINE_METRICS]
    part = paired[(paired["scenario"] == scenario) & (paired["backend"] == backend)]
    part = part[part["metric"].isin(headline)]
    pairs = list(dict.fromkeys(zip(part["treatment"], part["baseline"])))
    part = part.assign(
        _p=[pairs.index(pair) for pair in zip(part["treatment"], part["baseline"])],
        _h=part["metric"].map(headline.index),
    ).sort_values(["_p", "_h"])
    if part.empty:
        return []
    lines = ["| Comparison | Metric | n | Mean diff | W/T/L | Sign-test p |", "|---|---|---:|---:|---:|---:|"]
    for _, r in part.iterrows():
        lines.append(
            f"| {r['treatment']} vs {r['baseline']} | {r['metric']} | {int(r['n'])} | {r['mean_diff']:+.3f} "
            f"| {int(r['wins'])}/{int(r['ties'])}/{int(r['losses'])} | {r['sign_test_p']:.4f} |"
        )
    return lines


def summary_markdown(summary: pd.DataFrame, paired: pd.DataFrame) -> str:
    """Renders the human-readable report (headline tables first).

    Args:
        summary: Output of ``summarise``.
        paired: Output of ``paired_differences``.

    Returns:
        Markdown text.
    """
    lines = [
        "# Evaluation summary",
        "",
        "Cells are mean ± std (ddof=1) across seeds and repeats; n = number of runs.",
        "Paired rows compare conditions on identical worlds (same scenario, seed, repeat).",
        "W/T/L counts runs where the first condition is better / equal / worse; higher is",
        f"better only for {', '.join(sorted(HIGHER_IS_BETTER))}; lower is better otherwise.",
        "All metrics, including those that get worse, are in summary.csv and paired.csv.",
    ]
    groups = summary[["scenario", "backend"]].drop_duplicates().itertuples(index=False)
    for scenario, backend in groups:
        lines += ["", f"## {scenario} ({backend})", "", *_headline_table(summary, scenario, backend)]
        paired_lines = _paired_table(paired, scenario, backend)
        if paired_lines:
            lines += ["", *paired_lines]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- files


def write_report(results: Sequence[RunResult], out_dir: Path) -> None:
    """Writes ``summary.csv``, ``paired.csv`` and ``summary.md`` into ``out_dir``.

    Args:
        results: Run results.
        out_dir: Results root (created if missing).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarise(results)
    paired = paired_differences(results)
    summary.to_csv(out_dir / SUMMARY_CSV, index=False, lineterminator="\n")
    paired.to_csv(out_dir / PAIRED_CSV, index=False, lineterminator="\n")
    (out_dir / SUMMARY_MD).write_text(summary_markdown(summary, paired), encoding="utf-8")


def load_results(out_dir: Path) -> list[RunResult]:
    """Loads every run's ``metrics.json`` below ``out_dir`` (sorted by path).

    Args:
        out_dir: Results root (or any folder containing run folders).

    Returns:
        The results; empty if none are found.
    """
    if not out_dir.is_dir():
        return []
    return [read_result(path) for path in sorted(out_dir.rglob(METRICS_FILE))]
