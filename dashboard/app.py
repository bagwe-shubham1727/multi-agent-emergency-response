"""Streamlit app that replays emergency-response runs minute by minute.

Run from the project root::

    RESULTS_DIR=results .venv/bin/streamlit run dashboard/app.py

The app is deliberately thin: it picks runs, owns the widgets and lays out
five tabs (map, resources, agent activity, negotiation history, KPIs). All
data shaping lives in ``dashboard.replay`` and every chart in
``dashboard.figures``.
"""

import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # `streamlit run dashboard/app.py` only puts dashboard/ on the path
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard import figures, replay  # noqa: E402
from emergency_sim.domain.log_records import AgentCallRecord  # noqa: E402

RESULTS_ENV: Final = "RESULTS_DIR"
DEFAULT_RESULTS_DIR: Final = "results"
AUTOPLAY_INTERVAL_S: Final = 1.0
AUTOPLAY_STEPS: Final = (1, 2, 5, 10)
RECENT_MESSAGES: Final = 25
RECENT_EVENTS: Final = 15
AGENT_CARD_COLUMNS: Final = 3
CLI_HINT: Final = ("python -m emergency_sim.cli grid --scenarios normal surge constrained "
                   "--conditions b0 b0-cues b1 full --seeds 7380 --keep-logs-for 7380 --out results")
TAB_NAMES: Final = ("Map & incidents", "Resources", "Agent activity", "Negotiation history", "KPIs")

# Which way a KPI improves; decides the colour of its delta in compare mode.
HIGHER: Final = "higher"
LOWER: Final = "lower"
NEUTRAL: Final = "neutral"  # e.g. utilisation or message volume: neither better nor worse
DELTA_COLORS: Final = {HIGHER: "normal", LOWER: "inverse", NEUTRAL: "off"}
PERCENT_PATTERN: Final = "{:.0%}"
TABLE_DECIMALS: Final = 3

# (key, label, better, format)
KPI_AT_T: Final = (
    ("served", "Served", HIGHER, "{:.0f}"),
    ("red_in_window", "RED in window", HIGHER, "{:.0f}"),
    ("red_missed", "RED missed", LOWER, "{:.0f}"),
    ("unserved_critical_overdue", "RED overdue", LOWER, "{:.0f}"),
    ("mean_response_min", "Response (min)", LOWER, "{:.1f}"),
    ("conflicts", "Conflicts", LOWER, "{:.0f}"),
    ("messages", "Messages", NEUTRAL, "{:.0f}"),
)
KPI_FINAL: Final = (
    ("served_rate", "Served", HIGHER, PERCENT_PATTERN),
    ("red_in_window_rate", "RED in window", HIGHER, PERCENT_PATTERN),
    ("unserved_critical", "Unserved RED", LOWER, "{:.0f}"),
    ("response_time_mean_min", "Response (min)", LOWER, "{:.1f}"),
    ("ambulance_utilisation", "Ambulance use", NEUTRAL, PERCENT_PATTERN),
    ("er_utilisation", "ER use", NEUTRAL, PERCENT_PATTERN),
    ("conflicts", "Conflicts", LOWER, "{:.0f}"),
)
THREAD_COLUMNS: Final = {
    "thread": st.column_config.TextColumn(width="small"),
    "t (min)": st.column_config.NumberColumn(width="small"),
    "incident": st.column_config.TextColumn(width="small"),
    "kind": st.column_config.TextColumn(width="medium"),
    "resolution": st.column_config.TextColumn(width="medium"),
    "conflict": st.column_config.CheckboxColumn(width="small"),
    "contested resource": st.column_config.TextColumn(width="large"),
    "sequence": st.column_config.TextColumn(width="large"),
    "resolution path": st.column_config.TextColumn(width="medium"),
    "rounds": st.column_config.NumberColumn(width="small"),
    "hospitals asked": st.column_config.TextColumn(width="small"),
}


@dataclass(frozen=True)
class Selection:
    """What the sidebar selected.

    Attributes:
        results_dir: Results folder.
        refs: Runs of the selected scenario, seed, backend and repeat, by condition.
        condition_a: Primary condition.
        condition_b: Comparison condition, or None.
        autoplay: Whether the replay advances on its own.
        step: Minutes advanced per autoplay tick.
    """

    results_dir: Path
    refs: Mapping[str, replay.RunRef]
    condition_a: str
    condition_b: str | None
    autoplay: bool
    step: int


@dataclass(frozen=True)
class Context:
    """Everything the tabs render from.

    Attributes:
        runs: Displayed runs (condition A, then B) by condition.
        all_runs: Every condition of the selected scenario/seed/backend/repeat.
        summary: Cross-seed summary rows.
        backend: Selected backend label.
        t: Replay minute.
    """

    runs: Mapping[str, replay.RunLog]
    all_runs: Mapping[str, replay.RunLog]
    summary: pd.DataFrame
    backend: str
    t: float


# --- Cached loading -------------------------------------------------------------


@st.cache_data(ttl=30, show_spinner=False)
def cached_refs(results_dir: str) -> list[replay.RunRef]:
    """Discovers run logs (re-scanned at most every 30 s)."""
    return replay.discover_runs(Path(results_dir))


@st.cache_data(ttl=30, show_spinner=False)
def cached_summary(results_dir: str) -> pd.DataFrame:
    """Loads the cross-seed summary (re-read at most every 30 s)."""
    return replay.load_summary(Path(results_dir))


@st.cache_resource(max_entries=24, show_spinner="Loading run log…")
def cached_run(path: str, mtime: float) -> replay.RunLog:
    """Parses one run log.

    ``cache_resource`` (not ``cache_data``) because a ``RunLog`` is immutable
    and can be several MB: sharing it avoids re-copying it on every autoplay
    tick. ``mtime`` invalidates the entry when the file is rewritten.
    """
    del mtime  # cache key only
    return replay.load_run(Path(path))


def load(ref: replay.RunRef) -> replay.RunLog:
    """Loads a run through the cache."""
    return cached_run(str(ref.path), ref.path.stat().st_mtime)


# --- Sidebar ----------------------------------------------------------------------


def _pick(label: str, options: Sequence[Any], key: str, **kwargs: Any) -> Any:
    """Sidebar selectbox that tolerates an option list that changed."""
    if key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]
    return st.sidebar.selectbox(label, options, key=key, **kwargs)


def sidebar(results_dir: Path, refs: Sequence[replay.RunRef]) -> Selection:
    """Renders the run selectors and returns the selection.

    Args:
        results_dir: Results folder.
        refs: All discovered runs (non-empty).

    Returns:
        The selection.
    """
    scenario = _pick("Scenario", sorted({r.scenario for r in refs}, key=replay.scenario_sort_key), "scenario")
    refs = [r for r in refs if r.scenario == scenario]
    seed = _pick("Seed", sorted({r.seed for r in refs}), "seed")
    refs = [r for r in refs if r.seed == seed]
    backend = _pick("Backend", sorted({r.backend for r in refs}), "backend")
    refs = [r for r in refs if r.backend == backend]
    repeat = _pick("Repeat", sorted({r.repeat for r in refs}), "repeat")
    by_condition = {r.condition: r for r in refs if r.repeat == repeat}
    conditions = sorted(by_condition, key=replay.condition_sort_key)
    default = conditions.index("full") if "full" in conditions else len(conditions) - 1
    condition_a = _pick("Condition", conditions, "condition_a", index=default,
                        format_func=figures.condition_label)
    others = [c for c in conditions if c != condition_a]
    compare = st.sidebar.toggle("Compare with another condition", key="compare", disabled=not others)
    condition_b = None
    if compare and others:
        condition_b = _pick("Compare with", others, "condition_b", format_func=figures.condition_label)
    st.sidebar.divider()
    autoplay = st.sidebar.toggle("Autoplay", key="autoplay", help="Advance the replay every second.")
    step = st.sidebar.select_slider("Minutes per tick", AUTOPLAY_STEPS, value=AUTOPLAY_STEPS[1],
                                    key="step", disabled=not autoplay)
    return Selection(results_dir, by_condition, condition_a, condition_b, autoplay, int(step))


def time_slider(horizon: float) -> float:
    """Renders the replay-minute slider (0..horizon, step 1)."""
    top = int(horizon)
    if st.session_state.get("t", 0) > top:
        st.session_state["t"] = top
    return float(st.sidebar.slider("Simulation minute", 0, top, key="t"))


def playback_minute(selection: Selection, slider_t: float, horizon: float) -> float:
    """The minute to show: the slider's, or the autoplay position (advanced by one tick)."""
    if not selection.autoplay:
        st.session_state.pop("play_t", None)
        return slider_t
    current = st.session_state.get("play_t", slider_t)
    st.session_state["play_t"] = current + selection.step if current < horizon else 0.0
    return min(current, horizon)


# --- Small rendering helpers ---------------------------------------------------------


def chart(fig: go.Figure, key: str) -> None:
    """Renders a Plotly figure at full width with a unique key."""
    st.plotly_chart(fig, width="stretch", key=key, config={"displaylogo": False})


def table(rows: Sequence[Mapping[str, Any]] | pd.DataFrame, empty: str, **kwargs: Any) -> None:
    """Renders rows as a dataframe, or a caption when there are none."""
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    if frame.empty:
        st.caption(empty)
    else:
        st.dataframe(frame, hide_index=True, width="stretch", **kwargs)


def _fmt(value: float | None, pattern: str) -> str:
    """Formats a KPI value; '—' for None."""
    return "—" if value is None else pattern.format(value)


def format_delta(diff: float, pattern: str) -> str:
    """Formats a KPI difference: percentage points for shares, '0' when it rounds to nothing.

    Args:
        diff: Compared value minus baseline value.
        pattern: The KPI's display format (``PERCENT_PATTERN`` for shares).

    Returns:
        E.g. ``"-8 pp"``, ``"1.5"`` or ``"0"`` (Streamlit draws no arrow for ``"0"``).
    """
    if pattern == PERCENT_PATTERN:
        points = round(100 * diff)
        return "0" if points == 0 else f"{points:+d} pp"
    text = pattern.format(diff)
    return "0" if float(text) == 0 else text


def metric_row(spec: Sequence[tuple[str, str, str, str]], values: Mapping[str, Any],
               baseline: Mapping[str, Any] | None) -> None:
    """A row of KPI tiles; with a baseline, each tile shows the difference to it."""
    for col, (key, label, better, pattern) in zip(st.columns(len(spec)), spec, strict=True):
        value, base = values.get(key), (baseline or {}).get(key)
        delta = None
        if baseline is not None and value is not None and base is not None:
            delta = format_delta(value - base, pattern)
        col.metric(label, _fmt(value, pattern), delta=delta, delta_color=DELTA_COLORS[better], border=True)


def follow_minute(key: str, t: float, latest: Any) -> None:
    """Resets a drill-down widget to ``latest`` whenever the replay minute changes.

    Picking a decision point or thread is a drill-down at the current minute;
    without this, the widget kept its old choice (e.g. t = 29) after the
    slider moved to t = 180.
    """
    marker = f"{key}::t"
    if st.session_state.get(marker) != t:
        st.session_state[key] = latest
        st.session_state[marker] = t


def rounded(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Rounds float cells for display (``TABLE_DECIMALS`` places)."""
    return [{k: round(v, TABLE_DECIMALS) if isinstance(v, float) else v for k, v in r.items()} for r in rows]


def per_run(ctx: Context, render: Callable[[str, replay.RunLog], None], *, side_by_side: bool) -> None:
    """Renders ``render`` for each displayed run (columns or sub-tabs in compare mode)."""
    if len(ctx.runs) == 1:
        label, run = next(iter(ctx.runs.items()))
        render(label, run)
        return
    labels = [figures.condition_label(c) for c in ctx.runs]
    holders = st.columns(len(ctx.runs)) if side_by_side else st.tabs(labels)
    for holder, label, (condition, run) in zip(holders, labels, ctx.runs.items(), strict=True):
        with holder:
            if side_by_side:
                st.markdown(f"**{label}**")
            render(condition, run)


# --- Tab 1: map -------------------------------------------------------------------------


def render_map(ctx: Context) -> None:
    """Map, incident table and recent events."""
    def one(condition: str, run: replay.RunLog) -> None:
        snap = replay.snapshot_at(run, ctx.t)
        chart(figures.incident_map(snap, run.start), f"map-{condition}")
        st.markdown("**Incidents**")
        table(replay.incident_rows(snap.view), "No incidents reported yet.")
        st.markdown("**Recent events** (newest first; t=0 baseline bed/staff releases hidden)")
        events = replay.foreground_events(replay.events_until(run, ctx.t))[-RECENT_EVENTS:]
        table(list(reversed(replay.event_rows(events))), "No events yet.")

    per_run(ctx, one, side_by_side=True)


# --- Tab 2: resources ---------------------------------------------------------------------


def resource_tiles(summary: replay.ResourceSummary) -> None:
    """Availability tiles for every resource class."""
    rows = (
        (("Ambulances free", f"{summary.ambulances_available} / {summary.ambulances_in_service}"),
         ("Responders free", f"{summary.responders_available} / {summary.responders_total}"),
         ("ER beds free", f"{summary.beds_free} / {summary.er_capacity}"),
         ("Queued for a bed", f"{summary.queue}")),
        (("Doctor load", f"{summary.doctor_load} / {summary.doctor_capacity}"),
         ("Nurse load", f"{summary.nurse_load} / {summary.nurse_capacity}"),
         ("Float staff free / lent", f"{summary.float_available} / {summary.float_lent}")),
    )
    for tiles in rows:
        for col, (label, value) in zip(st.columns(len(tiles)), tiles, strict=True):
            col.metric(label, value, border=True)


def render_resources(ctx: Context) -> None:
    """Resource tiles, ER and staff charts, utilisation and unit timelines."""
    def one(condition: str, run: replay.RunLog) -> None:
        snap = replay.snapshot_at(run, ctx.t)
        resource_tiles(replay.resource_summary(snap.view))
        left, right = st.columns(2)
        with left:
            chart(figures.er_occupancy(snap), f"er-{condition}")
        with right:
            chart(figures.staff_load(snap), f"staff-{condition}")
        chart(figures.resource_timeseries(run, ctx.t), f"util-{condition}")
        chart(figures.ambulance_timeline(run, ctx.t), f"timeline-{condition}")

    per_run(ctx, one, side_by_side=False)


# --- Tab 3: agent activity -------------------------------------------------------------------


def agent_card(status: replay.AgentStatus) -> None:
    """One bordered card with an agent's state at t."""
    with st.container(border=True):
        role = "" if status.role == status.agent else f"  \n:gray[{status.role}]"
        st.markdown(f"**{status.agent}**{role}")
        last = "never" if status.last_call_t is None else f"t = {status.last_call_t:g}"
        if status.role == replay.CONTROLLER_ROLE:  # rule-based: no LLM calls or fallbacks to count
            st.caption(f"rule-based, acts directly · last action {last}")
        else:
            st.caption(f"{replay.plural(status.calls, 'call')} · {status.fallbacks} with fallback · last {last}")
        if status.last_triggers:
            st.caption("triggers: " + ", ".join(status.last_triggers))
        st.markdown(f"Last decision: {status.last_decision}")
        pending = "; ".join(status.pending) if status.pending else "nothing pending"
        st.markdown(f"Pending: {pending}")


def call_details(call: AgentCallRecord | None) -> None:
    """Rationale, tool calls and fallbacks of one agent call."""
    if call is None:
        st.caption("No call yet.")
        return
    st.caption(f"t = {call.t_min:g} · source: {call.source} · backend: {call.backend}")
    for rationale in replay.call_rationales(call):
        st.markdown(f"- {rationale}")
    table([{"tool": tc.name, "arguments": tc.arguments, "output": tc.output} for tc in call.tool_calls],
          "No tool calls.")
    table([{"scope": f.scope, "key": f.key, "rule": f.rule, "detail": f.detail} for f in call.fallbacks],
          "No fallbacks.")


def decision_stepper(run: replay.RunLog, t: float, key: str) -> None:
    """Step through one decision point's messages, calls and actions in protocol order."""
    times = [x for x in replay.decision_times(run) if x <= t]
    if not times:
        st.caption("No decisions yet.")
        return
    widget_key = f"dp-{key}"
    follow_minute(widget_key, t, times[-1])
    chosen = st.select_slider("Decision point (minute)", times, key=widget_key, format_func=lambda x: f"{x:g}")
    rows = [{"seq": r.seq, "kind": r.kind, "actor": r.actor, "what": r.summary, "detail": r.detail,
             "unanswered": r.open_request} for r in replay.decision_trace(run, float(chosen))]
    table(rows, "Nothing logged at this minute.")


def render_agents(ctx: Context) -> None:
    """Agent status cards, recent messages, decision step-through and last decisions."""
    def one(condition: str, run: replay.RunLog) -> None:
        statuses = replay.agent_status_at(run, ctx.t)
        if not run.agent_calls:
            st.info("This condition has no agents: the greedy controller acts directly (no messages).")
        cols = st.columns(AGENT_CARD_COLUMNS)
        for i, status in enumerate(statuses):
            with cols[i % AGENT_CARD_COLUMNS]:
                agent_card(status)
        st.markdown("**Recent messages** (newest first)")
        recent = replay.messages_until(run, ctx.t)[-RECENT_MESSAGES:]
        table([replay.message_row(m) for m in reversed(recent)], "No messages yet.")
        st.markdown("**Step through a decision point**")
        decision_stepper(run, ctx.t, condition)
        if run.agent_calls:
            st.markdown("**Last decision of each agent** (rationale, tool calls, fallbacks)")
            for status in statuses:
                with st.expander(f"{status.agent} · {replay.plural(status.calls, 'call')}"):
                    call_details(replay.last_call_of(run, status.agent, ctx.t))

    per_run(ctx, one, side_by_side=False)


# --- Tab 4: negotiation ----------------------------------------------------------------------


def render_negotiation(ctx: Context) -> None:
    """Thread table, resolution breakdown and one thread's message sequence."""
    def one(condition: str, run: replay.RunLog) -> None:
        threads = replay.threads_from_messages(replay.messages_until(run, ctx.t))
        conflicts = [th for th in threads if replay.is_conflict(th)]
        cols = st.columns(3)
        cols[0].metric("Negotiation threads", len(threads), border=True)
        cols[1].metric("Allocation conflicts", len(conflicts), border=True)
        cols[2].metric("Still open", sum(th.resolution is None for th in threads), border=True)
        table([replay.thread_row(th) for th in threads], "No negotiation threads up to this minute.",
              column_config=THREAD_COLUMNS)
        chart(figures.resolution_breakdown(threads), f"resolutions-{condition}")
        if threads:
            by_id = {th.conversation_id: th for th in threads}
            widget_key = f"thread-{condition}"
            follow_minute(widget_key, ctx.t, threads[-1].conversation_id)
            chosen = st.selectbox("Show thread (newest by default)", list(by_id), key=widget_key,
                                  format_func=lambda cid: replay.thread_label(by_id[cid]))
            thread = by_id[chosen]
            path = " > ".join(replay.resolution_path(thread.messages)) or "none"
            st.caption(f"{thread.kind} · {thread.contested_resource} · resolution: {thread.resolution or 'open'}"
                       f" · resolution path: {path}")
            table([replay.message_row(m) for m in thread.messages], "Empty thread.")

    per_run(ctx, one, side_by_side=False)


# --- Tab 5: KPIs --------------------------------------------------------------------------------


def kpi_tiles(ctx: Context) -> None:
    """KPI tiles at t and final, per displayed condition (B shows its difference to A)."""
    baseline_t: Mapping[str, Any] | None = None
    baseline_final: Mapping[str, Any] | None = None
    first = figures.condition_label(next(iter(ctx.runs), ""))
    for i, (condition, run) in enumerate(ctx.runs.items()):
        note = f" (arrows: difference to {first})" if i else ""
        st.markdown(f"**{figures.condition_label(condition)}**{note}")
        at_t = replay.kpis_at_time(run, ctx.t)
        st.caption(f"At t = {ctx.t:g} min")
        metric_row(KPI_AT_T, at_t, baseline_t)
        final = run.end.metrics.model_dump() if run.end else {}
        st.caption("Final (end of horizon)" if run.end else "Final metrics unavailable (log has no run_end).")
        if run.end:
            metric_row(KPI_FINAL, final, baseline_final)
        baseline_t, baseline_final = at_t, (final or None)


def cross_seed(ctx: Context) -> None:
    """Cross-seed comparison from ``summary.csv`` (chart plus table view)."""
    st.subheader("Across seeds")
    if ctx.summary.empty:
        st.caption("No summary yet. Run `python -m emergency_sim.cli report --results <dir>` to write summary.csv.")
        return
    metrics = sorted(ctx.summary["metric"].unique())
    default = metrics.index("served_rate") if "served_rate" in metrics else 0
    metric = st.selectbox("Metric", metrics, index=default, key="summary-metric",
                          format_func=lambda m: figures.METRIC_LABELS.get(m, m))
    backends = set(ctx.summary["backend"])
    backend = ctx.backend if ctx.backend in backends else None
    chart(figures.kpi_comparison(ctx.summary, metric, backend), "kpi-comparison")
    table(summary_rows(ctx.summary, metric, backend), "No rows.")


def summary_rows(summary: pd.DataFrame, metric: str, backend: str | None) -> list[dict[str, Any]]:
    """Cross-seed rows of one metric in chart order (scenario, then condition), labelled and rounded."""
    rows = summary[summary["metric"] == metric]
    if backend is not None:
        rows = rows[rows["backend"] == backend]
    records = sorted(rows.to_dict("records"), key=lambda r: (
        replay.scenario_sort_key(r["scenario"]), replay.condition_sort_key(r["condition"]), r["backend"]))
    shown = [{"scenario": r["scenario"], "condition": figures.condition_label(r["condition"]),
              "backend": r["backend"], "n": r["n"], "mean": r["mean"], "std": r["std"],
              "min": r["min"], "max": r["max"]} for r in records]
    return rounded(shown)


def render_kpis(ctx: Context) -> None:
    """KPI tiles, all-condition comparison for this seed, and the cross-seed summary."""
    kpi_tiles(ctx)
    st.subheader("All conditions, this scenario and seed")
    chart(figures.cumulative_served(ctx.all_runs, ctx.t), "cumulative-served")
    rows = replay.final_metrics_rows(ctx.all_runs)
    labelled = [{"metric": figures.METRIC_LABELS.get(r["metric"], r["metric"].replace("_", " "))}
                | {figures.condition_label(k): v for k, v in r.items() if k != "metric"} for r in rows]
    table(rounded(labelled), "No final metrics.")
    cross_seed(ctx)


# --- Page -----------------------------------------------------------------------------------------


def render_tabs(ctx: Context) -> None:
    """The five tabs."""
    renderers = (render_map, render_resources, render_agents, render_negotiation, render_kpis)
    for tab, render in zip(st.tabs(list(TAB_NAMES)), renderers, strict=True):
        with tab:
            render(ctx)


def render_empty(results_dir: Path, summary: pd.DataFrame) -> None:
    """Shown when no run logs exist."""
    st.info(f"No run logs found under `{results_dir}/runs`. Generate some from the project root:\n\n"
            f"`{CLI_HINT}`\n\nor point the `{RESULTS_ENV}` environment variable (or the sidebar) "
            "at another results folder.")
    if not summary.empty:
        cross_seed(Context(runs={}, all_runs={}, summary=summary, backend="", t=0.0))


def header(run: replay.RunLog, t: float) -> None:
    """Title line with the run identity and the replay minute."""
    start = run.start
    st.caption(f"Run `{start.run_id}` · scenario **{start.scenario}** · seed {start.seed} · "
               f"backend {start.backend} · t = **{t:g}** of {replay.horizon_of(run):g} min")


def replay_body(selection: Selection, summary: pd.DataFrame, slider_t: float) -> None:
    """Loads the selected runs and renders the tabs (an autoplay fragment when playing)."""
    all_runs = {c: load(ref) for c, ref in selection.refs.items()}
    shown = [selection.condition_a] + ([selection.condition_b] if selection.condition_b else [])
    runs = {c: all_runs[c] for c in shown}
    horizon = replay.horizon_of(runs[selection.condition_a])
    t = playback_minute(selection, slider_t, horizon)
    header(runs[selection.condition_a], t)
    backend = runs[selection.condition_a].start.backend
    render_tabs(Context(runs, all_runs, summary, backend, t))


def main() -> None:
    """Entry point executed by Streamlit on every rerun."""
    st.set_page_config(page_title="Emergency Response MAS replay", page_icon=":material/emergency:", layout="wide")
    st.title("Emergency response coordination: run replay")
    default_dir = os.environ.get(RESULTS_ENV, DEFAULT_RESULTS_DIR)
    results_dir = Path(st.sidebar.text_input("Results folder", value=default_dir, key="results_dir"))
    refs = cached_refs(str(results_dir))
    summary = cached_summary(str(results_dir))
    if not refs:
        render_empty(results_dir, summary)
        return
    selection = sidebar(results_dir, refs)
    horizon = replay.horizon_of(load(selection.refs[selection.condition_a]))
    slider_t = time_slider(horizon)
    run_every = AUTOPLAY_INTERVAL_S if selection.autoplay else None
    st.fragment(replay_body, run_every=run_every)(selection, summary, slider_t)


main()
