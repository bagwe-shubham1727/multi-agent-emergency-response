"""Pure Plotly figure builders for the dashboard.

Every function takes replayed data and returns a ``go.Figure``; nothing here
touches Streamlit or files. Colour rules (one system across all charts):

* Severity uses the START triage convention with accessible steps (status
  palette): RED, YELLOW, GREEN, plus a dark grey for BLACK and a hollow grey
  ring for untriaged incidents. Severity is never colour-only: the legend,
  hover text and marker shape (``x`` = cleared) carry it too.
* Conditions and other categories use a fixed categorical order (colour
  follows the entity, never its rank), validated for colour-vision
  deficiency on adjacent pairs.
* Magnitude (ER bed use on the map) uses a single-hue blue ramp.
* Backgrounds are transparent and grid/axis ink is a mid grey, so the same
  figure reads in Streamlit's light and dark themes.
"""

import math
import zlib
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Final

import pandas as pd
import plotly.graph_objects as go

from dashboard.replay import (
    STATUS_GROUP_ORDER,
    UNTRIAGED,
    UTILISATION_SERIES,
    RunLog,
    ThreadSummary,
    condition_sort_key,
    incident_severity,
    scenario_sort_key,
    triage_counts,
    unit_intervals,
    utilisation_series,
)
from emergency_sim import geo
from emergency_sim.domain.enums import IncidentStatus, StaffRole, UnitClass, UnitStatus
from emergency_sim.domain.log_records import RunStartRecord, SnapshotRecord
from emergency_sim.domain.scenario import StationSpec
from emergency_sim.domain.views import HospitalView, IncidentView, PublicView, UnitView

FONT_FAMILY: Final = 'system-ui, -apple-system, "Segoe UI", sans-serif'
MUTED: Final = "#898781"
GRID: Final = "rgba(137, 135, 129, 0.22)"
AXIS_LINE: Final = "rgba(137, 135, 129, 0.55)"
RING: Final = "rgba(252, 252, 251, 0.85)"
TRANSPARENT: Final = "rgba(0, 0, 0, 0)"

# Categorical slots (fixed order; adjacent pairs pass the CVD check).
SLOT: Final = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")

SEVERITY_ORDER: Final = ("RED", "YELLOW", "GREEN", "BLACK", UNTRIAGED)
TRIAGE_COLORS: Final = MappingProxyType({
    "RED": "#d03b3b", "YELLOW": "#fab219", "GREEN": "#0ca30c", "BLACK": "#4a4946", UNTRIAGED: MUTED,
})
TRIAGE_LABELS: Final = MappingProxyType({
    "RED": "RED (immediate)", "YELLOW": "YELLOW (delayed)", "GREEN": "GREEN (minor)",
    "BLACK": "BLACK (expectant)", UNTRIAGED: "Untriaged",
})
CONDITION_COLORS: Final = MappingProxyType({
    "b0": SLOT[0], "b1": SLOT[1], "full": SLOT[2],
    # Violet, not SLOT[5] (green): next to Full's green the two step lines were hard to tell apart.
    "full-no-float": SLOT[3], "full-no-diversion": SLOT[4], "full-no-escalation": SLOT[6],
})
CONDITION_LABELS: Final = MappingProxyType({
    "b0": "B0 greedy", "b1": "B1 agents", "full": "Full negotiation",
    "full-no-float": "Full − float staff", "full-no-diversion": "Full − diversion",
    "full-no-escalation": "Full − escalation", "b0-cues": "B0 greedy + cues",
})
# Conditions without a fixed colour (e.g. added after this module) take one of
# the slots no known condition uses, picked by a stable hash of the name.
SPARE_CONDITION_SLOTS: Final = tuple(c for c in SLOT if c not in CONDITION_COLORS.values())
STATUS_GROUP_COLORS: Final = MappingProxyType({
    "Available": "rgba(137, 135, 129, 0.28)", "To scene": SLOT[0], "On scene": SLOT[1],
    "Transporting": SLOT[2], "At hospital": SLOT[3], "Unavailable": MUTED,
})
UTILISATION_COLORS: Final = MappingProxyType(dict(zip(UTILISATION_SERIES, SLOT, strict=False)))
BED_COLORS: Final = MappingProxyType({
    "Occupied": SLOT[0], "Reserved (inbound)": SLOT[1], "Free": "rgba(137, 135, 129, 0.30)", "Queue": SLOT[6],
})
ROLE_COLORS: Final = MappingProxyType({StaffRole.DOCTOR: (SLOT[0], "rgba(42, 120, 214, 0.22)"),
                                        StaffRole.NURSE: (SLOT[1], "rgba(235, 104, 52, 0.22)")})
BED_SCALE: Final = ((0.0, "#cde2fb"), (0.5, "#5598e7"), (1.0, "#104281"))
UNIT_STYLES: Final = MappingProxyType({
    "ALS": ("diamond", "#6250d6", "ALS ambulance"),
    "BLS": ("circle", "#6250d6", "BLS ambulance"),
    "responder": ("hexagon", "#e87ba4", "Responder team"),
})
METRIC_LABELS: Final = MappingProxyType({
    "served_rate": "Share of victims served",
    "victims_served": "Victims served",
    "red_in_window": "RED treated within window",
    "red_in_window_rate": "RED within-window rate",
    "unserved_critical": "Unserved critical (RED)",
    "survival_weighted_score": "Survival-weighted score",
    "response_time_mean_min": "Mean response time (min)",
    "response_time_p90_min": "P90 response time (min)",
    "red_response_mean_min": "RED ALS response (min)",
    "door_to_doctor_mean_min": "Door-to-doctor (min)",
    "ambulance_utilisation": "Ambulance utilisation",
    "er_utilisation": "ER utilisation",
    "doctor_utilisation": "Doctor utilisation",
    "nurse_utilisation": "Nurse utilisation",
    "conflicts": "Allocation conflicts",
    "messages_total": "Messages",
    "hospital_full_on_arrival": "Arrived to a full ER",
})
SHARE_SUFFIXES: Final = ("_rate", "_utilisation")
MAP_HEIGHT: Final = 720
CLEARED_SIZE: Final = 10
CLEARED_OPACITY: Final = 0.35
CHART_HEIGHT: Final = 360


# --- Shared styling -----------------------------------------------------------


def _style(fig: go.Figure, title: str, height: int = CHART_HEIGHT, legend_rows: int = 1) -> go.Figure:
    """Applies the shared, theme-neutral chart chrome.

    The title sits at the very top; the legend grows upwards from the plot
    area, so the top margin reserves one 22 px band per legend row.
    """
    fig.update_layout(
        title={"text": title, "x": 0, "xanchor": "left", "y": 0.995, "yanchor": "top", "yref": "container",
               "font": {"size": 15}},
        font={"family": FONT_FAMILY, "size": 12},
        paper_bgcolor=TRANSPARENT,
        plot_bgcolor=TRANSPARENT,
        height=height,
        margin={"l": 8, "r": 8, "t": 48 + 22 * legend_rows, "b": 8},
        legend={"orientation": "h", "x": 0, "xanchor": "left", "y": 1.02, "yanchor": "bottom", "title": {"text": ""}},
        hoverlabel={"font": {"family": FONT_FAMILY}},
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS_LINE, zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS_LINE, zeroline=False, automargin=True)
    return fig


# Two side-by-side bar slots per category, kept thin (explicit width + offset).
_LEFT_BAR: Final = MappingProxyType({"width": 0.26, "offset": -0.28})
_RIGHT_BAR: Final = MappingProxyType({"width": 0.26, "offset": 0.02})
SMALL_COUNT_MAX: Final = 10


def _integer_ticks(update_axis: Callable[..., go.Figure], max_value: float) -> None:
    """Uses whole-number ticks for small counts (no 0.5 patients)."""
    if max_value <= SMALL_COUNT_MAX:
        update_axis(dtick=1, tick0=0)


def empty_figure(title: str, message: str) -> go.Figure:
    """A placeholder figure with a centred note (used when there is no data).

    Args:
        title: Chart title.
        message: Note shown in the plot area.

    Returns:
        The figure (no traces).
    """
    fig = _style(go.Figure(), title, height=220)
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font={"color": MUTED, "size": 13})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def _add_time_marker(fig: go.Figure, t: float | None) -> None:
    """Draws a thin vertical line at the current replay minute."""
    if t is not None:
        fig.add_vline(x=t, line_width=1, line_color=MUTED,
                      annotation_text=f"t = {t:g}", annotation_position="top left",
                      annotation_font_color=MUTED)


# --- Incident map -------------------------------------------------------------


def _positions(view: PublicView) -> dict[str, tuple[float, float]]:
    """Coordinates of every incident and hospital, keyed by id."""
    places = {i.incident_id: (i.x, i.y) for i in view.incidents}
    places.update({h.hospital_id: (h.x, h.y) for h in view.hospitals})
    return places


def _route_trace(view: PublicView) -> go.Scatter:
    """Lines from every busy unit to its destination (incident or hospital)."""
    places = _positions(view)
    xs: list[float | None] = []
    ys: list[float | None] = []
    for unit in view.units:
        target = places.get(unit.destination_id or "")
        if target is not None:
            xs += [unit.x, target[0], None]
            ys += [unit.y, target[1], None]
    return go.Scatter(x=xs, y=ys, mode="lines", name="Route to destination", hoverinfo="skip",
                      line={"color": MUTED, "width": 1.5})


def _station_trace(stations: Sequence[StationSpec]) -> go.Scatter:
    """Stations as hollow grey triangles."""
    return go.Scatter(
        x=[s.x for s in stations], y=[s.y for s in stations], mode="markers", name="Station",
        marker={"symbol": "triangle-up-open", "size": 12, "color": MUTED, "line": {"width": 2}},
        text=[f"<b>{s.id}</b> {s.name}" for s in stations], hovertemplate="%{text}<extra></extra>",
    )


def _hospital_hover(h: HospitalView) -> str:
    """Hover text of a hospital marker."""
    inbound = ", ".join(h.inbound_unit_ids) or "none"
    return (f"<b>{h.hospital_id}</b> {h.name}<br>ER beds: {h.beds_occupied} occupied, "
            f"{h.beds_reserved} reserved, {h.beds_free_unreserved} free of {h.er_capacity}"
            f"<br>queue {h.queue_length} · waiting for staff {h.waiting_for_staff}"
            f"<br>inbound: {inbound}<br>capabilities: {', '.join(h.capabilities)}")


def _hospital_trace(hospitals: Sequence[HospitalView]) -> go.Scatter:
    """Hospitals as squares shaded by ER bed use (occupied + reserved), labelled with the share.

    The share is printed next to each hospital instead of a colour bar: with
    the map kept square, a colour bar ended up far from the plot.
    """
    use = [(h.beds_occupied + h.beds_reserved) / h.er_capacity if h.er_capacity else 0.0 for h in hospitals]
    return go.Scatter(
        x=[h.x for h in hospitals], y=[h.y for h in hospitals], mode="markers+text",
        name="Hospital (darker = fuller ER)",
        text=[f"{h.hospital_id} · {u:.0%} beds" for h, u in zip(hospitals, use, strict=True)],
        textposition="top center",
        hovertext=[_hospital_hover(h) for h in hospitals], hovertemplate="%{hovertext}<extra></extra>",
        marker={"symbol": "square", "size": 20, "color": use, "cmin": 0, "cmax": 1, "showscale": False,
                "colorscale": [list(stop) for stop in BED_SCALE], "line": {"width": 1.5, "color": MUTED}},
    )


def _incident_hover(i: IncidentView) -> str:
    """Hover text of an incident marker."""
    units = ", ".join(i.assigned_unit_ids) or "none"
    return (f"<b>{i.incident_id}</b> · {i.incident_type}<br>{i.intersection}<br>"
            f"reported t={i.t_report_min:g} · {i.reported_victims} victim(s) reported<br>"
            f"triage: {triage_counts(i)}<br>status: {i.status.value} · units: {units}")


def _incident_trace(severity: str, incidents: Sequence[IncidentView]) -> go.Scatter:
    """Incidents of one severity; size = reported victims, faded ``x`` = cleared.

    Cleared incidents are drawn smaller and faded so the open ones stand out
    late in a run, when most incidents are already cleared. Open incidents
    come first because the legend shows the first point's style.
    """
    hollow = severity == UNTRIAGED
    base = "circle-open" if hollow else "circle"
    incidents = sorted(incidents, key=lambda i: i.status is IncidentStatus.CLEARED)
    cleared = [i.status is IncidentStatus.CLEARED for i in incidents]
    size = [14 + 4 * math.sqrt(max(i.reported_victims, 1)) for i in incidents]
    return go.Scatter(
        x=[i.x for i in incidents], y=[i.y for i in incidents], mode="markers",
        name=TRIAGE_LABELS[severity],
        hovertext=[_incident_hover(i) for i in incidents], hovertemplate="%{hovertext}<extra></extra>",
        marker={
            "symbol": ["x" if c else base for c in cleared],
            "size": [CLEARED_SIZE if c else s for c, s in zip(cleared, size, strict=True)],
            "opacity": [CLEARED_OPACITY if c else 1.0 for c in cleared],
            "color": TRIAGE_COLORS[severity],
            "line": {"width": 2.5 if hollow else 1, "color": TRIAGE_COLORS[severity] if hollow else RING},
        },
    )


def _unit_kind(unit: UnitView) -> str:
    """``"ALS"``, ``"BLS"`` or ``"responder"``."""
    if unit.unit_class is UnitClass.RESPONDER:
        return "responder"
    return "ALS" if unit.unit_type == "ALS" else "BLS"


def _unit_hover(u: UnitView) -> str:
    """Hover text of a unit marker."""
    load = ", ".join(u.load_victim_ids) or "none"
    target = u.destination_id or u.incident_id or u.hospital_id or "-"
    return (f"<b>{u.unit_id}</b> ({u.unit_type})<br>status: {u.status.value}<br>"
            f"destination: {target}<br>patients aboard: {load}")


def _unit_trace(kind: str, units: Sequence[UnitView]) -> go.Scatter:
    """Units of one kind; hollow symbol = available, filled = busy."""
    symbol, color, label = UNIT_STYLES[kind]
    return go.Scatter(
        x=[u.x for u in units], y=[u.y for u in units], mode="markers", name=f"{label} (hollow = free)",
        hovertext=[_unit_hover(u) for u in units], hovertemplate="%{hovertext}<extra></extra>",
        marker={"symbol": [f"{symbol}-open" if u.status is UnitStatus.AVAILABLE else symbol for u in units],
                "size": 9, "color": color, "line": {"width": 2, "color": color}},
    )


def _map_axes(fig: go.Figure) -> None:
    """Schematic 20 x 20 km grid labelled with avenues (x) and streets (y)."""
    centres = [2 * k + 1 for k in range(len(geo.AVENUES))]
    edges = list(range(0, int(geo.MAP_SIZE_KM) + 1, 2))
    common = {"range": [-0.5, geo.MAP_SIZE_KM + 0.5], "showgrid": False, "tickvals": centres,
              "minor": {"tickvals": edges, "showgrid": True, "gridcolor": GRID}, "fixedrange": False,
              # Keep the square city square by shrinking the plot area, not by
              # widening the range into empty space beyond 1st / 10th Ave.
              "constrain": "domain"}
    fig.update_xaxes(ticktext=geo.AVENUES, tickangle=-45, **common)
    fig.update_yaxes(ticktext=geo.STREETS, scaleanchor="x", scaleratio=1, **common)
    cx, cy = geo.DOWNTOWN
    r = geo.DOWNTOWN_RADIUS_KM
    fig.add_shape(type="circle", x0=cx - r, y0=cy - r, x1=cx + r, y1=cy + r, layer="below",
                  line={"color": GRID, "width": 1}, fillcolor="rgba(137, 135, 129, 0.06)")
    fig.add_annotation(x=cx, y=cy + r, text="downtown (slower traffic)", showarrow=False,
                       yshift=8, font={"color": MUTED, "size": 11})


def incident_map(snapshot: SnapshotRecord, start: RunStartRecord) -> go.Figure:
    """Schematic city map at one snapshot.

    Shows hospitals (squares coloured by ER bed use), stations (triangles),
    incidents (worst known triage colour, grey ring if untriaged, size =
    reported victims, ``x`` when cleared), units (ALS / BLS / responder
    symbols, hollow when free) and lines from busy units to their destination.

    Args:
        snapshot: The snapshot to draw.
        start: The run's start record (for the stations).

    Returns:
        The figure; traces are added only for non-empty groups (routes,
        stations and hospitals always).
    """
    view = snapshot.view
    fig = go.Figure()
    fig.add_trace(_route_trace(view))
    fig.add_trace(_station_trace(start.scenario_data.stations))
    fig.add_trace(_hospital_trace(view.hospitals))
    for severity in SEVERITY_ORDER:
        group = [i for i in view.incidents if incident_severity(i) == severity]
        if group:
            fig.add_trace(_incident_trace(severity, group))
    for kind in UNIT_STYLES:
        group = [u for u in view.units if _unit_kind(u) == kind]
        if group:
            fig.add_trace(_unit_trace(kind, group))
    _style(fig, f"City map at t = {snapshot.t_min:g} min · faded × = cleared incident", height=MAP_HEIGHT,
           legend_rows=0)
    # Legend under the map: it wraps to many rows in a narrow (compare-mode)
    # column, and below the plot it pushes the bottom margin instead of
    # running into the title.
    fig.update_layout(legend={"y": -0.14, "yanchor": "top"})
    _map_axes(fig)
    return fig


# --- Resources ------------------------------------------------------------------


def ambulance_timeline(run: RunLog, t: float | None = None) -> go.Figure:
    """Gantt-style status timeline of every ambulance and responder team.

    Args:
        run: The run log.
        t: Current replay minute (drawn as a vertical line), or None.

    Returns:
        One horizontal-bar trace per status group present.
    """
    intervals = unit_intervals(run)
    if not intervals:
        return empty_figure("Unit status timeline", "No snapshots in this log.")
    labels = sorted({i.unit_label for i in intervals}, reverse=True)
    fig = go.Figure()
    for group in STATUS_GROUP_ORDER:
        rows = [i for i in intervals if i.group == group]
        if rows:
            fig.add_trace(go.Bar(
                orientation="h", name=group, y=[i.unit_label for i in rows], x=[i.end - i.start for i in rows],
                base=[i.start for i in rows], marker={"color": STATUS_GROUP_COLORS[group], "line": {"width": 0}},
                width=0.62, customdata=[[i.status, i.start, i.end] for i in rows],
                hovertemplate="%{y}<br>%{customdata[0]}: t=%{customdata[1]:g}–%{customdata[2]:g}<extra></extra>",
            ))
    _style(fig, "Unit status timeline", height=110 + 24 * len(labels))
    fig.update_layout(barmode="overlay")
    fig.update_yaxes(categoryorder="array", categoryarray=labels, showgrid=False)
    fig.update_xaxes(title_text="minute", range=[0, max(i.end for i in intervals)])
    _add_time_marker(fig, t)
    return fig


def er_occupancy(snapshot: SnapshotRecord) -> go.Figure:
    """ER beds per hospital: occupied / reserved / free stacked, queue beside.

    Args:
        snapshot: The snapshot to draw.

    Returns:
        Four bar traces (Occupied, Reserved, Free, Queue).
    """
    hospitals = snapshot.view.hospitals
    ids = [h.hospital_id for h in hospitals]
    occupied = [h.beds_occupied for h in hospitals]
    reserved = [h.beds_reserved for h in hospitals]
    free = [h.beds_free_unreserved for h in hospitals]
    stacks = (("Occupied", occupied, [0] * len(ids)),
              ("Reserved (inbound)", reserved, occupied),
              ("Free", free, [o + r for o, r in zip(occupied, reserved, strict=True)]))
    fig = go.Figure()
    for name, values, base in stacks:
        fig.add_trace(go.Bar(name=name, x=ids, y=values, base=base, offsetgroup="beds", **_LEFT_BAR,
                             marker={"color": BED_COLORS[name]},
                             hovertemplate=f"%{{x}}: %{{y}} {name.lower()}<extra></extra>"))
    fig.add_trace(go.Bar(name="Queue", x=ids, y=[h.queue_length for h in hospitals], offsetgroup="queue",
                         **_RIGHT_BAR,
                         marker={"color": BED_COLORS["Queue"]},
                         hovertemplate="%{x}: %{y} queued for a bed<extra></extra>"))
    _style(fig, f"ER beds and queue at t = {snapshot.t_min:g} min")
    fig.update_layout(barmode="overlay")
    fig.update_yaxes(title_text="beds / patients", rangemode="tozero")
    _integer_ticks(fig.update_yaxes, max((h.er_capacity for h in hospitals), default=0))
    return fig


def _staff_bars(view: PublicView, role: StaffRole) -> tuple[list[str], list[int], list[int], list[str]]:
    """Per-hospital (plus float pool) categories, load, capacity and float notes of one role."""
    names, load, cap, notes = [], [], [], []
    for h in view.hospitals:
        members = [s for s in h.staff if s.role is role and (s.on_shift or s.load > 0)]
        lent = sum(s.home != h.hospital_id for s in members)
        names.append(h.hospital_id)
        load.append(sum(s.load for s in members))
        cap.append(sum(s.max_patients for s in members))
        notes.append(f"+{lent} float" if lent else "")
    idle = [s for s in view.float_pool if s.role is role and s.hospital_id is None]
    if any(s.hospital_id is None for s in view.float_pool):
        names.append("Float pool")
        load.append(0)
        cap.append(sum(s.max_patients for s in idle))
        notes.append(f"{len(idle)} free" if idle else "")
    return names, load, cap, notes


def staff_load(snapshot: SnapshotRecord) -> go.Figure:
    """Doctor and nurse load against capacity per hospital (and free float staff).

    Capacity is drawn as a pale bar behind the load bar; lent float staff are
    labelled ``+n float`` on the hospital's bar.

    Args:
        snapshot: The snapshot to draw.

    Returns:
        Four bar traces (capacity and load for doctors and nurses).
    """
    fig = go.Figure()
    for role in (StaffRole.DOCTOR, StaffRole.NURSE):
        names, load, cap, notes = _staff_bars(snapshot.view, role)
        solid, pale = ROLE_COLORS[role]
        slot = _LEFT_BAR if role is StaffRole.DOCTOR else _RIGHT_BAR
        label = role.value.capitalize() + "s"
        fig.add_trace(go.Bar(name=f"{label}: capacity", x=names, y=cap, offsetgroup=role.value, **slot,
                             # Outlined so the pale fill still reads on a dark background.
                             marker={"color": pale, "line": {"color": solid, "width": 1.5}},
                             hovertemplate=f"%{{x}}: {label.lower()} capacity %{{y}} patients<extra></extra>"))
        fig.add_trace(go.Bar(name=f"{label}: load", x=names, y=load, offsetgroup=role.value, **slot,
                             marker={"color": solid}, text=notes, textposition="outside", cliponaxis=False,
                             hovertemplate=f"%{{x}}: {label.lower()} carry %{{y}} patients<extra></extra>"))
    _style(fig, f"Staff load vs capacity at t = {snapshot.t_min:g} min", legend_rows=2)
    fig.update_layout(barmode="overlay")
    fig.update_yaxes(title_text="patients", rangemode="tozero")
    return fig


def resource_timeseries(run: RunLog, t: float | None = None) -> go.Figure:
    """Utilisation (percent) of ambulances, responders, ER beds, doctors and nurses over time.

    Args:
        run: The run log.
        t: Current replay minute (vertical line), or None.

    Returns:
        One line trace per resource class.
    """
    rows = utilisation_series(run)
    if not rows:
        return empty_figure("Resource utilisation over time", "No snapshots in this log.")
    fig = go.Figure()
    times = [r["t"] for r in rows]
    for name in UTILISATION_SERIES:
        fig.add_trace(go.Scatter(x=times, y=[r[name] for r in rows], mode="lines", line_shape="hv", name=name,
                                 line={"color": UTILISATION_COLORS[name], "width": 2},
                                 hovertemplate=f"{name}: %{{y:.0f}}% at t=%{{x:g}}<extra></extra>"))
    _style(fig, "Resource utilisation over time")
    fig.update_yaxes(range=[0, 105], ticksuffix="%", title_text="in use")
    fig.update_xaxes(title_text="minute")
    _add_time_marker(fig, t)
    return fig


# --- Comparison and negotiation -------------------------------------------------


def condition_label(condition: str) -> str:
    """Display name of a condition value (unknown values pass through)."""
    return CONDITION_LABELS.get(condition, condition)


def condition_color(condition: str) -> str:
    """Fixed colour of a condition; unknown conditions get a stable spare slot.

    Args:
        condition: Condition value, e.g. ``"full"``.

    Returns:
        A hex colour.
    """
    if condition in CONDITION_COLORS:
        return CONDITION_COLORS[condition]
    return SPARE_CONDITION_SLOTS[zlib.crc32(condition.encode()) % len(SPARE_CONDITION_SLOTS)]


def kpi_comparison(summary_df: pd.DataFrame, metric: str, backend: str | None = None) -> go.Figure:
    """Cross-seed mean ± std of one metric, grouped by scenario, one bar per condition.

    Args:
        summary_df: ``results/summary.csv`` rows (scenario, condition, backend,
            metric, n, mean, std, min, max).
        metric: Metric name to plot.
        backend: Keep only this backend label, or None for all.

    Returns:
        One bar trace per condition present (with std error bars).
    """
    title = f"{METRIC_LABELS.get(metric, metric)}: mean ± std across seeds"
    rows = summary_df[summary_df["metric"] == metric] if not summary_df.empty else summary_df
    if backend is not None and not rows.empty:
        rows = rows[rows["backend"] == backend]
    if rows.empty:
        return empty_figure(title, "No summary rows for this metric.")
    fig = go.Figure()
    scenarios = sorted(rows["scenario"].unique(), key=scenario_sort_key)
    for condition in sorted(rows["condition"].unique(), key=condition_sort_key):
        part = rows[rows["condition"] == condition]
        part = part.sort_values("scenario", key=lambda col: col.map(scenario_sort_key))
        fig.add_trace(go.Bar(
            name=condition_label(condition), x=part["scenario"], y=part["mean"],
            marker={"color": condition_color(condition)},
            error_y={"type": "data", "array": part["std"].fillna(0.0), "visible": True,
                     "color": MUTED, "thickness": 1.5, "width": 4},
            customdata=part[["n", "std"]].fillna(0.0).to_numpy(),
            hovertemplate="%{x}: %{y:.3g} ± %{customdata[1]:.2g} (n=%{customdata[0]})<extra></extra>",
        ))
    _style(fig, title)
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08, barcornerradius=4)
    fig.update_xaxes(categoryorder="array", categoryarray=scenarios)
    fig.update_yaxes(rangemode="tozero", tickformat=".0%" if is_share_metric(metric) else None)
    return fig


def is_share_metric(metric: str) -> bool:
    """True for metrics that are fractions in [0, 1] (rates and utilisations)."""
    return metric.endswith(SHARE_SUFFIXES)


def cumulative_served(runs_by_condition: Mapping[str, RunLog], t: float | None = None) -> go.Figure:
    """Victims whose care has started (cumulative), one step line per condition.

    Args:
        runs_by_condition: Condition value to run log (same scenario and seed).
        t: Current replay minute (vertical line), or None.

    Returns:
        One line trace per condition with snapshots.
    """
    title = "Victims whose care has started (cumulative)"
    fig = go.Figure()
    for condition in sorted(runs_by_condition, key=condition_sort_key):
        snaps = runs_by_condition[condition].snapshots
        if snaps:
            fig.add_trace(go.Scatter(
                x=[s.t_min for s in snaps], y=[s.counts.care_started_cum for s in snaps], mode="lines",
                line_shape="hv", name=condition_label(condition),
                line={"color": condition_color(condition), "width": 2},
                hovertemplate=f"{condition_label(condition)}: %{{y}} at t=%{{x:g}}<extra></extra>",
            ))
    if not fig.data:
        return empty_figure(title, "No snapshots to compare.")
    _style(fig, title)
    fig.update_yaxes(rangemode="tozero", title_text="victims")
    _integer_ticks(fig.update_yaxes, max((max(trace.y) for trace in fig.data), default=0))
    fig.update_xaxes(title_text="minute")
    _add_time_marker(fig, t)
    return fig


def resolution_breakdown(threads: Sequence[ThreadSummary]) -> go.Figure:
    """How negotiation threads ended (count per resolution; ``open`` if unresolved).

    Args:
        threads: Negotiation threads.

    Returns:
        A single horizontal-bar trace, largest count on top.
    """
    title = "How negotiation threads were resolved"
    counts = Counter(th.resolution or "open" for th in threads)
    if not counts:
        return empty_figure(title, "No negotiation threads (B0 does not negotiate).")
    ordered = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]))
    fig = go.Figure(go.Bar(
        orientation="h", y=[k for k, _ in ordered], x=[v for _, v in ordered], marker={"color": SLOT[0]},
        text=[v for _, v in ordered], textposition="outside", cliponaxis=False, width=0.4,
        hovertemplate="%{y}: %{x} thread(s)<extra></extra>",
    ))
    _style(fig, title, height=max(220, 110 + 34 * len(ordered)))
    fig.update_layout(showlegend=False)
    fig.update_xaxes(rangemode="tozero", title_text="threads")
    _integer_ticks(fig.update_xaxes, max(counts.values()))
    fig.update_yaxes(showgrid=False)
    return fig
