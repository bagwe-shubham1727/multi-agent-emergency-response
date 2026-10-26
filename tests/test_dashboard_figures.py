"""Tests for ``dashboard.figures``: each builder returns the expected traces."""

import pandas as pd
import plotly.graph_objects as go
import pytest

from dashboard import figures, replay
from emergency_sim import geo
from emergency_sim.domain.enums import Condition
from tests.test_dashboard_replay import HORIZON, fixture_run, summary_frame


@pytest.fixture(scope="module")
def run() -> replay.RunLog:
    return fixture_run()


def _names(fig: go.Figure) -> list[str]:
    return [trace.name for trace in fig.data]


def _assert_theme_neutral(fig: go.Figure) -> None:
    assert fig.layout.paper_bgcolor == figures.TRANSPARENT
    assert fig.layout.plot_bgcolor == figures.TRANSPARENT


# --- Map ------------------------------------------------------------------------------


def test_incident_map_before_triage(run: replay.RunLog) -> None:
    fig = figures.incident_map(replay.snapshot_at(run, 0), run.start)
    assert _names(fig) == ["Route to destination", "Station", "Hospital (darker = fuller ER)", "Untriaged",
                           "ALS ambulance (hollow = free)", "BLS ambulance (hollow = free)",
                           "Responder team (hollow = free)"]
    routes = fig.data[0]
    assert list(routes.x) == [6.0, 7.0, None] and list(routes.y) == [5.0, 5.0, None]  # AMB-01 -> INC-01
    untriaged = fig.data[3]
    assert untriaged.marker.color == figures.TRIAGE_COLORS[replay.UNTRIAGED]
    assert tuple(untriaged.marker.symbol) == ("circle-open",)
    assert tuple(fig.data[5].marker.symbol) == ("circle-open",)  # BLS free -> hollow
    _assert_theme_neutral(fig)


def test_incident_map_colours_by_worst_triage_and_marks_cleared(run: replay.RunLog) -> None:
    mid = figures.incident_map(replay.snapshot_at(run, 10), run.start)
    assert _names(mid)[3:5] == ["RED (immediate)", "YELLOW (delayed)"]
    assert mid.data[3].marker.color == figures.TRIAGE_COLORS["RED"]
    hospitals = mid.data[2]
    assert list(hospitals.text) == ["H1 · 50% beds", "H2 · 33% beds"]
    assert list(hospitals.marker.color) == pytest.approx([2 / 4, 1 / 3])  # (occupied + reserved) / capacity
    late = figures.incident_map(replay.snapshot_at(run, 12), run.start)
    yellow = next(t for t in late.data if t.name == "YELLOW (delayed)")
    assert tuple(yellow.marker.symbol) == ("x",)


def test_incident_map_axes_use_street_names(run: replay.RunLog) -> None:
    fig = figures.incident_map(replay.snapshot_at(run, 0), run.start)
    assert tuple(fig.layout.xaxis.ticktext) == tuple(geo.AVENUES)
    assert tuple(fig.layout.yaxis.ticktext) == tuple(geo.STREETS)
    assert fig.layout.yaxis.scaleanchor == "x"


# --- Resources ------------------------------------------------------------------------------


def test_ambulance_timeline_has_one_trace_per_status_group(run: replay.RunLog) -> None:
    fig = figures.ambulance_timeline(run, t=7)
    assert _names(fig) == list(replay.STATUS_GROUP_ORDER[:5])  # nobody unavailable in the fixture
    assert all(trace.orientation == "h" for trace in fig.data)
    assert sum(len(trace.x) for trace in fig.data) == len(replay.unit_intervals(run))
    assert any(shape.x0 == 7 for shape in fig.layout.shapes)  # time marker


def test_er_occupancy_stacks_beds_and_shows_queue(run: replay.RunLog) -> None:
    fig = figures.er_occupancy(replay.snapshot_at(run, 13))
    assert _names(fig) == ["Occupied", "Reserved (inbound)", "Free", "Queue"]
    occupied, reserved, free, queue = fig.data
    assert list(occupied.y) == [2, 1] and list(free.base) == [2, 1] and list(queue.y) == [0, 1]
    assert occupied.offsetgroup == reserved.offsetgroup == free.offsetgroup != queue.offsetgroup


def test_staff_load_labels_lent_float_staff(run: replay.RunLog) -> None:
    fig = figures.staff_load(replay.snapshot_at(run, 15))
    assert _names(fig) == ["Doctors: capacity", "Doctors: load", "Nurses: capacity", "Nurses: load"]
    doctor_load = fig.data[1]
    assert list(doctor_load.x) == ["H1", "H2"] and list(doctor_load.text) == ["", "+1 float"]
    before = figures.staff_load(replay.snapshot_at(run, 10))
    assert list(before.data[0].x) == ["H1", "H2", "Float pool"] and list(before.data[0].y) == [4, 4, 4]


def test_resource_timeseries_has_one_line_per_resource(run: replay.RunLog) -> None:
    fig = figures.resource_timeseries(run, t=3)
    assert _names(fig) == list(replay.UTILISATION_SERIES)
    assert len(fig.data[0].x) == int(HORIZON)
    assert fig.layout.yaxis.ticksuffix == "%"


def test_run_charts_without_snapshots_are_placeholders(run: replay.RunLog) -> None:
    bare = replay.run_log_from_records([replay.all_records(run)[0]])
    assert figures.ambulance_timeline(bare).data == ()
    assert figures.resource_timeseries(bare).data == ()
    assert figures.cumulative_served({"full": bare}).data == ()


# --- Comparison ----------------------------------------------------------------------------------


def test_kpi_comparison_bars_with_error_bars() -> None:
    fig = figures.kpi_comparison(summary_frame(), "served_rate")
    assert _names(fig) == ["B0 greedy", "B1 agents", "Full negotiation"]
    full = fig.data[2]
    assert list(full.x) == ["surge", "tiny"] and full.error_y.visible
    assert full.marker.color == figures.CONDITION_COLORS["full"]


def test_kpi_comparison_colour_follows_condition_not_rank() -> None:
    frame = summary_frame()
    fig = figures.kpi_comparison(frame[frame["condition"] != "b0"], "served_rate")
    assert [t.marker.color for t in fig.data] == [figures.CONDITION_COLORS["b1"], figures.CONDITION_COLORS["full"]]


@pytest.mark.parametrize(("metric", "backend"), [("missing", None), ("served_rate", "llm-x")])
def test_kpi_comparison_without_rows_is_a_placeholder(metric: str, backend: str | None) -> None:
    fig = figures.kpi_comparison(summary_frame(), metric, backend)
    assert fig.data == () and fig.layout.annotations


def test_kpi_comparison_on_empty_frame() -> None:
    assert figures.kpi_comparison(pd.DataFrame(), "served_rate").data == ()


def test_cumulative_served_one_line_per_condition() -> None:
    runs = {"full": fixture_run(), "b0": fixture_run(Condition.B0)}
    fig = figures.cumulative_served(runs, t=5)
    assert _names(fig) == ["B0 greedy", "Full negotiation"]
    assert max(fig.data[1].y) == 2 and fig.data[1].line.shape == "hv"


def test_resolution_breakdown_counts_threads(run: replay.RunLog) -> None:
    threads = replay.threads_from_messages(replay.messages_until(run, 12))
    fig = figures.resolution_breakdown(threads)
    assert len(fig.data) == 1 and fig.layout.showlegend is False
    assert dict(zip(fig.data[0].y, fig.data[0].x, strict=True)) == {
        "open": 1, "ACCEPTED_AFTER_REJECT": 1, "COUNTER_GRANTED": 1}


def test_resolution_breakdown_without_threads() -> None:
    fig = figures.resolution_breakdown([])
    assert fig.data == () and "B0" in fig.layout.annotations[0].text


def test_condition_label_passes_unknown_through() -> None:
    assert figures.condition_label("full") == "Full negotiation"
    assert figures.condition_label("custom") == "custom"


def test_condition_color_is_stable_and_spare_for_unknown_conditions() -> None:
    assert figures.condition_color("full") == figures.CONDITION_COLORS["full"]
    unknown = figures.condition_color("b0-new-variant")
    assert unknown in figures.SPARE_CONDITION_SLOTS and unknown == figures.condition_color("b0-new-variant")
    assert not set(figures.SPARE_CONDITION_SLOTS) & set(figures.CONDITION_COLORS.values())


def test_kpi_comparison_orders_scenarios_and_formats_shares() -> None:
    frame = summary_frame()
    frame = pd.concat([frame, frame[frame["scenario"] == "tiny"].assign(scenario="normal")], ignore_index=True)
    shares = figures.kpi_comparison(frame, "served_rate")
    assert list(shares.layout.xaxis.categoryarray) == ["normal", "surge", "tiny"]
    assert list(shares.data[0].x) == ["normal", "surge", "tiny"]
    assert shares.layout.yaxis.tickformat == ".0%"
    assert figures.kpi_comparison(frame, "response_time_mean_min").layout.yaxis.tickformat is None


def test_incident_map_fades_cleared_incidents_and_lists_open_first(run: replay.RunLog) -> None:
    late = figures.incident_map(replay.snapshot_at(run, 12), run.start)
    yellow = next(t for t in late.data if t.name == "YELLOW (delayed)")
    assert tuple(yellow.marker.opacity) == (figures.CLEARED_OPACITY,)
    assert tuple(yellow.marker.size) == (figures.CLEARED_SIZE,)
    red = next(t for t in late.data if t.name == "RED (immediate)")
    assert tuple(red.marker.opacity) == (1.0,)  # INC-01 still open


def test_incident_map_keeps_the_city_square_and_the_legend_below(run: replay.RunLog) -> None:
    fig = figures.incident_map(replay.snapshot_at(run, 0), run.start)
    assert fig.layout.xaxis.constrain == "domain" and fig.layout.yaxis.constrain == "domain"
    assert fig.layout.legend.yanchor == "top" and fig.layout.legend.y < 0
    assert fig.data[2].marker.showscale is False  # bed use is printed next to each hospital instead
