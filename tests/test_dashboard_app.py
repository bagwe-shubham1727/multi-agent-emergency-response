"""Smoke tests for the Streamlit app with ``AppTest`` on fixture and freshly simulated results in ``tmp_path``."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from dashboard import replay
from emergency_sim.domain.enums import Condition
from emergency_sim.evaluation import runner
from emergency_sim.evaluation.runner import RunSpec
from tests.test_dashboard_replay import write_fixture_results

APP_PATH = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")
TIMEOUT_S = 60
EXPECTED_TABS = ["Map & incidents", "Resources", "Agent activity", "Negotiation history", "KPIs"]


def _app(results_dir: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    """Runs the app once against ``results_dir``."""
    monkeypatch.setenv("RESULTS_DIR", str(results_dir))
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT_S)
    at.run()
    return at


def _assert_ok(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]


def test_app_renders_all_tabs_for_fixture_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(write_fixture_results(tmp_path), monkeypatch)
    _assert_ok(at)
    assert [tab.label for tab in at.tabs][:5] == EXPECTED_TABS
    assert at.sidebar.selectbox(key="condition_a").value == "full"
    assert len(at.get("plotly_chart")) >= 8  # map, 4 resource charts, resolutions, cumulative, cross-seed
    assert {"Served", "RED in window", "Ambulances free"} <= {m.label for m in at.metric}
    assert any("tiny-full-offline-s0-r0" in c.value for c in at.caption)


def test_time_slider_moves_the_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(write_fixture_results(tmp_path), monkeypatch)
    at.sidebar.slider(key="t").set_value(13).run()
    _assert_ok(at)
    assert any("t = **13**" in c.value for c in at.caption)


def test_compare_mode_renders_two_conditions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(write_fixture_results(tmp_path), monkeypatch)
    at.sidebar.toggle(key="compare").set_value(True).run()
    _assert_ok(at)
    assert at.sidebar.selectbox(key="condition_b").value == "b0"
    labels = [tab.label for tab in at.tabs]
    assert "B0 greedy" in labels and "Full negotiation" in labels  # per-condition sub-tabs
    assert any(m.delta is not None for m in at.metric)  # B shows its difference to A


def test_b0_condition_explains_absence_of_agents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(write_fixture_results(tmp_path), monkeypatch)
    at.sidebar.selectbox(key="condition_a").set_value("b0").run()
    _assert_ok(at)
    assert any("no agents" in i.value for i in at.info)


def test_autoplay_toggle_runs_without_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(write_fixture_results(tmp_path), monkeypatch)
    at.sidebar.toggle(key="autoplay").set_value(True).run()
    _assert_ok(at)
    assert at.session_state["play_t"] == 2.0  # started at the slider's 0 and advanced one tick


def test_empty_results_show_guidance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(tmp_path / "no-results", monkeypatch)
    _assert_ok(at)
    assert not at.tabs
    assert "No run logs found" in at.info[0].value


def test_summary_without_logs_still_shows_cross_seed_chart(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = write_fixture_results(tmp_path, conditions=())
    at = _app(root, monkeypatch)
    _assert_ok(at)
    assert len(at.get("plotly_chart")) == 1


# --- A real simulation log (generated in tmp_path, never read from results/) -----------------


REAL_SEED = 1
REAL_T = 90


@pytest.fixture(scope="module")
def real_results(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Simulates ``normal`` seed 1 under Full and B0 into a fresh results folder."""
    root = tmp_path_factory.mktemp("real-results")
    for condition in (Condition.FULL, Condition.B0):
        spec = RunSpec("normal", REAL_SEED, condition)
        log_path = runner.run_dir(root, spec, runner.backend_label(spec)) / replay.LOG_FILE_NAME
        runner.run_simulation(spec, log_path=log_path)
    return root


def _real_full_run(root: Path) -> replay.RunLog:
    ref = next(r for r in replay.discover_runs(root) if r.condition == "full")
    return replay.load_run(ref.path)


def test_app_replays_a_real_simulation_log(real_results: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(real_results, monkeypatch)
    _assert_ok(at)
    assert [tab.label for tab in at.tabs][:5] == EXPECTED_TABS
    assert any(f"normal-full-offline-s{REAL_SEED}-r0" in c.value for c in at.caption)

    at.sidebar.slider(key="t").set_value(REAL_T).run()
    _assert_ok(at)
    run = _real_full_run(real_results)
    threads = replay.threads_from_messages(replay.messages_until(run, REAL_T))
    tiles = {m.label: m.value for m in at.metric}
    assert tiles["Negotiation threads"] == str(len(threads))
    assert tiles["Still open"] == str(sum(th.resolution is None for th in threads))
    # Drill-downs follow the replay minute: newest decision point and thread at t.
    latest = max(x for x in replay.decision_times(run) if x <= REAL_T)
    assert at.select_slider(key="dp-full").value == latest
    assert threads, "normal seed 1 negotiates before t = 90"
    assert at.selectbox(key="thread-full").value == threads[-1].conversation_id


def test_real_log_compare_mode_shows_percentage_point_deltas(
        real_results: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _app(real_results, monkeypatch)
    at.sidebar.toggle(key="compare").set_value(True).run()
    _assert_ok(at)
    assert at.sidebar.selectbox(key="condition_b").value == "b0"
    served = [m for m in at.metric if m.label == "Served"]  # A at t, A final, B at t, B final
    assert len(served) == 4 and not served[1].delta
    final_delta = served[3].delta
    assert final_delta == "0" or final_delta.endswith(" pp")  # a share differs in percentage points
