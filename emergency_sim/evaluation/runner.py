"""Composition root: runs one simulation or a grid of them and persists results.

``run_simulation`` wires the scenario, the engine, the controller for the
condition (B0 / B0-cues greedy or the agent coordinator) and the decision backend, runs
the event loop to the horizon, computes the metrics and writes:

* ``metrics.json`` (always, when a result directory is known), and
* ``events.jsonl`` (only when a log path is given).

Output layout (deterministic, no absolute paths inside files, so two grids
can be compared with ``diff -r``)::

    out_dir/runs/{scenario}/{condition}/{backend_label}/seed-{seed}_r{repeat}/

The engine, control and SDK packages are imported inside the functions that
need them, so importing this module (e.g. from the dashboard or the CLI's
``report`` command) never requires them.
"""

import dataclasses
import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

from emergency_sim.domain.enums import GREEDY_CONDITIONS, Condition
from emergency_sim.domain.log_records import RunEndRecord, RunStartRecord
from emergency_sim.domain.outcomes import RunMetrics
from emergency_sim.domain.params import SimParams
from emergency_sim.domain.ports import Controller, DecisionBackend, EventSink
from emergency_sim.domain.scenario import Scenario
from emergency_sim.evaluation.metrics import compute_metrics
from emergency_sim.eventlog import MemoryEventSink, write_jsonl
from emergency_sim.scenarios import scenario_for

if TYPE_CHECKING:
    from emergency_sim.policies.params import PolicyParams

LOGGER: Final = logging.getLogger(__name__)

BackendKind = Literal["offline", "llm"]
BACKEND_KINDS: Final = ("offline", "llm")
METRICS_FILE: Final = "metrics.json"
EVENTS_FILE: Final = "events.jsonl"
RUNS_DIR: Final = "runs"


@dataclass(frozen=True)
class RunSpec:
    """Identity of one run.

    Attributes:
        scenario: Scenario name (``normal``, ``surge`` or ``constrained``).
        seed: Scenario generator seed.
        condition: Control condition.
        backend: Decision backend kind (ignored by B0, which has no agents).
        model: LLM model name for ``backend == "llm"``; None means the default.
        repeat: Repeat index (only meaningful for the non-deterministic LLM).
    """

    scenario: str
    seed: int
    condition: Condition
    backend: BackendKind = "offline"
    model: str | None = None
    repeat: int = 0


@dataclass(frozen=True)
class RunResult:
    """Metrics of one finished run.

    Attributes:
        spec: The run's identity (``model`` resolved for LLM runs).
        metrics: Headline metrics.
        log_path: ``events.jsonl`` of the run, if it was kept.
    """

    spec: RunSpec
    metrics: RunMetrics
    log_path: Path | None


ProgressCallback = Callable[[int, int, RunResult], None]
"""Called after each grid run with ``(index starting at 1, total, result)``."""


# --------------------------------------------------------------------------- labels and paths


def resolve_spec(spec: RunSpec) -> RunSpec:
    """Fills in the default LLM model so labels and paths are explicit.

    Args:
        spec: A run spec.

    Returns:
        The same spec, with ``model`` set to ``mas.backend.DEFAULT_LLM_MODEL``
        when it is an LLM spec without a model.
    """
    if spec.backend != "llm" or spec.model is not None:
        return spec
    from emergency_sim.mas.backend import DEFAULT_LLM_MODEL

    return dataclasses.replace(spec, model=DEFAULT_LLM_MODEL)


def backend_label(spec: RunSpec) -> str:
    """Returns the backend label used in run ids, paths and reports.

    Args:
        spec: A run spec (call ``resolve_spec`` first for LLM specs).

    Returns:
        ``"offline"``, or ``"llm-<model>"`` (``"llm"`` if no model is known).
        B0 runs carry the label of their grid's backend so they pair with it.
    """
    if spec.backend == "offline":
        return "offline"
    return f"llm-{spec.model}" if spec.model else "llm"


def run_id(spec: RunSpec, label: str) -> str:
    """Returns the deterministic run id ``{scenario}-{condition}-{label}-s{seed}-r{repeat}``.

    Args:
        spec: The run spec.
        label: Backend label.

    Returns:
        The run id.
    """
    return f"{spec.scenario}-{spec.condition.value}-{label}-s{spec.seed}-r{spec.repeat}"


def run_dir(out_dir: Path, spec: RunSpec, backend_label: str) -> Path:
    """Returns ``out_dir/runs/{scenario}/{condition}/{backend_label}/seed-{seed}_r{repeat}``.

    Args:
        out_dir: Results root.
        spec: The run spec.
        backend_label: Backend label (path separators are replaced by ``_``).

    Returns:
        The run's directory (not created).
    """
    safe_label = backend_label.replace("/", "_").replace("\\", "_")
    return (
        out_dir / RUNS_DIR / spec.scenario / spec.condition.value / safe_label
        / f"seed-{spec.seed}_r{spec.repeat}"
    )


# --------------------------------------------------------------------------- wiring


def build_controller(
    condition: Condition, backend: DecisionBackend | None, sink: EventSink, params: PolicyParams
) -> Controller:
    """Builds the controller of a condition.

    Args:
        condition: Control condition.
        backend: Decision backend; required for every condition except the
            greedy baselines (B0, B0-cues).
        sink: Log sink for messages and agent calls.
        params: Policy parameters.

    Returns:
        ``GreedyController`` for B0 and B0-cues, else ``AgentCoordinator`` with the
        condition's ``NegotiationConfig``.

    Raises:
        ValueError: If an agent condition has no backend.
    """
    if condition in GREEDY_CONDITIONS:
        from emergency_sim.control.greedy import GreedyController

        return GreedyController(params, use_caller_cues=condition == Condition.B0_CUES)
    if backend is None:
        raise ValueError(f"condition {condition.value!r} needs a decision backend")
    from emergency_sim.control.conditions import NEGOTIATION
    from emergency_sim.control.coordinator import AgentCoordinator

    return AgentCoordinator(backend, NEGOTIATION[condition], sink, params)


def make_backend(spec: RunSpec, scenario: Scenario, params: PolicyParams) -> DecisionBackend | None:
    """Creates the run's decision backend (one per run).

    Args:
        spec: The run spec (LLM specs should be resolved).
        scenario: The run's scenario (its hospitals define the hospital agents).
        params: Policy parameters.

    Returns:
        None for the greedy baselines; otherwise the offline or LLM SDK backend.

    Raises:
        ValueError: For an unknown backend kind.
    """
    if spec.condition in GREEDY_CONDITIONS:
        return None
    if spec.backend == "offline":
        from emergency_sim.mas.backend import make_offline_backend

        return make_offline_backend(scenario.hospitals, params)
    if spec.backend == "llm":
        from emergency_sim.mas.backend import make_llm_backend

        model = resolve_spec(spec).model
        return make_llm_backend(scenario.hospitals, params, model=model)
    raise ValueError(f"unknown backend kind {spec.backend!r}")


def _negotiation_dict(condition: Condition) -> dict[str, int | bool] | None:
    """The condition's negotiation config as a plain dict (None for greedy baselines)."""
    if condition in GREEDY_CONDITIONS:
        return None
    from emergency_sim.control.conditions import NEGOTIATION

    return dict(dataclasses.asdict(NEGOTIATION[condition]))


def _start_record(spec: RunSpec, label: str, scenario: Scenario, sim_params: SimParams) -> RunStartRecord:
    """Builds the run's first log record."""
    return RunStartRecord(
        t_min=0.0,
        run_id=run_id(spec, label),
        scenario=spec.scenario,
        seed=spec.seed,
        condition=spec.condition,
        backend=label,
        repeat=spec.repeat,
        sim_params={k: float(v) for k, v in dataclasses.asdict(sim_params).items()},
        negotiation=_negotiation_dict(spec.condition),
        scenario_data=scenario,
    )


# --------------------------------------------------------------------------- persistence


def result_payload(result: RunResult, label: str) -> dict[str, Any]:
    """Returns the JSON-ready content of ``metrics.json`` for a result.

    Args:
        result: The run result.
        label: Backend label of the run.

    Returns:
        ``{"run_id", "backend_label", "spec", "metrics", "events_file"}``;
        paths are relative so the file is identical across output roots.
    """
    spec = result.spec
    return {
        "run_id": run_id(spec, label),
        "backend_label": label,
        "spec": {
            "scenario": spec.scenario, "seed": spec.seed, "condition": spec.condition.value,
            "backend": spec.backend, "model": spec.model, "repeat": spec.repeat,
        },
        "metrics": result.metrics.model_dump(mode="json"),
        "events_file": result.log_path.name if result.log_path is not None else None,
    }


def write_result(result: RunResult, directory: Path, label: str) -> Path:
    """Writes ``metrics.json`` into ``directory``.

    Args:
        result: The run result.
        directory: Run directory (created if missing).
        label: Backend label of the run.

    Returns:
        Path of the written file.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / METRICS_FILE
    text = json.dumps(result_payload(result, label), indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def read_result(path: Path) -> RunResult:
    """Reads a ``metrics.json`` written by ``write_result``.

    Args:
        path: The ``metrics.json`` file.

    Returns:
        The result; ``log_path`` points to the sibling ``events.jsonl`` if it exists.

    Raises:
        KeyError: If a required key is missing.
        ValueError: If a value is invalid (e.g. unknown condition).
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload["spec"]
    spec = RunSpec(
        scenario=raw["scenario"], seed=int(raw["seed"]), condition=Condition(raw["condition"]),
        backend=raw["backend"], model=raw.get("model"), repeat=int(raw.get("repeat", 0)),
    )
    events = path.parent / EVENTS_FILE
    return RunResult(
        spec=spec,
        metrics=RunMetrics.model_validate(payload["metrics"]),
        log_path=events if events.is_file() else None,
    )


# --------------------------------------------------------------------------- running


def run_simulation(
    spec: RunSpec,
    *,
    sim_params: SimParams = SimParams(),
    policy_params: PolicyParams | None = None,
    backend: DecisionBackend | None = None,
    log_path: Path | None = None,
    result_dir: Path | None = None,
) -> RunResult:
    """Runs one simulation to the horizon.

    Args:
        spec: The run spec.
        sim_params: Engine constants.
        policy_params: Policy parameters; None means ``PolicyParams()``.
        backend: Decision backend to use instead of creating one from
            ``spec.backend`` (tests inject fakes here). Ignored by the greedy baselines (B0, B0-cues).
        log_path: Where to write ``events.jsonl``; None keeps no log.
        result_dir: Where to write ``metrics.json``; defaults to the folder
            of ``log_path``. With neither, nothing is written.

    Returns:
        The run result.
    """
    if policy_params is None:
        from emergency_sim.policies.params import PolicyParams

        policy_params = PolicyParams()
    from emergency_sim.engine.engine import Engine
    from emergency_sim.engine.port import EnginePortAdapter

    spec = resolve_spec(spec)
    scenario = scenario_for(spec.scenario, spec.seed)
    if spec.condition in GREEDY_CONDITIONS:
        backend = None
    elif backend is None:
        backend = make_backend(spec, scenario, policy_params)
    label = backend.label if backend is not None else backend_label(spec)
    sink = MemoryEventSink()
    sink.emit(_start_record(spec, label, scenario, sim_params))
    engine = Engine(scenario, sim_params, sink)
    port = EnginePortAdapter(engine)
    controller = build_controller(spec.condition, backend, sink, policy_params)
    while (point := engine.advance()) is not None:
        controller.handle(point, port)
    outcomes = engine.outcomes()
    metrics = compute_metrics(outcomes, sink.records)
    end_t = max(scenario.horizon_min, sink.records[-1].t_min)
    sink.emit(RunEndRecord(t_min=end_t, outcomes=outcomes, metrics=metrics))
    if log_path is not None:
        write_jsonl(sink.records, log_path)
    result = RunResult(spec=spec, metrics=metrics, log_path=log_path)
    target = result_dir if result_dir is not None else (log_path.parent if log_path else None)
    if target is not None:
        write_result(result, target, label)
    return result


def run_grid(
    specs: Sequence[RunSpec],
    out_dir: Path,
    *,
    keep_logs_for_seeds: frozenset[int],
    progress: ProgressCallback | None = None,
) -> list[RunResult]:
    """Runs specs sequentially, each into its ``run_dir``.

    Args:
        specs: Runs in execution order.
        out_dir: Results root.
        keep_logs_for_seeds: Seeds whose ``events.jsonl`` is kept.
        progress: Optional callback after each run.

    Returns:
        Results in the order of ``specs``.
    """
    results: list[RunResult] = []
    for index, raw_spec in enumerate(specs, start=1):
        spec = resolve_spec(raw_spec)
        directory = run_dir(out_dir, spec, backend_label(spec))
        log_path = directory / EVENTS_FILE if spec.seed in keep_logs_for_seeds else None
        LOGGER.info("run %d/%d: %s", index, len(specs), run_id(spec, backend_label(spec)))
        result = run_simulation(spec, log_path=log_path, result_dir=directory)
        results.append(result)
        if progress is not None:
            progress(index, len(specs), result)
    return results
