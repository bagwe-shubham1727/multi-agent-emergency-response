"""End-to-end runs of the composition root on the committed scenarios (WP6 step 1).

Every scenario (``normal``, ``surge``, ``constrained``) is run once per
condition with seed 7380 and the offline backend, through the real
``evaluation.runner.run_simulation``. The only test hook is a controller
decorator (installed by patching ``runner.build_controller``) that checks the
engine invariants after every ``advance`` and after every ``handle``; it does
not change what is logged, which ``test_identical_runs_are_byte_identical``
confirms by comparing against unwrapped runs.

The tests assert structural, protocol and consistency properties only (never
outcome numbers such as served rates), so they stay valid while the policies
are tuned.

Message-type coverage, seed 7380 offline (checked when this file was written):
every ``MessageType`` occurs in the Full runs of ``surge`` and ``constrained``;
``normal`` Full uses only PRIORITY_REPORT, INFORM, PROPOSE and ACCEPT (no
hospital ever refuses there). B1 never sends COUNTER, ESCALATE, REQUEST or
COMMIT. The ablations also negotiate; ``full-no-escalation`` sends no ESCALATE
at all (RESERVE_RELEASE counters need escalation), and its COMMITs come only
from granted float-staff REQUESTs.
"""

import collections
import dataclasses
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from emergency_sim.engine import port as engine_port
from emergency_sim.engine.engine import Engine
from emergency_sim.engine.port import EnginePortAdapter

from emergency_sim import cli
from emergency_sim.domain.actions import Action
from emergency_sim.domain.enums import (
    GREEDY_CONDITIONS,
    Condition,
    MessageType,
    Resolution,
    SimEventType,
)
from emergency_sim.domain.log_records import (
    ActionRecord,
    AgentCallRecord,
    LogRecord,
    MessageRecord,
    RunEndRecord,
    RunStartRecord,
    SimEventRecord,
)
from emergency_sim.domain.messages import CommitAction, Message
from emergency_sim.domain.ports import Controller, DecisionBackend, EnginePort, EventSink
from emergency_sim.domain.views import DecisionPoint
from emergency_sim.evaluation import report, runner
from emergency_sim.eventlog import read_log
from emergency_sim.policies.params import PolicyParams

SEED = 7380
SCENARIOS = ("normal", "surge", "constrained")
MAIN_CONDITIONS = (Condition.B0, Condition.B0_CUES, Condition.B1, Condition.FULL)
ABLATIONS = (Condition.FULL_NO_FLOAT, Condition.FULL_NO_DIVERSION, Condition.FULL_NO_ESCALATION)
ALL_CONDITIONS = MAIN_CONDITIONS + ABLATIONS
AGENT_CONDITIONS = tuple(c for c in ALL_CONDITIONS if c not in GREEDY_CONDITIONS)
B1_FORBIDDEN_TYPES = frozenset(
    {MessageType.COUNTER, MessageType.ESCALATE, MessageType.REQUEST, MessageType.COMMIT}
)
RESOLVING_TYPES = frozenset({MessageType.ACCEPT, MessageType.COMMIT})

RunKey = tuple[str, Condition]


def _key_id(key: RunKey) -> str:
    """Readable pytest id ``scenario-condition`` for a run key."""
    scenario, condition = key
    return f"{scenario}-{condition.value}"


def _keys(conditions: Sequence[Condition]) -> list[RunKey]:
    """All (scenario, condition) pairs for the given conditions."""
    return [(s, c) for s in SCENARIOS for c in conditions]


# --------------------------------------------------------------------------- harness


@dataclasses.dataclass(frozen=True)
class InvariantViolation:
    """Engine invariant problems seen at one decision point.

    Attributes:
        t_min: Decision-point time.
        stage: ``"after advance"`` or ``"after handle"``.
        problems: The engine's ``check_invariants`` output.
    """

    t_min: float
    stage: str
    problems: tuple[str, ...]


class InvariantCheckingController:
    """``Controller`` decorator that checks engine invariants around each ``handle``.

    ``handle`` is called right after ``Engine.advance`` returned a decision
    point, so checking before delegating covers "after every advance". The
    controller only receives the narrow port, so the harness records the
    engine itself when the runner builds that port.
    """

    def __init__(self, inner: Controller, engines: list[Engine]) -> None:
        """Wraps a controller.

        Args:
            inner: The real controller of the condition.
            engines: Engines recorded when ports were built (the last is live).
        """
        self._inner = inner
        self._engines = engines
        self.engine: Engine | None = None
        self.decision_points = 0
        self.violations: list[InvariantViolation] = []

    def handle(self, point: DecisionPoint, port: EnginePort) -> None:
        """Checks invariants, delegates, then checks again.

        Args:
            point: The decision point.
            port: The engine.
        """
        self.engine = self._engines[-1]
        self.decision_points += 1
        self._check(self.engine, point.t_min, "after advance")
        self._inner.handle(point, port)
        self._check(self.engine, point.t_min, "after handle")

    def final_problems(self) -> tuple[str, ...]:
        """Invariant problems of the finished engine (empty if it never decided)."""
        return self._invariants(self.engine) if self.engine is not None else ()

    def _check(self, engine: Engine, t_min: float, stage: str) -> None:
        problems = self._invariants(engine)
        if problems:
            self.violations.append(InvariantViolation(t_min, stage, problems))

    @staticmethod
    def _invariants(engine: Engine) -> tuple[str, ...]:
        return tuple(engine.check_invariants())


@dataclasses.dataclass(frozen=True)
class CheckedRun:
    """One finished end-to-end run and what the harness observed.

    Attributes:
        spec: The run spec.
        result: ``run_simulation``'s result.
        log_path: The run's ``events.jsonl``.
        records: ``read_log(log_path)``.
        decision_points: Decision points handled.
        violations: Invariant problems during the run.
        final_problems: Invariant problems after the horizon.
    """

    spec: runner.RunSpec
    result: runner.RunResult
    log_path: Path
    records: tuple[LogRecord, ...]
    decision_points: int
    violations: tuple[InvariantViolation, ...]
    final_problems: tuple[str, ...]

    @property
    def messages(self) -> list[Message]:
        """Messages in log order."""
        return [r.message for r in self.records if isinstance(r, MessageRecord)]

    @property
    def agent_calls(self) -> list[AgentCallRecord]:
        """Agent-call records in log order."""
        return [r for r in self.records if isinstance(r, AgentCallRecord)]


def run_checked(spec: runner.RunSpec, out_dir: Path) -> CheckedRun:
    """Runs ``run_simulation`` with an invariant-checking controller.

    Args:
        spec: The run spec (offline backend).
        out_dir: Directory for ``events.jsonl`` and ``metrics.json``.

    Returns:
        The checked run.
    """
    original = runner.build_controller
    wrappers: list[InvariantCheckingController] = []
    engines: list[Engine] = []

    class RecordingAdapter(EnginePortAdapter):
        """Port adapter that also records the engine it wraps."""

        __slots__ = ()

        def __init__(self, engine: Engine) -> None:
            super().__init__(engine)
            engines.append(engine)

    def build(
        condition: Condition, backend: DecisionBackend | None, sink: EventSink, params: PolicyParams
    ) -> Controller:
        wrapper = InvariantCheckingController(original(condition, backend, sink, params), engines)
        wrappers.append(wrapper)
        return wrapper

    log_path = out_dir / runner.EVENTS_FILE
    with pytest.MonkeyPatch.context() as patch:
        patch.delenv("OPENAI_API_KEY", raising=False)
        patch.setattr(runner, "build_controller", build)
        patch.setattr(engine_port, "EnginePortAdapter", RecordingAdapter)
        result = runner.run_simulation(spec, log_path=log_path)
    (wrapper,) = wrappers
    return CheckedRun(
        spec=spec,
        result=result,
        log_path=log_path,
        records=tuple(read_log(log_path)),
        decision_points=wrapper.decision_points,
        violations=tuple(wrapper.violations),
        final_problems=wrapper.final_problems(),
    )


@pytest.fixture(scope="session")
def e2e_runs(tmp_path_factory: pytest.TempPathFactory) -> Mapping[RunKey, CheckedRun]:
    """Every scenario x condition (seed 7380, offline), each run exactly once."""
    root = tmp_path_factory.mktemp("e2e")
    return {
        key: run_checked(runner.RunSpec(key[0], SEED, key[1]), root / _key_id(key))
        for key in _keys(ALL_CONDITIONS)
    }


@pytest.fixture(params=_keys(ALL_CONDITIONS), ids=_key_id)
def any_run(request: pytest.FixtureRequest, e2e_runs: Mapping[RunKey, CheckedRun]) -> CheckedRun:
    """Each run in turn (all six conditions)."""
    return e2e_runs[request.param]


@pytest.fixture(params=_keys(AGENT_CONDITIONS), ids=_key_id)
def agent_run(request: pytest.FixtureRequest, e2e_runs: Mapping[RunKey, CheckedRun]) -> CheckedRun:
    """Each run of an agent condition (B1, Full and the ablations) in turn."""
    return e2e_runs[request.param]


def _message_types(runs: Sequence[CheckedRun]) -> collections.Counter[MessageType]:
    """Counts message types over runs."""
    return collections.Counter(m.type for run in runs for m in run.messages)


# --------------------------------------------------------------------------- run health


def test_run_completes_with_clean_invariants(any_run: CheckedRun) -> None:
    assert any_run.decision_points > 0
    assert not any_run.violations, "\n".join(
        f"t={v.t_min} {v.stage}: {'; '.join(v.problems)}" for v in any_run.violations[:10]
    )
    assert not any_run.final_problems, any_run.final_problems
    end = any_run.records[-1]
    assert isinstance(end, RunEndRecord)
    assert end.metrics == any_run.result.metrics


def test_no_fallbacks(any_run: CheckedRun) -> None:
    replaced = [
        (c.seq, c.agent, c.source, [n.rule for n in c.fallbacks])
        for c in any_run.agent_calls if c.source == "policy_fallback" or c.fallbacks
    ]
    assert not replaced, f"offline agent outputs needed fallbacks: {replaced[:10]}"
    assert any_run.result.metrics.fallbacks == 0


def test_no_controller_omissions(any_run: CheckedRun) -> None:
    omissions = [
        f"t={r.t_min} {r.unit_id} ({r.incident_id}) -> {r.hospital_id}"
        for r in any_run.records
        if isinstance(r, SimEventRecord) and r.event is SimEventType.CONTROLLER_OMISSION
    ]
    assert not omissions, (
        "the controller left loaded units without a destination and the engine guard "
        f"had to route them: {omissions}"
    )
    assert any_run.result.metrics.controller_omissions == 0


def test_metrics_agree_with_the_log(any_run: CheckedRun) -> None:
    metrics = any_run.result.metrics
    types = _message_types([any_run])
    assert metrics.messages_total == len(any_run.messages)
    assert metrics.agent_calls == len(any_run.agent_calls)
    assert {k: v for k, v in metrics.messages_by_type.items() if v} == {t.value: n for t, n in types.items()}


# --------------------------------------------------------------------------- condition protocol


@pytest.mark.parametrize("condition", sorted(GREEDY_CONDITIONS))
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_greedy_baselines_have_no_messages_or_agent_calls(
    scenario: str, condition: Condition, e2e_runs: Mapping[RunKey, CheckedRun]
) -> None:
    run = e2e_runs[scenario, condition]
    assert run.messages == [] and run.agent_calls == []
    assert run.result.metrics.messages_total == 0 and run.result.metrics.agent_calls == 0
    start = run.records[0]
    assert isinstance(start, RunStartRecord) and start.negotiation is None


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_b1_never_counters_escalates_requests_or_commits(
    scenario: str, e2e_runs: Mapping[RunKey, CheckedRun]
) -> None:
    run = e2e_runs[scenario, Condition.B1]
    assert run.agent_calls, "B1 should call agents"
    forbidden = sorted(t.value for t in _message_types([run]) if t in B1_FORBIDDEN_TYPES)
    assert not forbidden, f"B1 sent {forbidden}"


def test_every_message_type_appears_across_full_runs(e2e_runs: Mapping[RunKey, CheckedRun]) -> None:
    full = _message_types([e2e_runs[s, Condition.FULL] for s in SCENARIOS])
    missing = [t for t in MessageType if full[t] == 0]
    if missing:
        elsewhere = {
            t.value: [_key_id(k) for k, run in e2e_runs.items() if _message_types([run])[t]]
            for t in missing
        }
        pytest.fail(
            f"message types never sent in the seed-{SEED} Full runs: {[t.value for t in missing]}; "
            f"runs of other conditions that do send them: {elsewhere}"
        )


# --------------------------------------------------------------------------- message protocol


def test_messages_are_well_formed(agent_run: CheckedRun) -> None:
    messages = agent_run.messages
    assert messages, "agent conditions always exchange messages"
    by_id: dict[str, Message] = {}
    for index, message in enumerate(messages, start=1):
        assert message.message_id == f"M-{index:04d}"
        assert message.sender and message.recipient and message.incident_id, message
        assert message.conversation_id.startswith("NEG-"), message
        assert message.proposed_action is not None and isinstance(message.type, MessageType)
        assert message.sender != message.recipient, message
        if message.in_reply_to is not None:
            parent = by_id.get(message.in_reply_to)
            assert parent is not None, f"{message.message_id} replies to unknown/later {message.in_reply_to}"
            assert parent.conversation_id == message.conversation_id, (
                f"{message.message_id} ({message.conversation_id}) replies to "
                f"{parent.message_id} ({parent.conversation_id})"
            )
            assert parent.t_min <= message.t_min
        by_id[message.message_id] = message


def test_replies_reference_the_right_message_types(agent_run: CheckedRun) -> None:
    by_id = {m.message_id: m for m in agent_run.messages}
    for message in agent_run.messages:
        if message.type in (MessageType.ACCEPT, MessageType.COUNTER):
            assert message.in_reply_to is not None, message
            assert by_id[message.in_reply_to].type is MessageType.PROPOSE, message
        if message.type is MessageType.PRIORITY_REPORT:
            assert message.in_reply_to is None and message.round == 0, message


def test_accept_and_commit_actions_were_applied(agent_run: CheckedRun) -> None:
    applied: dict[tuple[float, str, str], int] = {}
    for record in agent_run.records:
        if isinstance(record, ActionRecord) and record.ok:
            applied.setdefault((record.t_min, record.actor, record.action.model_dump_json()), record.seq)
    checked = 0
    for record in agent_run.records:
        if not isinstance(record, MessageRecord) or record.message.type not in RESOLVING_TYPES:
            continue
        message = record.message
        content = message.proposed_action
        assert isinstance(content, CommitAction), f"{message.message_id} carries {content!r}"
        if message.type is MessageType.ACCEPT:
            assert content.resolution in (Resolution.ACCEPTED, Resolution.ACCEPTED_AFTER_REJECT), message
            assert len(content.actions) == 1, message
        if content.resolution is Resolution.ENGINE_REJECTED:
            continue
        for action in content.actions:
            seq = applied.get((message.t_min, message.sender, _action_json(action)))
            assert seq is not None, (
                f"{message.message_id} {message.type.value} claims {action!r} by {message.sender} "
                f"at t={message.t_min}, but no successful ActionRecord matches"
            )
            assert seq < record.seq, f"{message.message_id} was sent before its action was applied"
            checked += 1
    resolving = sum(1 for m in agent_run.messages if m.type in RESOLVING_TYPES)
    assert resolving == 0 or checked > 0


def _action_json(action: Action) -> str:
    """Canonical JSON of an action, for matching against ``ActionRecord``s."""
    return action.model_dump_json()


# --------------------------------------------------------------------------- log format


def test_log_ordering(any_run: CheckedRun) -> None:
    records = any_run.records
    assert isinstance(records[0], RunStartRecord) and isinstance(records[-1], RunEndRecord)
    assert sum(isinstance(r, RunStartRecord) for r in records) == 1
    assert sum(isinstance(r, RunEndRecord) for r in records) == 1
    assert [r.seq for r in records] == list(range(len(records)))
    times = [r.t_min for r in records]
    regressions = [(records[i].seq, times[i - 1], times[i]) for i in range(1, len(times)) if times[i] < times[i - 1]]
    assert not regressions, f"t_min went backwards at (seq, prev, now): {regressions[:5]}"
    for record in records:
        if isinstance(record, MessageRecord):
            assert record.t_min == record.message.t_min


def test_read_log_round_trips_every_line(any_run: CheckedRun) -> None:
    lines = any_run.log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(any_run.records)
    assert [r.model_dump_json() for r in any_run.records] == lines
    start = any_run.records[0]
    assert isinstance(start, RunStartRecord)
    assert (start.scenario, start.seed, start.condition) == (
        any_run.spec.scenario, SEED, any_run.spec.condition,
    )


@pytest.mark.parametrize("key", _keys(MAIN_CONDITIONS), ids=_key_id)
def test_identical_runs_are_byte_identical(
    key: RunKey, e2e_runs: Mapping[RunKey, CheckedRun], tmp_path: Path
) -> None:
    first = e2e_runs[key]
    second = tmp_path / runner.EVENTS_FILE
    runner.run_simulation(runner.RunSpec(key[0], SEED, key[1]), log_path=second)
    assert second.read_bytes() == first.log_path.read_bytes()
    assert (tmp_path / runner.METRICS_FILE).read_bytes() == (first.log_path.parent / runner.METRICS_FILE).read_bytes()


# --------------------------------------------------------------------------- CLI grid


@pytest.mark.slow
def test_cli_grid_and_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    conditions = [c.value for c in ALL_CONDITIONS]
    code = cli.main([
        "grid", "--scenarios", "surge", "--conditions", *conditions, "--seeds", "1-2",
        "--keep-logs-for", "1", "--backend", "offline", "--out", str(tmp_path),
    ])
    assert code == 0, capsys.readouterr().err
    metrics_files = sorted(tmp_path.rglob(runner.METRICS_FILE))
    assert len(metrics_files) == len(conditions) * 2
    logs = sorted(tmp_path.rglob(runner.EVENTS_FILE))
    assert len(logs) == len(conditions)
    assert all(p.parent.name == "seed-1_r0" for p in logs)
    for log in logs:
        assert isinstance(read_log(log)[-1], RunEndRecord)

    outputs = [tmp_path / name for name in (report.SUMMARY_CSV, report.PAIRED_CSV, report.SUMMARY_MD)]
    for path in outputs:
        assert path.is_file(), f"grid did not write {path.name}"
        path.unlink()
    assert cli.main(["report", "--results", str(tmp_path)]) == 0, capsys.readouterr().err
    for path in outputs:
        assert path.is_file() and path.stat().st_size > 0, f"report did not write {path.name}"
    summary = (tmp_path / report.SUMMARY_CSV).read_text(encoding="utf-8")
    assert all(c in summary for c in conditions)
    assert len((tmp_path / report.PAIRED_CSV).read_text(encoding="utf-8").splitlines()) > 1


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_no_escalation_ablation_never_escalates(
    scenario: str, e2e_runs: Mapping[RunKey, CheckedRun]
) -> None:
    run = e2e_runs[scenario, Condition.FULL_NO_ESCALATION]
    assert MessageType.ESCALATE not in _message_types([run])
