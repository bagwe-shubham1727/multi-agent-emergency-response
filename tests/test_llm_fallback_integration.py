"""A Full run with misbehaving LLM-style agents stays consistent (WP6 step 2).

The backend is an ``SdkBackend`` in LLM mode (``fallback_on_error=True``)
whose agents run through the real ``Runner``:

* ``dispatch`` is an ``agents.testing.ScriptedModel`` that answers with
  schema-valid JSON containing illegal orders (an unknown unit and a busy
  unit) on top of otherwise sensible orders;
* every ``hospital_*`` agent is a ``ScriptedModel`` that answers with
  malformed JSON;
* ``incident_assessment`` and ``resource_allocation`` use the offline policy
  model (valid answers).

The run is driven like ``run_simulation`` (engine + ``AgentCoordinator``) on a
tiny factory scenario, through an auditing port that records the engine
state before every action. Nothing reaches the network.
"""

import socket
from collections.abc import Callable
from dataclasses import dataclass, field

import pytest
from agents import RunConfig
from agents.items import TResponseOutputItem
from agents.testing import ModelCall, ModelStep, ScriptedModel, assistant_message

from emergency_sim.control.conditions import NEGOTIATION
from emergency_sim.control.coordinator import AgentCoordinator
from emergency_sim.domain.actions import Action, ActionCheck, AssignTransport, DispatchUnit, DivertUnit
from emergency_sim.domain.agent_io import DispatchInput, DispatchOutput, UnitOrder
from emergency_sim.domain.enums import Condition, UnitStatus
from emergency_sim.domain.log_records import ActionRecord, AgentCallRecord
from emergency_sim.domain.messages import DISPATCH_AGENT
from emergency_sim.domain.outcomes import RunMetrics
from emergency_sim.domain.params import SimParams
from emergency_sim.domain.scenario import Scenario
from emergency_sim.domain.views import PublicView
from emergency_sim.engine.engine import Engine
from emergency_sim.evaluation.metrics import compute_metrics
from emergency_sim.evaluation.runner import RunSpec, run_simulation
from emergency_sim.eventlog import MemoryEventSink
from emergency_sim.mas import BackendUnavailableError, SdkBackend, make_llm_backend
from emergency_sim.mas.backend import build_agents, make_offline_backend
from emergency_sim.mas.policy_model import PolicyModel
from emergency_sim.mas.specs import AgentSpec, agent_specs, specs_by_name
from emergency_sim.policies.params import PolicyParams
from tests import factories as f

PARAMS = PolicyParams()
LABEL = "llm-scripted"
BUSY_UNIT = "AMB-03"
BUSY_UNTIL_MIN = 150
UNKNOWN_UNIT = "AMB-99"
REQUIRED_STATUS: dict[type, UnitStatus] = {
    DispatchUnit: UnitStatus.AVAILABLE,
    DivertUnit: UnitStatus.EN_ROUTE_SCENE,
    AssignTransport: UnitStatus.AWAITING_DESTINATION,
}
"""Status a unit must have (before the action) for a successful action on it."""

Responder = Callable[[ModelCall], list[TResponseOutputItem]]


def fallback_scenario() -> Scenario:
    """The tiny world plus a unit busy until minute 150 and a second, later incident."""
    return f.make_scenario(
        hospitals=[f.hospital("H1", caps=("trauma", "general"), er_capacity=4, reserve=1)],
        ambulances=[
            f.ambulance("AMB-01", "ALS"),
            f.ambulance("AMB-02", "BLS"),
            f.ambulance(BUSY_UNIT, "ALS", status="busy", busy_until=BUSY_UNTIL_MIN),
        ],
        staff=[
            f.doctor("DR-H1-1", "H1", ("emergency", "trauma")),
            f.nurse("RN-H1-1", "H1", ("emergency", "trauma")),
        ],
        incidents=[
            f.incident("INC-01", 0, 7.0, 5.0, [
                f.victim("INC-01-V1", "RED", "trauma", condition="stab wound"),
                f.victim("INC-01-V2", "GREEN", "general"),
            ]),
            f.incident("INC-02", 20, 3.0, 6.0, [f.victim("INC-02-V1", "YELLOW", "general")]),
        ],
    )


# --------------------------------------------------------------------------- scripted models


def payload_text(call: ModelCall) -> str:
    """Returns the JSON payload the backend sent as the run's user message.

    Args:
        call: A recorded model call.

    Returns:
        The payload JSON.
    """
    if isinstance(call.input, str):
        return call.input
    content = call.input[0]["content"]
    if not isinstance(content, str):
        raise AssertionError(f"unexpected first input item {call.input[0]!r}")
    return content


def endless_model(respond: Responder) -> ScriptedModel:
    """A ``ScriptedModel`` that answers every call with ``respond`` (it re-enqueues itself).

    Args:
        respond: Builds the output items of one model turn.

    Returns:
        The model.
    """
    model = ScriptedModel()

    def step(call: ModelCall) -> list[TResponseOutputItem]:
        model.enqueue(ModelStep(responder=step))
        return respond(call)

    model.enqueue(ModelStep(responder=step))
    return model


def illegal_dispatch(policy: Callable[[DispatchInput, PolicyParams], DispatchOutput]) -> Responder:
    """Dispatch answers: the policy's output plus orders for an unknown and a busy unit.

    Args:
        policy: The dispatch policy (source of the legal part).

    Returns:
        The responder.
    """

    def respond(call: ModelCall) -> list[TResponseOutputItem]:
        payload = DispatchInput.model_validate_json(payload_text(call))
        answer = policy(payload, PARAMS)
        target = payload.incidents[0].incident_id if payload.incidents else "INC-01"
        busy = [u.unit_id for u in payload.units if u.status is not UnitStatus.AVAILABLE]
        illegal = [UnitOrder(unit_id=UNKNOWN_UNIT, incident_id=target, rationale="scripted: unknown unit")]
        illegal += [UnitOrder(unit_id=u, incident_id=target, rationale="scripted: busy unit") for u in busy[:1]]
        bad = answer.model_copy(update={"orders": [*answer.orders, *illegal]})
        return [assistant_message(bad.model_dump_json())]

    return respond


def malformed_json(_call: ModelCall) -> list[TResponseOutputItem]:
    """Hospital answers: text that is not JSON."""
    return [assistant_message("Sure! We can take the patient {beds: maybe")]


@dataclass
class ScriptedRoster:
    """The LLM-mode backend and its scripted models.

    Attributes:
        backend: The ``SdkBackend`` (``fallback_on_error=True``).
        scripted: Scripted model per agent name.
    """

    backend: SdkBackend
    scripted: dict[str, ScriptedModel]


def scripted_backend(scenario: Scenario) -> ScriptedRoster:
    """Builds the LLM-mode backend described in the module docstring.

    Args:
        scenario: The run's scenario (one hospital agent per hospital).

    Returns:
        The backend and its scripted models.
    """
    specs = agent_specs(scenario.hospitals, PARAMS)
    by_name = specs_by_name(specs)
    scripted = {DISPATCH_AGENT: endless_model(illegal_dispatch(by_name[DISPATCH_AGENT].policy))}
    scripted |= {s.name: endless_model(malformed_json) for s in specs if s.name.startswith("hospital_")}

    def model_for(spec: AgentSpec) -> ScriptedModel | PolicyModel:
        return scripted.get(spec.name) or PolicyModel(spec, PARAMS)

    backend = SdkBackend(
        LABEL, specs, build_agents(specs, model_for), PARAMS,
        run_config=RunConfig(tracing_disabled=True, workflow_name="ems-test-llm-fallback"),
        max_turns=6, fallback_on_error=True,
    )
    return ScriptedRoster(backend, scripted)


# --------------------------------------------------------------------------- driving


@dataclass(frozen=True)
class AppliedAction:
    """One ``apply`` through the auditing port.

    Attributes:
        t_min: Time of the call.
        actor: Submitting agent.
        action: The action.
        ok: Whether the engine accepted it.
        status_before: The unit's status before the call (None if unknown unit).
    """

    t_min: float
    actor: str
    action: Action
    ok: bool
    status_before: UnitStatus | None


@dataclass
class AuditingPort:
    """``EnginePort`` that forwards to the engine and records every ``apply``."""

    engine: Engine
    applied: list[AppliedAction] = field(default_factory=list)

    @property
    def now(self) -> float:
        """Current simulation time."""
        return self.engine.now

    def view(self) -> PublicView:
        """The engine's public view."""
        return self.engine.view()

    def check(self, action: Action) -> ActionCheck:
        """Validates without applying."""
        return self.engine.check(action)

    def apply(self, action: Action, actor: str) -> ActionCheck:
        """Records the unit's prior status, then applies through the engine."""
        unit_id = getattr(action, "unit_id", None)
        before = next((u.status for u in self.engine.view().units if u.unit_id == unit_id), None)
        result = self.engine.apply(action, actor)
        self.applied.append(AppliedAction(self.engine.now, actor, action, result.ok, before))
        return result


@dataclass(frozen=True)
class FallbackRun:
    """Everything a fallback-run test inspects.

    Attributes:
        roster: Backend and scripted models.
        sink: The run log.
        port: The auditing port.
        invariant_problems: ``check_invariants`` output collected around every point.
        metrics: Metrics of the finished run.
    """

    roster: ScriptedRoster
    sink: MemoryEventSink
    port: AuditingPort
    invariant_problems: tuple[str, ...]
    metrics: RunMetrics

    def calls(self, prefix: str) -> list[AgentCallRecord]:
        """Agent-call records of agents whose name starts with ``prefix``."""
        return [r for r in self.sink.records if isinstance(r, AgentCallRecord) and r.agent.startswith(prefix)]


@pytest.fixture(scope="module")
def fallback_run() -> FallbackRun:
    """One Full run of ``fallback_scenario`` with the scripted LLM-mode backend."""
    scenario = fallback_scenario()
    roster = scripted_backend(scenario)
    sink = MemoryEventSink()
    engine = Engine(scenario, SimParams(), sink)
    port = AuditingPort(engine)
    coordinator = AgentCoordinator(roster.backend, NEGOTIATION[Condition.FULL], sink, PARAMS)
    problems: list[str] = []
    while (point := engine.advance()) is not None:
        problems += [f"t={point.t_min} after advance: {p}" for p in engine.check_invariants()]
        coordinator.handle(point, port)
        problems += [f"t={point.t_min} after handle: {p}" for p in engine.check_invariants()]
    problems += [f"final: {p}" for p in engine.check_invariants()]
    metrics = compute_metrics(engine.outcomes(), sink.records)
    return FallbackRun(roster, sink, port, tuple(problems), metrics)


# --------------------------------------------------------------------------- tests


def test_run_completes_with_consistent_engine_state(fallback_run: FallbackRun) -> None:
    assert fallback_run.port.engine.finished
    assert not fallback_run.invariant_problems, fallback_run.invariant_problems
    for name, model in fallback_run.roster.scripted.items():
        assert model.calls, f"scripted model of {name} was never called"


def test_illegal_dispatch_orders_are_replaced_by_the_policy(fallback_run: FallbackRun) -> None:
    calls = fallback_run.calls(DISPATCH_AGENT)
    assert calls and all(c.source == "model" for c in calls)
    for call in calls:
        notes = [(n.scope, n.key) for n in call.fallbacks]
        assert ("section", "orders") in notes or any(n.scope == "call" for n in call.fallbacks), (
            f"dispatch call at t={call.t_min} has illegal orders but no fallback note: {call.fallbacks}"
        )
        acted_on = {o["unit_id"] for o in call.output["orders"]}
        assert UNKNOWN_UNIT not in acted_on
        input_units = {u["unit_id"]: u["status"] for u in call.input["units"]}
        busy_ordered = {u for u in acted_on if input_units.get(u) != UnitStatus.AVAILABLE.value}
        assert not busy_ordered, f"orders for unavailable units survived validation: {busy_ordered}"


def test_malformed_hospital_answers_fall_back_to_the_policy(fallback_run: FallbackRun) -> None:
    calls = fallback_run.calls("hospital_")
    assert calls, "no hospital was ever asked; the scenario should produce a transport"
    for call in calls:
        assert call.source == "policy_fallback" and call.tool_calls == () and call.usage is None
        rules = {(n.scope, n.rule) for n in call.fallbacks}
        assert ("call", "backend_fallback") in rules
        assert any("ModelBehaviorError" in n.detail for n in call.fallbacks)
        assert call.output["responses"], "the fallback policy must answer every request"


def test_fallback_metric_counts_every_replaced_call(fallback_run: FallbackRun) -> None:
    replaced = [c for c in fallback_run.calls("") if c.source == "policy_fallback" or c.fallbacks]
    assert fallback_run.metrics.fallbacks == len(replaced) > 0
    assert fallback_run.metrics.agent_calls == len(fallback_run.calls(""))


def test_no_illegal_action_was_applied(fallback_run: FallbackRun) -> None:
    applied = [a for a in fallback_run.port.applied if a.ok]
    assert any(isinstance(a.action, AssignTransport) for a in applied), "no transport was ever applied"
    for record in applied:
        unit_id = getattr(record.action, "unit_id", None)
        assert unit_id != UNKNOWN_UNIT, record
        if unit_id == BUSY_UNIT:
            assert record.t_min >= BUSY_UNTIL_MIN, f"busy unit used before it was free: {record}"
        required = REQUIRED_STATUS.get(type(record.action))
        if required is not None:
            assert record.status_before is required, f"{record} applied while unit was {record.status_before}"
    logged = [r for r in fallback_run.sink.records if isinstance(r, ActionRecord)]
    assert [(r.action, r.ok) for r in logged if r.actor != "engine_guard"] == [
        (a.action, a.ok) for a in fallback_run.port.applied
    ]


def test_no_unit_is_dispatched_twice(fallback_run: FallbackRun) -> None:
    dispatched = [
        (a.t_min, a.action.unit_id) for a in fallback_run.port.applied
        if a.ok and isinstance(a.action, DispatchUnit)
    ]
    assert dispatched, "no unit was ever dispatched"
    assert len(dispatched) == len(set(dispatched)), f"double dispatch: {dispatched}"


# --------------------------------------------------------------------------- network isolation


def test_llm_backend_is_unavailable_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(BackendUnavailableError):
        make_llm_backend(fallback_scenario().hospitals, PARAMS)
    with pytest.raises(BackendUnavailableError):
        run_simulation(RunSpec("normal", 7380, Condition.FULL, backend="llm", model="gpt-test"))


def test_offline_backend_never_touches_the_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    attempts: list[object] = []

    def refuse(*args: object, **kwargs: object) -> None:
        attempts.append(args)
        raise OSError("network access is disabled in this test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)

    scenario = fallback_scenario()
    backend = make_offline_backend(scenario.hospitals, PARAMS)
    sink = MemoryEventSink()
    engine = Engine(scenario, SimParams(), sink)
    coordinator = AgentCoordinator(backend, NEGOTIATION[Condition.FULL], sink, PARAMS)
    while (point := engine.advance()) is not None:
        coordinator.handle(point, engine)
    assert attempts == []
    calls = [r for r in sink.records if isinstance(r, AgentCallRecord)]
    assert calls and all(c.backend == "offline" and c.source == "model" and not c.fallbacks for c in calls)
    assert all(c.latency_ms is None for c in calls)
