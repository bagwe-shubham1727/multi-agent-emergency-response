"""The JSONL run-log contract that the dashboard and evaluation consume.

One record per line. Ordering guarantee: ``seq`` is strictly increasing and
``t_min`` is non-decreasing; ``run_start`` comes first and ``run_end`` last.
The sink (``EventSink.emit``) assigns ``seq``; producers leave it at 0.
"""

from typing import Annotated, Any, Final, Literal

from pydantic import Field, TypeAdapter

from emergency_sim.domain.actions import Action
from emergency_sim.domain.agent_io import TokenUsage, ToolCallTrace
from emergency_sim.domain.base import FrozenModel
from emergency_sim.domain.enums import (
    AgentRole,
    Condition,
    RejectReason,
    SimEventType,
    TriggerKind,
)
from emergency_sim.domain.messages import Message
from emergency_sim.domain.outcomes import RunMetrics, RunOutcomes
from emergency_sim.domain.scenario import Scenario
from emergency_sim.domain.views import PublicView


class _Record(FrozenModel):
    """Common fields of every log record."""

    seq: int = 0
    t_min: float


class RunStartRecord(_Record):
    """First record: run identity, parameters and the full scenario truth.

    ``scenario_data`` is for analysis only; controllers never read the log.
    """

    kind: Literal["run_start"] = "run_start"
    run_id: str
    scenario: str
    seed: int
    condition: Condition
    backend: str
    repeat: int
    sim_params: dict[str, float]
    negotiation: dict[str, int | bool] | None
    scenario_data: Scenario


class SnapshotCounts(FrozenModel):
    """Aggregate victim counts at a snapshot minute."""

    incidents_reported: int
    incidents_open: int
    waiting_on_scene: int
    in_transport: int
    queued: int
    waiting_for_staff: int
    in_treatment: int
    treated: int
    care_started_cum: int
    expectant: int


class SnapshotRecord(_Record):
    """The public view at a whole snapshot minute."""

    kind: Literal["snapshot"] = "snapshot"
    view: PublicView
    counts: SnapshotCounts


class SimEventRecord(_Record):
    """A simulation event emitted by the engine."""

    kind: Literal["sim_event"] = "sim_event"
    event: SimEventType
    incident_id: str | None = None
    unit_id: str | None = None
    hospital_id: str | None = None
    staff_id: str | None = None
    victim_ids: tuple[str, ...] = ()
    detail: str = ""


class MessageRecord(_Record):
    """An inter-agent message."""

    kind: Literal["message"] = "message"
    message: Message


class FallbackNote(FrozenModel):
    """Where the coordinator replaced (part of) an agent output with the policy's.

    Attributes:
        scope: Granularity of the replacement.
        key: Section name or item id, None for a whole call.
        rule: The validation rule that failed.
        detail: Free-text explanation.
    """

    scope: Literal["call", "section", "item"]
    key: str | None
    rule: str
    detail: str


class AgentCallRecord(_Record):
    """One agent call: input and output payloads plus provenance."""

    kind: Literal["agent_call"] = "agent_call"
    agent: str
    role: AgentRole
    triggers: tuple[TriggerKind, ...]
    backend: str
    input: dict[str, Any]
    output: dict[str, Any]
    source: Literal["model", "policy_fallback"]
    tool_calls: tuple[ToolCallTrace, ...]
    fallbacks: tuple[FallbackNote, ...]
    usage: TokenUsage | None
    latency_ms: float | None


class ActionRecord(_Record):
    """An action submitted to the engine and its validation result."""

    kind: Literal["action"] = "action"
    actor: str
    action: Action
    ok: bool
    reason: RejectReason | None
    detail: str


class RunEndRecord(_Record):
    """Last record: outcomes and metrics."""

    kind: Literal["run_end"] = "run_end"
    outcomes: RunOutcomes
    metrics: RunMetrics


LogRecord = Annotated[
    RunStartRecord
    | SnapshotRecord
    | SimEventRecord
    | MessageRecord
    | AgentCallRecord
    | ActionRecord
    | RunEndRecord,
    Field(discriminator="kind"),
]
"""Any log record, discriminated by ``kind``."""

LOG_RECORD_ADAPTER: Final = TypeAdapter(LogRecord)
"""Validates one JSONL line (``validate_json``) or a dict (``validate_python``)."""
