"""Inter-agent messages (the contract-net protocol of section 5.4)."""

from typing import Annotated, Final, Literal

from pydantic import Field

from emergency_sim.domain.actions import Action
from emergency_sim.domain.base import FrozenModel
from emergency_sim.domain.enums import (
    CaseKind,
    Capability,
    CounterCondition,
    MessageType,
    Resolution,
    StaffRole,
)

ASSESSMENT_AGENT: Final = "incident_assessment"
DISPATCH_AGENT: Final = "dispatch"
ALLOCATION_AGENT: Final = "resource_allocation"
BROADCAST: Final = "all"


def hospital_agent_name(hospital_id: str) -> str:
    """Returns the agent name of a hospital's coordination agent.

    Args:
        hospital_id: Hospital id, e.g. ``"H1"``.

    Returns:
        The agent name, e.g. ``"hospital_H1"``.
    """
    return f"hospital_{hospital_id}"


class AssessAction(FrozenModel):
    """Content of a PRIORITY_REPORT: the assessed priority of an incident."""

    action: Literal["assess"] = "assess"
    urgency: int
    suspected_red: bool
    suspected_cardiac_arrest: bool
    est_patients: int
    needed_capabilities: tuple[Capability, ...]
    als_units_needed: int
    deadline_in_min: float


class DispatchAction(FrozenModel):
    """Units dispatched to the message's incident."""

    action: Literal["dispatch"] = "dispatch"
    unit_ids: tuple[str, ...]


class AdmitAction(FrozenModel):
    """Proposal (or acceptance) to admit a unit's patients at a hospital."""

    action: Literal["admit"] = "admit"
    hospital_id: str
    unit_id: str
    victim_ids: tuple[str, ...]
    eta_min: float


class ConditionalAdmitAction(FrozenModel):
    """COUNTER: admit only if a condition (float staff, reserve release) is met."""

    action: Literal["admit_if"] = "admit_if"
    hospital_id: str
    unit_id: str
    victim_ids: tuple[str, ...]
    condition: CounterCondition
    skill: str | None


class DeclineAction(FrozenModel):
    """Refusal with a reason."""

    action: Literal["decline"] = "decline"
    reason: str


class StaffRequestAction(FrozenModel):
    """Request to lend a float staff member with a skill to a hospital."""

    action: Literal["lend_staff"] = "lend_staff"
    hospital_id: str
    role: StaffRole
    skill: str


class ResolveAction(FrozenModel):
    """ESCALATE: ask Resource Allocation to choose among case options."""

    action: Literal["resolve"] = "resolve"
    case_id: str
    case_kind: CaseKind
    option_ids: tuple[str, ...]


class CommitAction(FrozenModel):
    """COMMIT: the engine actions chosen to close a thread or case."""

    action: Literal["commit"] = "commit"
    case_id: str | None
    option_id: str | None
    actions: tuple[Action, ...]
    resolution: Resolution


ProposedAction = Annotated[
    AssessAction
    | DispatchAction
    | AdmitAction
    | ConditionalAdmitAction
    | DeclineAction
    | StaffRequestAction
    | ResolveAction
    | CommitAction,
    Field(discriminator="action"),
]
"""Structured content of a message, discriminated by ``action``."""


class Message(FrozenModel):
    """One inter-agent message. Every field is required.

    Attributes:
        message_id: ``"M-0001"``, sequential per run.
        t_min: Simulation time of the decision point.
        sender: Agent name.
        recipient: Agent name or ``BROADCAST``.
        type: Performative.
        incident_id: The triggering incident (always set).
        proposed_action: Structured content.
        rationale: Short human-readable reason.
        in_reply_to: ``message_id`` this message answers, if any.
        conversation_id: ``"NEG-0001"``: one per negotiation thread / case.
        round: 0 for non-negotiation messages, 1..3 in transport negotiation.
    """

    message_id: str
    t_min: float
    sender: str
    recipient: str
    type: MessageType
    incident_id: str
    proposed_action: ProposedAction
    rationale: str
    in_reply_to: str | None
    conversation_id: str
    round: int
