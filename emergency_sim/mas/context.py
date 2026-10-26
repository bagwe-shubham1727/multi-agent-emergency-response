"""Run context handed to every agent run and its function tools."""

from dataclasses import dataclass

from emergency_sim.domain.base import AgentIOModel
from emergency_sim.policies.params import PolicyParams


@dataclass(frozen=True)
class AgentContext:
    """What a tool may read during one agent call.

    Tools see only the call's own payload (the same JSON the model sees) and
    the policy parameters, never the engine, so they are read-only by
    construction.

    Attributes:
        payload: The agent's ``*Input`` model for this call.
        params: Policy parameters used by the shared estimates.
    """

    payload: AgentIOModel
    params: PolicyParams
