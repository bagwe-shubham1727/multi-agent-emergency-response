"""OpenAI Agents SDK layer: the six agents, the offline policy model and the backends.

This is the only package that imports ``agents`` (the SDK) or ``openai``.

Modules:
    context: ``AgentContext`` passed to runs and tools.
    tools: Read-only ``@function_tool`` factories over the payload.
    instructions: System prompts per role.
    specs: ``AgentSpec`` and ``agent_specs(hospitals)`` (the roster).
    policy_model: ``PolicyModel``, the deterministic offline ``Model``.
    backend: ``SdkBackend``, ``make_offline_backend``, ``make_llm_backend``
        and ``BackendUnavailableError``.
"""

from emergency_sim.mas.backend import (
    BackendUnavailableError,
    SdkBackend,
    make_llm_backend,
    make_offline_backend,
)
from emergency_sim.mas.context import AgentContext
from emergency_sim.mas.specs import AgentSpec, agent_specs

__all__ = [
    "AgentContext",
    "AgentSpec",
    "BackendUnavailableError",
    "SdkBackend",
    "agent_specs",
    "make_llm_backend",
    "make_offline_backend",
]
