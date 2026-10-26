"""Protocols (ports) that decouple the layers.

The control layer sees the engine only through ``EnginePort`` and the agent
backend only through ``DecisionBackend``; every producer of log records writes
through an ``EventSink``. All are ``runtime_checkable`` so tests can assert
that fakes and implementations conform structurally.
"""

from typing import Protocol, runtime_checkable

from emergency_sim.domain.actions import Action, ActionCheck
from emergency_sim.domain.agent_io import AgentDecision
from emergency_sim.domain.base import AgentIOModel
from emergency_sim.domain.log_records import LogRecord
from emergency_sim.domain.views import DecisionPoint, PublicView


@runtime_checkable
class EventSink(Protocol):
    """Destination of log records."""

    def emit(self, record: LogRecord) -> None:
        """Stores a record; the sink assigns ``seq``.

        Args:
            record: The record to store (its ``seq`` is overwritten).
        """
        ...


@runtime_checkable
class EnginePort(Protocol):
    """The controllers' only access to the simulation."""

    @property
    def now(self) -> float:
        """Current simulation time in minutes."""
        ...

    def view(self) -> PublicView:
        """Returns the current public view."""
        ...

    def check(self, action: Action) -> ActionCheck:
        """Validates an action without applying or logging it.

        Args:
            action: The action to check.

        Returns:
            Whether the action is currently legal, and why not.
        """
        ...

    def apply(self, action: Action, actor: str) -> ActionCheck:
        """Validates, logs an ``ActionRecord`` and mutates state only if legal.

        Args:
            action: The action to apply.
            actor: Name of the agent or controller submitting it.

        Returns:
            The validation result; state is unchanged when ``ok`` is False.
        """
        ...


@runtime_checkable
class Controller(Protocol):
    """Decides what to do at each decision point (B0, B1 or Full)."""

    def handle(self, point: DecisionPoint, port: EnginePort) -> None:
        """Reacts to a decision point by applying actions through ``port``.

        Args:
            point: Why and when the engine stopped.
            port: Access to the view and to action application.
        """
        ...


@runtime_checkable
class DecisionBackend(Protocol):
    """Runs one agent call (offline policy model or LLM)."""

    @property
    def label(self) -> str:
        """Backend label: ``"offline"``, ``"llm-<model>"`` or a test label."""
        ...

    def decide(self, agent_name: str, payload: AgentIOModel) -> AgentDecision:
        """Runs the named agent on a payload.

        Args:
            agent_name: Agent name, e.g. ``"dispatch"`` or ``"hospital_H1"``.
            payload: The agent's ``*Input`` model.

        Returns:
            The structured output and its provenance.
        """
        ...
