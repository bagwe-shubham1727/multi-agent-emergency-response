"""Narrow engine port handed to controllers.

Controllers must see the world only through ``domain.ports.EnginePort``. Passing
the ``Engine`` itself would satisfy the type but still let a controller reach
engine internals (hidden victim truth included). ``EnginePortAdapter`` exposes
exactly the four port members and nothing else.
"""

from emergency_sim.domain.actions import Action, ActionCheck
from emergency_sim.domain.views import PublicView
from emergency_sim.engine.engine import Engine


class EnginePortAdapter:
    """Implements ``EnginePort`` by delegating to an engine it keeps private."""

    __slots__ = ("_engine",)

    def __init__(self, engine: Engine) -> None:
        """Wraps an engine.

        Args:
            engine: The engine to expose through the port.
        """
        self._engine = engine

    @property
    def now(self) -> float:
        """Current simulation time in minutes."""
        return self._engine.now

    def view(self) -> PublicView:
        """Returns the public view of the world (no hidden truth).

        Returns:
            The current public view.
        """
        return self._engine.view()

    def check(self, action: Action) -> ActionCheck:
        """Checks an action without applying it.

        Args:
            action: The proposed action.

        Returns:
            Whether the action is legal now, with a reason if not.
        """
        return self._engine.check(action)

    def apply(self, action: Action, actor: str) -> ActionCheck:
        """Validates, logs and (if legal) applies an action.

        Args:
            action: The action.
            actor: Name of the agent or controller applying it.

        Returns:
            The validation result; state changes only when it is ok.
        """
        return self._engine.apply(action, actor)
