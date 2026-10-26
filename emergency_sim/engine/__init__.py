"""Deterministic discrete-event simulation engine (plan section 4).

The engine is the single source of truth of a run. Controllers reach it only
through ``domain.ports.EnginePort`` (which ``Engine`` satisfies); its mutable
state (``engine.state``) is private to this package.
"""

from emergency_sim.engine.engine import GUARD_ACTOR, Engine
from emergency_sim.engine.state import EngineError

__all__ = ["GUARD_ACTOR", "Engine", "EngineError"]
