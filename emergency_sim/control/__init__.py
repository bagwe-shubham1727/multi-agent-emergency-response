"""Control layer: the B0 greedy baseline and the agent coordinator (B1/Full).

Controllers see the engine only through ``domain.ports.EnginePort`` and the
agents only through ``domain.ports.DecisionBackend``. Public entry points:

* ``control.greedy.GreedyController``: B0, no messages, no agent calls.
* ``control.coordinator.AgentCoordinator``: B1, Full and the ablations,
  configured by ``control.conditions.NegotiationConfig``.
"""


class ProtocolError(RuntimeError):
    """A programming error in the control layer (broken invariant or contract).

    Raised for bugs only; illegal agent decisions are handled by validation
    fallbacks and engine rejections, never by this exception.
    """
