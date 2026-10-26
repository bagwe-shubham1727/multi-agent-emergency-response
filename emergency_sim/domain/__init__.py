"""Domain contracts shared by every package of the emergency-response MAS.

The modules in this package hold only immutable value types, enums and
Protocols: the frozen interfaces that the engine, policies, agents, control
layer, evaluation and dashboard code against. They depend on nothing but the
standard library, pydantic and ``emergency_sim.geo``.

Import from the defining module (for example
``from emergency_sim.domain.views import PublicView``). This package
deliberately re-exports nothing, which keeps the import graph acyclic.
"""
