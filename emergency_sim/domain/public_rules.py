"""Public scene-access rules shared by the engine and the controllers.

These rules are part of the published game (plan section 4.3), so the engine
that enforces them and the controllers that plan around them import the same
definitions instead of keeping copies that could drift apart.
"""

from collections.abc import Iterable
from typing import Final

from emergency_sim.domain.enums import ResponderCapability

HEAVY_ACCESS_CAPABILITIES: Final = frozenset({
    ResponderCapability.EXTRICATION,
    ResponderCapability.HEAVY_RESCUE,
    ResponderCapability.FIRE_SUPPRESSION,
    ResponderCapability.HAZMAT,
})
"""Required capabilities whose work delays scene access by ``heavy_access_min``."""

DECON_COVERED_BY: Final = ResponderCapability.HAZMAT
"""A hazmat team on scene also covers decontamination."""


def effective_capabilities(present: Iterable[str]) -> set[str]:
    """Returns responder capabilities with hazmat also covering decontamination.

    Args:
        present: Capabilities of the teams considered (enum members or their
            string values; ``ResponderCapability`` is a ``StrEnum``).

    Returns:
        The effective set used by the access rule.
    """
    effective = set(present)
    if DECON_COVERED_BY in effective:
        effective.add(ResponderCapability.DECONTAMINATION)
    return effective
