"""Engine actions: the only ways a controller can change the world.

Controllers submit these through ``EnginePort.apply``; the engine validates
each one (section 4.6 of the plan) and answers with an ``ActionCheck``.
"""

from typing import Annotated, Literal

from pydantic import Field

from emergency_sim.domain.base import FrozenModel
from emergency_sim.domain.enums import RejectReason


class DispatchUnit(FrozenModel):
    """Send an available unit (ambulance or responder team) to an incident."""

    kind: Literal["dispatch_unit"] = "dispatch_unit"
    unit_id: str
    incident_id: str


class DivertUnit(FrozenModel):
    """Redirect an ambulance that is en route to a scene to another incident."""

    kind: Literal["divert_unit"] = "divert_unit"
    unit_id: str
    incident_id: str


class AssignTransport(FrozenModel):
    """Give a loaded ambulance its destination hospital, optionally reserving beds."""

    kind: Literal["assign_transport"] = "assign_transport"
    unit_id: str
    hospital_id: str
    reserve_beds: bool


class LendStaff(FrozenModel):
    """Lend a float-pool staff member to a hospital (permanent for the run)."""

    kind: Literal["lend_staff"] = "lend_staff"
    staff_id: str
    hospital_id: str


Action = Annotated[
    DispatchUnit | DivertUnit | AssignTransport | LendStaff,
    Field(discriminator="kind"),
]
"""Any engine action, discriminated by ``kind``."""


class ActionCheck(FrozenModel):
    """Result of validating (and possibly applying) an action.

    Attributes:
        ok: Whether the action is legal (and, for ``apply``, was applied).
        reason: Machine-readable rejection reason when ``ok`` is False.
        detail: Free-text explanation for logs.
    """

    ok: bool
    reason: RejectReason | None = None
    detail: str = ""
