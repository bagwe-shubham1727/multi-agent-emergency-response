"""Negotiation configuration of each agent condition (B1, Full and ablations).

B0 and B0-cues have no entry: they are the ``GreedyController`` and never
negotiate.

The mechanisms nest. Every escalation to Resource Allocation (unit contention,
diversion, no-accepting-hospital cases and RESERVE_RELEASE counters) needs
``allow_escalation``. A float-staff counter is a staff REQUEST, not an
escalation, so ``full-no-escalation`` keeps float-staff counters but has no
ESCALATE messages at all.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Final, Literal

from emergency_sim.domain.enums import Condition, CounterCondition

type HospitalResponse = Literal["ACCEPT", "REJECT", "COUNTER"]


@dataclass(frozen=True)
class NegotiationConfig:
    """Which negotiation mechanisms a condition may use.

    Attributes:
        allow_counter: Hospitals may answer COUNTER (conditional acceptance).
        allow_float_staff: COUNTER may ask for float staff (FLOAT_DOCTOR /
            FLOAT_NURSE), settled through a staff REQUEST.
        allow_escalation: Agents may escalate to Resource Allocation: unit
            contention, no-accepting-hospital cases and RESERVE_RELEASE
            counters (a hospital asking to use its critical reserve).
        allow_diversion: Dispatch may request diverting an en-route ALS;
            diversions are settled by Resource Allocation, so this requires
            ``allow_escalation``.
        max_rounds: Transport negotiation rounds per decision point.
    """

    allow_counter: bool
    allow_float_staff: bool
    allow_escalation: bool
    allow_diversion: bool
    max_rounds: int = 3

    def __post_init__(self) -> None:
        """Rejects configurations whose mechanisms do not nest.

        Raises:
            ValueError: If diversion is allowed without escalation.
        """
        if self.allow_diversion and not self.allow_escalation:
            raise ValueError("diversion is settled by Resource Allocation; it needs escalation")

    def allowed_responses(self) -> list[HospitalResponse]:
        """Returns the responses a hospital agent may give.

        Returns:
            ``["ACCEPT", "REJECT", "COUNTER"]`` if counters are allowed, else
            ``["ACCEPT", "REJECT"]``.
        """
        if self.allow_counter:
            return ["ACCEPT", "REJECT", "COUNTER"]
        return ["ACCEPT", "REJECT"]

    def allowed_counter_conditions(self) -> list[CounterCondition]:
        """Returns the counter conditions a hospital agent may use.

        Returns:
            FLOAT_DOCTOR and FLOAT_NURSE if counters and float staff are
            allowed, plus RESERVE_RELEASE if counters and escalation are
            allowed (it is decided by Resource Allocation); empty otherwise.
        """
        conditions: list[CounterCondition] = []
        if self.allow_counter and self.allow_float_staff:
            conditions += [CounterCondition.FLOAT_DOCTOR, CounterCondition.FLOAT_NURSE]
        if self.allow_counter and self.allow_escalation:
            conditions.append(CounterCondition.RESERVE_RELEASE)
        return conditions

    def as_log_dict(self) -> dict[str, int | bool]:
        """Returns the config as the ``RunStartRecord.negotiation`` mapping.

        Returns:
            Field name to value.
        """
        return dict(asdict(self))


NEGOTIATION: Final[Mapping[Condition, NegotiationConfig]] = MappingProxyType({
    Condition.B1: NegotiationConfig(False, False, False, False),
    Condition.FULL: NegotiationConfig(True, True, True, True),
    Condition.FULL_NO_FLOAT: NegotiationConfig(True, False, True, True),
    Condition.FULL_NO_DIVERSION: NegotiationConfig(True, True, True, False),
    Condition.FULL_NO_ESCALATION: NegotiationConfig(True, True, False, False),
})
"""Negotiation config per agent condition (the greedy baselines have none)."""
