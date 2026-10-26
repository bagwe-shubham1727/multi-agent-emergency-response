"""Resource Allocation policy: arbitrate contested resources by expected utility.

The coordinator precomputes each option's utility (``estimates.utility``) when
it builds a case; this policy picks the best option per case and makes sure a
float staff member is granted at most once per call (plan 5.3).

Option-id convention for ``FLOAT_STAFF`` cases (shared with the coordinator):
``grant_option_id(staff_id)`` (``"GRANT:<staff_id>"``) lends that staff member;
``DECLINE_OPTION_ID`` (``"DECLINE"``) refuses the request.
"""

from collections.abc import Sequence
from typing import Final

from emergency_sim.domain.agent_io import (
    AllocationInput,
    AllocationOutput,
    CaseDecision,
    CaseOption,
    EscalationCase,
)
from emergency_sim.domain.enums import CaseKind
from emergency_sim.policies.params import PolicyParams

GRANT_PREFIX: Final = "GRANT:"
DECLINE_OPTION_ID: Final = "DECLINE"
MAX_RATIONALE_CHARS: Final = 200


def grant_option_id(staff_id: str) -> str:
    """Option id that lends ``staff_id`` in a ``FLOAT_STAFF`` case.

    Args:
        staff_id: Float staff member.

    Returns:
        ``"GRANT:<staff_id>"``.
    """
    return f"{GRANT_PREFIX}{staff_id}"


def granted_staff(option_id: str) -> str | None:
    """Staff id lent by a ``FLOAT_STAFF`` option, or None for any other option.

    Args:
        option_id: Option id.

    Returns:
        The staff id after ``"GRANT:"``, else None.
    """
    return option_id.removeprefix(GRANT_PREFIX) if option_id.startswith(GRANT_PREFIX) else None


def rank_options(options: Sequence[CaseOption]) -> list[CaseOption]:
    """Sorts options by utility (highest first), ties broken by option id.

    Args:
        options: Case options.

    Returns:
        Best first.
    """
    return sorted(options, key=lambda o: (-o.utility, o.option_id))


def _best_utility(case: EscalationCase) -> float:
    return max((o.utility for o in case.options), default=float("-inf"))


def _is_green_only(case: EscalationCase, params: PolicyParams) -> bool:
    return all(o.triage_weight <= params.green_weight for o in case.options)


def _choose_float(
    case: EscalationCase, granted: set[str], params: PolicyParams
) -> tuple[CaseOption, str]:
    ranked = rank_options(case.options)
    decline = next((o for o in ranked if granted_staff(o.option_id) is None), ranked[0])
    if _is_green_only(case, params):
        return decline, "GREEN-only request: float staff are kept for sicker patients"
    for option in ranked:
        staff_id = granted_staff(option.option_id)
        if staff_id is None:
            return option, f"declining has the highest utility {option.utility:.3f}"
        if staff_id not in granted:
            granted.add(staff_id)
            return option, f"lend {staff_id}: highest utility {option.utility:.3f}"
    return decline, "every requested float staff member is already granted"


def choose_option(case: EscalationCase, granted: set[str], params: PolicyParams) -> CaseDecision:
    """Picks the best option of one case.

    Args:
        case: The case (must have at least one option).
        granted: Float staff already granted in this call; updated in place
            when a GRANT option is chosen.
        params: Policy parameters.

    Returns:
        The decision.
    """
    if case.kind is CaseKind.FLOAT_STAFF:
        option, why = _choose_float(case, granted, params)
    else:
        option = rank_options(case.options)[0]
        why = f"highest expected utility {option.utility:.3f} (slack {option.slack_min:.1f} min)"
    return CaseDecision(
        case_id=case.case_id, chosen_option_id=option.option_id, rationale=why[:MAX_RATIONALE_CHARS]
    )


def allocate(inp: AllocationInput, params: PolicyParams) -> AllocationOutput:
    """Resource Allocation policy (plan 5.3).

    Cases are processed by descending best-option utility, then ``case_id``.
    Cases without options are skipped (the coordinator auto-resolves them).

    Args:
        inp: Open cases and the float pool.
        params: Policy parameters.

    Returns:
        One decision per case that has options.
    """
    granted: set[str] = set()
    cases = sorted((c for c in inp.cases if c.options), key=lambda c: (-_best_utility(c), c.case_id))
    return AllocationOutput(decisions=[choose_option(c, granted, params) for c in cases])
