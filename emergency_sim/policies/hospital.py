"""Hospital Coordination policy: protect the critical reserve, admit only what can be treated.

Requests are answered in ``proposal_id`` order with a running count of free
beds, so two proposals cannot both take the last bed (plan 5.3). A COUNTER is a
conditional acceptance: "yes if I get a float doctor/nurse" or "yes only by
using my critical reserve". It is only used when the payload allows it.
"""

from typing import Final, Literal

from emergency_sim.domain.agent_io import AdmissionResponse, HospitalInput, HospitalOutput
from emergency_sim.domain.enums import CounterCondition
from emergency_sim.policies.estimates import NURSE_SKILL, CapacityCheck, capacity_check
from emergency_sim.policies.params import PolicyParams

type Reason = Literal["OK", "NO_BEDS", "RESERVE_HELD", "NO_DOCTOR", "NO_NURSE", "NOT_CAPABLE"]

COUNTER: Final = "COUNTER"
MAX_RATIONALE_CHARS: Final = 200


def _can_counter(inp: HospitalInput, condition: CounterCondition) -> bool:
    return COUNTER in inp.allowed_responses and condition in inp.allowed_counter_conditions


def _reject(check: CapacityCheck, reason: Reason, why: str) -> AdmissionResponse:
    return AdmissionResponse(
        proposal_id=check.proposal_id, response="REJECT", reason=reason,
        counter_condition=None, counter_skill=None, rationale=why[:MAX_RATIONALE_CHARS],
    )


def _counter(
    check: CapacityCheck, reason: Reason, condition: CounterCondition, skill: str | None, why: str
) -> AdmissionResponse:
    return AdmissionResponse(
        proposal_id=check.proposal_id, response="COUNTER", reason=reason,
        counter_condition=condition, counter_skill=skill, rationale=why[:MAX_RATIONALE_CHARS],
    )


def _staff_ok(check: CapacityCheck) -> bool:
    return not check.missing_doctor_skills and not check.nurse_short


def _answer_reserve(inp: HospitalInput, check: CapacityCheck) -> AdmissionResponse:
    why = f"non-RED admission leaves {check.beds_free_after} bed(s), below reserve {check.critical_reserve}"
    if _can_counter(inp, CounterCondition.RESERVE_RELEASE) and _staff_ok(check):
        why += "; accept if the reserve is released"
        return _counter(check, "RESERVE_HELD", CounterCondition.RESERVE_RELEASE, None, why)
    return _reject(check, "RESERVE_HELD", why)


def _answer_staff(inp: HospitalInput, check: CapacityCheck) -> AdmissionResponse | None:
    if check.missing_doctor_skills:
        skill = check.missing_doctor_skills[0]
        why = f"no doctor slot with skill {skill}"
        if _can_counter(inp, CounterCondition.FLOAT_DOCTOR):
            why += "; accept with a float doctor"
            return _counter(check, "NO_DOCTOR", CounterCondition.FLOAT_DOCTOR, skill, why)
        return _reject(check, "NO_DOCTOR", why)
    if check.nurse_short:
        why = f"{check.nurse_slots} nurse slot(s) for {check.patients} patient(s)"
        if _can_counter(inp, CounterCondition.FLOAT_NURSE):
            why += "; accept with a float nurse"
            return _counter(check, "NO_NURSE", CounterCondition.FLOAT_NURSE, NURSE_SKILL, why)
        return _reject(check, "NO_NURSE", why)
    return None


def answer_request(inp: HospitalInput, check: CapacityCheck) -> AdmissionResponse:
    """Applies the admission rule to one capacity check.

    Order: capability, beds, critical reserve (non-RED only), doctor skill,
    nurse slots; otherwise ACCEPT.

    Args:
        inp: The hospital payload (allowed responses and counter conditions).
        check: Capacity check computed with the running bed count.

    Returns:
        The response.
    """
    if check.missing_capabilities:
        missing = ", ".join(c.value for c in check.missing_capabilities)
        return _reject(check, "NOT_CAPABLE", f"cannot treat {missing}")
    if check.beds_free_unreserved < check.patients:
        why = f"{check.beds_free_unreserved} free bed(s) for {check.patients} patient(s)"
        return _reject(check, "NO_BEDS", why)
    if check.reserve_breached:
        return _answer_reserve(inp, check)
    staff_answer = _answer_staff(inp, check)
    if staff_answer is not None:
        return staff_answer
    return AdmissionResponse(
        proposal_id=check.proposal_id, response="ACCEPT", reason="OK", counter_condition=None,
        counter_skill=None, rationale=f"beds and staff available; {check.beds_free_after} bed(s) left",
    )


def hospital(inp: HospitalInput, params: PolicyParams) -> HospitalOutput:
    """Hospital Coordination policy (plan 5.3).

    Args:
        inp: The hospital's state and the admission requests to answer.
        params: Policy parameters (unused; kept for the uniform policy signature).

    Returns:
        One response per request, in ``proposal_id`` order.
    """
    del params  # every rule here is structural; no tunable constant applies
    free = inp.beds_free_unreserved
    responses = []
    for request in sorted(inp.requests, key=lambda r: r.proposal_id):
        response = answer_request(inp, capacity_check(inp, request.proposal_id, beds_free=free))
        if response.response == "ACCEPT":
            free -= len(request.victims)
        responses.append(response)
    return HospitalOutput(responses=responses)
