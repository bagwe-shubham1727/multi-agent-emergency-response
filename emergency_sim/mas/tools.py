"""Read-only function tools of the agents.

Each tool reads the current call's payload from ``RunContextWrapper[AgentContext]``
and returns a JSON string computed by the same pure functions in
``policies.estimates`` that the offline policies use. Tools have no engine
reference, so they cannot change the world. Unknown ids or a payload of the
wrong type produce ``{"error": ...}`` instead of an exception, so an LLM can
recover.

The plain functions are module-level; ``FunctionTool`` objects are built fresh
by the ``*_tools()`` factories so no mutable SDK object lives at module level.
"""

import dataclasses
import json
from collections.abc import Callable
from types import MappingProxyType
from typing import Any, Final, Literal

from agents import FunctionTool, RunContextWrapper, function_tool

from emergency_sim.domain.agent_io import AllocationInput, DispatchInput, HospitalInput
from emergency_sim.domain.base import AgentIOModel
from emergency_sim.domain.enums import AmbulanceType, Triage
from emergency_sim.mas.context import AgentContext
from emergency_sim.policies import estimates
from emergency_sim.policies.allocation import rank_options

INCIDENT_TYPE_PROFILE: Final = "incident_type_profile"
NOMINAL_DEADLINE: Final = "nominal_deadline"
RANK_UNITS_FOR_INCIDENT: Final = "rank_units_for_incident"
RANK_HOSPITALS_FOR_UNIT: Final = "rank_hospitals_for_unit"
CAPACITY_CHECK: Final = "capacity_check"
SCORE_CASE_OPTIONS: Final = "score_case_options"


def _dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _error(message: str) -> str:
    return _dumps({"error": message})


def _payload[T: AgentIOModel](ctx: RunContextWrapper[AgentContext], kind: type[T]) -> T | None:
    payload = ctx.context.payload
    return payload if isinstance(payload, kind) else None


def incident_type_profile(ctx: RunContextWrapper[AgentContext], incident_type: str) -> str:
    """Public profile of an incident type: base urgency and typical hospital capabilities.

    Args:
        incident_type: Incident type label, e.g. "cardiac_event" or "building_fire".
    """
    del ctx  # the profile table is public and payload-independent
    return _dumps(dataclasses.asdict(estimates.incident_type_profile(incident_type)))


def nominal_deadline(
    ctx: RunContextWrapper[AgentContext], triage: Literal["RED", "YELLOW", "GREEN"], condition: str
) -> str:
    """Nominal care window in minutes for a triage category and condition.

    Args:
        triage: START triage category of the victim.
        condition: Condition label, e.g. "cardiac arrest" or "stab wound".
    """
    minutes = estimates.nominal_deadline(Triage(triage), condition, ctx.context.params)
    return _dumps({"triage": triage, "condition": condition, "deadline_min": minutes})


def rank_units_for_incident(
    ctx: RunContextWrapper[AgentContext], incident_id: str, ambulance_type: Literal["ALS", "BLS", "ANY"]
) -> str:
    """Free ambulances ranked by ETA (turnout included) to an incident, nearest first.

    Args:
        incident_id: Incident id from the input.
        ambulance_type: "ALS", "BLS" or "ANY".
    """
    inp = _payload(ctx, DispatchInput)
    if inp is None:
        return _error("rank_units_for_incident needs a dispatch payload")
    kind = None if ambulance_type == "ANY" else AmbulanceType(ambulance_type)
    try:
        ranks = estimates.rank_units(inp, incident_id, kind)
    except KeyError:
        return _error(f"unknown incident {incident_id}")
    return _dumps({"incident_id": incident_id, "units": [dataclasses.asdict(r) for r in ranks]})


def rank_hospitals_for_unit(ctx: RunContextWrapper[AgentContext], unit_id: str) -> str:
    """Hospitals capable of every patient on a loaded unit: free bed first, then ETA, then id.

    Args:
        unit_id: Id of a loaded unit from the input's loaded_units.
    """
    inp = _payload(ctx, DispatchInput)
    if inp is None:
        return _error("rank_hospitals_for_unit needs a dispatch payload")
    unit = next((u for u in inp.loaded_units if u.unit_id == unit_id), None)
    if unit is None:
        return _error(f"unknown loaded unit {unit_id}")
    ranks = estimates.rank_hospitals(inp.hospitals, unit)
    return _dumps({"unit_id": unit_id, "hospitals": [dataclasses.asdict(r) for r in ranks]})


def capacity_check(ctx: RunContextWrapper[AgentContext], proposal_id: str) -> str:
    """Capability, free beds, critical reserve and doctor/nurse slots for one admission request.

    Args:
        proposal_id: Id of an admission request from the input.
    """
    inp = _payload(ctx, HospitalInput)
    if inp is None:
        return _error("capacity_check needs a hospital payload")
    try:
        check = estimates.capacity_check(inp, proposal_id)
    except KeyError:
        return _error(f"unknown proposal {proposal_id}")
    return _dumps(dataclasses.asdict(check))


def score_case_options(ctx: RunContextWrapper[AgentContext], case_id: str) -> str:
    """Options of an escalated case ranked by expected utility (ties by option id).

    Args:
        case_id: Id of a case from the input.
    """
    inp = _payload(ctx, AllocationInput)
    if inp is None:
        return _error("score_case_options needs an allocation payload")
    case = next((c for c in inp.cases if c.case_id == case_id), None)
    if case is None:
        return _error(f"unknown case {case_id}")
    params = ctx.context.params
    options = [
        {**o.model_dump(), "p_window": round(estimates.p_window(o.slack_min, params), 4)}
        for o in rank_options(case.options)
    ]
    return _dumps({"case_id": case_id, "kind": case.kind.value, "options": options})


TOOL_FUNCTIONS: Final[MappingProxyType[str, Callable[..., str]]] = MappingProxyType({
    INCIDENT_TYPE_PROFILE: incident_type_profile,
    NOMINAL_DEADLINE: nominal_deadline,
    RANK_UNITS_FOR_INCIDENT: rank_units_for_incident,
    RANK_HOSPITALS_FOR_UNIT: rank_hospitals_for_unit,
    CAPACITY_CHECK: capacity_check,
    SCORE_CASE_OPTIONS: score_case_options,
})
"""Tool name -> plain function (the tool name equals the function name)."""


def make_tools(*names: str) -> tuple[FunctionTool, ...]:
    """Builds fresh strict ``FunctionTool`` objects.

    Args:
        *names: Tool names from ``TOOL_FUNCTIONS``.

    Returns:
        The tools, in the given order.

    Raises:
        KeyError: If a name is unknown.
    """
    return tuple(function_tool(TOOL_FUNCTIONS[name]) for name in names)


def assessment_tools() -> tuple[FunctionTool, ...]:
    """Tools of the Incident Assessment agent."""
    return make_tools(INCIDENT_TYPE_PROFILE, NOMINAL_DEADLINE)


def dispatch_tools() -> tuple[FunctionTool, ...]:
    """Tools of the Dispatch agent."""
    return make_tools(RANK_UNITS_FOR_INCIDENT, RANK_HOSPITALS_FOR_UNIT)


def hospital_tools() -> tuple[FunctionTool, ...]:
    """Tools of a Hospital Coordination agent."""
    return make_tools(CAPACITY_CHECK)


def allocation_tools() -> tuple[FunctionTool, ...]:
    """Tools of the Resource Allocation agent."""
    return make_tools(SCORE_CASE_OPTIONS)
