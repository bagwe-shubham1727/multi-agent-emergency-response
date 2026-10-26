"""Agent specifications: the single description each backend builds its SDK agents from.

An ``AgentSpec`` ties an agent name to its role, instructions, read-only tools,
input/output types, offline policy, and the "evidence" tool call the offline
model makes before answering. The roster (plan 5.1) is: incident_assessment,
dispatch, one hospital_<id> per scenario hospital, and resource_allocation.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

from agents import FunctionTool

from emergency_sim.domain.agent_io import (
    AllocationInput,
    AllocationOutput,
    AssessmentInput,
    AssessmentOutput,
    DispatchInput,
    DispatchOutput,
    HospitalInput,
    HospitalOutput,
)
from emergency_sim.domain.base import AgentIOModel
from emergency_sim.domain.enums import AgentRole
from emergency_sim.domain.messages import (
    ALLOCATION_AGENT,
    ASSESSMENT_AGENT,
    DISPATCH_AGENT,
    hospital_agent_name,
)
from emergency_sim.domain.scenario import HospitalSpec
from emergency_sim.mas import instructions, tools
from emergency_sim.policies.allocation import allocate
from emergency_sim.policies.assessment import assess
from emergency_sim.policies.dispatch import dispatch, priority_order
from emergency_sim.policies.hospital import hospital
from emergency_sim.policies.params import PolicyParams

type Policy = Callable[[AgentIOModel, PolicyParams], AgentIOModel]
type Evidence = tuple[str, dict[str, str]]
type EvidenceFn = Callable[[AgentIOModel], Evidence | None]


@dataclass(frozen=True)
class AgentSpec:
    """Everything needed to build and run one agent.

    Attributes:
        name: Agent name (also the SDK ``Agent.name``).
        role: Agent role.
        instructions: System prompt.
        tools: Read-only function tools.
        input_type: The ``*Input`` model the agent receives.
        output_type: The ``*Output`` model it must return.
        policy: Offline rule-based policy ``(payload, params) -> output``.
        evidence: ``(tool name, arguments)`` of the offline model's single
            tool call for a payload, or None when there is nothing to look up.
    """

    name: str
    role: AgentRole
    instructions: str
    tools: tuple[FunctionTool, ...]
    input_type: type[AgentIOModel]
    output_type: type[AgentIOModel]
    policy: Policy
    evidence: EvidenceFn


def assessment_evidence(payload: AgentIOModel) -> Evidence | None:
    """Looks up the type profile of the first incident (by id)."""
    inp = cast(AssessmentInput, payload)
    if not inp.incidents:
        return None
    first = min(inp.incidents, key=lambda r: r.incident_id)
    return (tools.INCIDENT_TYPE_PROFILE, {"incident_type": first.incident_type})


def dispatch_evidence(payload: AgentIOModel) -> Evidence | None:
    """Ranks hospitals for the first loaded unit, else units for the top incident."""
    inp = cast(DispatchInput, payload)
    if inp.loaded_units:
        first = min(inp.loaded_units, key=lambda u: u.unit_id)
        return (tools.RANK_HOSPITALS_FOR_UNIT, {"unit_id": first.unit_id})
    if inp.incidents:
        top = priority_order(inp.incidents)[0]
        return (tools.RANK_UNITS_FOR_INCIDENT, {"incident_id": top.incident_id, "ambulance_type": "ANY"})
    return None


def hospital_evidence(payload: AgentIOModel) -> Evidence | None:
    """Checks capacity for the first admission request (by proposal id)."""
    inp = cast(HospitalInput, payload)
    if not inp.requests:
        return None
    first = min(inp.requests, key=lambda r: r.proposal_id)
    return (tools.CAPACITY_CHECK, {"proposal_id": first.proposal_id})


def allocation_evidence(payload: AgentIOModel) -> Evidence | None:
    """Scores the options of the first case (by case id)."""
    inp = cast(AllocationInput, payload)
    if not inp.cases:
        return None
    first = min(inp.cases, key=lambda c: c.case_id)
    return (tools.SCORE_CASE_OPTIONS, {"case_id": first.case_id})


def _hospital_spec(spec: HospitalSpec, params: PolicyParams) -> AgentSpec:
    return AgentSpec(
        name=hospital_agent_name(spec.id),
        role=AgentRole.HOSPITAL,
        instructions=instructions.hospital_instructions(spec, params),
        tools=tools.hospital_tools(),
        input_type=HospitalInput,
        output_type=HospitalOutput,
        policy=cast(Policy, hospital),
        evidence=hospital_evidence,
    )


def agent_specs(
    hospitals: Sequence[HospitalSpec], params: PolicyParams = PolicyParams()
) -> tuple[AgentSpec, ...]:
    """Builds the agent roster for a scenario's hospitals.

    Args:
        hospitals: Scenario hospitals; one Hospital Coordination agent each.
        params: Policy parameters quoted in the instructions.

    Returns:
        Specs in the order assessment, dispatch, hospitals by id, allocation.
    """
    assessment = AgentSpec(
        ASSESSMENT_AGENT, AgentRole.ASSESSMENT, instructions.assessment_instructions(params),
        tools.assessment_tools(), AssessmentInput, AssessmentOutput, cast(Policy, assess),
        assessment_evidence,
    )
    dispatcher = AgentSpec(
        DISPATCH_AGENT, AgentRole.DISPATCH, instructions.dispatch_instructions(params),
        tools.dispatch_tools(), DispatchInput, DispatchOutput, cast(Policy, dispatch),
        dispatch_evidence,
    )
    allocation = AgentSpec(
        ALLOCATION_AGENT, AgentRole.ALLOCATION, instructions.allocation_instructions(params),
        tools.allocation_tools(), AllocationInput, AllocationOutput, cast(Policy, allocate),
        allocation_evidence,
    )
    hospital_specs = tuple(_hospital_spec(h, params) for h in sorted(hospitals, key=lambda h: h.id))
    return (assessment, dispatcher, *hospital_specs, allocation)


def specs_by_name(specs: Sequence[AgentSpec]) -> Mapping[str, AgentSpec]:
    """Indexes specs by agent name.

    Args:
        specs: Agent specs.

    Returns:
        A read-only name -> spec mapping.

    Raises:
        ValueError: If two specs share a name.
    """
    index = {s.name: s for s in specs}
    if len(index) != len(specs):
        raise ValueError("duplicate agent names in specs")
    return MappingProxyType(index)
