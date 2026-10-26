"""Agent inputs/outputs (LLM-facing DTOs) and the backend decision record.

Every ``*Input`` is the full JSON payload one agent call sees; every
``*Output`` is the structured output the agent must return. Output models are
strict-schema compatible (no dict fields, no defaults, no numeric or length
constraints), so they can be used as an OpenAI strict ``output_type``. Ranges
are checked by the coordinator's semantic validation instead.

Class docstrings become JSON-schema descriptions seen by the LLM, so they are
written for that reader.
"""

from dataclasses import dataclass
from typing import Literal

from emergency_sim.domain.base import AgentIOModel, FrozenModel
from emergency_sim.domain.enums import (
    CaseKind,
    Capability,
    CounterCondition,
    ResponderCapability,
    StaffRole,
    Triage,
    UnitClass,
    UnitStatus,
    VictimStatus,
)


class TriagedVictim(AgentIOModel):
    """A victim whose scene has been triaged.

    Attributes:
        minutes_since_deterioration: For a YELLOW that turned RED, minutes
            since the deterioration was first seen (its shorter RED window
            counts from then); None for a victim that has not deteriorated.
    """

    victim_id: str
    triage: Triage
    condition: str
    required_capability: Capability
    status: VictimStatus
    minutes_since_deterioration: float | None


# --- Incident Assessment ---------------------------------------------------


class IncidentReport(AgentIOModel):
    """An open incident to assess: the caller's report plus any triage results."""

    incident_id: str
    incident_type: str
    t_report_min: float
    minutes_since_report: float
    x: float
    y: float
    intersection: str
    reported_victims: int
    caller_report: str
    required_responder_capabilities: list[ResponderCapability]
    triaged_victims: list[TriagedVictim]


class AssessmentInput(AgentIOModel):
    """Incidents that need a (re)assessment at time t_min."""

    t_min: float
    incidents: list[IncidentReport]


class PriorityReport(AgentIOModel):
    """Assessed priority of one incident (urgency 0..100, higher is more urgent)."""

    incident_id: str
    urgency: int
    suspected_red: bool
    suspected_cardiac_arrest: bool
    est_patients: int
    needed_capabilities: list[Capability]
    als_units_needed: int
    deadline_in_min: float
    rationale: str


class AssessmentOutput(AgentIOModel):
    """One priority report per input incident."""

    reports: list[PriorityReport]


# --- Dispatch --------------------------------------------------------------


class IncidentNeed(AgentIOModel):
    """An open incident with its assessed priority and unmet resource needs.

    Attributes:
        access_wait_min: Estimated minutes until ambulances may load on scene:
            0 if accessible, else the arrival of the last required responder
            team en route plus the heavy-rescue delay when one applies; None
            if a required responder capability is neither on scene nor en
            route (access time unknown).
    """

    incident_id: str
    x: float
    y: float
    urgency: int
    suspected_red: bool
    suspected_cardiac_arrest: bool
    deadline_in_min: float
    triaged: bool
    est_patients_remaining: int
    red_waiting: int
    ambulances_assigned: int
    seats_en_route: int
    als_en_route: int
    als_needed: int
    responder_caps_missing: list[ResponderCapability]
    access_wait_min: float | None


class UnitStatusView(AgentIOModel):
    """A unit's type, status and position."""

    unit_id: str
    unit_class: UnitClass
    unit_type: str
    capabilities: list[str]
    status: UnitStatus
    x: float
    y: float
    incident_id: str | None
    seats: int


class HospitalBoardEntry(AgentIOModel):
    """Public bed board entry of one hospital."""

    hospital_id: str
    x: float
    y: float
    capabilities: list[Capability]
    beds_free_unreserved: int
    queue_length: int


class LoadedUnit(AgentIOModel):
    """An ambulance with patients on board that needs a destination."""

    unit_id: str
    incident_id: str
    x: float
    y: float
    victims: list[TriagedVictim]


class DispatchInput(AgentIOModel):
    """Everything the dispatch agent sees at time t_min."""

    t_min: float
    incidents: list[IncidentNeed]
    units: list[UnitStatusView]
    hospitals: list[HospitalBoardEntry]
    loaded_units: list[LoadedUnit]
    allow_diversion: bool
    allow_escalation: bool


class UnitOrder(AgentIOModel):
    """Send an available unit to an incident."""

    unit_id: str
    incident_id: str
    rationale: str


class DiversionRequest(AgentIOModel):
    """Divert an ambulance en route to a scene to a more urgent incident."""

    unit_id: str
    to_incident_id: str
    rationale: str


class ContentionFlag(AgentIOModel):
    """A unit that several incidents compete for; Resource Allocation decides."""

    unit_id: str
    incident_ids: list[str]
    rationale: str


class TransportPlan(AgentIOModel):
    """Hospitals to propose, in order, for one loaded unit."""

    unit_id: str
    hospital_ranking: list[str]
    rationale: str


class DispatchOutput(AgentIOModel):
    """Dispatch orders, diversion requests, contention flags and transport plans."""

    orders: list[UnitOrder]
    diversions: list[DiversionRequest]
    contentions: list[ContentionFlag]
    transports: list[TransportPlan]


# --- Hospital Coordination (one agent per hospital) ------------------------


class StaffMemberView(AgentIOModel):
    """A doctor or nurse with current load."""

    staff_id: str
    role: StaffRole
    skills: list[str]
    load: int
    max_patients: int
    on_shift: bool
    lent: bool


class AdmissionRequest(AgentIOModel):
    """A proposal to admit a unit's patients."""

    proposal_id: str
    incident_id: str
    unit_id: str
    victims: list[TriagedVictim]
    eta_min: float


class HospitalInput(AgentIOModel):
    """The hospital's private state plus the admission requests to answer."""

    t_min: float
    hospital_id: str
    capabilities: list[Capability]
    er_capacity: int
    beds_occupied: int
    beds_reserved: int
    beds_free_unreserved: int
    critical_reserve: int
    queue_length: int
    waiting_for_staff: int
    staff: list[StaffMemberView]
    requests: list[AdmissionRequest]
    allowed_responses: list[Literal["ACCEPT", "REJECT", "COUNTER"]]
    allowed_counter_conditions: list[CounterCondition]


class AdmissionResponse(AgentIOModel):
    """Answer to one admission request; counter fields are null unless COUNTER."""

    proposal_id: str
    response: Literal["ACCEPT", "REJECT", "COUNTER"]
    reason: Literal["OK", "NO_BEDS", "RESERVE_HELD", "NO_DOCTOR", "NO_NURSE", "NOT_CAPABLE"]
    counter_condition: CounterCondition | None
    counter_skill: str | None
    rationale: str


class HospitalOutput(AgentIOModel):
    """One response per admission request."""

    responses: list[AdmissionResponse]


# --- Resource Allocation ---------------------------------------------------


class CaseOption(AgentIOModel):
    """One way to resolve a case, with its precomputed expected utility."""

    option_id: str
    summary: str
    triage_weight: float
    slack_min: float
    travel_min: float
    utility: float


class EscalationCase(AgentIOModel):
    """A conflict escalated for arbitration."""

    case_id: str
    kind: CaseKind
    incident_id: str
    hospital_id: str | None
    summary: str
    options: list[CaseOption]


class AllocationInput(AgentIOModel):
    """Open cases and the float staff pool at time t_min."""

    t_min: float
    cases: list[EscalationCase]
    float_pool: list[StaffMemberView]


class CaseDecision(AgentIOModel):
    """The option chosen for one case."""

    case_id: str
    chosen_option_id: str
    rationale: str


class AllocationOutput(AgentIOModel):
    """One decision per case."""

    decisions: list[CaseDecision]


# --- Decision record returned by any backend -------------------------------


class ToolCallTrace(FrozenModel):
    """One function-tool call made during an agent run."""

    name: str
    arguments: str
    output: str


class TokenUsage(FrozenModel):
    """Model usage of one agent call."""

    requests: int
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class AgentDecision:
    """What a ``DecisionBackend`` returns for one agent call.

    Attributes:
        output: The validated structured output (an ``*Output`` model).
        source: ``"model"`` if the model answered, ``"policy_fallback"`` if
            the rule-based policy replaced a failed model call.
        error: Exception summary when ``source == "policy_fallback"``.
        tool_calls: Function-tool calls made during the run, in order.
        usage: Token usage, if known.
        latency_ms: Wall-clock latency; None for offline runs (keeps logs
            byte-identical).
    """

    output: AgentIOModel
    source: Literal["model", "policy_fallback"]
    error: str | None
    tool_calls: tuple[ToolCallTrace, ...]
    usage: TokenUsage | None
    latency_ms: float | None
