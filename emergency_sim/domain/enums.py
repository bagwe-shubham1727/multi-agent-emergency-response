"""Enumerations shared by every layer.

All enums are ``StrEnum`` so that they serialise to their plain string value in
JSON logs and LLM payloads. Values that appear in the scenario data files match
those files exactly.
"""

from enum import StrEnum
from types import MappingProxyType
from typing import Final


class Triage(StrEnum):
    """START triage category of a victim."""

    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"
    BLACK = "BLACK"


TRIAGE_RANK: Final = MappingProxyType(
    {Triage.RED: 0, Triage.YELLOW: 1, Triage.GREEN: 2, Triage.BLACK: 3}
)
"""Sort key for triage: lower is more urgent (RED first, BLACK last)."""


class Capability(StrEnum):
    """Hospital capability a victim requires (and hospitals offer)."""

    TRAUMA = "trauma"
    BURNS = "burns"
    NEURO = "neuro"
    CARDIAC = "cardiac"
    PEDIATRIC = "pediatric"
    GENERAL = "general"


class ResponderCapability(StrEnum):
    """Capability a non-medical responder team brings to a scene."""

    FIRE_SUPPRESSION = "fire_suppression"
    EXTRICATION = "extrication"
    HEAVY_RESCUE = "heavy_rescue"
    HAZMAT = "hazmat"
    DECONTAMINATION = "decontamination"
    SCENE_CONTROL = "scene_control"


class AmbulanceType(StrEnum):
    """Advanced or basic life support ambulance."""

    ALS = "ALS"
    BLS = "BLS"


class UnitClass(StrEnum):
    """Kind of mobile unit."""

    AMBULANCE = "ambulance"
    RESPONDER = "responder"


class UnitStatus(StrEnum):
    """Lifecycle state of a unit inside the engine (section 4.2)."""

    AVAILABLE = "available"
    BUSY_T0 = "busy_t0"
    OUT_OF_SERVICE = "out_of_service"
    EN_ROUTE_SCENE = "en_route_scene"
    ON_SCENE = "on_scene"
    LOADING = "loading"
    AWAITING_DESTINATION = "awaiting_destination"
    TRANSPORTING = "transporting"
    AT_HOSPITAL_FULL = "at_hospital_full"
    HANDOVER = "handover"


class VictimStatus(StrEnum):
    """Lifecycle state of a victim inside the engine."""

    UNTRIAGED = "untriaged"
    EXPECTANT = "expectant"
    WAITING = "waiting"
    LOADING = "loading"
    LOADED = "loaded"
    TRANSPORTING = "transporting"
    QUEUED = "queued"
    WAITING_FOR_STAFF = "waiting_for_staff"
    IN_TREATMENT = "in_treatment"
    TREATED = "treated"


class IncidentStatus(StrEnum):
    """Whether an incident still has victims on scene."""

    OPEN = "open"
    CLEARED = "cleared"


class StaffRole(StrEnum):
    """Role of an ER staff member."""

    DOCTOR = "doctor"
    NURSE = "nurse"


class TriggerKind(StrEnum):
    """Why the engine stopped at a decision point."""

    NEW_INCIDENT = "new_incident"
    SCENE_TRIAGED = "scene_triaged"
    PATIENTS_LOADED = "patients_loaded"
    UNIT_FREED = "unit_freed"
    HOSPITAL_FULL_ON_ARRIVAL = "hospital_full_on_arrival"
    DETERIORATION = "deterioration"


class Condition(StrEnum):
    """Experimental condition (baseline, full MAS or an ablation)."""

    B0 = "b0"
    B0_CUES = "b0-cues"
    B1 = "b1"
    FULL = "full"
    FULL_NO_FLOAT = "full-no-float"
    FULL_NO_DIVERSION = "full-no-diversion"
    FULL_NO_ESCALATION = "full-no-escalation"


GREEDY_CONDITIONS: Final = frozenset({Condition.B0, Condition.B0_CUES})
"""Baselines run by the greedy controller: no messages and no agent calls."""


class AgentRole(StrEnum):
    """Role of an agent in the multi-agent system."""

    ASSESSMENT = "incident_assessment"
    DISPATCH = "dispatch"
    HOSPITAL = "hospital_coordination"
    ALLOCATION = "resource_allocation"


class MessageType(StrEnum):
    """Performative of an inter-agent message."""

    PRIORITY_REPORT = "PRIORITY_REPORT"
    PROPOSE = "PROPOSE"
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    COUNTER = "COUNTER"
    REQUEST = "REQUEST"
    ESCALATE = "ESCALATE"
    COMMIT = "COMMIT"
    INFORM = "INFORM"


class CaseKind(StrEnum):
    """Kind of conflict escalated to the Resource Allocation agent."""

    UNIT_CONTENTION = "UNIT_CONTENTION"
    NO_ACCEPTING_HOSPITAL = "NO_ACCEPTING_HOSPITAL"
    FLOAT_STAFF = "FLOAT_STAFF"
    RESERVE_RELEASE = "RESERVE_RELEASE"


class CounterCondition(StrEnum):
    """Condition attached to a hospital's conditional acceptance (COUNTER)."""

    FLOAT_DOCTOR = "FLOAT_DOCTOR"
    FLOAT_NURSE = "FLOAT_NURSE"
    RESERVE_RELEASE = "RESERVE_RELEASE"


class Resolution(StrEnum):
    """How a negotiation thread or case was closed."""

    ACCEPTED = "ACCEPTED"
    ACCEPTED_AFTER_REJECT = "ACCEPTED_AFTER_REJECT"
    COUNTER_GRANTED = "COUNTER_GRANTED"
    COUNTER_DECLINED = "COUNTER_DECLINED"
    RA_COMMIT = "RA_COMMIT"
    FORCED_ADMISSION = "FORCED_ADMISSION"
    KEPT_ASSIGNMENT = "KEPT_ASSIGNMENT"
    ENGINE_REJECTED = "ENGINE_REJECTED"


class RejectReason(StrEnum):
    """Why the engine refused an action (section 4.6)."""

    UNKNOWN_UNIT = "unknown_unit"
    UNKNOWN_INCIDENT = "unknown_incident"
    UNKNOWN_HOSPITAL = "unknown_hospital"
    UNKNOWN_STAFF = "unknown_staff"
    UNIT_NOT_AVAILABLE = "unit_not_available"
    UNIT_NOT_EN_ROUTE = "unit_not_en_route"
    NOT_AN_AMBULANCE = "not_an_ambulance"
    INCIDENT_NOT_OPEN = "incident_not_open"
    SAME_TARGET = "same_target"
    NOT_AWAITING_DESTINATION = "not_awaiting_destination"
    HOSPITAL_NOT_CAPABLE = "hospital_not_capable"
    INSUFFICIENT_BEDS = "insufficient_beds"
    SAME_HOSPITAL = "same_hospital"
    STAFF_NOT_FLOAT = "staff_not_float"
    STAFF_ALREADY_LENT = "staff_already_lent"
    STAFF_OFF_SHIFT = "staff_off_shift"


class SimEventType(StrEnum):
    """Kind of simulation event logged by the engine (values are lowercase names)."""

    INCIDENT_REPORTED = "incident_reported"
    UNIT_DISPATCHED = "unit_dispatched"
    UNIT_DIVERTED = "unit_diverted"
    UNIT_ARRIVED_SCENE = "unit_arrived_scene"
    SCENE_TRIAGED = "scene_triaged"
    ACCESS_GRANTED = "access_granted"
    LOADING_STARTED = "loading_started"
    PATIENTS_LOADED = "patients_loaded"
    BEDS_RESERVED = "beds_reserved"
    TRANSPORT_STARTED = "transport_started"
    ARRIVED_HOSPITAL = "arrived_hospital"
    HOSPITAL_FULL_ON_ARRIVAL = "hospital_full_on_arrival"
    PATIENT_BEDDED = "patient_bedded"
    PATIENT_QUEUED = "patient_queued"
    FIELD_CARE_STARTED = "field_care_started"
    TREATMENT_STARTED = "treatment_started"
    TREATMENT_ENDED = "treatment_ended"
    UNIT_AVAILABLE = "unit_available"
    VICTIM_DETERIORATED = "victim_deteriorated"
    STAFF_LENT = "staff_lent"
    SHIFT_ENDED = "shift_ended"
    T0_BED_RELEASED = "t0_bed_released"
    T0_STAFF_RELEASED = "t0_staff_released"
    INCIDENT_CLEARED = "incident_cleared"
    CONTROLLER_OMISSION = "controller_omission"
