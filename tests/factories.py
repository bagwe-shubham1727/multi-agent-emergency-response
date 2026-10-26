"""Builders for tiny, hand-made scenarios and log records used across the test suite.

Every builder returns a validated, immutable domain object with sensible
defaults, so a test only spells out what it is about::

    from tests import factories as f

    scenario = f.make_scenario(
        hospitals=[f.hospital("H1", caps=("trauma", "general"))],
        ambulances=[f.ambulance("AMB-01", "ALS", x=2.0, y=2.0)],
        staff=[f.doctor("DR-1", "H1", ("emergency", "trauma")), f.nurse("RN-1", "H1")],
        incidents=[f.incident("INC-01", 0, 4.0, 4.0, [f.victim("V1", "RED", "trauma")])],
    )

Conventions:
    * Ids are free-form, but keep the real data's style (``H1``, ``AMB-01``,
      ``INC-01``) so log output reads naturally.
    * Victim ids passed to ``incident`` are used verbatim; ``victim("V1", ...)``
      inside ``incident("INC-01", ...)`` stays ``V1``.
    * Defaults mirror ``datagen``: doctors carry 4 patients, nurses 3, every
      ER doctor and nurse has the ``"emergency"`` skill, ambulances seat 2 with
      at most 1 RED per trip.
    * ``tiny_scenario()`` is a ready-made one-hospital, two-ambulance,
      one-incident world for smoke tests; ``sample_log_records()`` returns one
      record of every ``LogRecord`` kind for log/dashboard tests.
"""

from collections.abc import Sequence
from types import MappingProxyType

from emergency_sim import geo
from emergency_sim.domain.actions import AssignTransport, DispatchUnit
from emergency_sim.domain.agent_io import TokenUsage, ToolCallTrace
from emergency_sim.domain.enums import (
    TRIAGE_RANK,
    AgentRole,
    Capability,
    Condition,
    IncidentStatus,
    MessageType,
    RejectReason,
    Resolution,
    SimEventType,
    StaffRole,
    Triage,
    TriggerKind,
)
from emergency_sim.domain.log_records import (
    ActionRecord,
    AgentCallRecord,
    FallbackNote,
    LogRecord,
    MessageRecord,
    RunEndRecord,
    RunStartRecord,
    SimEventRecord,
    SnapshotCounts,
    SnapshotRecord,
)
from emergency_sim.domain.messages import (
    DISPATCH_AGENT,
    AdmitAction,
    CommitAction,
    Message,
    ProposedAction,
    hospital_agent_name,
)
from emergency_sim.domain.outcomes import (
    IncidentOutcome,
    RunMetrics,
    RunOutcomes,
    VictimOutcome,
)
from emergency_sim.domain.scenario import (
    AmbulanceSpec,
    HospitalSpec,
    IncidentSpec,
    ResponderSpec,
    Scenario,
    StaffSpec,
    StationSpec,
    VictimSpec,
)
from emergency_sim.domain.views import IncidentView, PublicView

DEFAULT_HORIZON_MIN = 180.0
FLOAT_HOME = "FLOAT"
DEFAULT_STATION = StationSpec(id="S1", name="Station 1 (Test)", x=5.0, y=5.0)

# Default care windows by triage when ``victim(window=None)`` (datagen ranges:
# RED 20-60, YELLOW 90-240, GREEN 480, BLACK 0).
DEFAULT_WINDOW_MIN = MappingProxyType({"RED": 40, "YELLOW": 165, "GREEN": 480, "BLACK": 0})


# --- Scenario entities ------------------------------------------------------


def hospital(
    id: str = "H1",
    x: float = 5.0,
    y: float = 5.0,
    caps: Sequence[str] = ("general",),
    er_capacity: int = 4,
    occupied: int = 0,
    reserve: int = 1,
    *,
    name: str | None = None,
) -> HospitalSpec:
    """Builds a hospital ER.

    Args:
        id: Hospital id.
        x: Position in km.
        y: Position in km.
        caps: Capability values, e.g. ``("trauma", "general")``.
        er_capacity: ER beds.
        occupied: Beds occupied at t=0 (released one per 15 min by the engine).
        reserve: ``critical_reserve`` held back for RED patients.
        name: Display name; defaults to ``"<id> Test Hospital"``.

    Returns:
        The hospital spec (no ICU beds; trauma level 1 iff it offers trauma).
    """
    return HospitalSpec(
        id=id,
        name=name or f"{id} Test Hospital",
        x=x,
        y=y,
        trauma_level=1 if "trauma" in caps else 3,
        capabilities=tuple(Capability(c) for c in caps),
        er_capacity=er_capacity,
        icu_beds=0,
        critical_reserve=reserve,
        er_occupied_t0=occupied,
        icu_occupied_t0=0,
        er_free_t0=er_capacity - occupied,
        intersection=geo.nearest_intersection((x, y)),
    )


def station(id: str = "S1", x: float = 5.0, y: float = 5.0) -> StationSpec:
    """Builds a station.

    Args:
        id: Station id.
        x: Position in km.
        y: Position in km.

    Returns:
        The station spec.
    """
    return StationSpec(id=id, name=f"Station {id} (Test)", x=x, y=y)


def ambulance(
    id: str,
    type: str = "ALS",
    x: float = 5.0,
    y: float = 5.0,
    status: str = "available",
    busy_until: int = 0,
    *,
    station: str = "S1",
    capacity: int = 2,
    max_red_per_trip: int = 1,
) -> AmbulanceSpec:
    """Builds an ambulance.

    Args:
        id: Unit id, e.g. ``"AMB-01"``.
        type: ``"ALS"`` or ``"BLS"``.
        x: Position in km.
        y: Position in km.
        status: ``"available"``, ``"busy"`` (busy at t=0 until ``busy_until``)
            or ``"out_of_service"``.
        busy_until: Minute a busy unit becomes available.
        station: Home station id.
        capacity: Patient seats.
        max_red_per_trip: RED patients allowed per trip.

    Returns:
        The ambulance spec.
    """
    return AmbulanceSpec.model_validate(
        {
            "id": id,
            "type": type,
            "station": station,
            "x": x,
            "y": y,
            "capacity": capacity,
            "max_red_per_trip": max_red_per_trip,
            "status": status,
            "busy_until_min": busy_until,
        }
    )


def doctor(
    id: str,
    home: str,
    skills: Sequence[str] = ("emergency",),
    load: int = 0,
    shift_end: int = 300,
    *,
    max_patients: int = 4,
) -> StaffSpec:
    """Builds an ER doctor.

    Args:
        id: Staff id, e.g. ``"DR-H1-1"``.
        home: Hospital id, or ``"FLOAT"`` for the float pool.
        skills: Skills; include ``"emergency"`` for general patients and the
            capability name (``"trauma"``, ``"cardiac"``...) for specialist care.
        load: Patients carried at t=0 (shed one per 20 min by the engine).
        shift_end: Shift end minute.
        max_patients: Concurrent patient limit.

    Returns:
        The staff spec.
    """
    return StaffSpec(
        id=id,
        role=StaffRole.DOCTOR,
        home=home,
        skills=tuple(sorted(skills)),
        max_patients=max_patients,
        load_t0=load,
        shift_end_min=shift_end,
    )


def nurse(
    id: str,
    home: str,
    skills: Sequence[str] = ("emergency",),
    load: int = 0,
    shift_end: int = 300,
    *,
    max_patients: int = 3,
) -> StaffSpec:
    """Builds an ER nurse.

    Args:
        id: Staff id, e.g. ``"RN-H1-1"``.
        home: Hospital id, or ``"FLOAT"`` for the float pool.
        skills: Skills; every nurse needs ``"emergency"`` to treat anyone.
        load: Patients carried at t=0.
        shift_end: Shift end minute.
        max_patients: Concurrent patient limit.

    Returns:
        The staff spec.
    """
    return StaffSpec(
        id=id,
        role=StaffRole.NURSE,
        home=home,
        skills=tuple(sorted(skills)),
        max_patients=max_patients,
        load_t0=load,
        shift_end_min=shift_end,
    )


def responder(
    id: str,
    caps: Sequence[str],
    x: float = 5.0,
    y: float = 5.0,
    *,
    type: str = "test_team",
    base: str = "S1",
    crew: int = 4,
) -> ResponderSpec:
    """Builds a responder team.

    Args:
        id: Team id, e.g. ``"FR-ENG1"``.
        caps: Responder capability values, e.g. ``("extrication",)``.
        x: Position in km.
        y: Position in km.
        type: Team type label.
        base: Home station id.
        crew: Crew size.

    Returns:
        The responder spec (available at t=0).
    """
    return ResponderSpec.model_validate(
        {
            "id": id,
            "type": type,
            "base": base,
            "capabilities": list(caps),
            "x": x,
            "y": y,
            "crew": crew,
            "status": "available",
            "busy_until_min": 0,
        }
    )


def victim(
    vid: str,
    triage: str,
    capability: str = "general",
    window: int | None = None,
    deteriorates: int | None = None,
    condition: str = "lacerations",
    *,
    needs_als: bool | None = None,
    needs_decon: bool = False,
) -> VictimSpec:
    """Builds a victim (ground truth).

    Args:
        vid: Victim id.
        triage: ``"RED"``, ``"YELLOW"``, ``"GREEN"`` or ``"BLACK"``.
        capability: Required hospital capability value.
        window: Golden window in minutes after the report; defaults by triage
            (RED 40, YELLOW 165, GREEN 480, BLACK 0).
        deteriorates: Minutes after the report at which an untreated YELLOW
            turns RED; None for no deterioration.
        condition: Condition label (``"cardiac arrest"`` triggers field care).
        needs_als: Defaults to ``triage == "RED"``.
        needs_decon: Whether decontamination is needed.

    Returns:
        The victim spec.
    """
    return VictimSpec(
        victim_id=vid,
        condition=condition,
        triage=Triage(triage),
        required_capability=Capability(capability),
        needs_als=triage == "RED" if needs_als is None else needs_als,
        needs_decon=needs_decon,
        golden_window_min=DEFAULT_WINDOW_MIN[triage] if window is None else window,
        deteriorates_to_red_at_min=deteriorates,
    )


def incident(
    iid: str,
    t: int,
    x: float,
    y: float,
    victims: Sequence[VictimSpec],
    reported: int | None = None,
    responders: Sequence[str] = (),
    caller_report: str = "Caller reports people hurt; someone is bleeding.",
    *,
    type: str = "fall",
    mci_cluster: bool = False,
) -> IncidentSpec:
    """Builds an incident; hidden summary fields are derived from ``victims``.

    Args:
        iid: Incident id, e.g. ``"INC-01"``.
        t: Report minute.
        x: Position in km.
        y: Position in km.
        victims: Victims at the scene (at least one).
        reported: Victims the caller reports; defaults to ``len(victims)``.
        responders: Required responder capability values, e.g. ``("extrication",)``.
        caller_report: The caller's free-text report.
        type: Incident type (see ``datagen.INCIDENT_TYPES``).
        mci_cluster: Whether the incident belongs to a mass-casualty cluster.

    Returns:
        The incident spec.

    Raises:
        ValueError: If ``victims`` is empty.
    """
    if not victims:
        raise ValueError("an incident needs at least one victim")
    alive = [v for v in victims if v.triage is not Triage.BLACK]
    worst = min((v.triage for v in alive), key=TRIAGE_RANK.__getitem__, default=Triage.BLACK)
    return IncidentSpec(
        incident_id=iid,
        type=type,
        t_report_min=t,
        x=x,
        y=y,
        intersection=geo.nearest_intersection((x, y)),
        congestion=round(geo.congestion((x, y)), 2),
        reported_victims=len(victims) if reported is None else reported,
        caller_report=caller_report,
        required_responder_capabilities=tuple(responders),
        true_victims=len(victims),
        worst_triage=worst,
        time_sensitivity_min=min((v.golden_window_min for v in alive), default=0),
        mci_cluster=mci_cluster,
        victims=tuple(victims),
    )


def make_scenario(
    *,
    hospitals: Sequence[HospitalSpec] = (),
    ambulances: Sequence[AmbulanceSpec] = (),
    staff: Sequence[StaffSpec] = (),
    responders: Sequence[ResponderSpec] = (),
    incidents: Sequence[IncidentSpec] = (),
    horizon: float = DEFAULT_HORIZON_MIN,
    stations: Sequence[StationSpec] = (DEFAULT_STATION,),
    name: str = "tiny",
    seed: int = 0,
) -> Scenario:
    """Assembles a scenario from entity specs.

    Args:
        hospitals: Hospitals.
        ambulances: Ambulances.
        staff: Doctors and nurses (home hospital or ``"FLOAT"``).
        responders: Responder teams.
        incidents: Incidents.
        horizon: Simulation horizon in minutes.
        stations: Stations (one default station ``S1``).
        name: Scenario name.
        seed: Seed recorded in the scenario (informational).

    Returns:
        The scenario.
    """
    return Scenario(
        name=name,
        seed=seed,
        horizon_min=horizon,
        hospitals=tuple(hospitals),
        stations=tuple(stations),
        ambulances=tuple(ambulances),
        staff=tuple(staff),
        responders=tuple(responders),
        incidents=tuple(incidents),
    )


def tiny_scenario(*, horizon: float = DEFAULT_HORIZON_MIN) -> Scenario:
    """Returns a small, fully specified world for smoke tests.

    One general/trauma hospital ``H1`` at (5, 5) with 4 beds, one ALS
    (``AMB-01``) and one BLS (``AMB-02``) at the hospital, one doctor and one
    nurse, and incident ``INC-01`` at (7, 5) reported at t=0 with a RED trauma
    victim ``INC-01-V1`` and a GREEN general victim ``INC-01-V2``.

    Args:
        horizon: Simulation horizon in minutes.

    Returns:
        The scenario.
    """
    return make_scenario(
        hospitals=[hospital("H1", caps=("trauma", "general"), er_capacity=4, reserve=1)],
        ambulances=[ambulance("AMB-01", "ALS"), ambulance("AMB-02", "BLS")],
        staff=[
            doctor("DR-H1-1", "H1", ("emergency", "trauma")),
            nurse("RN-H1-1", "H1", ("emergency", "trauma")),
        ],
        incidents=[
            incident(
                "INC-01",
                0,
                7.0,
                5.0,
                [
                    victim("INC-01-V1", "RED", "trauma", condition="stab wound"),
                    victim("INC-01-V2", "GREEN", "general"),
                ],
            )
        ],
        horizon=horizon,
    )


# --- Log records ------------------------------------------------------------


def empty_public_view(t: float = 0.0, horizon: float = DEFAULT_HORIZON_MIN) -> PublicView:
    """Returns a view with no incidents, units, hospitals or float staff.

    Args:
        t: View time.
        horizon: Horizon in minutes.

    Returns:
        The view.
    """
    return PublicView(t_min=t, horizon_min=horizon, incidents=(), units=(), hospitals=(), float_pool=())


def untriaged_incident_view(
    iid: str = "INC-01", t: float = 0.0, x: float = 7.0, y: float = 5.0
) -> IncidentView:
    """Returns an open, untriaged incident view (no victims visible).

    Args:
        iid: Incident id.
        t: Report minute.
        x: Position in km.
        y: Position in km.

    Returns:
        The incident view.
    """
    return IncidentView(
        incident_id=iid,
        incident_type="fall",
        t_report_min=t,
        x=x,
        y=y,
        intersection=geo.nearest_intersection((x, y)),
        reported_victims=1,
        caller_report="Caller reports one person hurt.",
        required_responder_capabilities=(),
        status=IncidentStatus.OPEN,
        triaged=False,
        accessible=True,
        victims=(),
        assigned_unit_ids=(),
        responder_capabilities_present=(),
    )


def message(
    message_id: str = "M-0001",
    *,
    t: float = 0.0,
    sender: str = DISPATCH_AGENT,
    recipient: str = "hospital_H1",
    type: MessageType = MessageType.PROPOSE,
    incident_id: str = "INC-01",
    proposed_action: ProposedAction | None = None,
    rationale: str = "nearest capable hospital",
    in_reply_to: str | None = None,
    conversation_id: str = "NEG-0001",
    round: int = 1,
) -> Message:
    """Builds a message; defaults to a round-1 PROPOSE from dispatch to H1.

    Args:
        message_id: Message id.
        t: Simulation time.
        sender: Sender agent name.
        recipient: Recipient agent name or ``BROADCAST``.
        type: Performative.
        incident_id: Triggering incident.
        proposed_action: Content; defaults to admitting ``INC-01-V1`` from
            ``AMB-01`` at ``recipient``'s hospital with ETA 6.
        rationale: Free text.
        in_reply_to: Answered message id.
        conversation_id: Thread id.
        round: Negotiation round.

    Returns:
        The message.
    """
    hospital_id = recipient.removeprefix("hospital_")
    content = proposed_action or AdmitAction(
        hospital_id=hospital_id, unit_id="AMB-01", victim_ids=("INC-01-V1",), eta_min=6.0
    )
    return Message(
        message_id=message_id,
        t_min=t,
        sender=sender,
        recipient=recipient,
        type=type,
        incident_id=incident_id,
        proposed_action=content,
        rationale=rationale,
        in_reply_to=in_reply_to,
        conversation_id=conversation_id,
        round=round,
    )


def empty_outcomes(horizon: float = DEFAULT_HORIZON_MIN) -> RunOutcomes:
    """Returns outcomes with no victims, incidents or units.

    Args:
        horizon: Horizon in minutes.

    Returns:
        The outcomes.
    """
    return RunOutcomes(
        horizon_min=horizon,
        victims=(),
        incidents=(),
        ambulance_busy_min=(),
        responder_busy_min=(),
        ambulances_in_service=0,
    )


def zero_metrics(**overrides: object) -> RunMetrics:
    """Returns a ``RunMetrics`` with every count zero and every optional None.

    Args:
        **overrides: Field values to set instead of the zero defaults.

    Returns:
        The metrics.
    """
    values: dict[str, object] = {}
    for field_name, info in RunMetrics.model_fields.items():
        annotation = str(info.annotation)
        if "None" in annotation:
            values[field_name] = None
        elif "dict" in annotation:
            values[field_name] = {}
        elif info.annotation is float:
            values[field_name] = 0.0
        else:
            values[field_name] = 0
    values.update(overrides)
    return RunMetrics.model_validate(values)


def sample_log_records(scenario: Scenario | None = None) -> tuple[LogRecord, ...]:
    """Returns one record of every ``LogRecord`` kind, in a valid log order.

    ``seq`` is left at 0 (a sink assigns it); ``t_min`` is non-decreasing.

    Args:
        scenario: Scenario embedded in the ``RunStartRecord``; defaults to
            ``tiny_scenario()``.

    Returns:
        ``(run_start, snapshot, sim_event, message, agent_call, action, run_end)``.
    """
    scenario = scenario or tiny_scenario()
    view = empty_public_view(t=0.0, horizon=scenario.horizon_min).model_copy(
        update={"incidents": (untriaged_incident_view(),)}
    )
    counts = SnapshotCounts(
        incidents_reported=1, incidents_open=1, waiting_on_scene=0, in_transport=0, queued=0,
        waiting_for_staff=0, in_treatment=0, treated=0, care_started_cum=0, expectant=0,
    )
    outcome = VictimOutcome(
        victim_id="INC-01-V1", incident_id="INC-01", t_report_min=0.0, condition="stab wound",
        required_capability=Capability.TRAUMA, initial_triage=Triage.RED, final_triage=Triage.RED,
        deteriorated_at_min=None, deadline_min=40.0, als_on_scene_min=3.3, loaded_min=6.3,
        hospital_id="H1", arrived_hospital_min=8.5, bedded_min=8.5, treatment_start_min=8.5,
        care_start_min=8.5, care_mode="hospital",
    )
    outcomes = empty_outcomes(scenario.horizon_min).model_copy(
        update={
            "victims": (outcome,),
            "incidents": (IncidentOutcome(
                incident_id="INC-01", t_report_min=0.0, mci_cluster=False,
                first_unit_on_scene_min=3.3, first_ambulance_on_scene_min=3.3, cleared_min=6.3,
            ),),
            "ambulance_busy_min": (("AMB-01", 20.0), ("AMB-02", 0.0)),
            "ambulances_in_service": 2,
        }
    )
    proposal = message()
    return (
        RunStartRecord(
            t_min=0.0, run_id="tiny-full-offline-s0-r0", scenario=scenario.name,
            seed=scenario.seed, condition=Condition.FULL, backend="offline", repeat=0,
            sim_params={"handover_min": 5.0}, negotiation={"allow_counter": True, "max_rounds": 3},
            scenario_data=scenario,
        ),
        SnapshotRecord(t_min=0.0, view=view, counts=counts),
        SimEventRecord(t_min=0.0, event=SimEventType.INCIDENT_REPORTED, incident_id="INC-01"),
        MessageRecord(t_min=6.3, message=proposal),
        AgentCallRecord(
            t_min=6.3, agent=hospital_agent_name("H1"), role=AgentRole.HOSPITAL,
            triggers=(TriggerKind.PATIENTS_LOADED,), backend="offline",
            input={"hospital_id": "H1", "requests": []}, output={"responses": []},
            source="model",
            tool_calls=(ToolCallTrace(name="capacity_check", arguments='{"proposal_id": "P-1"}', output="{}"),),
            fallbacks=(FallbackNote(scope="item", key="P-1", rule="unknown_proposal", detail=""),),
            usage=TokenUsage(requests=2, input_tokens=0, output_tokens=0), latency_ms=None,
        ),
        ActionRecord(
            t_min=6.3, actor="hospital_H1",
            action=AssignTransport(unit_id="AMB-01", hospital_id="H1", reserve_beds=True),
            ok=False, reason=RejectReason.INSUFFICIENT_BEDS, detail="0 free beds",
        ),
        RunEndRecord(
            t_min=scenario.horizon_min, outcomes=outcomes,
            metrics=zero_metrics(messages_by_type={"PROPOSE": 1}, resolutions={"ACCEPTED": 1}),
        ),
    )


def commit_message(message_id: str = "M-0002", *, in_reply_to: str = "M-0001") -> Message:
    """Returns an RA COMMIT carrying a nested engine action (for round-trip tests).

    Args:
        message_id: Message id.
        in_reply_to: Answered message id.

    Returns:
        The message.
    """
    return message(
        message_id,
        sender="resource_allocation",
        recipient=DISPATCH_AGENT,
        type=MessageType.COMMIT,
        proposed_action=CommitAction(
            case_id="CASE-0001",
            option_id="OPT-1",
            actions=(DispatchUnit(unit_id="AMB-01", incident_id="INC-01"),),
            resolution=Resolution.RA_COMMIT,
        ),
        in_reply_to=in_reply_to,
    )
