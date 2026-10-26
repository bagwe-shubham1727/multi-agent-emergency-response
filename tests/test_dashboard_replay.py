"""Tests for ``dashboard.replay`` on a hand-built fixture log (no engine).

The fixture world (``fixture_records``) is a 20-minute run with two hospitals,
an ALS, a BLS and a fire team:

* t=0  ``INC-01`` (RED trauma + GREEN) reported; AMB-01 (ALS) dispatched.
* t=3  AMB-01 on scene, triage.
* t=5  ``INC-02`` (YELLOW, needs fire suppression) reported; AMB-02 + FIRE-01 sent.
* t=6  AMB-01 loads the RED: PROPOSE to H1 -> COUNTER (reserve) -> ESCALATE ->
       COMMIT ``COUNTER_GRANTED`` (thread NEG-0003).
* t=9  AMB-02 loads the YELLOW: PROPOSE H1 -> REJECT -> PROPOSE H2 -> ACCEPT
       (NEG-0004); the RED starts treatment at H1.
* t=12 YELLOW treated at H2, INC-02 cleared; H2 REQUESTs a float doctor (NEG-0005).
* t=14 RA COMMITs the float doctor (``COUNTER_GRANTED``).

``write_fixture_results`` lays these out as a results folder for the app
tests (B0 / B1 / Full), and the figure tests reuse ``fixture_run``.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from emergency_sim.evaluation.metrics import kpis_at

from dashboard import replay
from emergency_sim import geo
from emergency_sim.domain.actions import AssignTransport, DispatchUnit, LendStaff
from emergency_sim.domain.agent_io import TokenUsage, ToolCallTrace
from emergency_sim.domain.enums import (
    AgentRole,
    Capability,
    CaseKind,
    Condition,
    CounterCondition,
    IncidentStatus,
    MessageType,
    RejectReason,
    ResponderCapability,
    Resolution,
    SimEventType,
    StaffRole,
    Triage,
    TriggerKind,
    UnitClass,
    UnitStatus,
    VictimStatus,
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
    ALLOCATION_AGENT,
    ASSESSMENT_AGENT,
    BROADCAST,
    DISPATCH_AGENT,
    AdmitAction,
    AssessAction,
    CommitAction,
    ConditionalAdmitAction,
    DeclineAction,
    DispatchAction,
    Message,
    ProposedAction,
    ResolveAction,
    StaffRequestAction,
)
from emergency_sim.domain.outcomes import IncidentOutcome, RunOutcomes, VictimOutcome
from emergency_sim.domain.scenario import Scenario
from emergency_sim.domain.views import (
    HospitalView,
    IncidentView,
    PublicView,
    StaffView,
    UnitView,
    VictimView,
)
from emergency_sim.eventlog import MemoryEventSink, write_jsonl
from tests import factories as f

HORIZON = 20.0
H1, H2 = "hospital_H1", "hospital_H2"


# --- Fixture world --------------------------------------------------------------


def fixture_scenario() -> Scenario:
    """Ground truth embedded in ``run_start``."""
    return f.make_scenario(
        hospitals=[f.hospital("H1", 5.0, 5.0, ("trauma", "general"), er_capacity=4, occupied=1, reserve=1),
                   f.hospital("H2", 15.0, 15.0, ("general",), er_capacity=3, reserve=0)],
        ambulances=[f.ambulance("AMB-01", "ALS"), f.ambulance("AMB-02", "BLS")],
        staff=[f.doctor("DR-H1-1", "H1", ("emergency", "trauma")), f.nurse("RN-H1-1", "H1"),
               f.doctor("DR-H2-1", "H2"), f.doctor("DR-F-1", f.FLOAT_HOME)],
        responders=[f.responder("FIRE-01", ("fire_suppression",))],
        incidents=[
            f.incident("INC-01", 0, 7.0, 5.0, [f.victim("V1", "RED", "trauma", condition="stab wound"),
                                               f.victim("V2", "GREEN")]),
            f.incident("INC-02", 5, 12.0, 12.0, [f.victim("V3", "YELLOW")], responders=("fire_suppression",)),
        ],
        horizon=HORIZON,
    )


def _unit(uid: str, status: UnitStatus, x: float, y: float, *, kind: str = "ALS", incident: str | None = None,
          hospital: str | None = None, dest: str | None = None, load: tuple[str, ...] = ()) -> UnitView:
    responder = kind == "responder"
    return UnitView(
        unit_id=uid, unit_class=UnitClass.RESPONDER if responder else UnitClass.AMBULANCE,
        unit_type="fire_engine" if responder else kind,
        capabilities=("fire_suppression",) if responder else (), status=status, x=x, y=y,
        incident_id=incident, hospital_id=hospital, destination_id=dest, load_victim_ids=load,
        seats=0 if responder else 2 - len(load), available_at_min=None,
    )


def _amb01(t: float) -> UnitView:
    if t < 3:
        return _unit("AMB-01", UnitStatus.EN_ROUTE_SCENE, 6.0, 5.0, incident="INC-01", dest="INC-01")
    if t < 6:
        return _unit("AMB-01", UnitStatus.ON_SCENE, 7.0, 5.0, incident="INC-01")
    if t < 9:
        return _unit("AMB-01", UnitStatus.TRANSPORTING, 6.0, 5.0, hospital="H1", dest="H1", load=("V1",))
    if t < 12:
        return _unit("AMB-01", UnitStatus.HANDOVER, 5.0, 5.0, hospital="H1")
    return _unit("AMB-01", UnitStatus.AVAILABLE, 5.0, 5.0)


def _amb02(t: float) -> UnitView:
    if t < 5:
        return _unit("AMB-02", UnitStatus.AVAILABLE, 5.0, 5.0, kind="BLS")
    if t < 8:
        return _unit("AMB-02", UnitStatus.EN_ROUTE_SCENE, 9.0, 9.0, kind="BLS", incident="INC-02", dest="INC-02")
    if t < 9:
        return _unit("AMB-02", UnitStatus.ON_SCENE, 12.0, 12.0, kind="BLS", incident="INC-02")
    if t < 12:
        return _unit("AMB-02", UnitStatus.TRANSPORTING, 13.0, 13.0, kind="BLS", hospital="H2", dest="H2",
                     load=("V3",))
    if t < 15:
        return _unit("AMB-02", UnitStatus.HANDOVER, 15.0, 15.0, kind="BLS", hospital="H2")
    return _unit("AMB-02", UnitStatus.AVAILABLE, 15.0, 15.0, kind="BLS")


def _fire01(t: float) -> UnitView:
    if 5 <= t < 8:
        return _unit("FIRE-01", UnitStatus.EN_ROUTE_SCENE, 9.0, 9.0, kind="responder", incident="INC-02",
                     dest="INC-02")
    if 8 <= t < 14:
        return _unit("FIRE-01", UnitStatus.ON_SCENE, 12.0, 12.0, kind="responder", incident="INC-02")
    return _unit("FIRE-01", UnitStatus.AVAILABLE, 5.0, 5.0, kind="responder")


def _victim(vid: str, iid: str, triage: Triage, status: VictimStatus, *, cap: Capability = Capability.GENERAL,
            unit: str | None = None, hospital: str | None = None) -> VictimView:
    return VictimView(victim_id=vid, incident_id=iid, triage=triage, condition="test", required_capability=cap,
                      needs_als=triage is Triage.RED, status=status, unit_id=unit, hospital_id=hospital,
                      deteriorated=False)


def _v1(t: float) -> VictimView:
    if t < 6:
        return _victim("V1", "INC-01", Triage.RED, VictimStatus.WAITING, cap=Capability.TRAUMA)
    if t < 9:
        return _victim("V1", "INC-01", Triage.RED, VictimStatus.TRANSPORTING, cap=Capability.TRAUMA, unit="AMB-01")
    return _victim("V1", "INC-01", Triage.RED, VictimStatus.IN_TREATMENT, cap=Capability.TRAUMA, hospital="H1")


def _v3(t: float) -> VictimView:
    if t < 9:
        return _victim("V3", "INC-02", Triage.YELLOW, VictimStatus.WAITING)
    if t < 12:
        return _victim("V3", "INC-02", Triage.YELLOW, VictimStatus.TRANSPORTING, unit="AMB-02")
    return _victim("V3", "INC-02", Triage.YELLOW, VictimStatus.IN_TREATMENT, hospital="H2")


def _incident(iid: str, t_report: float, x: float, y: float, *, triaged: bool, victims: tuple[VictimView, ...],
              units: tuple[str, ...], status: IncidentStatus = IncidentStatus.OPEN,
              responders: tuple[ResponderCapability, ...] = ()) -> IncidentView:
    return IncidentView(
        incident_id=iid, incident_type="fall", t_report_min=t_report, x=x, y=y,
        intersection=geo.nearest_intersection((x, y)), reported_victims=max(len(victims), 1),
        caller_report="Caller reports people hurt.", required_responder_capabilities=responders,
        status=status, triaged=triaged, accessible=True, victims=victims if triaged else (),
        assigned_unit_ids=units, responder_capabilities_present=(),
    )


def _incidents(t: float) -> tuple[IncidentView, ...]:
    triaged1 = t >= 3
    v2 = _victim("V2", "INC-01", Triage.GREEN, VictimStatus.WAITING)
    units1 = ("AMB-01",) if t < 6 else ()
    inc1 = _incident("INC-01", 0.0, 7.0, 5.0, triaged=triaged1, victims=(_v1(t), v2), units=units1)
    if t < 5:
        return (inc1,)
    cleared = t >= 12
    inc2 = _incident("INC-02", 5.0, 12.0, 12.0, triaged=t >= 8, victims=(_v3(t),),
                     units=() if cleared else ("AMB-02", "FIRE-01"),
                     status=IncidentStatus.CLEARED if cleared else IncidentStatus.OPEN,
                     responders=(ResponderCapability.FIRE_SUPPRESSION,))
    return (inc1, inc2)


def _staff(sid: str, role: StaffRole, home: str, hospital: str | None, load: int) -> StaffView:
    return StaffView(staff_id=sid, role=role, skills=("emergency",), home=home, hospital_id=hospital, load=load,
                     max_patients=4 if role is StaffRole.DOCTOR else 3, on_shift=True)


def _hospital(hid: str, x: float, y: float, cap: int, occupied: int, reserved: int, *, queue: int = 0,
              inbound: tuple[str, ...] = (), staff: tuple[StaffView, ...] = (),
              caps: tuple[Capability, ...] = (Capability.GENERAL,)) -> HospitalView:
    return HospitalView(
        hospital_id=hid, name=f"{hid} Test Hospital", x=x, y=y, capabilities=caps, er_capacity=cap,
        critical_reserve=1 if hid == "H1" else 0, beds_occupied=occupied, beds_reserved=reserved,
        beds_free_unreserved=cap - occupied - reserved, queue_length=queue, waiting_for_staff=0,
        in_treatment=max(occupied - 1, 0) if hid == "H1" else occupied, inbound_unit_ids=inbound, staff=staff,
    )


def _hospitals(t: float) -> tuple[HospitalView, ...]:
    treating1 = int(t >= 9)
    h1 = _hospital("H1", 5.0, 5.0, 4, 1 + treating1, int(6 <= t < 9), inbound=("AMB-01",) if 6 <= t < 9 else (),
                   caps=(Capability.TRAUMA, Capability.GENERAL),
                   staff=(_staff("DR-H1-1", StaffRole.DOCTOR, "H1", "H1", treating1),
                          _staff("RN-H1-1", StaffRole.NURSE, "H1", "H1", treating1)))
    lent = t >= 14
    h2_staff = (_staff("DR-H2-1", StaffRole.DOCTOR, "H2", "H2", int(t >= 12)),)
    if lent:
        h2_staff += (_staff("DR-F-1", StaffRole.DOCTOR, f.FLOAT_HOME, "H2", 0),)
    h2 = _hospital("H2", 15.0, 15.0, 3, int(t >= 12), int(9 <= t < 12), queue=int(12 <= t < 15),
                   inbound=("AMB-02",) if 9 <= t < 12 else (), staff=h2_staff)
    return (h1, h2)


def fixture_view(t: float) -> PublicView:
    """The public view of the fixture world at minute ``t``."""
    floater = _staff("DR-F-1", StaffRole.DOCTOR, f.FLOAT_HOME, "H2" if t >= 14 else None, 0)
    return PublicView(t_min=t, horizon_min=HORIZON, incidents=_incidents(t),
                      units=(_amb01(t), _amb02(t), _fire01(t)), hospitals=_hospitals(t), float_pool=(floater,))


def _counts(view: PublicView) -> SnapshotCounts:
    victims = [v for i in view.incidents for v in i.victims]
    treating = sum(v.status is VictimStatus.IN_TREATMENT for v in victims)
    return SnapshotCounts(
        incidents_reported=len(view.incidents),
        incidents_open=sum(i.status is IncidentStatus.OPEN for i in view.incidents),
        waiting_on_scene=sum(v.status is VictimStatus.WAITING for v in victims),
        in_transport=sum(v.status is VictimStatus.TRANSPORTING for v in victims),
        queued=sum(h.queue_length for h in view.hospitals), waiting_for_staff=0, in_treatment=treating,
        treated=0, care_started_cum=treating, expectant=0,
    )


def _msg(mid: str, t: float, sender: str, recipient: str, mtype: MessageType, content: ProposedAction, conv: str,
         *, incident: str = "INC-01", reply: str | None = None, rnd: int = 1, why: str = "test") -> MessageRecord:
    return MessageRecord(t_min=t, message=Message(
        message_id=mid, t_min=t, sender=sender, recipient=recipient, type=mtype, incident_id=incident,
        proposed_action=content, rationale=why, in_reply_to=reply, conversation_id=conv, round=rnd))


def fixture_messages() -> list[MessageRecord]:
    """Every message of the Full fixture run."""
    assess = AssessAction(urgency=80, suspected_red=True, suspected_cardiac_arrest=False, est_patients=2,
                          needed_capabilities=(Capability.TRAUMA,), als_units_needed=1, deadline_in_min=30.0)
    admit1 = AdmitAction(hospital_id="H1", unit_id="AMB-01", victim_ids=("V1",), eta_min=3.0)
    admit3 = AdmitAction(hospital_id="H1", unit_id="AMB-02", victim_ids=("V3",), eta_min=4.0)
    admit3b = AdmitAction(hospital_id="H2", unit_id="AMB-02", victim_ids=("V3",), eta_min=5.0)
    return [
        _msg("M-0001", 0, ASSESSMENT_AGENT, DISPATCH_AGENT, MessageType.PRIORITY_REPORT, assess, "NEG-0001", rnd=0),
        _msg("M-0002", 0, DISPATCH_AGENT, BROADCAST, MessageType.INFORM, DispatchAction(unit_ids=("AMB-01",)),
             "NEG-0002", rnd=0),
        _msg("M-0003", 6, DISPATCH_AGENT, H1, MessageType.PROPOSE, admit1, "NEG-0003"),
        _msg("M-0004", 6, H1, DISPATCH_AGENT, MessageType.COUNTER, ConditionalAdmitAction(
            hospital_id="H1", unit_id="AMB-01", victim_ids=("V1",), condition=CounterCondition.RESERVE_RELEASE,
            skill=None), "NEG-0003", reply="M-0003", why="only the reserve bed is free"),
        _msg("M-0005", 6, H1, ALLOCATION_AGENT, MessageType.ESCALATE, ResolveAction(
            case_id="CASE-0001", case_kind=CaseKind.RESERVE_RELEASE, option_ids=("OPT-1", "OPT-2")),
             "NEG-0003", reply="M-0004"),
        _msg("M-0006", 6, ALLOCATION_AGENT, H1, MessageType.COMMIT, CommitAction(
            case_id="CASE-0001", option_id="OPT-1", resolution=Resolution.COUNTER_GRANTED,
            actions=(AssignTransport(unit_id="AMB-01", hospital_id="H1", reserve_beds=True),)),
             "NEG-0003", reply="M-0005"),
        _msg("M-0007", 9, DISPATCH_AGENT, H1, MessageType.PROPOSE, admit3, "NEG-0004", incident="INC-02"),
        _msg("M-0008", 9, H1, DISPATCH_AGENT, MessageType.REJECT, DeclineAction(reason="NO_BEDS"), "NEG-0004",
             incident="INC-02", reply="M-0007"),
        _msg("M-0009", 9, DISPATCH_AGENT, H2, MessageType.PROPOSE, admit3b, "NEG-0004", incident="INC-02", rnd=2),
        _msg("M-0010", 9, H2, DISPATCH_AGENT, MessageType.ACCEPT, admit3b, "NEG-0004", incident="INC-02",
             reply="M-0009", rnd=2),
        _msg("M-0011", 12, H2, ALLOCATION_AGENT, MessageType.REQUEST, StaffRequestAction(
            hospital_id="H2", role=StaffRole.DOCTOR, skill="emergency"), "NEG-0005", incident="INC-02"),
        _msg("M-0012", 14, ALLOCATION_AGENT, H2, MessageType.COMMIT, CommitAction(
            case_id="CASE-0002", option_id="GRANT-DR-F-1", resolution=Resolution.COUNTER_GRANTED,
            actions=(LendStaff(staff_id="DR-F-1", hospital_id="H2"),)), "NEG-0005", incident="INC-02",
             reply="M-0011"),
    ]


def _call(t: float, agent: str, role: AgentRole, output: dict[str, Any], *, fallback: bool = False,
          tools: tuple[ToolCallTrace, ...] = ()) -> AgentCallRecord:
    notes = (FallbackNote(scope="call", key=None, rule="model_error", detail="ModelBehaviorError"),) if fallback else ()
    return AgentCallRecord(
        t_min=t, agent=agent, role=role, triggers=(TriggerKind.NEW_INCIDENT,), backend="offline", input={},
        output=output, source="policy_fallback" if fallback else "model", tool_calls=tools, fallbacks=notes,
        usage=TokenUsage(requests=2, input_tokens=0, output_tokens=0), latency_ms=None)


def fixture_calls() -> list[AgentCallRecord]:
    """Every agent call of the Full fixture run."""
    hosp, ra = AgentRole.HOSPITAL, AgentRole.ALLOCATION
    tool = ToolCallTrace(name="nearest_units", arguments='{"incident_id": "INC-01"}', output='["AMB-01"]')
    return [
        _call(0, ASSESSMENT_AGENT, AgentRole.ASSESSMENT, {"reports": [
            {"incident_id": "INC-01", "urgency": 80, "suspected_red": True, "rationale": "bleeding reported"}]}),
        _call(0, DISPATCH_AGENT, AgentRole.DISPATCH, {"orders": [
            {"unit_id": "AMB-01", "incident_id": "INC-01", "rationale": "nearest ALS"}],
            "diversions": [], "contentions": [], "transports": []}, tools=(tool,)),
        _call(6, H1, hosp, {"responses": [{"proposal_id": "M-0003", "response": "COUNTER",
                                           "reason": "RESERVE_HELD", "rationale": "only the reserve bed is free"}]}),
        _call(6, ALLOCATION_AGENT, ra, {"decisions": [{"case_id": "CASE-0001", "chosen_option_id": "OPT-1",
                                                       "rationale": "no RED expected"}]}, fallback=True),
        _call(9, H1, hosp, {"responses": [{"proposal_id": "M-0007", "response": "REJECT", "reason": "NO_BEDS",
                                           "rationale": "full"}]}),
        _call(9, H2, hosp, {"responses": [{"proposal_id": "M-0009", "response": "ACCEPT", "reason": "OK",
                                           "rationale": "beds free"}]}),
        _call(14, ALLOCATION_AGENT, ra, {"decisions": [{"case_id": "CASE-0002", "chosen_option_id": "GRANT-DR-F-1",
                                                        "rationale": "queue forming at H2"}]}),
    ]


def fixture_actions(actor_override: str | None = None) -> list[ActionRecord]:
    """Engine actions (all by ``actor_override`` when given, as in B0)."""
    def act(t: float, actor: str, action: Any, ok: bool = True, reason: RejectReason | None = None) -> ActionRecord:
        return ActionRecord(t_min=t, actor=actor_override or actor, action=action, ok=ok, reason=reason,
                            detail="" if ok else "busy")
    return [
        act(0, DISPATCH_AGENT, DispatchUnit(unit_id="AMB-01", incident_id="INC-01")),
        act(5, DISPATCH_AGENT, DispatchUnit(unit_id="AMB-01", incident_id="INC-02"), False,
            RejectReason.UNIT_NOT_AVAILABLE),
        act(5, DISPATCH_AGENT, DispatchUnit(unit_id="AMB-02", incident_id="INC-02")),
        act(5, DISPATCH_AGENT, DispatchUnit(unit_id="FIRE-01", incident_id="INC-02")),
        act(6, ALLOCATION_AGENT, AssignTransport(unit_id="AMB-01", hospital_id="H1", reserve_beds=True)),
        act(9, H2, AssignTransport(unit_id="AMB-02", hospital_id="H2", reserve_beds=True)),
        act(14, ALLOCATION_AGENT, LendStaff(staff_id="DR-F-1", hospital_id="H2")),
    ]


def fixture_events() -> list[SimEventRecord]:
    """A few simulation events."""
    e = SimEventType
    return [
        SimEventRecord(t_min=0, event=e.INCIDENT_REPORTED, incident_id="INC-01"),
        SimEventRecord(t_min=3, event=e.SCENE_TRIAGED, incident_id="INC-01", unit_id="AMB-01"),
        SimEventRecord(t_min=5, event=e.INCIDENT_REPORTED, incident_id="INC-02"),
        SimEventRecord(t_min=6, event=e.PATIENTS_LOADED, unit_id="AMB-01", victim_ids=("V1",)),
        SimEventRecord(t_min=9, event=e.TREATMENT_STARTED, hospital_id="H1", victim_ids=("V1",)),
        SimEventRecord(t_min=12, event=e.INCIDENT_CLEARED, incident_id="INC-02"),
        SimEventRecord(t_min=14, event=e.STAFF_LENT, staff_id="DR-F-1", hospital_id="H2"),
    ]


def _victim_outcome(vid: str, iid: str, triage: Triage, deadline: float, care: float | None) -> VictimOutcome:
    return VictimOutcome(
        victim_id=vid, incident_id=iid, t_report_min=0.0, condition="test", required_capability=Capability.GENERAL,
        initial_triage=triage, final_triage=triage, deteriorated_at_min=None, deadline_min=deadline,
        als_on_scene_min=None, loaded_min=None, hospital_id=None, arrived_hospital_min=None, bedded_min=care,
        treatment_start_min=care, care_start_min=care, care_mode="hospital" if care is not None else None)


def fixture_end(served: int = 2) -> RunEndRecord:
    """``run_end`` with outcomes consistent with the snapshots."""
    outcomes = RunOutcomes(
        horizon_min=HORIZON,
        victims=(_victim_outcome("V1", "INC-01", Triage.RED, 8.0, 9.0),
                 _victim_outcome("V2", "INC-01", Triage.GREEN, 480.0, None),
                 _victim_outcome("V3", "INC-02", Triage.YELLOW, 170.0, 12.0 if served > 1 else None)),
        incidents=(IncidentOutcome(incident_id="INC-01", t_report_min=0.0, mci_cluster=False,
                                   first_unit_on_scene_min=3.0, first_ambulance_on_scene_min=3.0, cleared_min=None),
                   IncidentOutcome(incident_id="INC-02", t_report_min=5.0, mci_cluster=False,
                                   first_unit_on_scene_min=8.0, first_ambulance_on_scene_min=8.0, cleared_min=12.0)),
        ambulance_busy_min=(("AMB-01", 12.0), ("AMB-02", 10.0)), responder_busy_min=(("FIRE-01", 9.0),),
        ambulances_in_service=2)
    metrics = f.zero_metrics(
        patients_total=3, victims_served=served, served_rate=served / 3, red_total=1, red_in_window=0,
        red_in_window_rate=0.0, unserved_critical=0, response_time_mean_min=3.0, conflicts=2,
        messages_total=12, ambulance_utilisation=0.55, er_utilisation=0.3, agent_calls=7, fallbacks=1)
    return RunEndRecord(t_min=HORIZON, outcomes=outcomes, metrics=metrics)


def fixture_records(condition: Condition = Condition.FULL) -> tuple[LogRecord, ...]:
    """A complete, ``seq``-numbered log of the fixture world under ``condition``.

    B0 has no messages or agent calls; its actions are by ``b0_greedy``.
    """
    scenario = fixture_scenario()
    b0 = condition is Condition.B0
    start = RunStartRecord(
        t_min=0.0, run_id=f"tiny-{condition}-offline-s0-r0", scenario=scenario.name, seed=scenario.seed,
        condition=condition, backend="offline", repeat=0, sim_params={"handover_min": 5.0},
        negotiation=None if b0 else {"allow_counter": True, "max_rounds": 3}, scenario_data=scenario)
    body: list[LogRecord] = [*fixture_events()]
    for minute in range(int(HORIZON)):
        view = fixture_view(float(minute))
        body.append(SnapshotRecord(t_min=float(minute), view=view, counts=_counts(view)))
    if not b0:
        body += [*fixture_calls(), *fixture_messages()]
    body += fixture_actions("b0_greedy" if b0 else None)
    body.sort(key=lambda r: r.t_min)  # stable: keeps insertion order within a minute
    sink = MemoryEventSink()
    for record in (start, *body, fixture_end(served=1 if b0 else 2)):
        sink.emit(record)
    return sink.records


def fixture_run(condition: Condition = Condition.FULL) -> replay.RunLog:
    """The fixture log as a ``RunLog``."""
    return replay.run_log_from_records(fixture_records(condition))


def summary_frame() -> pd.DataFrame:
    """A small cross-seed summary in the ``summary.csv`` layout."""
    rows = []
    for scenario in ("tiny", "surge"):
        for k, condition in enumerate(("b0", "b1", "full")):
            rows.append({"scenario": scenario, "condition": condition, "backend": "offline",
                         "metric": "served_rate", "n": 3, "mean": 0.5 + 0.1 * k, "std": 0.05,
                         "min": 0.4, "max": 0.8})
            rows.append({"scenario": scenario, "condition": condition, "backend": "offline",
                         "metric": "response_time_mean_min", "n": 3, "mean": 9.0 - k, "std": 1.0,
                         "min": 7.0, "max": 10.0})
    return pd.DataFrame(rows)


def write_fixture_results(root: Path, *, conditions: Sequence[Condition] = (
        Condition.B0, Condition.B1, Condition.FULL), with_summary: bool = True) -> Path:
    """Writes fixture runs (and ``summary.csv``) in the results layout under ``root``."""
    for condition in conditions:
        path = root / "runs" / "tiny" / condition.value / "offline" / "seed-0_r0" / "events.jsonl"
        write_jsonl(fixture_records(condition), path)
    if with_summary:
        summary_frame().to_csv(root / "summary.csv", index=False)
    return root


@pytest.fixture(scope="module")
def run() -> replay.RunLog:
    return fixture_run()


# --- Loading --------------------------------------------------------------------


def test_discover_runs_parses_layout_and_ignores_strays(tmp_path: Path) -> None:
    write_fixture_results(tmp_path)
    (tmp_path / "runs" / "tiny" / "full" / "offline" / "not-a-run").mkdir()
    (tmp_path / "runs" / "tiny" / "full" / "offline" / "not-a-run" / "events.jsonl").write_text("")
    refs = replay.discover_runs(tmp_path)
    assert [r.condition for r in refs] == ["b0", "b1", "full"]
    assert {(r.scenario, r.backend, r.seed, r.repeat) for r in refs} == {("tiny", "offline", 0, 0)}
    assert all(r.path.name == "events.jsonl" for r in refs)


def test_discover_runs_on_missing_folder_is_empty(tmp_path: Path) -> None:
    assert replay.discover_runs(tmp_path / "nope") == []


def test_load_run_round_trips_the_fixture(tmp_path: Path) -> None:
    write_fixture_results(tmp_path, conditions=(Condition.FULL,), with_summary=False)
    loaded = replay.load_run(tmp_path / "runs/tiny/full/offline/seed-0_r0/events.jsonl")
    assert loaded == fixture_run()
    assert len(loaded.snapshots) == 20 and len(loaded.messages) == 12 and len(loaded.agent_calls) == 7
    assert loaded.end is not None and replay.horizon_of(loaded) == HORIZON


def test_run_without_start_is_rejected() -> None:
    with pytest.raises(replay.RunLogError):
        replay.run_log_from_records(fixture_records()[1:])


def test_all_records_restores_log_order(run: replay.RunLog) -> None:
    records = replay.all_records(run)
    assert records[0].kind == "run_start" and records[-1].kind == "run_end"
    assert [r.seq for r in records] == sorted(r.seq for r in records)
    assert len(records) == len(fixture_records())


# --- Time slicing ---------------------------------------------------------------


@pytest.mark.parametrize(("t", "expected"), [(-1.0, 0.0), (0.0, 0.0), (7.5, 7.0), (12.0, 12.0), (99.0, 19.0)])
def test_snapshot_at_returns_latest_not_after_t(run: replay.RunLog, t: float, expected: float) -> None:
    assert replay.snapshot_at(run, t).t_min == expected


def test_snapshot_at_without_snapshots_raises() -> None:
    bare = replay.run_log_from_records([fixture_records()[0]])
    with pytest.raises(replay.RunLogError):
        replay.snapshot_at(bare, 0.0)


def test_slices_until_t(run: replay.RunLog) -> None:
    assert [m.message_id for m in replay.messages_until(run, 6)][-1] == "M-0006"
    assert len(replay.messages_until(run, 5.9)) == 2
    assert len(replay.events_until(run, 5)) == 3
    assert len(replay.agent_calls_until(run, 9)) == 6
    assert len(replay.actions_until(run, 0)) == 1


# --- Threads ------------------------------------------------------------------------


def test_negotiation_threads_summarise_each_conversation(run: replay.RunLog) -> None:
    threads = {th.conversation_id: th for th in replay.negotiation_threads(run)}
    assert list(threads) == ["NEG-0003", "NEG-0004", "NEG-0005"]  # PRIORITY_REPORT / INFORM skipped
    reserve, beds, staff = threads["NEG-0003"], threads["NEG-0004"], threads["NEG-0005"]
    assert reserve.kind == "RESERVE_RELEASE"
    assert reserve.contested_resource == "critical-reserve bed at H1 for AMB-01"
    assert reserve.resolution == "COUNTER_GRANTED"
    assert [m.type for m in reserve.messages] == [MessageType.PROPOSE, MessageType.COUNTER,
                                                  MessageType.ESCALATE, MessageType.COMMIT]
    assert beds.kind == replay.BED_REJECTION and beds.resolution == "ACCEPTED_AFTER_REJECT"
    assert beds.incident_id == "INC-02" and beds.t_min == 9
    assert staff.kind == "FLOAT_STAFF" and staff.contested_resource == "float doctor (emergency) for H2"


def test_thread_is_open_until_answered(run: replay.RunLog) -> None:
    threads = replay.threads_from_messages(replay.messages_until(run, 12))
    assert threads[-1].conversation_id == "NEG-0005" and threads[-1].resolution is None
    assert replay.thread_row(threads[-1])["resolution"] == "open"


def test_thread_row_shows_sequence_and_conflict(run: replay.RunLog) -> None:
    row = replay.thread_row(replay.negotiation_threads(run)[1])
    assert row["sequence"] == "PROPOSE > REJECT > PROPOSE > ACCEPT"
    assert row["hospitals asked"] == "H1, H2" and row["rounds"] == 2 and row["conflict"] is True


@pytest.mark.parametrize(
    ("last", "expected"),
    [
        (f.message("M-2", type=MessageType.INFORM, sender=DISPATCH_AGENT), "FORCED_ADMISSION"),
        (f.message("M-2", type=MessageType.INFORM, rationale="ENGINE_REJECTED: beds raced"), "ENGINE_REJECTED"),
        (f.message("M-2", type=MessageType.REJECT, sender=ALLOCATION_AGENT,
                   proposed_action=DeclineAction(reason="no float")), "COUNTER_DECLINED"),
        (f.message("M-2", type=MessageType.ACCEPT), "ACCEPTED"),
        (f.commit_message(), "RA_COMMIT"),
        (f.message("M-2", type=MessageType.REJECT, proposed_action=DeclineAction(reason="NO_BEDS")), None),
    ],
)
def test_thread_resolution_rules(last: Message, expected: str | None) -> None:
    assert replay.thread_resolution([f.message("M-1"), last]) == expected


def test_thread_resolution_of_empty_thread_is_none() -> None:
    assert replay.thread_resolution([]) is None


# --- Coordinator conventions (every resolution travels in a CommitAction) ----------------


def _commit(resolution: Resolution, option_id: str | None = None, case_id: str | None = None) -> CommitAction:
    return CommitAction(case_id=case_id, option_id=option_id, actions=(), resolution=resolution)


def _reserve_release_declined_then_accepted() -> list[Message]:
    """PROPOSE > COUNTER > ESCALATE > REJECT (COUNTER_DECLINED) > PROPOSE > ACCEPT, as in the real logs."""
    admit = AdmitAction(hospital_id="H1", unit_id="AMB-04", victim_ids=("V1", "V2"), eta_min=4.0)
    counter = ConditionalAdmitAction(hospital_id="H1", unit_id="AMB-04", victim_ids=("V1", "V2"),
                                     condition=CounterCondition.RESERVE_RELEASE, skill=None)
    resolve = ResolveAction(case_id="CASE-0001", case_kind=CaseKind.RESERVE_RELEASE, option_ids=("APPROVE", "DECLINE"))
    return [
        f.message("M-1", proposed_action=admit),
        f.message("M-2", type=MessageType.COUNTER, sender=H1, recipient=DISPATCH_AGENT, proposed_action=counter),
        f.message("M-3", type=MessageType.ESCALATE, sender=H1, recipient=ALLOCATION_AGENT, proposed_action=resolve),
        f.message("M-4", type=MessageType.REJECT, sender=ALLOCATION_AGENT, recipient=H1,
                  proposed_action=_commit(Resolution.COUNTER_DECLINED, "DECLINE", "CASE-0001")),
        f.message("M-5", recipient=H2, proposed_action=admit.model_copy(update={"hospital_id": "H2"}), round=2),
        f.message("M-6", type=MessageType.ACCEPT, sender=H2, recipient=DISPATCH_AGENT, round=2,
                  proposed_action=_commit(Resolution.ACCEPTED_AFTER_REJECT)),
    ]


def test_thread_resolution_follows_commit_actions_through_a_thread() -> None:
    thread = _reserve_release_declined_then_accepted()
    states = [replay.thread_resolution(thread[:k]) for k in range(1, len(thread) + 1)]
    assert states == [None, None, None, "COUNTER_DECLINED", None, "ACCEPTED_AFTER_REJECT"]
    assert replay.resolution_path(thread) == ("COUNTER_DECLINED", "ACCEPTED_AFTER_REJECT")


def test_commit_threads_ignore_legacy_heuristics() -> None:
    commit = f.message("M-1", type=MessageType.INFORM, proposed_action=_commit(Resolution.ENGINE_REJECTED))
    reproposal = f.message("M-2", rationale="previous hospital ACCEPTED but the engine refused")
    assert replay.thread_resolution([commit]) == "ENGINE_REJECTED"
    assert replay.thread_resolution([commit, reproposal]) is None  # no rationale-token guess


def test_hospital_reject_with_decline_keeps_a_commit_thread_open() -> None:
    decline = f.message("M-3", type=MessageType.REJECT, sender=H1, proposed_action=DeclineAction(reason="NO_DOCTOR"))
    thread = [*_reserve_release_declined_then_accepted()[:4], decline]
    assert replay.thread_resolution(thread) is None


def test_granted_counter_commit_sent_twice_counts_once() -> None:
    granted = _commit(Resolution.COUNTER_GRANTED, "GRANT:DR-FLOAT-1", "CASE-0002")
    thread = [f.message("M-1"), f.message("M-2", type=MessageType.COMMIT, proposed_action=granted),
              f.message("M-3", type=MessageType.COMMIT, proposed_action=granted)]
    assert replay.resolution_path(thread) == ("COUNTER_GRANTED",)
    assert replay.thread_row(replay.threads_from_messages(thread)[0])["resolution path"] == ""


@pytest.mark.parametrize(
    ("option_id", "unit"),
    [("ASSIGN:AMB-01->INC-04", "AMB-01"), ("DIVERT:AMB-03->INC-09", "AMB-03"), ("KEEP", None),
     ("GRANT:DR-FLOAT-1", None), ("HOSPITAL:H2", None), ("APPROVE", None), ("ASSIGN:broken", None)],
)
def test_option_unit(option_id: str, unit: str | None) -> None:
    assert replay.option_unit(option_id) == unit


def test_unit_contention_kept_assignment_names_the_contested_units() -> None:
    resolve = ResolveAction(case_id="CASE-0007", case_kind=CaseKind.UNIT_CONTENTION,
                            option_ids=("ASSIGN:AMB-03->INC-08", "ASSIGN:AMB-01->INC-08", "KEEP"))
    thread = replay.threads_from_messages([
        f.message("M-1", type=MessageType.ESCALATE, recipient=ALLOCATION_AGENT, proposed_action=resolve,
                  conversation_id="NEG-0021", round=0),
        f.message("M-2", type=MessageType.COMMIT, sender=ALLOCATION_AGENT, recipient=DISPATCH_AGENT,
                  proposed_action=_commit(Resolution.KEPT_ASSIGNMENT, "KEEP", "CASE-0007"),
                  conversation_id="NEG-0021", round=0),
    ])[0]
    assert thread.kind == "UNIT_CONTENTION" and thread.resolution == "KEPT_ASSIGNMENT"
    assert thread.contested_resource == "ambulance AMB-01 or AMB-03"
    assert replay.thread_label(thread) == "NEG-0021 · t=0 · UNIT_CONTENTION · KEPT_ASSIGNMENT"


def test_contested_bed_pluralises_patients() -> None:
    thread = replay.threads_from_messages(_reserve_release_declined_then_accepted()[4:5])[0]
    assert thread.contested_resource == "ER bed for AMB-04 (2 patients)"
    assert replay.plural(1, "patient") == "1 patient"


def test_foreground_events_hide_t0_baseline_releases(run: replay.RunLog) -> None:
    released = run.events[0].model_copy(update={"event": SimEventType.T0_BED_RELEASED})
    assert replay.foreground_events([released, *run.events]) == list(run.events)


def test_scenario_sort_key_orders_by_difficulty() -> None:
    assert sorted(["tiny", "constrained", "surge", "normal"], key=replay.scenario_sort_key) == [
        "normal", "surge", "constrained", "tiny"]


# --- Agents ----------------------------------------------------------------------------


def test_agent_roster_lists_roles_and_hospitals(run: replay.RunLog) -> None:
    assert [name for name, _ in replay.agent_roster(run)] == [
        ASSESSMENT_AGENT, DISPATCH_AGENT, H1, H2, ALLOCATION_AGENT]


def test_agent_status_at_reports_calls_fallbacks_and_pending(run: replay.RunLog) -> None:
    status = {s.agent: s for s in replay.agent_status_at(run, 13)}
    ra = status[ALLOCATION_AGENT]
    assert ra.calls == 1 and ra.fallbacks == 1 and ra.last_call_t == 6
    assert ra.pending == ("NEG-0005: request float doctor (emergency) for H2",)
    assert status[DISPATCH_AGENT].pending == ("INC-01: 1 waiting, 0 seats en route",)
    assert status[DISPATCH_AGENT].last_decision.startswith("AMB-01->INC-01")
    assert status[H2].pending == ("1 queued for a bed",)
    assert status[H1].last_decision == "M-0007 REJECT (NO_BEDS)"
    assert status[ASSESSMENT_AGENT].last_triggers == ("new_incident",)


def test_agent_status_pending_items_over_time(run: replay.RunLog) -> None:
    at6 = {s.agent: s for s in replay.agent_status_at(run, 6)}
    assert at6[ASSESSMENT_AGENT].pending == ("INC-02: awaiting triage",)
    assert at6[H1].pending == ("AMB-01 inbound",)
    assert {s.agent: s for s in replay.agent_status_at(run, 14)}[ALLOCATION_AGENT].pending == ()


def test_agent_status_before_any_call() -> None:
    first = replay.agent_status_at(fixture_run(), -1)
    assert all(s.calls == 0 and s.last_decision == "no decision yet" for s in first)
    assert first[1].pending == ()  # AMB-01 is already en route at t=0


def test_b0_shows_the_controller_instead_of_agents() -> None:
    statuses = replay.agent_status_at(fixture_run(Condition.B0), 5)
    assert [(s.agent, s.role) for s in statuses] == [("b0_greedy", replay.CONTROLLER_ROLE)]
    assert statuses[0].last_decision == "dispatch FIRE-01 -> INC-02" and statuses[0].calls == 0


def test_last_call_of_and_rationales(run: replay.RunLog) -> None:
    call = replay.last_call_of(run, ALLOCATION_AGENT, 10)
    assert call is not None and replay.used_fallback(call)
    assert replay.call_rationales(call) == ["no RED expected"]
    assert replay.last_call_of(run, ALLOCATION_AGENT, 1) is None


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ({"reports": [{"incident_id": "INC-01", "urgency": 80, "suspected_red": True}]},
         "INC-01 urgency 80 (RED suspected)"),
        ({"orders": [], "diversions": [], "contentions": [], "transports": [
            {"unit_id": "AMB-01", "hospital_ranking": ["H1", "H2"]}]}, "AMB-01: H1 > H2 | 0 diversions, 0 contentions"),
        ({"decisions": [{"case_id": "C-1", "chosen_option_id": "OPT-2"}]}, "C-1 -> OPT-2"),
        ({"responses": []}, "no items"),
        ({"other": 1}, '{"other": 1}'),
    ],
)
def test_summarise_output(output: dict[str, Any], expected: str) -> None:
    assert replay.summarise_output(output) == expected


def test_summarise_output_truncates_long_text() -> None:
    long = {"reports": [{"incident_id": f"INC-{i:02d}", "urgency": 50} for i in range(40)]}
    assert len(replay.summarise_output(long)) == replay.MAX_SUMMARY_CHARS


# --- Step-through -------------------------------------------------------------------------


def test_decision_times(run: replay.RunLog) -> None:
    assert replay.decision_times(run) == [0, 5, 6, 9, 12, 14]


def test_decision_trace_is_in_protocol_order(run: replay.RunLog) -> None:
    rows = replay.decision_trace(run, 6)
    assert [r.seq for r in rows] == sorted(r.seq for r in rows)
    assert {r.kind for r in rows} == {"message", "agent call", "action"}
    assert not any(r.open_request for r in rows)
    fallback = next(r for r in rows if r.kind == "agent call" and r.actor == ALLOCATION_AGENT)
    assert fallback.summary.startswith("[policy_fallback]") and "call:model_error" in fallback.detail


def test_decision_trace_flags_unanswered_requests(run: replay.RunLog) -> None:
    rows = replay.decision_trace(run, 12)
    assert [(r.kind, r.open_request) for r in rows] == [("message", True)]
    rejected = [r for r in replay.decision_trace(run, 5) if r.kind == "action" and "rejected" in r.detail]
    assert rejected and "unit_not_available" in rejected[0].detail


def test_unanswered_requests() -> None:
    proposal = f.message("M-1")
    assert replay.unanswered_requests([proposal]) == [proposal]
    assert replay.unanswered_requests([proposal, f.message("M-2", type=MessageType.ACCEPT, in_reply_to="M-1")]) == []


# --- Incidents and resources ----------------------------------------------------------------


def test_incident_severity_and_rows(run: replay.RunLog) -> None:
    early, late = replay.snapshot_at(run, 1).view, replay.snapshot_at(run, 10).view
    assert replay.incident_severity(early.incidents[0]) == replay.UNTRIAGED
    assert [replay.incident_severity(i) for i in late.incidents] == ["RED", "YELLOW"]
    rows = replay.incident_rows(late)
    assert rows[0]["victims"] == "1 RED, 1 GREEN" and rows[0]["waiting on scene"] == 1
    assert rows[1]["units"] == "AMB-02, FIRE-01" and rows[1]["location"] == geo.nearest_intersection((12, 12))
    assert replay.triage_counts(early.incidents[0]) == "untriaged"


def test_event_and_message_rows(run: replay.RunLog) -> None:
    assert replay.event_rows(run.events[:1])[0]["event"] == "incident_reported"
    row = replay.message_row(run.messages[2].message)
    assert row["content"] == "admit V1 from AMB-01 at H1 (ETA 3.0 min)" and row["type"] == "PROPOSE"


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (DispatchAction(unit_ids=("AMB-01", "AMB-02")), "dispatch AMB-01, AMB-02"),
        (StaffRequestAction(hospital_id="H2", role=StaffRole.NURSE, skill="burns"),
         "request float nurse (burns) for H2"),
        (ResolveAction(case_id="C-1", case_kind=CaseKind.UNIT_CONTENTION, option_ids=("A", "B")),
         "UNIT_CONTENTION C-1: options A, B"),
        (CommitAction(case_id=None, option_id=None, actions=(), resolution=Resolution.KEPT_ASSIGNMENT),
         "KEPT_ASSIGNMENT: no change"),
        (ConditionalAdmitAction(hospital_id="H1", unit_id="AMB-01", victim_ids=("V1",),
                                condition=CounterCondition.FLOAT_DOCTOR, skill="trauma"),
         "admit AMB-01 at H1 if FLOAT_DOCTOR (trauma)"),
    ],
)
def test_describe_content(content: ProposedAction, expected: str) -> None:
    assert replay.describe_content(content) == expected


def test_resource_summary(run: replay.RunLog) -> None:
    s = replay.resource_summary(replay.snapshot_at(run, 10).view)
    assert (s.ambulances_available, s.ambulances_busy, s.ambulances_in_service) == (0, 2, 2)
    assert (s.responders_available, s.responders_busy, s.responders_total) == (0, 1, 1)
    assert (s.er_capacity, s.beds_occupied, s.beds_reserved, s.beds_free) == (7, 2, 1, 4)
    assert (s.doctor_load, s.doctor_capacity, s.nurse_load, s.nurse_capacity) == (1, 8, 1, 3)
    assert (s.float_available, s.float_lent) == (1, 0)
    lent = replay.resource_summary(replay.snapshot_at(run, 15).view)
    assert (lent.float_available, lent.float_lent, lent.doctors_on_shift) == (0, 1, 3)


def test_utilisation_series(run: replay.RunLog) -> None:
    rows = replay.utilisation_series(run)
    assert len(rows) == 20 and set(rows[0]) == {"t", *replay.UTILISATION_SERIES}
    assert rows[10]["Ambulances"] == 100.0 and rows[0]["Ambulances"] == 50.0
    assert replay.ratio(1, 0) == 0.0


def test_unit_intervals_run_length_encode_statuses(run: replay.RunLog) -> None:
    amb01 = [(i.group, i.start, i.end) for i in replay.unit_intervals(run) if i.unit_id == "AMB-01"]
    assert amb01 == [("To scene", 0, 3), ("On scene", 3, 6), ("Transporting", 6, 9),
                     ("At hospital", 9, 12), ("Available", 12, HORIZON)]


# --- KPIs ---------------------------------------------------------------------------------


def test_kpis_at_time_uses_evaluation_metrics(run: replay.RunLog) -> None:
    assert run.end is not None
    for t in (2, 10, 13):
        expected = kpis_at(run.end.outcomes, replay.all_records(run), t).model_dump()
        assert replay.kpis_at_time(run, t) == expected


def test_kpis_at_time_without_run_end_uses_snapshots() -> None:
    truncated = replay.run_log_from_records(fixture_records()[:-1])
    kpis = replay.kpis_at_time(truncated, 13)
    assert kpis["served"] == 2 and kpis["red_missed"] is None


def test_final_metrics_rows_side_by_side() -> None:
    rows = replay.final_metrics_rows({"b0": fixture_run(Condition.B0), "full": fixture_run()})
    served = next(r for r in rows if r["metric"] == "victims_served")
    assert served == {"metric": "victims_served", "b0": 1, "full": 2}
    assert all(r["metric"] != "messages_by_type" for r in rows)


def test_condition_sort_key_orders_known_conditions_first() -> None:
    labels = ["zzz", "full", "b1", "full-no-float", "b0"]
    assert sorted(labels, key=replay.condition_sort_key) == ["b0", "b1", "full", "full-no-float", "zzz"]


# --- Summary --------------------------------------------------------------------------------


def test_load_summary_reads_csv(tmp_path: Path) -> None:
    summary_frame().to_csv(tmp_path / "summary.csv", index=False)
    frame = replay.load_summary(tmp_path)
    assert len(frame) == 12 and list(frame.columns) == list(replay.SUMMARY_COLUMNS)


def test_load_summary_degrades_gracefully(tmp_path: Path) -> None:
    assert replay.load_summary(tmp_path / "missing").empty
    pd.DataFrame({"scenario": ["x"]}).to_csv(tmp_path / "summary.csv", index=False)
    assert replay.load_summary(tmp_path).empty


def test_load_summary_falls_back_to_metrics_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "runs").mkdir()
    monkeypatch.setattr(replay, "_summary_from_metrics", lambda results_dir: summary_frame())
    assert len(replay.load_summary(tmp_path)) == 12
