"""Test doubles and view builders for the control-layer tests (WP3).

* ``FakePort``: an in-memory ``EnginePort`` over a hand-built ``PublicView``.
  It applies a simplified version of the engine rules (enough to observe
  reservations, lending and status changes), records every attempt and can
  be told to reject actions.
* ``PolicyBackend``: a ``DecisionBackend`` that calls the offline policies
  directly (no SDK), with optional per-agent overrides.
"""

from collections.abc import Callable, Mapping, Sequence

from emergency_sim.domain.actions import (
    Action,
    ActionCheck,
    AssignTransport,
    DispatchUnit,
    DivertUnit,
    LendStaff,
)
from emergency_sim.domain.agent_io import (
    AgentDecision,
    AllocationInput,
    AssessmentInput,
    DispatchInput,
    HospitalInput,
)
from emergency_sim.domain.base import AgentIOModel
from emergency_sim.domain.enums import (
    Capability,
    IncidentStatus,
    RejectReason,
    ResponderCapability,
    StaffRole,
    Triage,
    UnitClass,
    UnitStatus,
    VictimStatus,
)
from emergency_sim.domain.log_records import AgentCallRecord, MessageRecord
from emergency_sim.domain.messages import Message
from emergency_sim.domain.views import (
    HospitalView,
    IncidentView,
    PublicView,
    StaffView,
    UnitView,
    VictimView,
)
from emergency_sim.eventlog import MemoryEventSink
from emergency_sim.policies.allocation import allocate
from emergency_sim.policies.assessment import assess
from emergency_sim.policies.dispatch import dispatch
from emergency_sim.policies.hospital import hospital
from emergency_sim.policies.params import PolicyParams

# --- View builders --------------------------------------------------------------


def victim_view(
    vid: str,
    incident_id: str = "INC-01",
    triage: str = "YELLOW",
    cap: str = "general",
    *,
    status: VictimStatus = VictimStatus.LOADED,
    unit_id: str | None = None,
    condition: str = "lacerations",
    deteriorated: bool = False,
) -> VictimView:
    """Builds a victim view."""
    return VictimView(
        victim_id=vid, incident_id=incident_id, triage=Triage(triage), condition=condition,
        required_capability=Capability(cap), needs_als=triage == "RED", status=status,
        unit_id=unit_id, hospital_id=None, deteriorated=deteriorated,
    )


def incident_view(
    iid: str = "INC-01",
    *,
    t: float = 0.0,
    x: float = 5.0,
    y: float = 5.0,
    victims: Sequence[VictimView] = (),
    triaged: bool | None = None,
    status: IncidentStatus = IncidentStatus.OPEN,
    reported: int = 1,
    report: str = "Caller reports one person hurt.",
    incident_type: str = "fall",
    required: Sequence[str] = (),
    present: Sequence[str] = (),
    accessible: bool | None = None,
    assigned: Sequence[str] = (),
) -> IncidentView:
    """Builds an incident view (triaged iff victims are given, unless overridden)."""
    return IncidentView(
        incident_id=iid, incident_type=incident_type, t_report_min=t, x=x, y=y,
        intersection="Elm St & 3rd Ave", reported_victims=reported, caller_report=report,
        required_responder_capabilities=tuple(ResponderCapability(c) for c in required),
        status=status, triaged=bool(victims) if triaged is None else triaged,
        accessible=(not required) if accessible is None else accessible,
        victims=tuple(victims), assigned_unit_ids=tuple(assigned),
        responder_capabilities_present=tuple(ResponderCapability(c) for c in present),
    )


def unit_view(
    uid: str,
    *,
    unit_type: str = "ALS",
    x: float = 5.0,
    y: float = 5.0,
    status: UnitStatus = UnitStatus.AVAILABLE,
    incident_id: str | None = None,
    hospital_id: str | None = None,
    load: Sequence[str] = (),
    caps: Sequence[str] = (),
) -> UnitView:
    """Builds an ambulance (ALS/BLS) or, with ``caps``, a responder team."""
    responder = bool(caps)
    return UnitView(
        unit_id=uid, unit_class=UnitClass.RESPONDER if responder else UnitClass.AMBULANCE,
        unit_type="team" if responder else unit_type, capabilities=tuple(caps), status=status,
        x=x, y=y, incident_id=incident_id, hospital_id=hospital_id,
        destination_id=incident_id if status is UnitStatus.EN_ROUTE_SCENE else None,
        load_victim_ids=tuple(load), seats=0 if responder else 2, available_at_min=None,
    )


def staff_view(
    sid: str,
    role: str = "doctor",
    home: str = "H1",
    skills: Sequence[str] = ("emergency",),
    *,
    load: int = 0,
    max_patients: int = 4,
    on_shift: bool = True,
) -> StaffView:
    """Builds a staff member; float staff (home FLOAT) start unlent."""
    return StaffView(
        staff_id=sid, role=StaffRole(role), skills=tuple(skills), home=home,
        hospital_id=None if home == "FLOAT" else home, load=load, max_patients=max_patients,
        on_shift=on_shift,
    )


def hospital_view(
    hid: str = "H1",
    *,
    x: float = 5.0,
    y: float = 5.0,
    caps: Sequence[str] = ("general",),
    capacity: int = 6,
    free: int | None = None,
    reserve: int = 0,
    queue: int = 0,
    staff: Sequence[StaffView] | None = None,
) -> HospitalView:
    """Builds a hospital; default staff is one emergency doctor and one nurse."""
    free = capacity if free is None else free
    if staff is None:
        staff = (staff_view(f"DR-{hid}", "doctor", hid), staff_view(f"RN-{hid}", "nurse", hid))
    return HospitalView(
        hospital_id=hid, name=f"Hospital {hid}", x=x, y=y,
        capabilities=tuple(Capability(c) for c in caps), er_capacity=capacity,
        critical_reserve=reserve, beds_occupied=capacity - free, beds_reserved=0,
        beds_free_unreserved=free, queue_length=queue, waiting_for_staff=0, in_treatment=0,
        inbound_unit_ids=(), staff=tuple(staff),
    )


def public_view(
    *,
    t: float = 10.0,
    incidents: Sequence[IncidentView] = (),
    units: Sequence[UnitView] = (),
    hospitals: Sequence[HospitalView] = (),
    float_pool: Sequence[StaffView] = (),
) -> PublicView:
    """Builds a view with every collection sorted by id."""
    return PublicView(
        t_min=t, horizon_min=180.0,
        incidents=tuple(sorted(incidents, key=lambda i: i.incident_id)),
        units=tuple(sorted(units, key=lambda u: u.unit_id)),
        hospitals=tuple(sorted(hospitals, key=lambda h: h.hospital_id)),
        float_pool=tuple(sorted(float_pool, key=lambda s: s.staff_id)),
    )


# --- FakePort ---------------------------------------------------------------------

type Rejector = Callable[[Action], RejectReason | None]


class FakePort:
    """In-memory ``EnginePort`` with simplified engine rules.

    Attributes:
        attempts: Every ``apply`` as ``(actor, action, ok)``.
        reject: Optional hook forcing a rejection (checked before the rules).
    """

    def __init__(self, view: PublicView, *, reject: Rejector | None = None) -> None:
        self._view = view
        self.reject = reject
        self.attempts: list[tuple[str, Action, bool]] = []

    @property
    def now(self) -> float:
        return self._view.t_min

    def view(self) -> PublicView:
        return self._view

    @property
    def applied(self) -> list[tuple[str, Action]]:
        """Successfully applied ``(actor, action)`` pairs."""
        return [(actor, action) for actor, action, ok in self.attempts if ok]

    def applied_actions(self, kind: type | None = None) -> list[Action]:
        """Applied actions, optionally of one class."""
        return [a for _, a in self.applied if kind is None or isinstance(a, kind)]

    def check(self, action: Action) -> ActionCheck:
        if self.reject is not None and (reason := self.reject(action)) is not None:
            return ActionCheck(ok=False, reason=reason, detail="forced by test")
        match action:
            case DispatchUnit():
                return self._check_dispatch(action)
            case DivertUnit():
                return self._check_divert(action)
            case AssignTransport():
                return self._check_transport(action)
            case LendStaff():
                return self._check_lend(action)
        raise AssertionError(action)

    def apply(self, action: Action, actor: str) -> ActionCheck:
        check = self.check(action)
        self.attempts.append((actor, action, check.ok))
        if check.ok:
            self._mutate(action)
        return check

    # rules -----------------------------------------------------------------------

    def _unit(self, unit_id: str) -> UnitView | None:
        return next((u for u in self._view.units if u.unit_id == unit_id), None)

    def _incident(self, incident_id: str) -> IncidentView | None:
        return next((i for i in self._view.incidents if i.incident_id == incident_id), None)

    def _hospital(self, hospital_id: str) -> HospitalView | None:
        return next((h for h in self._view.hospitals if h.hospital_id == hospital_id), None)

    def _check_dispatch(self, a: DispatchUnit) -> ActionCheck:
        unit, incident = self._unit(a.unit_id), self._incident(a.incident_id)
        if unit is None:
            return ActionCheck(ok=False, reason=RejectReason.UNKNOWN_UNIT)
        if unit.status is not UnitStatus.AVAILABLE:
            return ActionCheck(ok=False, reason=RejectReason.UNIT_NOT_AVAILABLE)
        if incident is None or incident.status is not IncidentStatus.OPEN:
            return ActionCheck(ok=False, reason=RejectReason.INCIDENT_NOT_OPEN)
        return ActionCheck(ok=True)

    def _check_divert(self, a: DivertUnit) -> ActionCheck:
        unit = self._unit(a.unit_id)
        if unit is None or unit.status is not UnitStatus.EN_ROUTE_SCENE:
            return ActionCheck(ok=False, reason=RejectReason.UNIT_NOT_EN_ROUTE)
        if unit.incident_id == a.incident_id:
            return ActionCheck(ok=False, reason=RejectReason.SAME_TARGET)
        return ActionCheck(ok=True)

    def _check_transport(self, a: AssignTransport) -> ActionCheck:
        unit, hosp = self._unit(a.unit_id), self._hospital(a.hospital_id)
        if unit is None or unit.status not in (UnitStatus.AWAITING_DESTINATION, UnitStatus.AT_HOSPITAL_FULL):
            return ActionCheck(ok=False, reason=RejectReason.NOT_AWAITING_DESTINATION)
        if hosp is None:
            return ActionCheck(ok=False, reason=RejectReason.UNKNOWN_HOSPITAL)
        victims = {v.victim_id: v for i in self._view.incidents for v in i.victims}
        needed = {victims[v].required_capability for v in unit.load_victim_ids if v in victims}
        if not needed <= set(hosp.capabilities):
            return ActionCheck(ok=False, reason=RejectReason.HOSPITAL_NOT_CAPABLE)
        if a.reserve_beds and hosp.beds_free_unreserved < len(unit.load_victim_ids):
            return ActionCheck(ok=False, reason=RejectReason.INSUFFICIENT_BEDS)
        if unit.status is UnitStatus.AT_HOSPITAL_FULL and unit.hospital_id == a.hospital_id:
            return ActionCheck(ok=False, reason=RejectReason.SAME_HOSPITAL)
        return ActionCheck(ok=True)

    def _check_lend(self, a: LendStaff) -> ActionCheck:
        staff = next((s for s in self._view.float_pool if s.staff_id == a.staff_id), None)
        if staff is None:
            return ActionCheck(ok=False, reason=RejectReason.STAFF_NOT_FLOAT)
        if staff.hospital_id is not None:
            return ActionCheck(ok=False, reason=RejectReason.STAFF_ALREADY_LENT)
        return ActionCheck(ok=True)

    # effects ---------------------------------------------------------------------

    def _mutate(self, action: Action) -> None:
        match action:
            case DispatchUnit(unit_id=uid, incident_id=iid):
                self._set_unit(uid, status=UnitStatus.EN_ROUTE_SCENE, incident_id=iid, destination_id=iid)
            case DivertUnit(unit_id=uid, incident_id=iid):
                self._set_unit(uid, incident_id=iid, destination_id=iid)
            case AssignTransport(unit_id=uid, hospital_id=hid, reserve_beds=reserve):
                unit = self._unit(uid)
                assert unit is not None
                self._set_unit(uid, status=UnitStatus.TRANSPORTING, destination_id=hid)
                hosp = self._hospital(hid)
                assert hosp is not None
                n = len(unit.load_victim_ids) if reserve else 0
                self._set_hospital(
                    hid, beds_reserved=hosp.beds_reserved + n,
                    beds_free_unreserved=hosp.beds_free_unreserved - n,
                    inbound_unit_ids=tuple(sorted({*hosp.inbound_unit_ids, uid})),
                )
            case LendStaff(staff_id=sid, hospital_id=hid):
                staff = next(s for s in self._view.float_pool if s.staff_id == sid)
                lent = staff.model_copy(update={"hospital_id": hid})
                pool = tuple(lent if s.staff_id == sid else s for s in self._view.float_pool)
                hosp = self._hospital(hid)
                assert hosp is not None
                self._view = self._view.model_copy(update={"float_pool": pool})
                self._set_hospital(hid, staff=tuple(sorted((*hosp.staff, lent), key=lambda s: s.staff_id)))

    def _set_unit(self, unit_id: str, **update: object) -> None:
        units = tuple(u.model_copy(update=update) if u.unit_id == unit_id else u for u in self._view.units)
        self._view = self._view.model_copy(update={"units": units})

    def _set_hospital(self, hospital_id: str, **update: object) -> None:
        hospitals = tuple(
            h.model_copy(update=update) if h.hospital_id == hospital_id else h for h in self._view.hospitals
        )
        self._view = self._view.model_copy(update={"hospitals": hospitals})


# --- PolicyBackend ----------------------------------------------------------------

POLICIES: Mapping[type, Callable[..., AgentIOModel]] = {
    AssessmentInput: assess,
    DispatchInput: dispatch,
    HospitalInput: hospital,
    AllocationInput: allocate,
}


class PolicyBackend:
    """``DecisionBackend`` calling the offline policies directly (no SDK).

    Attributes:
        calls: ``(agent_name, payload)`` of every call.
        overrides: Per-agent functions returning a scripted output instead.
        fallback_agents: Agents whose decisions are marked ``policy_fallback``.
    """

    def __init__(
        self,
        params: PolicyParams | None = None,
        *,
        overrides: Mapping[str, Callable[[AgentIOModel], AgentIOModel]] | None = None,
        fallback_agents: frozenset[str] = frozenset(),
    ) -> None:
        self.params = params or PolicyParams()
        self.overrides = dict(overrides or {})
        self.fallback_agents = fallback_agents
        self.calls: list[tuple[str, AgentIOModel]] = []

    @property
    def label(self) -> str:
        return "policy-test"

    def decide(self, agent_name: str, payload: AgentIOModel) -> AgentDecision:
        self.calls.append((agent_name, payload))
        if agent_name in self.overrides:
            output = self.overrides[agent_name](payload)
        else:
            output = POLICIES[type(payload)](payload, self.params)
        fallback = agent_name in self.fallback_agents
        return AgentDecision(
            output=output, source="policy_fallback" if fallback else "model",
            error="ModelBehaviorError: scripted" if fallback else None,
            tool_calls=(), usage=None, latency_ms=None,
        )


# --- Log helpers ------------------------------------------------------------------


def messages(sink: MemoryEventSink) -> list[Message]:
    """Messages emitted to a sink, in order."""
    return [r.message for r in sink.records if isinstance(r, MessageRecord)]


def agent_calls(sink: MemoryEventSink) -> list[AgentCallRecord]:
    """Agent-call records emitted to a sink, in order."""
    return [r for r in sink.records if isinstance(r, AgentCallRecord)]
