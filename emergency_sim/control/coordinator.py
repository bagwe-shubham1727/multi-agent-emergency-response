"""Agent coordinator: the contract-net protocol of B1, Full and the ablations.

Plan section 5.4. Per decision point:

* **Phase A, assessment.** One Incident Assessment call for every open
  incident that is new, newly triaged, has a deteriorated victim or just had
  patients loaded (its waiting set changed); one
  PRIORITY_REPORT per incident (assessment -> dispatch).
* **Phase B, dispatch.** One Dispatch call when some incident has an unmet
  need and a unit is free, a diversion is possible, or a unit awaits a
  destination. Orders are applied (INFORM per incident); contention flags and
  diversion requests become UNIT_CONTENTION cases (ESCALATE -> RA -> COMMIT).
* **Phase C, transport negotiation.** Each loaded unit owns a thread and
  walks its hospital ranking for at most ``max_rounds`` rounds: PROPOSE ->
  ACCEPT / REJECT / COUNTER (-> REQUEST or ESCALATE -> RA -> COMMIT or
  REJECT). Units left without a destination get a NO_ACCEPTING_HOSPITAL case
  (escalation allowed) or a FORCED_ADMISSION (otherwise, e.g. B1).

Message conventions (relied on by metrics and the dashboard):

* Every resolution is carried by a ``CommitAction`` in ``proposed_action``:
  ACCEPT (``ACCEPTED`` / ``ACCEPTED_AFTER_REJECT``), COMMIT (``RA_COMMIT``,
  ``KEPT_ASSIGNMENT``, ``COUNTER_GRANTED``), RA's REJECT of a counter
  (``COUNTER_DECLINED``) and INFORM (``FORCED_ADMISSION``,
  ``ENGINE_REJECTED``). A hospital's REJECT carries a ``DeclineAction``.
* An outcome the engine refused is always an INFORM whose ``CommitAction``
  carries the attempted actions and ``ENGINE_REJECTED``. A COMMIT carries
  only actions the engine applied (a partly refused contention commit lists
  the applied ones and names the refused ones in its rationale).
* Conversations: one per loaded unit's transport thread (including the
  counter and no-accepting cases it spawns), one per UNIT_CONTENTION case, and
  one per incident for round-0 PRIORITY_REPORT / dispatch INFORM messages.
"""

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from emergency_sim.control import ProtocolError
from emergency_sim.control.bus import MessageBus
from emergency_sim.control.cases import (
    NO_FEASIBLE_ALTERNATIVE,
    CasePlan,
    CounterContext,
    OptionPlan,
    contention_case,
    counter_case,
    diversion_case,
    hospital_eta,
    no_accepting_case,
)
from emergency_sim.control.conditions import NegotiationConfig
from emergency_sim.control.needs import has_unmet_need, incident_needs, unit_victims
from emergency_sim.control.observations import (
    DeteriorationLog,
    allocation_input,
    assessment_input,
    awaiting_units,
    dispatch_input,
    find_hospital,
    hospital_input,
    triaged_victim,
)
from emergency_sim.control.validation import (
    Validated,
    validate_allocation,
    validate_assessment,
    validate_dispatch,
    validate_hospital,
)
from emergency_sim.domain.actions import Action, ActionCheck, AssignTransport, DispatchUnit
from emergency_sim.domain.agent_io import (
    AdmissionRequest,
    AdmissionResponse,
    AllocationOutput,
    AssessmentOutput,
    DispatchOutput,
    HospitalOutput,
    IncidentNeed,
    PriorityReport,
    UnitOrder,
)
from emergency_sim.domain.base import AgentIOModel
from emergency_sim.domain.enums import (
    AgentRole,
    AmbulanceType,
    CounterCondition,
    IncidentStatus,
    MessageType,
    Resolution,
    StaffRole,
    TriggerKind,
    UnitStatus,
)
from emergency_sim.domain.log_records import AgentCallRecord, FallbackNote
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
    hospital_agent_name,
)
from emergency_sim.domain.ports import DecisionBackend, EnginePort, EventSink
from emergency_sim.domain.views import DecisionPoint, PublicView, UnitView
from emergency_sim.policies.allocation import allocate
from emergency_sim.policies.assessment import assess
from emergency_sim.policies.dispatch import dispatch as dispatch_policy
from emergency_sim.policies.estimates import NURSE_SKILL
from emergency_sim.policies.hospital import hospital as hospital_policy
from emergency_sim.policies.params import PolicyParams

LOGGER = logging.getLogger(__name__)

ASSESSMENT_TRIGGERS = frozenset({
    TriggerKind.NEW_INCIDENT,
    TriggerKind.SCENE_TRIAGED,
    TriggerKind.DETERIORATION,
    TriggerKind.PATIENTS_LOADED,
})
"""Triggers that make an incident's priority report stale.

A report describes the incident's waiting victims, so loading patients (which
removes them from the scene) makes it stale just like triage or deterioration.
"""

MAX_ERROR_CHARS = 300


@dataclass
class _Point:
    """Per-decision-point context with a refreshable view.

    Attributes:
        t_min: Decision-point time (stamped on every message and record).
        triggers: Trigger kinds of the point (sorted, unique).
        port: Engine access.
        view: Latest public view.
        deterioration_age: Minutes since each deteriorated victim was first
            seen (``DeteriorationLog``), for the victims' RED windows.
    """

    t_min: float
    triggers: tuple[TriggerKind, ...]
    port: EnginePort
    view: PublicView
    deterioration_age: Mapping[str, float] = field(default_factory=dict)

    def refresh(self) -> PublicView:
        """Re-reads the view after actions were applied.

        Returns:
            The new view.
        """
        self.view = self.port.view()
        return self.view

    def unit(self, unit_id: str) -> UnitView:
        """Returns a unit of the current view.

        Args:
            unit_id: Unit id.

        Returns:
            The unit.

        Raises:
            ProtocolError: If unknown.
        """
        for unit in self.view.units:
            if unit.unit_id == unit_id:
                return unit
        raise ProtocolError(f"unknown unit {unit_id}")


class AgentCaller:
    """Runs agent calls: backend, semantic validation, ``AgentCallRecord``."""

    def __init__(self, backend: DecisionBackend, sink: EventSink, params: PolicyParams) -> None:
        """Creates the caller.

        Args:
            backend: Offline or LLM backend.
            sink: Destination of the call records.
            params: Policy parameters (for the fallback policies).
        """
        self._backend = backend
        self._sink = sink
        self._params = params

    def call[I: AgentIOModel, O: AgentIOModel](
        self,
        point: _Point,
        agent: str,
        role: AgentRole,
        payload: I,
        output_type: type[O],
        policy: Callable[[I, PolicyParams], O],
        validate: Callable[[O, O], Validated[O]],
    ) -> O:
        """Calls one agent and returns its validated output.

        Args:
            point: Decision-point context.
            agent: Agent name.
            role: Agent role.
            payload: Input payload.
            output_type: Expected output class.
            policy: The role's offline policy (fallback source).
            validate: Semantic validation ``(agent_output, policy_output)``.

        Returns:
            The output to act on (agent output with invalid parts replaced).
        """
        decision = self._backend.decide(agent, payload)
        fallback = policy(payload, self._params)
        notes: list[FallbackNote] = []
        if decision.source == "policy_fallback":
            detail = (decision.error or "")[:MAX_ERROR_CHARS]
            notes.append(FallbackNote(scope="call", key=None, rule="backend_fallback", detail=detail))
        if isinstance(decision.output, output_type):
            result = validate(decision.output, fallback)
            output, notes = result.output, notes + list(result.notes)
        else:
            output = fallback
            detail = f"expected {output_type.__name__}, got {type(decision.output).__name__}"
            notes.append(FallbackNote(scope="call", key=None, rule="wrong_output_type", detail=detail))
        self._sink.emit(AgentCallRecord(
            t_min=point.t_min, agent=agent, role=role, triggers=point.triggers,
            backend=self._backend.label, input=payload.model_dump(mode="json"),
            output=output.model_dump(mode="json"), source=decision.source,
            tool_calls=decision.tool_calls, fallbacks=tuple(notes), usage=decision.usage,
            latency_ms=decision.latency_ms,
        ))
        if notes:
            LOGGER.info("%s output partly replaced: %s", agent, [n.rule for n in notes])
        return output


@dataclass(frozen=True)
class Choice:
    """Resource Allocation's choice for one case.

    Attributes:
        plan: The chosen option's plan.
        rationale: Why (``"no feasible alternative"`` if auto-resolved).
    """

    plan: OptionPlan
    rationale: str


class CaseDesk:
    """Resolves cases: auto-resolves single-option cases, one RA call for the rest."""

    def __init__(self, caller: AgentCaller) -> None:
        """Creates the desk.

        Args:
            caller: Agent caller.
        """
        self._caller = caller

    def decide(self, point: _Point, plans: Sequence[CasePlan]) -> dict[str, Choice]:
        """Chooses an option for every case that has at least one.

        Args:
            point: Decision-point context.
            plans: Case plans.

        Returns:
            Choice per case id (cases without options are absent).

        Raises:
            ProtocolError: If Resource Allocation leaves a case undecided.
        """
        choices: dict[str, Choice] = {}
        contested = []
        for plan in sorted(plans, key=lambda p: p.case.case_id):
            if plan.needs_arbitration:
                contested.append(plan)
            elif plan.plans:
                choices[plan.case.case_id] = Choice(plan.plans[0], NO_FEASIBLE_ALTERNATIVE)
        if not contested:
            return choices
        payload = allocation_input(point.view, [p.case for p in contested])
        output = self._caller.call(
            point, ALLOCATION_AGENT, AgentRole.ALLOCATION, payload, AllocationOutput, allocate,
            lambda out, pol: validate_allocation(out, payload, pol),
        )
        decided = {d.case_id: d for d in output.decisions}
        for plan in contested:
            decision = decided.get(plan.case.case_id)
            if decision is None:
                raise ProtocolError(f"{plan.case.case_id} left undecided")
            choices[plan.case.case_id] = Choice(plan.plan(decision.chosen_option_id), decision.rationale)
        return choices


def apply_all(port: EnginePort, actions: Sequence[Action], actor: str) -> ActionCheck | None:
    """Checks every action first, then applies them in order.

    Args:
        port: Engine access.
        actions: Actions to apply together.
        actor: Agent name.

    Returns:
        None if all were applied, else the first failing check (nothing is
        applied when a pre-check fails).
    """
    for action in actions:
        check = port.check(action)
        if not check.ok:
            return check
    for action in actions:
        check = port.apply(action, actor)
        if not check.ok:
            return check
    return None


def _rejection_text(check: ActionCheck) -> str:
    """Formats an engine rejection for a rationale.

    Args:
        check: The failed check.

    Returns:
        Short text.
    """
    return f"engine rejected: {check.reason} {check.detail}".strip()


@dataclass(frozen=True)
class _Services:
    """Collaborators shared by the phases of one coordinator."""

    bus: MessageBus
    caller: AgentCaller
    desk: CaseDesk
    config: NegotiationConfig
    params: PolicyParams


class AgentCoordinator:
    """B1/Full controller; implements ``domain.ports.Controller``."""

    def __init__(
        self,
        backend: DecisionBackend,
        config: NegotiationConfig,
        sink: EventSink,
        params: PolicyParams,
    ) -> None:
        """Creates the coordinator.

        Args:
            backend: Agent backend (offline policy models or LLM).
            config: Negotiation config of the condition.
            sink: Destination of message and agent-call records.
            params: Policy parameters.

        Raises:
            ProtocolError: If ``config.max_rounds`` is not positive.
        """
        if config.max_rounds < 1:
            raise ProtocolError(f"max_rounds must be >= 1, got {config.max_rounds}")
        caller = AgentCaller(backend, sink, params)
        self._services = _Services(MessageBus(sink), caller, CaseDesk(caller), config, params)
        self._board: dict[str, PriorityReport] = {}
        self._assessed_at: dict[str, float] = {}
        self._deteriorations = DeteriorationLog()

    @property
    def messages(self) -> tuple[Message, ...]:
        """All messages sent so far (also emitted to the sink)."""
        return self._services.bus.messages

    def handle(self, point: DecisionPoint, port: EnginePort) -> None:
        """Runs phases A, B and C for one decision point.

        Args:
            point: The decision point.
            port: Engine access.
        """
        triggers = tuple(sorted({t.kind for t in point.triggers}))
        view = port.view()
        ctx = _Point(point.t_min, triggers, port, view, self._deteriorations.observe(view))
        self._assess(ctx, point)
        rankings = _DispatchPhase(self._services, ctx, self._aged_board(ctx.t_min)).run()
        if rankings is not None:
            _TransportNegotiation(self._services, ctx, rankings).run()

    def _aged_board(self, t_min: float) -> dict[str, PriorityReport]:
        """Returns the board with each deadline aged to ``t_min``.

        Args:
            t_min: Current time.

        Returns:
            Reports by incident id.
        """
        return {
            iid: report.model_copy(update={
                "deadline_in_min": round(report.deadline_in_min - (t_min - self._assessed_at[iid]), 1)
            })
            for iid, report in self._board.items()
        }

    def _incidents_to_assess(self, ctx: _Point, point: DecisionPoint) -> list[str]:
        """Selects the open incidents whose assessment is missing or stale.

        Args:
            ctx: Decision-point context.
            point: The decision point.

        Returns:
            Sorted incident ids.
        """
        open_ids = {i.incident_id for i in ctx.view.incidents if i.status is IncidentStatus.OPEN}
        victim_incident = {v.victim_id: v.incident_id for i in ctx.view.incidents for v in i.victims}
        stale = {iid for iid in open_ids if iid not in self._board}
        for trigger in point.triggers:
            if trigger.kind not in ASSESSMENT_TRIGGERS:
                continue
            if trigger.incident_id is not None:
                stale.add(trigger.incident_id)
            stale.update(victim_incident[v] for v in trigger.victim_ids if v in victim_incident)
        return sorted(stale & open_ids)

    def _assess(self, ctx: _Point, point: DecisionPoint) -> None:
        """Phase A: assesses incidents and broadcasts PRIORITY_REPORTs.

        Args:
            ctx: Decision-point context.
            point: The decision point.
        """
        ids = self._incidents_to_assess(ctx, point)
        if not ids:
            return
        svc = self._services
        payload = assessment_input(ctx.view, ids, ctx.deterioration_age)
        output = svc.caller.call(
            ctx, ASSESSMENT_AGENT, AgentRole.ASSESSMENT, payload, AssessmentOutput, assess,
            lambda out, pol: validate_assessment(out, payload, pol),
        )
        for report in output.reports:
            self._board[report.incident_id] = report
            self._assessed_at[report.incident_id] = ctx.t_min
            svc.bus.send(
                t_min=ctx.t_min, sender=ASSESSMENT_AGENT, recipient=DISPATCH_AGENT,
                type=MessageType.PRIORITY_REPORT, incident_id=report.incident_id,
                action=AssessAction(
                    urgency=report.urgency, suspected_red=report.suspected_red,
                    suspected_cardiac_arrest=report.suspected_cardiac_arrest,
                    est_patients=report.est_patients,
                    needed_capabilities=tuple(report.needed_capabilities),
                    als_units_needed=report.als_units_needed, deadline_in_min=report.deadline_in_min,
                ),
                rationale=report.rationale,
                conversation_id=svc.bus.incident_conversation(report.incident_id),
            )


class _DispatchPhase:
    """Phase B: one Dispatch call, orders, and UNIT_CONTENTION cases."""

    def __init__(self, svc: _Services, ctx: _Point, board: Mapping[str, PriorityReport]) -> None:
        """Prepares the phase.

        Args:
            svc: Shared collaborators.
            ctx: Decision-point context.
            board: Priority reports aged to now.
        """
        self._svc = svc
        self._ctx = ctx
        self._needs = incident_needs(ctx.view, board, svc.params)

    def run(self) -> dict[str, tuple[str, ...]] | None:
        """Runs the phase.

        Returns:
            Hospital ranking per loaded unit, or None if Dispatch was not called.
        """
        if not self._needed():
            return None
        svc, ctx = self._svc, self._ctx
        payload = dispatch_input(ctx.view, self._needs, svc.config, ctx.deterioration_age)
        output = svc.caller.call(
            ctx, DISPATCH_AGENT, AgentRole.DISPATCH, payload, DispatchOutput, dispatch_policy,
            lambda out, pol: validate_dispatch(out, payload, pol, svc.params),
        )
        self._apply_orders(output.orders)
        if output.contentions or output.diversions:
            self._unit_cases(output)
        return {p.unit_id: tuple(p.hospital_ranking) for p in output.transports}

    def _needed(self) -> bool:
        """Tells whether Dispatch has anything to decide.

        Returns:
            True on an unmet need with a free unit, a possible diversion, or
            a unit awaiting a destination.
        """
        units = self._ctx.view.units
        free = any(u.status is UnitStatus.AVAILABLE for u in units)
        unmet = any(has_unmet_need(n, self._svc.params) for n in self._needs)
        als_short = any(n.als_needed > n.als_en_route for n in self._needs)
        als_en_route = any(
            u.unit_type == AmbulanceType.ALS and u.status is UnitStatus.EN_ROUTE_SCENE for u in units
        )
        diversion = self._svc.config.allow_diversion and als_short and als_en_route
        awaiting = any(u.status is UnitStatus.AWAITING_DESTINATION for u in units)
        return (unmet and free) or diversion or awaiting

    def _apply_orders(self, orders: Sequence[UnitOrder]) -> None:
        """Applies dispatch orders and sends one INFORM per incident.

        Args:
            orders: Validated orders.
        """
        sent: dict[str, list[UnitOrder]] = {}
        for order in orders:
            action = DispatchUnit(unit_id=order.unit_id, incident_id=order.incident_id)
            check = self._ctx.port.apply(action, DISPATCH_AGENT)
            if check.ok:
                sent.setdefault(order.incident_id, []).append(order)
            else:
                LOGGER.warning("dispatch order rejected: %s (%s)", action, check.reason)
        bus = self._svc.bus
        for incident_id in sorted(sent):
            units = sent[incident_id]
            bus.send(
                t_min=self._ctx.t_min, sender=DISPATCH_AGENT, recipient=BROADCAST,
                type=MessageType.INFORM, incident_id=incident_id,
                action=DispatchAction(unit_ids=tuple(o.unit_id for o in units)),
                rationale="; ".join(o.rationale for o in units),
                conversation_id=bus.incident_conversation(incident_id),
            )
        if sent:
            self._ctx.refresh()

    def _unit_cases(self, output: DispatchOutput) -> None:
        """Escalates contention flags and diversion requests, then commits RA's choices.

        Args:
            output: Validated dispatch output.
        """
        svc, ctx = self._svc, self._ctx
        needs: dict[str, IncidentNeed] = {n.incident_id: n for n in self._needs}
        contested = {c.unit_id for c in output.contentions}
        plans = [
            contention_case(svc.bus.new_case_id(), ctx.view, flag, needs, contested, svc.params)
            for flag in sorted(output.contentions, key=lambda c: c.unit_id)
        ] + [
            diversion_case(svc.bus.new_case_id(), ctx.view, req, needs, svc.params)
            for req in sorted(output.diversions, key=lambda d: d.unit_id)
        ]
        escalations = {plan.case.case_id: self._escalate(plan) for plan in plans}
        choices = svc.desk.decide(ctx, plans)
        for plan in plans:
            self._commit(plan, choices[plan.case.case_id], escalations[plan.case.case_id])
        ctx.refresh()

    def _escalate(self, plan: CasePlan) -> Message:
        """Sends the ESCALATE of a UNIT_CONTENTION case in a new conversation.

        Args:
            plan: The case plan.

        Returns:
            The ESCALATE message.
        """
        case = plan.case
        return self._svc.bus.send(
            t_min=self._ctx.t_min, sender=DISPATCH_AGENT, recipient=ALLOCATION_AGENT,
            type=MessageType.ESCALATE, incident_id=case.incident_id,
            action=ResolveAction(
                case_id=case.case_id, case_kind=case.kind,
                option_ids=tuple(o.option_id for o in case.options),
            ),
            rationale=case.summary, conversation_id=self._svc.bus.new_conversation(),
        )

    def _commit(self, plan: CasePlan, choice: Choice, escalation: Message) -> None:
        """Applies RA's choice for a UNIT_CONTENTION case and sends the COMMIT.

        Actions are applied one by one (the alternatives are independent);
        rejected ones are dropped from the COMMIT and named in its rationale.
        If every action failed, an INFORM carrying the attempted actions and
        ``ENGINE_REJECTED`` is sent instead.

        Args:
            plan: The case plan.
            choice: RA's choice.
            escalation: The ESCALATE message answered.
        """
        applied: list[Action] = []
        rejected: list[str] = []
        for action in choice.plan.actions:
            check = self._ctx.port.apply(action, ALLOCATION_AGENT)
            if check.ok:
                applied.append(action)
            else:
                rejected.append(_rejection_text(check))
        refused = bool(choice.plan.actions) and not applied
        resolution = Resolution.ENGINE_REJECTED if refused else choice.plan.resolution
        rationale = "; ".join([choice.rationale, *rejected])
        self._svc.bus.send(
            t_min=self._ctx.t_min, sender=ALLOCATION_AGENT, recipient=DISPATCH_AGENT,
            type=MessageType.INFORM if refused else MessageType.COMMIT,
            incident_id=escalation.incident_id,
            action=CommitAction(
                case_id=plan.case.case_id, option_id=choice.plan.option.option_id,
                actions=tuple(choice.plan.actions if refused else applied), resolution=resolution,
            ),
            rationale=rationale, in_reply_to=escalation.message_id,
            conversation_id=escalation.conversation_id,
        )


@dataclass
class _Thread:
    """Transport negotiation state of one loaded unit.

    Attributes:
        unit_id: The loaded unit.
        incident_id: Incident its patients come from.
        ranking: Hospitals to propose, in order.
        conversation_id: Thread id.
        pointer: Index of the hospital to propose next.
        done: Whether a destination was committed.
        last_message_id: Latest message of the thread.
    """

    unit_id: str
    incident_id: str
    ranking: tuple[str, ...]
    conversation_id: str
    pointer: int = 0
    done: bool = False
    last_message_id: str | None = None

    @property
    def current(self) -> str | None:
        """The hospital to propose to next, or None when the ranking is exhausted."""
        return self.ranking[self.pointer] if self.pointer < len(self.ranking) else None


@dataclass(frozen=True)
class _Counter:
    """An open COUNTER awaiting Resource Allocation.

    Attributes:
        thread: The unit's thread.
        plan: The FLOAT_STAFF / RESERVE_RELEASE case.
        counter: The COUNTER message.
        request: The REQUEST or ESCALATE message to RA.
    """

    thread: _Thread
    plan: CasePlan
    counter: Message
    request: Message


@dataclass
class _TransportNegotiation:
    """Phase C: rounds of PROPOSE / response / counter resolution, then settlement."""

    svc: _Services
    ctx: _Point
    rankings: Mapping[str, tuple[str, ...]]
    threads: list[_Thread] = field(default_factory=list)

    def run(self) -> None:
        """Runs up to ``max_rounds`` rounds, then settles units still without destination."""
        self.threads = self._open_threads()
        last_round = 1
        for rnd in range(1, self.svc.config.max_rounds + 1):
            active = [t for t in self.threads if not t.done and t.current is not None]
            if not active:
                break
            last_round = rnd
            counters: list[_Counter] = []
            for hospital_id in sorted({t.current for t in active if t.current is not None}):
                group = sorted((t for t in active if t.current == hospital_id), key=lambda t: t.unit_id)
                counters += self._negotiate(hospital_id, group, rnd)
                self.ctx.refresh()
            if counters:
                self._resolve_counters(counters, rnd)
        leftovers = [t for t in self.threads if not t.done]
        if leftovers:
            self._settle(leftovers, last_round)

    def _open_threads(self) -> list[_Thread]:
        """Opens one thread per unit awaiting a destination.

        Returns:
            Threads sorted by unit id.
        """
        threads = []
        for unit in awaiting_units(self.ctx.view):
            ranking = self.rankings.get(unit.unit_id)
            if ranking is None:
                LOGGER.warning("no transport plan for %s; engine guard will route it", unit.unit_id)
                continue
            victims = unit_victims(self.ctx.view, unit)
            incident_id = unit.incident_id or (victims[0].incident_id if victims else "")
            threads.append(_Thread(unit.unit_id, incident_id, ranking, self.svc.bus.new_conversation()))
        return threads

    def _send(
        self,
        thread: _Thread,
        rnd: int,
        *,
        sender: str,
        recipient: str,
        type: MessageType,
        action: ProposedAction,
        rationale: str,
        in_reply_to: str | None = None,
    ) -> Message:
        """Sends a message in a unit's thread and records it as the latest.

        Args:
            thread: The thread.
            rnd: Round number.
            sender: Sender agent.
            recipient: Recipient agent.
            type: Performative.
            action: Structured content.
            rationale: Reason.
            in_reply_to: Answered message id.

        Returns:
            The message.
        """
        message = self.svc.bus.send(
            t_min=self.ctx.t_min, sender=sender, recipient=recipient, type=type,
            incident_id=thread.incident_id, action=action, rationale=rationale,
            conversation_id=thread.conversation_id, in_reply_to=in_reply_to, round=rnd,
        )
        thread.last_message_id = message.message_id
        return message

    def _negotiate(self, hospital_id: str, group: Sequence[_Thread], rnd: int) -> list[_Counter]:
        """Proposes a group of units to one hospital and handles its answers.

        Args:
            hospital_id: The hospital.
            group: Threads currently pointing at it.
            rnd: Round number.

        Returns:
            The counters opened (Full only).
        """
        agent = hospital_agent_name(hospital_id)
        proposals = {self._propose(t, hospital_id, rnd): t for t in group}
        requests = [self._request(msg, t) for msg, t in proposals.items()]
        payload = hospital_input(self.ctx.view, hospital_id, requests, self.svc.config)
        output = self.svc.caller.call(
            self.ctx, agent, AgentRole.HOSPITAL, payload, HospitalOutput, hospital_policy,
            lambda out, pol: validate_hospital(out, payload, pol),
        )
        by_id = {msg.message_id: (msg, t) for msg, t in proposals.items()}
        counters = []
        for response in output.responses:
            proposal, thread = by_id[response.proposal_id]
            if response.response == "ACCEPT":
                self._accept(thread, proposal, response, rnd)
            elif response.response == "REJECT":
                self._reject(thread, proposal, response, rnd)
            else:
                counters.append(self._counter(thread, proposal, response, rnd))
        return counters

    def _propose(self, thread: _Thread, hospital_id: str, rnd: int) -> Message:
        """Sends a PROPOSE for one unit.

        Args:
            thread: The unit's thread.
            hospital_id: Target hospital.
            rnd: Round number.

        Returns:
            The PROPOSE message.
        """
        unit = self.ctx.unit(thread.unit_id)
        eta = hospital_eta(unit, find_hospital(self.ctx.view, hospital_id))
        return self._send(
            thread, rnd, sender=DISPATCH_AGENT, recipient=hospital_agent_name(hospital_id),
            type=MessageType.PROPOSE,
            action=AdmitAction(
                hospital_id=hospital_id, unit_id=unit.unit_id,
                victim_ids=unit.load_victim_ids, eta_min=eta,
            ),
            rationale=f"choice {thread.pointer + 1} of {len(thread.ranking)} for {unit.unit_id}, ETA {eta} min",
        )

    def _request(self, proposal: Message, thread: _Thread) -> AdmissionRequest:
        """Builds the hospital-facing admission request of a PROPOSE.

        Args:
            proposal: The PROPOSE message (its id is the proposal id).
            thread: The unit's thread.

        Returns:
            The request.
        """
        content = proposal.proposed_action
        if not isinstance(content, AdmitAction):
            raise ProtocolError(f"{proposal.message_id} is not an admission proposal")
        victims = unit_victims(self.ctx.view, self.ctx.unit(thread.unit_id))
        return AdmissionRequest(
            proposal_id=proposal.message_id, incident_id=thread.incident_id,
            unit_id=thread.unit_id, victims=[triaged_victim(v, self.ctx.deterioration_age) for v in victims],
            eta_min=content.eta_min,
        )

    def _accept(self, thread: _Thread, proposal: Message, response: AdmissionResponse, rnd: int) -> None:
        """Applies an ACCEPT (reserved transport) or reports the engine rejection.

        Args:
            thread: The unit's thread.
            proposal: The PROPOSE answered.
            response: The ACCEPT.
            rnd: Round number.
        """
        hospital_id = proposal.recipient.removeprefix("hospital_")
        action = AssignTransport(unit_id=thread.unit_id, hospital_id=hospital_id, reserve_beds=True)
        check = self.ctx.port.apply(action, proposal.recipient)
        if check.ok:
            resolution = Resolution.ACCEPTED if thread.pointer == 0 else Resolution.ACCEPTED_AFTER_REJECT
            msg_type, rationale = MessageType.ACCEPT, response.rationale
            thread.done = True
            self.ctx.refresh()
        else:
            resolution, msg_type, rationale = Resolution.ENGINE_REJECTED, MessageType.INFORM, _rejection_text(check)
            thread.pointer += 1
        self._send(
            thread, rnd, sender=proposal.recipient, recipient=DISPATCH_AGENT, type=msg_type,
            action=CommitAction(case_id=None, option_id=None, actions=(action,), resolution=resolution),
            rationale=rationale, in_reply_to=proposal.message_id,
        )

    def _reject(self, thread: _Thread, proposal: Message, response: AdmissionResponse, rnd: int) -> None:
        """Sends a REJECT and moves the unit to its next hospital.

        Args:
            thread: The unit's thread.
            proposal: The PROPOSE answered.
            response: The REJECT.
            rnd: Round number.
        """
        thread.pointer += 1
        self._send(
            thread, rnd, sender=proposal.recipient, recipient=DISPATCH_AGENT, type=MessageType.REJECT,
            action=DeclineAction(reason=response.reason), rationale=response.rationale,
            in_reply_to=proposal.message_id,
        )

    def _counter(self, thread: _Thread, proposal: Message, response: AdmissionResponse, rnd: int) -> _Counter:
        """Sends a COUNTER and the REQUEST / ESCALATE that opens its case.

        Args:
            thread: The unit's thread.
            proposal: The PROPOSE answered.
            response: The COUNTER (validated: condition allowed, skill consistent).
            rnd: Round number.

        Returns:
            The open counter.
        """
        if response.counter_condition is None:
            raise ProtocolError(f"COUNTER without condition for {proposal.message_id}")
        hospital_id = proposal.recipient.removeprefix("hospital_")
        unit = self.ctx.unit(thread.unit_id)
        condition, skill = response.counter_condition, response.counter_skill
        if condition is CounterCondition.FLOAT_NURSE:
            skill = skill or NURSE_SKILL
        counter = self._send(
            thread, rnd, sender=proposal.recipient, recipient=DISPATCH_AGENT, type=MessageType.COUNTER,
            action=ConditionalAdmitAction(
                hospital_id=hospital_id, unit_id=unit.unit_id, victim_ids=unit.load_victim_ids,
                condition=condition, skill=skill,
            ),
            rationale=response.rationale, in_reply_to=proposal.message_id,
        )
        plan = counter_case(self.ctx.view, CounterContext(
            self.svc.bus.new_case_id(), unit, tuple(unit_victims(self.ctx.view, unit)),
            thread.incident_id, hospital_id, condition, skill,
        ), self.svc.params)
        return _Counter(thread, plan, counter, self._open_case(thread, plan, counter, rnd))

    def _open_case(self, thread: _Thread, plan: CasePlan, counter: Message, rnd: int) -> Message:
        """Sends REQUEST (float staff) or ESCALATE (reserve release) to RA.

        Args:
            thread: The unit's thread.
            plan: The counter's case.
            counter: The COUNTER message.
            rnd: Round number.

        Returns:
            The message sent to Resource Allocation.

        Raises:
            ProtocolError: If the counter is not a COUNTER, or is a reserve
                release while escalation is disabled (validation must have
                removed it).
        """
        case, content = plan.case, counter.proposed_action
        if not isinstance(content, ConditionalAdmitAction):
            raise ProtocolError(f"{counter.message_id} is not a COUNTER")
        if content.condition is CounterCondition.RESERVE_RELEASE:
            if not self.svc.config.allow_escalation:
                raise ProtocolError(f"{counter.message_id}: reserve release needs escalation")
            msg_type = MessageType.ESCALATE
            action: ResolveAction | StaffRequestAction = ResolveAction(
                case_id=case.case_id, case_kind=case.kind,
                option_ids=tuple(o.option_id for o in case.options),
            )
        else:
            msg_type = MessageType.REQUEST
            role = StaffRole.DOCTOR if content.condition is CounterCondition.FLOAT_DOCTOR else StaffRole.NURSE
            action = StaffRequestAction(
                hospital_id=content.hospital_id, role=role, skill=content.skill or NURSE_SKILL
            )
        return self._send(
            thread, rnd, sender=counter.sender, recipient=ALLOCATION_AGENT, type=msg_type,
            action=action, rationale=f"{case.case_id}: {case.summary}", in_reply_to=counter.message_id,
        )

    def _resolve_counters(self, counters: Sequence[_Counter], rnd: int) -> None:
        """Runs Resource Allocation on the round's counters and commits the outcome.

        Args:
            counters: Open counters of this round.
            rnd: Round number.
        """
        choices = self.svc.desk.decide(self.ctx, [c.plan for c in counters])
        for counter in sorted(counters, key=lambda c: c.plan.case.case_id):
            choice = choices[counter.plan.case.case_id]
            failure = None
            if choice.plan.grants:
                failure = apply_all(self.ctx.port, choice.plan.actions, ALLOCATION_AGENT)
                if failure is None:
                    self._grant(counter, choice, rnd)
                    continue
            self._decline(counter, choice, rnd, failure)
        self.ctx.refresh()

    def _grant(self, counter: _Counter, choice: Choice, rnd: int) -> None:
        """Sends the two COMMITs of a granted counter (to the hospital and to dispatch).

        Args:
            counter: The counter.
            choice: RA's (applied) choice.
            rnd: Round number.
        """
        commit = CommitAction(
            case_id=counter.plan.case.case_id, option_id=choice.plan.option.option_id,
            actions=choice.plan.actions, resolution=Resolution.COUNTER_GRANTED,
        )
        thread = counter.thread
        self._send(thread, rnd, sender=ALLOCATION_AGENT, recipient=counter.counter.sender,
                   type=MessageType.COMMIT, action=commit, rationale=choice.rationale,
                   in_reply_to=counter.request.message_id)
        self._send(thread, rnd, sender=ALLOCATION_AGENT, recipient=DISPATCH_AGENT,
                   type=MessageType.COMMIT, action=commit, rationale=choice.rationale,
                   in_reply_to=counter.counter.message_id)
        thread.done = True

    def _decline(self, counter: _Counter, choice: Choice, rnd: int, failure: ActionCheck | None) -> None:
        """Closes a counter that was not granted and moves the unit to its next hospital.

        RA's decline is a REJECT carrying ``COUNTER_DECLINED``. A grant the
        engine refused follows the engine-refusal convention instead: an
        INFORM carrying the attempted actions and ``ENGINE_REJECTED``.

        Args:
            counter: The counter.
            choice: RA's choice (a decline, or a grant the engine refused).
            rnd: Round number.
            failure: The engine rejection of a granted option, if any.
        """
        counter.thread.pointer += 1
        if failure is not None:
            msg_type, rationale = MessageType.INFORM, _rejection_text(failure)
            outcome = CommitAction(
                case_id=counter.plan.case.case_id, option_id=choice.plan.option.option_id,
                actions=choice.plan.actions, resolution=Resolution.ENGINE_REJECTED,
            )
        else:
            decline = next((p for p in counter.plan.plans if not p.grants), choice.plan)
            msg_type, rationale = MessageType.REJECT, choice.rationale
            outcome = CommitAction(
                case_id=counter.plan.case.case_id, option_id=decline.option.option_id,
                actions=(), resolution=Resolution.COUNTER_DECLINED,
            )
        self._send(
            counter.thread, rnd, sender=ALLOCATION_AGENT, recipient=counter.counter.sender,
            type=msg_type, action=outcome, rationale=rationale,
            in_reply_to=counter.request.message_id,
        )

    def _settle(self, leftovers: Sequence[_Thread], rnd: int) -> None:
        """Places units that no hospital accepted.

        Args:
            leftovers: Threads still without destination.
            rnd: Last round number.
        """
        if self.svc.config.allow_escalation:
            self._no_accepting(leftovers, rnd)
        else:
            for thread in leftovers:
                self._force(thread, rnd)

    def _force(self, thread: _Thread, rnd: int) -> None:
        """B1 rule: unreserved transport to the top-ranked hospital (FORCED_ADMISSION).

        Args:
            thread: The unit's thread.
            rnd: Last round number.
        """
        if not thread.ranking:
            LOGGER.warning("%s has no capable hospital; engine guard will route it", thread.unit_id)
            return
        action = AssignTransport(unit_id=thread.unit_id, hospital_id=thread.ranking[0], reserve_beds=False)
        check = self.ctx.port.apply(action, DISPATCH_AGENT)
        resolution = Resolution.FORCED_ADMISSION if check.ok else Resolution.ENGINE_REJECTED
        self._send(
            thread, rnd, sender=DISPATCH_AGENT, recipient=hospital_agent_name(thread.ranking[0]),
            type=MessageType.INFORM,
            action=CommitAction(case_id=None, option_id=None, actions=(action,), resolution=resolution),
            rationale="no hospital accepted; forced admission to the top-ranked hospital"
            if check.ok else _rejection_text(check),
            in_reply_to=thread.last_message_id,
        )
        thread.done = check.ok

    def _no_accepting(self, leftovers: Sequence[_Thread], rnd: int) -> None:
        """Full rule: NO_ACCEPTING_HOSPITAL cases decided by Resource Allocation.

        Args:
            leftovers: Threads still without destination.
            rnd: Last round number.
        """
        plans: list[tuple[_Thread, CasePlan, Message]] = []
        for thread in leftovers:
            unit = self.ctx.unit(thread.unit_id)
            plan = no_accepting_case(
                self.svc.bus.new_case_id(), self.ctx.view, unit,
                unit_victims(self.ctx.view, unit), thread.incident_id, self.svc.params,
            )
            if not plan.plans:
                LOGGER.warning("%s has no capable hospital; engine guard will route it", unit.unit_id)
                continue
            escalation = self._send(
                thread, rnd, sender=DISPATCH_AGENT, recipient=ALLOCATION_AGENT, type=MessageType.ESCALATE,
                action=ResolveAction(
                    case_id=plan.case.case_id, case_kind=plan.case.kind,
                    option_ids=tuple(o.option_id for o in plan.case.options),
                ),
                rationale=plan.case.summary, in_reply_to=thread.last_message_id,
            )
            plans.append((thread, plan, escalation))
        choices = self.svc.desk.decide(self.ctx, [p for _, p, _ in plans])
        for thread, plan, escalation in plans:
            self._commit_destination(thread, plan, choices[plan.case.case_id], escalation, rnd)

    def _commit_destination(
        self, thread: _Thread, plan: CasePlan, choice: Choice, escalation: Message, rnd: int
    ) -> None:
        """Applies RA's destination (retrying without reservation if beds raced) and COMMITs.

        Args:
            thread: The unit's thread.
            plan: The NO_ACCEPTING_HOSPITAL case.
            choice: RA's choice.
            escalation: The ESCALATE answered.
            rnd: Last round number.
        """
        (action,) = choice.plan.actions
        if not isinstance(action, AssignTransport):
            raise ProtocolError(f"{plan.case.case_id} option is not a transport")
        check = self.ctx.port.apply(action, ALLOCATION_AGENT)
        if not check.ok and action.reserve_beds:
            action = action.model_copy(update={"reserve_beds": False})
            check = self.ctx.port.apply(action, ALLOCATION_AGENT)
        self._send(
            thread, rnd, sender=ALLOCATION_AGENT, recipient=DISPATCH_AGENT,
            type=MessageType.COMMIT if check.ok else MessageType.INFORM,
            action=CommitAction(
                case_id=plan.case.case_id, option_id=choice.plan.option.option_id, actions=(action,),
                resolution=Resolution.RA_COMMIT if check.ok else Resolution.ENGINE_REJECTED,
            ),
            rationale=choice.rationale if check.ok else _rejection_text(check),
            in_reply_to=escalation.message_id,
        )
        thread.done = check.ok
        self.ctx.refresh()
