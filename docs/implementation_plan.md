# Implementation Plan: Emergency Response MAS, Phase 2 (engine, agents, negotiation, evaluation, dashboard)

Project root (all paths below are relative to it):
`/Users/suhas/Suhas/WorkWithMe/CSYE-7380-Agentic AI/Intelligent Agents and Multi-Agent Systems for Emergency Response Coordination/Shubham_Bagwe_Emergency_Response_MAS`

## 0. What I checked (facts this plan relies on)

**Repo state.** Phase 1 is in `emergency_sim/geo.py`, `datagen.py` and `validate.py`, plus `data/{normal,surge,constrained}`. There is no `pyproject.toml`, no `tests/`, and the folder is not a git repo. Do not modify `geo.py`, `datagen.py`, `validate.py` or `data/`; reuse them.

**datagen supports `--seed`.** `datagen.generate_scenario(name, seed, out_dir)` writes files. I generated seeds 1, 2, 3 and 1001 in memory and all three scenarios look sane: 21 to 54 victims, 8 to 18 free ER beds, all capability types present. `validate.py` only scans `data/*/manifest.json`, so generating into a temp dir does not affect it.

**Data facts that matter for the design:**
- Doctor skills are narrow. H1 has only one trauma doctor (DR-H1-1). DR-H1-2 and DR-H1-3 end their shifts at 120 min.
- The float pool has trauma, cardiac and paediatric doctors (one doctor in `constrained`).
- An ambulance carries at most 1 RED per trip. So the design doc's counter example "take the 2 RED, send the YELLOW to H2" is not physically possible. This is one of the corrections below.

**SDK (openai-agents 0.24.0).** Verified by running read-only snippets in `.venv`:
- A custom `agents.models.interface.Model` subclass passed as `Agent(model=...)` works through `Runner.run` and `Runner.run_sync`, fully offline.
- The final structured output must be an assistant `ResponseOutputMessage` whose `ResponseOutputText.text` is JSON. The SDK validates it with `output_schema.validate_json(...)` in `run_internal/turn_resolution.py` (around line 1227). Invalid JSON raises `agents.ModelBehaviorError`.
- Two-turn tool loop: the model returns a `ResponseFunctionToolCall`. Runner executes the `@function_tool` with `RunContextWrapper[Ctx]`. The next `get_response` input contains `{'type': 'function_call_output', 'output': ...}`, and the model then returns the final message. `result.new_items` holds `ToolCallItem`, `ToolCallOutputItem` and `MessageOutputItem`.
- Measured overhead: about 2.6 ms per offline Runner call including one tool round-trip. A full offline grid takes minutes.
- Strict output schemas reject `dict[str, X]` fields (`UserError`). The root must not be a union. Do not use defaults or numeric/length constraints in LLM-facing output models (OpenAI strict mode may reject `minimum` etc.); check ranges in semantic validation instead.
- `RunConfig(tracing_disabled=True)` and `agents.set_tracing_disabled(True)` exist. `agents.testing.ScriptedModel`, `assistant_message` and `function_call` exist; use them in tests only.
- `Runner.run_sync` handles Python 3.14's event-loop change.

**Other packages.** Streamlit 1.66 has `streamlit.testing.v1.AppTest`, `st.fragment` and `st.cache_data`. Plotly 7.1 has `px.timeline`. pandas 3.1, numpy 2.5 and pytest 9.1 are installed.

## 1. Design changes vs `docs/simulation_design.md` (and why)

Everything else in the agreed design stays.

1. **Discrete-event engine with 1-minute snapshots, instead of fixed 1-minute ticks.** Travel times are 0.1-min floats from `geo.travel_time`. A priority queue is exact, deterministic and cheaper. Snapshots every minute still give the dashboard a per-minute timeline.
2. **Agents' function tools are read-only.** Proposals and commits are structured outputs (`output_type`), not tools. Only the coordinator mutates state, by applying validated engine `Action`s through one engine port.
   - Why: one validation point, no side effects hidden inside LLM tool calls (those are hard to roll back), and an identical contract for the offline and LLM backends.
3. **COUNTER means conditional acceptance only:** "accept if I get float staff" (`FLOAT_DOCTOR` / `FLOAT_NURSE`) or "accept only by using my critical reserve, needs approval" (`RESERVE_RELEASE`). Partial-load splitting is dropped: trips go to a single destination and carry at most 1 RED, so "take the 2 RED" cannot happen.
4. **Hospital bed board is public to every condition**, including B0: free unreserved beds and queue length. B0's rule "nearest capable hospital with a free bed" needs it. Staff loads are private to each hospital agent, so revealing them is the value of communication.
5. **Staff holdover rule.** A staff member past `shift_end_min` takes no new patients, unless no on-shift colleague of the same role at that hospital has the needed skill. Without this, H1 would lose all cardiac and burns cover at t=120 in every scenario. Patients would be stuck forever, B0 would look artificially bad, and the comparison would be unfair.
6. **"Care start" is defined precisely (golden window clock).** It is the start of ER treatment at a capable hospital. Cardiac arrest is the exception: care starts when an ALS unit reaches the scene (field defibrillation).
7. **Responder teams gate scene access, and the first unit on scene performs START triage.** Triage reveals victims to every controller.
8. **Not modelled (documented):** ICU dynamics, negotiation latency (decisions take zero simulated time and happen while loading), and the LLM rewrite of caller reports (design step 8).

## 2. Architecture

### 2.1 Layers and dependency direction

```
            dashboard/  (Streamlit + Plotly; replays JSONL; never calls a model)
                 │
   emergency_sim/cli.py ── emergency_sim/evaluation/   (composition root: runner, metrics, report)
                 │                    │
        ┌────────┴──────────┬─────────┴────────────┐
   emergency_sim/control/   emergency_sim/mas/     emergency_sim/engine/
   (B0 greedy, coordinator,  (SDK Agents, PolicyModel,  (deterministic DES,
    negotiation, validation)  tools, backends)           single source of truth)
        │        │                 │                      │
        │        └──── emergency_sim/policies/ ◄──────────┘ (no: engine does not import policies)
        │              (pure offline decision rules + scoring; SDK-free)
        └──────────────┬──────────────────────────────────┘
              emergency_sim/domain/   (contracts: Pydantic/dataclass types + Protocols)
              emergency_sim/scenarios.py, emergency_sim/eventlog.py
              emergency_sim/geo.py, emergency_sim/datagen.py  (existing)
```

Allowed imports (enforced by `tests/test_architecture.py` using `ast`):

| Package | May import |
|---|---|
| `domain` | stdlib, pydantic, `emergency_sim.geo` |
| `scenarios.py` | `domain`, `datagen` |
| `eventlog.py` | `domain` |
| `engine` | `domain`, `geo`. Not `policies`, `control`, `mas` or `agents` (SDK). |
| `policies` | `domain`, `geo`. Not the `agents` SDK. |
| `mas` | `domain`, `policies`, `agents` (SDK), `openai`. The **only** package that imports `agents`. |
| `control` | `domain`, `policies`, `geo`. Not `engine`, `mas` or `agents`. It sees the engine only through the `EnginePort` Protocol and the backend only through the `DecisionBackend` Protocol. |
| `evaluation`, `cli` | anything in `emergency_sim` |
| `dashboard` | `domain`, `eventlog`, `evaluation` (metrics/report, optional runner). Not `engine`, `control`, `mas` or `policies` directly. |

Why: the engine (truth) and the policies (decisions) are testable without the SDK. The SDK is isolated so API drift touches one package. The information boundary is structural: controllers can only reach `PublicView`. The dashboard depends only on the log contract.

### 2.2 Exact file layout and ownership (WP = work package, section 7)

```
pyproject.toml                         WP0  (metadata + [tool.pytest.ini_options] testpaths=["tests"], pythonpath=["."], markers=["slow"])
requirements.txt                       WP0  (documents installed pins: openai-agents==0.24.0, openai==3.37.0, pydantic==2.14.1, streamlit==1.66.1, plotly==7.1.0, pandas==3.1.0, numpy==2.5.4, pytest==9.1.1) – no installs
emergency_sim/scenarios.py             WP0
emergency_sim/eventlog.py              WP0
emergency_sim/domain/__init__.py       WP0  (docstring only; no re-exports, avoids cycles)
emergency_sim/domain/base.py           WP0
emergency_sim/domain/enums.py          WP0
emergency_sim/domain/scenario.py       WP0
emergency_sim/domain/params.py         WP0
emergency_sim/domain/actions.py        WP0
emergency_sim/domain/views.py          WP0
emergency_sim/domain/messages.py       WP0
emergency_sim/domain/agent_io.py       WP0
emergency_sim/domain/outcomes.py       WP0
emergency_sim/domain/log_records.py    WP0
emergency_sim/domain/ports.py          WP0
tests/conftest.py, tests/factories.py, tests/test_contracts.py, tests/test_architecture.py, tests/test_scenarios.py   WP0

emergency_sim/engine/{__init__,state,events,rules,care,observe,engine}.py                          WP1
tests/test_engine_lifecycle.py, test_engine_hospital.py, test_engine_rules.py, test_public_view.py, test_engine_determinism.py   WP1

emergency_sim/policies/{__init__,params,estimates,assessment,dispatch,hospital,allocation}.py      WP2
emergency_sim/mas/{__init__,context,tools,instructions,specs,policy_model,backend}.py               WP2
tests/test_policies.py, test_estimates.py, test_strict_schemas.py, test_mas_offline_runner.py, test_mas_backend_fallback.py   WP2

emergency_sim/control/{__init__,conditions,needs,greedy,observations,cases,validation,bus,coordinator}.py   WP3
tests/test_needs.py, test_greedy.py, test_validation.py, test_coordinator.py                       WP3

emergency_sim/evaluation/{__init__,metrics,runner,report}.py, emergency_sim/cli.py                 WP4
tests/test_metrics.py, test_report.py, test_cli.py                                                 WP4

dashboard/{__init__,replay,figures,app}.py                                                         WP5
tests/test_dashboard_replay.py, test_dashboard_figures.py, test_dashboard_app.py                   WP5

tests/test_end_to_end.py, tests/test_llm_fallback_integration.py, results/**, README.md,
docs/simulation_design.md (v2 section), docs/evaluation.md, docs/system_flow.mmd, prompts.md        WP6 (integration, lead)
```

### 2.3 Code standards for every package (Google Python Style)

- Module docstring. Google-style docstrings (Args/Returns/Raises) on every public function and class.
- Full type hints. Use Python 3.14's native lazy annotations (no `from __future__ import annotations`).
- `StrEnum` for all enums. Immutable value objects: `FrozenModel` with tuples, or `@dataclass(frozen=True)`.
- No mutable module globals. Pass params, sinks and backends explicitly (dependency injection).
- Keep functions under about 40 lines. Name constants in UPPER_CASE.
- No `print` outside `cli.py`; use `logging`.
- Determinism: sort before iterating over any collection, no wall-clock or uuid in logs, ties broken by id.
- Each package defines narrow exceptions: `engine.EngineError`, `mas.BackendUnavailableError`, `control.ProtocolError`.

## 3. Interface contracts (WP0 writes these verbatim; changes only via the lead)

### 3.1 `domain/base.py`

```python
from pydantic import BaseModel, ConfigDict

class FrozenModel(BaseModel):
    """Immutable value object; unknown fields rejected."""
    model_config = ConfigDict(frozen=True, extra="forbid")

class AgentIOModel(BaseModel):
    """LLM-facing DTO. Output subclasses must be strict-schema compatible:
    no dict fields, no defaults, no numeric/length constraints."""
    model_config = ConfigDict(extra="forbid")
```

### 3.2 `domain/enums.py`

```python
class Triage(StrEnum): RED="RED"; YELLOW="YELLOW"; GREEN="GREEN"; BLACK="BLACK"
TRIAGE_RANK: Final = MappingProxyType({Triage.RED:0, Triage.YELLOW:1, Triage.GREEN:2, Triage.BLACK:3})
class Capability(StrEnum): TRAUMA="trauma"; BURNS="burns"; NEURO="neuro"; CARDIAC="cardiac"; PEDIATRIC="pediatric"; GENERAL="general"
class ResponderCapability(StrEnum): FIRE_SUPPRESSION="fire_suppression"; EXTRICATION="extrication"; HEAVY_RESCUE="heavy_rescue"; HAZMAT="hazmat"; DECONTAMINATION="decontamination"; SCENE_CONTROL="scene_control"
class AmbulanceType(StrEnum): ALS="ALS"; BLS="BLS"
class UnitClass(StrEnum): AMBULANCE="ambulance"; RESPONDER="responder"
class UnitStatus(StrEnum): AVAILABLE="available"; BUSY_T0="busy_t0"; OUT_OF_SERVICE="out_of_service"; EN_ROUTE_SCENE="en_route_scene"; ON_SCENE="on_scene"; LOADING="loading"; AWAITING_DESTINATION="awaiting_destination"; TRANSPORTING="transporting"; AT_HOSPITAL_FULL="at_hospital_full"; HANDOVER="handover"
class VictimStatus(StrEnum): UNTRIAGED="untriaged"; EXPECTANT="expectant"; WAITING="waiting"; LOADING="loading"; LOADED="loaded"; TRANSPORTING="transporting"; QUEUED="queued"; WAITING_FOR_STAFF="waiting_for_staff"; IN_TREATMENT="in_treatment"; TREATED="treated"
class IncidentStatus(StrEnum): OPEN="open"; CLEARED="cleared"
class StaffRole(StrEnum): DOCTOR="doctor"; NURSE="nurse"
class TriggerKind(StrEnum): NEW_INCIDENT="new_incident"; SCENE_TRIAGED="scene_triaged"; PATIENTS_LOADED="patients_loaded"; UNIT_FREED="unit_freed"; HOSPITAL_FULL_ON_ARRIVAL="hospital_full_on_arrival"; DETERIORATION="deterioration"
class Condition(StrEnum): B0="b0"; B1="b1"; FULL="full"; FULL_NO_FLOAT="full-no-float"; FULL_NO_DIVERSION="full-no-diversion"; FULL_NO_ESCALATION="full-no-escalation"
class AgentRole(StrEnum): ASSESSMENT="incident_assessment"; DISPATCH="dispatch"; HOSPITAL="hospital_coordination"; ALLOCATION="resource_allocation"
class MessageType(StrEnum): PRIORITY_REPORT="PRIORITY_REPORT"; PROPOSE="PROPOSE"; ACCEPT="ACCEPT"; REJECT="REJECT"; COUNTER="COUNTER"; REQUEST="REQUEST"; ESCALATE="ESCALATE"; COMMIT="COMMIT"; INFORM="INFORM"
class CaseKind(StrEnum): UNIT_CONTENTION="UNIT_CONTENTION"; NO_ACCEPTING_HOSPITAL="NO_ACCEPTING_HOSPITAL"; FLOAT_STAFF="FLOAT_STAFF"; RESERVE_RELEASE="RESERVE_RELEASE"
class CounterCondition(StrEnum): FLOAT_DOCTOR="FLOAT_DOCTOR"; FLOAT_NURSE="FLOAT_NURSE"; RESERVE_RELEASE="RESERVE_RELEASE"
class Resolution(StrEnum): ACCEPTED="ACCEPTED"; ACCEPTED_AFTER_REJECT="ACCEPTED_AFTER_REJECT"; COUNTER_GRANTED="COUNTER_GRANTED"; COUNTER_DECLINED="COUNTER_DECLINED"; RA_COMMIT="RA_COMMIT"; FORCED_ADMISSION="FORCED_ADMISSION"; KEPT_ASSIGNMENT="KEPT_ASSIGNMENT"; ENGINE_REJECTED="ENGINE_REJECTED"
class RejectReason(StrEnum): UNKNOWN_UNIT="unknown_unit"; UNKNOWN_INCIDENT="unknown_incident"; UNKNOWN_HOSPITAL="unknown_hospital"; UNKNOWN_STAFF="unknown_staff"; UNIT_NOT_AVAILABLE="unit_not_available"; UNIT_NOT_EN_ROUTE="unit_not_en_route"; NOT_AN_AMBULANCE="not_an_ambulance"; INCIDENT_NOT_OPEN="incident_not_open"; SAME_TARGET="same_target"; NOT_AWAITING_DESTINATION="not_awaiting_destination"; HOSPITAL_NOT_CAPABLE="hospital_not_capable"; INSUFFICIENT_BEDS="insufficient_beds"; SAME_HOSPITAL="same_hospital"; STAFF_NOT_FLOAT="staff_not_float"; STAFF_ALREADY_LENT="staff_already_lent"; STAFF_OFF_SHIFT="staff_off_shift"
class SimEventType(StrEnum): INCIDENT_REPORTED; UNIT_DISPATCHED; UNIT_DIVERTED; UNIT_ARRIVED_SCENE; SCENE_TRIAGED; ACCESS_GRANTED; LOADING_STARTED; PATIENTS_LOADED; BEDS_RESERVED; TRANSPORT_STARTED; ARRIVED_HOSPITAL; HOSPITAL_FULL_ON_ARRIVAL; PATIENT_BEDDED; PATIENT_QUEUED; FIELD_CARE_STARTED; TREATMENT_STARTED; TREATMENT_ENDED; UNIT_AVAILABLE; VICTIM_DETERIORATED; STAFF_LENT; SHIFT_ENDED; T0_BED_RELEASED; T0_STAFF_RELEASED; INCIDENT_CLEARED; CONTROLLER_OMISSION   # values = lowercase names
```

### 3.3 `domain/scenario.py`

Field names must match the JSON files exactly (verified). `extra="forbid"` catches data drift.

```python
class HospitalSpec(FrozenModel): id: str; name: str; x: float; y: float; trauma_level: int; capabilities: tuple[Capability, ...]; er_capacity: int; icu_beds: int; critical_reserve: int; er_occupied_t0: int; icu_occupied_t0: int; er_free_t0: int; intersection: str
class StationSpec(FrozenModel): id: str; name: str; x: float; y: float
class AmbulanceSpec(FrozenModel): id: str; type: AmbulanceType; station: str; x: float; y: float; capacity: int; max_red_per_trip: int; status: Literal["available","busy","out_of_service"]; busy_until_min: int
class StaffSpec(FrozenModel): id: str; role: StaffRole; home: str; skills: tuple[str, ...]; max_patients: int; load_t0: int; shift_end_min: int
class ResponderSpec(FrozenModel): id: str; type: str; base: str; capabilities: tuple[ResponderCapability, ...]; x: float; y: float; crew: int; status: Literal["available"]; busy_until_min: int
class VictimSpec(FrozenModel): victim_id: str; condition: str; triage: Triage; required_capability: Capability; needs_als: bool; needs_decon: bool; golden_window_min: int; deteriorates_to_red_at_min: int | None
class IncidentSpec(FrozenModel): incident_id: str; type: str; t_report_min: int; x: float; y: float; intersection: str; congestion: float; reported_victims: int; caller_report: str; required_responder_capabilities: tuple[ResponderCapability, ...]; true_victims: int; worst_triage: Triage; time_sensitivity_min: int; mci_cluster: bool; victims: tuple[VictimSpec, ...]
class Scenario(FrozenModel): name: str; seed: int; horizon_min: float; hospitals: tuple[HospitalSpec, ...]; stations: tuple[StationSpec, ...]; ambulances: tuple[AmbulanceSpec, ...]; staff: tuple[StaffSpec, ...]; responders: tuple[ResponderSpec, ...]; incidents: tuple[IncidentSpec, ...]
```

### 3.4 `emergency_sim/scenarios.py` (I/O; WP0)

```python
DEFAULT_DATA_ROOT: Final[Path]  # <repo>/data
def load_scenario(directory: Path) -> Scenario:
    """Reads the six JSON files + manifest.json (seed, horizon_min)."""
def scenario_for(name: str, seed: int, data_root: Path = DEFAULT_DATA_ROOT) -> Scenario:
    """Committed data if data_root/name/manifest.json has this seed; otherwise
    datagen.generate_scenario(name, seed, tmp) in a TemporaryDirectory, load, discard. Raises KeyError for unknown name."""
```

### 3.5 `domain/params.py`

```python
@dataclass(frozen=True)
class SimParams:
    """Engine constants (documented assumptions). Horizon comes from Scenario."""
    snapshot_interval_min: float = 1.0
    ambulance_turnout_min: float = geo.TURNOUT_MIN["ambulance"]   # 1.0
    responder_turnout_min: float = geo.TURNOUT_MIN["responder"]   # 2.0
    scene_load_min: float = 3.0
    handover_min: float = 5.0
    heavy_access_min: float = 10.0          # extrication / heavy_rescue / fire_suppression / hazmat on scene
    treatment_red_min: float = 60.0
    treatment_yellow_min: float = 40.0
    treatment_green_min: float = 20.0
    deteriorated_red_window_min: float = 30.0
    t0_bed_release_interval_min: float = 15.0
    t0_staff_release_interval_min: float = 20.0
    def treatment_min(self, triage: Triage) -> float: ...
```

### 3.6 `domain/actions.py`: the only ways to change the world

```python
class DispatchUnit(FrozenModel): kind: Literal["dispatch_unit"] = "dispatch_unit"; unit_id: str; incident_id: str
class DivertUnit(FrozenModel): kind: Literal["divert_unit"] = "divert_unit"; unit_id: str; incident_id: str
class AssignTransport(FrozenModel): kind: Literal["assign_transport"] = "assign_transport"; unit_id: str; hospital_id: str; reserve_beds: bool
class LendStaff(FrozenModel): kind: Literal["lend_staff"] = "lend_staff"; staff_id: str; hospital_id: str
Action = Annotated[DispatchUnit | DivertUnit | AssignTransport | LendStaff, Field(discriminator="kind")]
class ActionCheck(FrozenModel): ok: bool; reason: RejectReason | None = None; detail: str = ""
```

### 3.7 `domain/views.py`: the information boundary

Contains no hidden truth: no `golden_window_min`, `deteriorates_to_red_at_min`, `true_victims`, `worst_triage` or `time_sensitivity_min`, and no victims of untriaged incidents.

```python
class VictimView(FrozenModel): victim_id: str; incident_id: str; triage: Triage; condition: str; required_capability: Capability; needs_als: bool; status: VictimStatus; unit_id: str | None; hospital_id: str | None; deteriorated: bool
class IncidentView(FrozenModel): incident_id: str; incident_type: str; t_report_min: float; x: float; y: float; intersection: str; reported_victims: int; caller_report: str; required_responder_capabilities: tuple[ResponderCapability, ...]; status: IncidentStatus; triaged: bool; accessible: bool; victims: tuple[VictimView, ...]; assigned_unit_ids: tuple[str, ...]; responder_capabilities_present: tuple[ResponderCapability, ...]
class UnitView(FrozenModel): unit_id: str; unit_class: UnitClass; unit_type: str; capabilities: tuple[str, ...]; status: UnitStatus; x: float; y: float; incident_id: str | None; hospital_id: str | None; destination_id: str | None; load_victim_ids: tuple[str, ...]; seats: int; available_at_min: float | None
class StaffView(FrozenModel): staff_id: str; role: StaffRole; skills: tuple[str, ...]; home: str; hospital_id: str | None; load: int; max_patients: int; on_shift: bool
class HospitalView(FrozenModel): hospital_id: str; name: str; x: float; y: float; capabilities: tuple[Capability, ...]; er_capacity: int; critical_reserve: int; beds_occupied: int; beds_reserved: int; beds_free_unreserved: int; queue_length: int; waiting_for_staff: int; in_treatment: int; inbound_unit_ids: tuple[str, ...]; staff: tuple[StaffView, ...]
class PublicView(FrozenModel): t_min: float; horizon_min: float; incidents: tuple[IncidentView, ...]; units: tuple[UnitView, ...]; hospitals: tuple[HospitalView, ...]; float_pool: tuple[StaffView, ...]   # all sorted by id; incidents = reported only
class Trigger(FrozenModel): kind: TriggerKind; incident_id: str | None = None; unit_id: str | None = None; hospital_id: str | None = None; victim_ids: tuple[str, ...] = ()
class DecisionPoint(FrozenModel): t_min: float; triggers: tuple[Trigger, ...]
```

### 3.8 `domain/messages.py`

```python
ASSESSMENT_AGENT: Final = "incident_assessment"; DISPATCH_AGENT: Final = "dispatch"; ALLOCATION_AGENT: Final = "resource_allocation"; BROADCAST: Final = "all"
def hospital_agent_name(hospital_id: str) -> str: return f"hospital_{hospital_id}"

class AssessAction(FrozenModel): action: Literal["assess"] = "assess"; urgency: int; suspected_red: bool; suspected_cardiac_arrest: bool; est_patients: int; needed_capabilities: tuple[Capability, ...]; als_units_needed: int; deadline_in_min: float
class DispatchAction(FrozenModel): action: Literal["dispatch"] = "dispatch"; unit_ids: tuple[str, ...]
class AdmitAction(FrozenModel): action: Literal["admit"] = "admit"; hospital_id: str; unit_id: str; victim_ids: tuple[str, ...]; eta_min: float
class ConditionalAdmitAction(FrozenModel): action: Literal["admit_if"] = "admit_if"; hospital_id: str; unit_id: str; victim_ids: tuple[str, ...]; condition: CounterCondition; skill: str | None
class DeclineAction(FrozenModel): action: Literal["decline"] = "decline"; reason: str
class StaffRequestAction(FrozenModel): action: Literal["lend_staff"] = "lend_staff"; hospital_id: str; role: StaffRole; skill: str
class ResolveAction(FrozenModel): action: Literal["resolve"] = "resolve"; case_id: str; case_kind: CaseKind; option_ids: tuple[str, ...]
class CommitAction(FrozenModel): action: Literal["commit"] = "commit"; case_id: str | None; option_id: str | None; actions: tuple[Action, ...]; resolution: Resolution
ProposedAction = Annotated[AssessAction | DispatchAction | AdmitAction | ConditionalAdmitAction | DeclineAction | StaffRequestAction | ResolveAction | CommitAction, Field(discriminator="action")]

class Message(FrozenModel):
    message_id: str          # "M-0001", sequential per run
    t_min: float
    sender: str              # agent name
    recipient: str           # agent name or BROADCAST
    type: MessageType
    incident_id: str         # always set (the triggering incident)
    proposed_action: ProposedAction
    rationale: str
    in_reply_to: str | None
    conversation_id: str     # "NEG-0001": one per negotiation thread / case
    round: int               # 0 for non-negotiation messages, 1..3 in transport negotiation
```

### 3.9 `domain/agent_io.py`: agent inputs/outputs (LLM-facing) and the decision record

```python
class TriagedVictim(AgentIOModel): victim_id: str; triage: Triage; condition: str; required_capability: Capability; status: VictimStatus
# Incident Assessment
class IncidentReport(AgentIOModel): incident_id: str; incident_type: str; t_report_min: float; minutes_since_report: float; x: float; y: float; intersection: str; reported_victims: int; caller_report: str; required_responder_capabilities: list[ResponderCapability]; triaged_victims: list[TriagedVictim]
class AssessmentInput(AgentIOModel): t_min: float; incidents: list[IncidentReport]
class PriorityReport(AgentIOModel): incident_id: str; urgency: int; suspected_red: bool; suspected_cardiac_arrest: bool; est_patients: int; needed_capabilities: list[Capability]; als_units_needed: int; deadline_in_min: float; rationale: str
class AssessmentOutput(AgentIOModel): reports: list[PriorityReport]
# Dispatch
class IncidentNeed(AgentIOModel): incident_id: str; x: float; y: float; urgency: int; suspected_red: bool; suspected_cardiac_arrest: bool; deadline_in_min: float; triaged: bool; est_patients_remaining: int; red_waiting: int; ambulances_assigned: int; seats_en_route: int; als_en_route: int; als_needed: int; responder_caps_missing: list[ResponderCapability]
class UnitStatusView(AgentIOModel): unit_id: str; unit_class: UnitClass; unit_type: str; capabilities: list[str]; status: UnitStatus; x: float; y: float; incident_id: str | None; seats: int
class HospitalBoardEntry(AgentIOModel): hospital_id: str; x: float; y: float; capabilities: list[Capability]; beds_free_unreserved: int; queue_length: int
class LoadedUnit(AgentIOModel): unit_id: str; incident_id: str; x: float; y: float; victims: list[TriagedVictim]
class DispatchInput(AgentIOModel): t_min: float; incidents: list[IncidentNeed]; units: list[UnitStatusView]; hospitals: list[HospitalBoardEntry]; loaded_units: list[LoadedUnit]; allow_diversion: bool; allow_escalation: bool
class UnitOrder(AgentIOModel): unit_id: str; incident_id: str; rationale: str
class DiversionRequest(AgentIOModel): unit_id: str; to_incident_id: str; rationale: str
class ContentionFlag(AgentIOModel): unit_id: str; incident_ids: list[str]; rationale: str
class TransportPlan(AgentIOModel): unit_id: str; hospital_ranking: list[str]; rationale: str
class DispatchOutput(AgentIOModel): orders: list[UnitOrder]; diversions: list[DiversionRequest]; contentions: list[ContentionFlag]; transports: list[TransportPlan]
# Hospital Coordination (one agent per hospital)
class StaffMemberView(AgentIOModel): staff_id: str; role: StaffRole; skills: list[str]; load: int; max_patients: int; on_shift: bool; lent: bool
class AdmissionRequest(AgentIOModel): proposal_id: str; incident_id: str; unit_id: str; victims: list[TriagedVictim]; eta_min: float
class HospitalInput(AgentIOModel): t_min: float; hospital_id: str; capabilities: list[Capability]; er_capacity: int; beds_occupied: int; beds_reserved: int; beds_free_unreserved: int; critical_reserve: int; queue_length: int; waiting_for_staff: int; staff: list[StaffMemberView]; requests: list[AdmissionRequest]; allowed_responses: list[Literal["ACCEPT","REJECT","COUNTER"]]; allowed_counter_conditions: list[CounterCondition]
class AdmissionResponse(AgentIOModel): proposal_id: str; response: Literal["ACCEPT","REJECT","COUNTER"]; reason: Literal["OK","NO_BEDS","RESERVE_HELD","NO_DOCTOR","NO_NURSE","NOT_CAPABLE"]; counter_condition: CounterCondition | None; counter_skill: str | None; rationale: str
class HospitalOutput(AgentIOModel): responses: list[AdmissionResponse]
# Resource Allocation
class CaseOption(AgentIOModel): option_id: str; summary: str; triage_weight: float; slack_min: float; travel_min: float; utility: float
class EscalationCase(AgentIOModel): case_id: str; kind: CaseKind; incident_id: str; hospital_id: str | None; summary: str; options: list[CaseOption]
class AllocationInput(AgentIOModel): t_min: float; cases: list[EscalationCase]; float_pool: list[StaffMemberView]
class CaseDecision(AgentIOModel): case_id: str; chosen_option_id: str; rationale: str
class AllocationOutput(AgentIOModel): decisions: list[CaseDecision]

# decision record returned by any backend
class ToolCallTrace(FrozenModel): name: str; arguments: str; output: str
class TokenUsage(FrozenModel): requests: int; input_tokens: int; output_tokens: int
@dataclass(frozen=True)
class AgentDecision:
    output: AgentIOModel
    source: Literal["model", "policy_fallback"]
    error: str | None                 # exception summary when source == "policy_fallback"
    tool_calls: tuple[ToolCallTrace, ...]
    usage: TokenUsage | None
    latency_ms: float | None          # None for offline (keeps logs byte-identical)
```

### 3.10 `domain/ports.py`

```python
class EventSink(Protocol):
    def emit(self, record: LogRecord) -> None: ...      # sink assigns seq
class EnginePort(Protocol):
    @property
    def now(self) -> float: ...
    def view(self) -> PublicView: ...
    def check(self, action: Action) -> ActionCheck: ...
    def apply(self, action: Action, actor: str) -> ActionCheck: ...   # validates, logs ActionRecord, mutates only if ok
class Controller(Protocol):
    def handle(self, point: DecisionPoint, port: EnginePort) -> None: ...
class DecisionBackend(Protocol):
    @property
    def label(self) -> str: ...                          # "offline" | "llm-<model>" | test labels
    def decide(self, agent_name: str, payload: AgentIOModel) -> AgentDecision: ...
```

### 3.11 `domain/outcomes.py`

```python
class VictimOutcome(FrozenModel):
    victim_id: str; incident_id: str; t_report_min: float; condition: str; required_capability: Capability
    initial_triage: Triage; final_triage: Triage; deteriorated_at_min: float | None
    deadline_min: float | None            # absolute; None for BLACK. RED: t_report+window; deteriorated: det_at+30; YELLOW: t_report+window; GREEN: t_report+480
    als_on_scene_min: float | None; loaded_min: float | None; hospital_id: str | None; arrived_hospital_min: float | None
    bedded_min: float | None; treatment_start_min: float | None
    care_start_min: float | None; care_mode: Literal["field", "hospital"] | None
class IncidentOutcome(FrozenModel): incident_id: str; t_report_min: float; mci_cluster: bool; first_unit_on_scene_min: float | None; first_ambulance_on_scene_min: float | None; cleared_min: float | None
class RunOutcomes(FrozenModel): horizon_min: float; victims: tuple[VictimOutcome, ...]; incidents: tuple[IncidentOutcome, ...]; ambulance_busy_min: tuple[tuple[str, float], ...]; responder_busy_min: tuple[tuple[str, float], ...]; ambulances_in_service: int
class RunMetrics(FrozenModel):   # formulas in section 6
    incidents: int; incidents_reached: int; response_time_mean_min: float | None; response_time_p90_min: float | None; red_response_mean_min: float | None
    patients_total: int; victims_served: int; served_rate: float; red_total: int; red_in_window: int; red_in_window_rate: float | None
    unserved_critical: int; deteriorated: int; black_victims: int; survival_weighted_score: float; door_to_doctor_mean_min: float | None
    ambulance_utilisation: float; responder_utilisation: float; er_utilisation: float; doctor_utilisation: float; nurse_utilisation: float
    hospital_full_on_arrival: int; controller_omissions: int
    messages_total: int; messages_by_type: dict[str, int]; conflicts: int; conflicts_by_kind: dict[str, int]; resolutions: dict[str, int]
    agent_calls: int; fallbacks: int; float_staff_lent: int; diversions: int
```

### 3.12 `domain/log_records.py`: the JSONL contract the dashboard consumes (one record per line)

```python
class _Record(FrozenModel): seq: int = 0; t_min: float
class RunStartRecord(_Record): kind: Literal["run_start"] = "run_start"; run_id: str; scenario: str; seed: int; condition: Condition; backend: str; repeat: int; sim_params: dict[str, float]; negotiation: dict[str, int | bool] | None; scenario_data: Scenario   # full truth, analysis only
class SnapshotCounts(FrozenModel): incidents_reported: int; incidents_open: int; waiting_on_scene: int; in_transport: int; queued: int; waiting_for_staff: int; in_treatment: int; treated: int; care_started_cum: int; expectant: int
class SnapshotRecord(_Record): kind: Literal["snapshot"] = "snapshot"; view: PublicView; counts: SnapshotCounts
class SimEventRecord(_Record): kind: Literal["sim_event"] = "sim_event"; event: SimEventType; incident_id: str | None = None; unit_id: str | None = None; hospital_id: str | None = None; staff_id: str | None = None; victim_ids: tuple[str, ...] = (); detail: str = ""
class MessageRecord(_Record): kind: Literal["message"] = "message"; message: Message
class FallbackNote(FrozenModel): scope: Literal["call", "section", "item"]; key: str | None; rule: str; detail: str
class AgentCallRecord(_Record): kind: Literal["agent_call"] = "agent_call"; agent: str; role: AgentRole; triggers: tuple[TriggerKind, ...]; backend: str; input: dict[str, Any]; output: dict[str, Any]; source: Literal["model", "policy_fallback"]; tool_calls: tuple[ToolCallTrace, ...]; fallbacks: tuple[FallbackNote, ...]; usage: TokenUsage | None; latency_ms: float | None
class ActionRecord(_Record): kind: Literal["action"] = "action"; actor: str; action: Action; ok: bool; reason: RejectReason | None; detail: str
class RunEndRecord(_Record): kind: Literal["run_end"] = "run_end"; outcomes: RunOutcomes; metrics: RunMetrics
LogRecord = Annotated[RunStartRecord | SnapshotRecord | SimEventRecord | MessageRecord | AgentCallRecord | ActionRecord | RunEndRecord, Field(discriminator="kind")]
LOG_RECORD_ADAPTER: Final = TypeAdapter(LogRecord)
```

Ordering guarantee: `seq` is strictly increasing and `t_min` is non-decreasing. `run_start` comes first and `run_end` last.

### 3.13 `emergency_sim/eventlog.py` (WP0)

```python
class MemoryEventSink:   # implements EventSink
    def __init__(self) -> None
    @property
    def records(self) -> tuple[LogRecord, ...]
    def emit(self, record: LogRecord) -> None      # record.model_copy(update={"seq": len(self._records)})
def write_jsonl(records: Iterable[LogRecord], path: Path) -> None   # one model_dump_json() per line, parent dirs created
def read_log(path: Path) -> list[LogRecord]                          # LOG_RECORD_ADAPTER.validate_json per line
```

## 4. Simulation semantics (engine, WP1)

### 4.1 Clock and event queue

- Time is float minutes. Every scheduled time is `round(t, 1)`.
- `EventQueue` is a heap of `(t, priority, seq, event)`. `seq` is a global insertion counter, so ties are deterministic.
- Priority at equal `t`, lowest first:
  - **0, resources free:** `TREATMENT_END`, `T0_BED_RELEASE`, `T0_STAFF_RELEASE`, `SHIFT_END`
  - **1:** `BUSY_T0_END`
  - **2:** `ARRIVE_HOSPITAL`, `HANDOVER_DONE`
  - **3:** `ARRIVE_SCENE`, `ACCESS_GRANTED`, `LOAD_DONE`
  - **4:** `INCIDENT_REPORTED`
  - **5:** `DETERIORATION`

`Engine.advance() -> DecisionPoint | None`:
1. **Liveness guard first.** Any unit still `AWAITING_DESTINATION` gets `AssignTransport(nearest hospital capable of all its victims by ETA, reserve_beds=False)` with actor `"engine_guard"`, plus a `CONTROLLER_OMISSION` event. Any unit still `AT_HOSPITAL_FULL` does its handover at the current hospital: bedded where possible, queued otherwise.
2. Before popping an event at time `t_next`, emit a `SnapshotRecord` for every snapshot minute `k` with `k < t_next` and `k <= horizon`. Each snapshot therefore reflects all events and decisions at times `<= k`.
3. Pop every event at `t_next`. If `t_next > horizon`, stop and go to step 5.
4. Collect triggers. If there are any, return `DecisionPoint(t_next, triggers sorted by (kind, ids))`. Otherwise loop.
5. **At horizon:** emit the remaining snapshots up to the horizon (180 from manifest) and return `None`. Events after the horizon are not processed.

Initial events:
- Every incident's report.
- Each `BUSY_T0` unit's end at `busy_until_min`.
- T0 bed releases: hospital `h` frees one bed at `15*k` for `k = 1..er_occupied_t0`.
- T0 staff releases: staff `s` sheds one load at `20*k` for `k = 1..load_t0`.
- Each staff `SHIFT_END`.
- Deterioration for each YELLOW at `t_report + deteriorates_to_red_at_min`.
- `OUT_OF_SERVICE` units are never available.

### 4.2 Ambulance lifecycle (identical crew protocol in all conditions)

`AVAILABLE` → (`DispatchUnit`) `EN_ROUTE_SCENE` → `ON_SCENE` (staged if no access) → `LOADING` → `AWAITING_DESTINATION` → (`AssignTransport`) `TRANSPORTING` → `HANDOVER`, or `AT_HOSPITAL_FULL` then `HANDOVER` → `AVAILABLE` at the hospital's location.

- **Dispatch.** Arrival = `now + turnout(1.0) + geo.travel_time(pos, incident, "ambulance")`.
- **Divert** (only from `EN_ROUTE_SCENE`). The current position is linear interpolation between origin and target by elapsed fraction after turnout (documented approximation). New arrival = `now + travel(pos, new incident)`, with no turnout.
- **On scene arrival:**
  - If the incident is untriaged, START triage happens now. All victims are revealed with their current triage, BLACK becomes `EXPECTANT`, and a `SCENE_TRIAGED` trigger fires.
  - If the unit is ALS, every "cardiac arrest" victim without care gets `care_start = now` (`FIELD_CARE_STARTED`).
  - If the incident is accessible, loading starts. Otherwise the unit waits staged in `ON_SCENE`.
- **Loading rule** (chosen at load start; the chosen victims become `LOADING` so no two units pick the same victim):
  - Candidates are `WAITING` victims sorted by `(TRIAGE_RANK, victim_id)`.
  - First pick: the first candidate the unit may carry. ALS may carry any; BLS may not carry RED.
  - Second pick: the next candidate that is not RED if the first was RED (respect `max_red_per_trip`), within capacity 2, and such that some hospital is capable of both victims.
  - If there is no first pick (for example a BLS where only RED remain), the unit clears immediately: `AVAILABLE` at the incident, `UNIT_FREED`.
  - `LOAD_DONE` comes at `+3.0`. Victims become `LOADED`, the unit becomes `AWAITING_DESTINATION`, and a `PATIENTS_LOADED` trigger fires.
- **`AssignTransport`.** Status becomes `TRANSPORTING`. With `reserve_beds`, reserve `n` beds keyed by `unit_id`. Arrival = `now + travel(pos, hospital)`.
- **Arrival at hospital:**
  - A reservation converts to occupied beds.
  - Otherwise, if free unreserved beds are at least the load size, occupy them.
  - Otherwise the unit goes to `AT_HOSPITAL_FULL` and a `HOSPITAL_FULL_ON_ARRIVAL` trigger fires. The controller may redirect with `AssignTransport` to another hospital in the same decision point. If it does not, the guard does the handover here: as many bedded as possible, the rest queued.
- **Handover.** The unit is busy for `+5.0`, then `AVAILABLE` at the hospital and `UNIT_FREED` fires.
- **Transport is never refused physically** (hospital rejections are advisory). The engine only requires the destination to be capable of every loaded victim.

### 4.3 Responder teams

- Dispatch: turnout 2.0, `"responder"` speed.
- On arrival they add their capabilities to the incident. The first unit of any kind triggers triage.
- **Access rule:** an incident is accessible once every `required_responder_capabilities` entry is present on scene. Access is granted `heavy_access_min` (10) after the last required team arrives if any of {extrication, heavy_rescue, fire_suppression, hazmat} is required, otherwise immediately. `scene_control` gates access with no delay. Hazmat on scene also covers decontamination for chemical exposure.
- Staged ambulances begin loading at `ACCESS_GRANTED`, in unit_id order.
- An incident is `CLEARED` when no victim is `UNTRIAGED`, `WAITING` or `LOADING`. Responders on scene then become `AVAILABLE` at the incident location (`UNIT_FREED`). Staged ambulances with nothing left also clear.

### 4.4 Hospitals: ER beds, queue, staff, treatment

- Beds: `occupied + reserved <= er_capacity` is an invariant. T0 occupants hold beds until their release times.
- `_assign_care(h)` runs on every resource change at `h` (arrival, bed release, treatment end, staff release, staff lent, shift end):
  1. While free unreserved beds > 0 and the queue is non-empty, bed the next patient by `(TRIAGE_RANK, arrival time, victim_id)`.
  2. For each bedded patient waiting for staff, in the same order, find:
     - **Doctor:** at hospital `h` (home or lent), skill = `"emergency"` if the capability is `general`, else the capability; `load < max_patients`; eligible on shift (see holdover below). Choose lowest load, then id.
     - **Nurse:** same rules with skill `"emergency"`. Prefer a nurse whose skill matches `datagen.NURSE_SKILL_FOR.get(cap)`, then lowest load, then id.
     - With both found: `TREATMENT_STARTED`, `care_start = now` (unless field care already set it), loads +1, `TREATMENT_END` at `now + treatment_min(current triage)`.
- **Holdover rule:** a staff member with `t >= shift_end_min` stays eligible for skill `k` only if no colleague of the same role at `h` with skill `k` is still on shift.
- `TREATMENT_END`: the bed is freed, loads -1, the victim becomes `TREATED`.
- **Float pool:** staff with `home == "FLOAT"` can only work after a `LendStaff` (permanent for the run). Lending runs `_assign_care`.
- ICU is not modelled.

### 4.5 Deterioration, BLACK, end-of-horizon

- **Deterioration** fires at `t_report + deteriorates_to_red_at_min` if care has not started and the victim is not BLACK: `triage = RED`, `needs_als = True`, `deadline = now + 30`, `deteriorated_at = now`.
  - Visible (`DETERIORATION` trigger) only if the incident is triaged or the victim is in transport or hospital.
  - If untriaged, it is hidden until triage reveals RED.
  - A YELLOW already loaded in a BLS continues; there is no reroute.
- **BLACK** (expectant): revealed at triage, never loaded, excluded from patients and served metrics, counted in `black_victims`.
- **Horizon (180):** victims without `care_start <= 180` are unserved. Victims in transport or queued are reported in snapshot counts only.

### 4.6 Validation rules (`engine/rules.py`, pure `check_action(state, action) -> ActionCheck`)

| Action | Legal only if |
|---|---|
| `DispatchUnit` | Unit exists and is `AVAILABLE`. Incident reported and `OPEN`. |
| `DivertUnit` | Ambulance, `EN_ROUTE_SCENE`, target `OPEN` and different from the current target. |
| `AssignTransport` | Unit is `AWAITING_DESTINATION` or `AT_HOSPITAL_FULL`. Hospital is capable of every loaded victim. `reserve_beds` requires `beds_free_unreserved >= n`. The full-on-arrival case requires a different hospital. |
| `LendStaff` | Staff home is FLOAT, not yet lent, `t < shift_end_min`; hospital exists. |

Illegal actions are logged as `ActionRecord(ok=False, reason)`, and state is unchanged.

### 4.7 Engine API (`engine/engine.py`)

```python
class Engine:   # satisfies EnginePort
    def __init__(self, scenario: Scenario, params: SimParams, sink: EventSink) -> None
    @property
    def now(self) -> float
    def advance(self) -> DecisionPoint | None
    def view(self) -> PublicView                     # engine/observe.build_public_view(state)
    def check(self, action: Action) -> ActionCheck
    def apply(self, action: Action, actor: str) -> ActionCheck
    def outcomes(self) -> RunOutcomes                # valid after advance() returned None
    def check_invariants(self) -> tuple[str, ...]    # used by tests; empty tuple when consistent
```

Internals:
- `state.py`: mutable dataclasses `UnitState`, `VictimState`, `IncidentState`, `HospitalState`, `StaffState`, `WorldState` and `from_scenario()`. Private to the engine.
- `events.py`: `EventQueue` and frozen `Event`.
- `rules.py`: legality, loading rule, capability helpers.
- `care.py`: beds, queue, staff selection, holdover.
- `observe.py`: `build_public_view`, `build_snapshot_counts`, `build_outcomes`.

Busy-minute accounting runs per unit across `[0, horizon]`. Busy means any status other than `AVAILABLE` and `OUT_OF_SERVICE`; `BUSY_T0` counts as busy.

## 5. Agents, policies, negotiation

### 5.1 Agent roster (4 roles, 6 SDK `Agent`s)

| Agent name | Role | Objective | Read-only tools (`mas/tools.py`) | Output type |
|---|---|---|---|---|
| `incident_assessment` | ASSESSMENT | sickest incidents seen first | `incident_type_profile(incident_type)`, `nominal_deadline(triage, condition)` | `AssessmentOutput` |
| `dispatch` | DISPATCH | send help now, minimise time-to-care | `rank_units_for_incident(incident_id, ambulance_type)`, `rank_hospitals_for_unit(unit_id)` | `DispatchOutput` |
| `hospital_H1`, `hospital_H2`, `hospital_H3` | HOSPITAL | protect `critical_reserve` for RED; only accept patients it can treat on arrival | `capacity_check(proposal_id)` | `HospitalOutput` |
| `resource_allocation` | ALLOCATION | maximise survival-weighted outcome across contested resources | `score_case_options(case_id)` | `AllocationOutput` |

- Every tool is a `@function_tool` taking `ctx: RunContextWrapper[AgentContext]`. `AgentContext` is a frozen dataclass of `(payload: AgentIOModel, params: PolicyParams)`.
- Tools compute from the payload only, via the same pure functions in `policies/estimates.py` that the policies use. They return JSON strings. They have no engine reference, so they are read-only by construction.
- Agents are stateless per call. All context is in the JSON input, which is the same payload for the offline and LLM backends, so both have the same information boundary.

### 5.2 `mas/` design (SDK-specific; WP2)

```python
# mas/specs.py
@dataclass(frozen=True)
class AgentSpec:
    name: str; role: AgentRole; instructions: str
    tools: tuple[FunctionTool, ...]
    input_type: type[AgentIOModel]; output_type: type[AgentIOModel]
    policy: Callable[[AgentIOModel, PolicyParams], AgentIOModel]
    evidence: Callable[[AgentIOModel], tuple[str, dict[str, str]] | None]   # (tool name, args) for the offline model's one tool call
def agent_specs(hospitals: Sequence[HospitalSpec]) -> tuple[AgentSpec, ...]

# mas/policy_model.py
class PolicyModel(Model):
    """Deterministic offline Model: answers with the agent's rule-based policy."""
    def __init__(self, spec: AgentSpec, params: PolicyParams) -> None
    async def get_response(self, system_instructions, input, model_settings, tools, output_schema, handoffs, tracing, *, previous_response_id, conversation_id, prompt) -> ModelResponse
    def stream_response(self, ...) -> AsyncIterator[TResponseStreamEvent]   # raises NotImplementedError (never streamed)
```

`get_response` algorithm:
1. `payload = spec.input_type.model_validate_json(<first user item text>)`. The input is a str, or a list whose first `{'role': 'user'}` item has string `content`.
2. If `spec.evidence(payload)` is not None and no input item has `type == "function_call_output"`, return `ModelResponse(output=[ResponseFunctionToolCall(id=f"{name}-evidence", call_id=f"{name}-evidence", type="function_call", name=tool, arguments=json.dumps(args))], usage=Usage(requests=1), response_id=None)`.
3. Otherwise `text = spec.policy(payload, params).model_dump_json()` and return `ModelResponse(output=[ResponseOutputMessage(id=f"{name}-final", type="message", role="assistant", status="completed", content=[ResponseOutputText(type="output_text", text=text, annotations=[], logprobs=[])])], usage=Usage(requests=1), response_id=None)`.
4. Build these from `openai.types.responses` directly. Do not import `agents.testing` in production code.

The evidence call exists so offline runs exercise the SDK tool loop and record real tool traces. The policy computes the same function, so the result does not depend on it.

```python
# mas/backend.py
class SdkBackend:   # implements DecisionBackend
    def __init__(self, label: str, specs: Sequence[AgentSpec], agents: Mapping[str, Agent[AgentContext]], params: PolicyParams, *, run_config: RunConfig, max_turns: int, fallback_on_error: bool) -> None
    def decide(self, agent_name: str, payload: AgentIOModel) -> AgentDecision
def make_offline_backend(hospitals: Sequence[HospitalSpec], params: PolicyParams) -> SdkBackend
    # Agent(model=PolicyModel(spec)); agents.set_tracing_disabled(True); RunConfig(tracing_disabled=True, workflow_name="ems-offline"); max_turns=4; fallback_on_error=False (bugs must fail loudly)
def make_llm_backend(hospitals: Sequence[HospitalSpec], params: PolicyParams, *, model: str, trace: bool = False, temperature: float | None = None) -> SdkBackend
    # requires OPENAI_API_KEY else BackendUnavailableError; Agent(model=model, model_settings=ModelSettings(temperature=temperature)); RunConfig(tracing_disabled=not trace); max_turns=6; fallback_on_error=True
DEFAULT_LLM_MODEL: Final = os.environ.get("EMS_LLM_MODEL", "gpt-4.1-mini")
```

`decide` behaviour:
- Calls `Runner.run_sync(agent, payload.model_dump_json(), context=AgentContext(payload, params), max_turns=..., run_config=...)`.
- Checks `isinstance(result.final_output, spec.output_type)`.
- Extracts tool traces from `result.new_items` (`ToolCallItem` / `ToolCallOutputItem`) and usage from `result.context_wrapper.usage`.
- Measures `latency_ms` only for the LLM backend.
- When `fallback_on_error` is set, it catches `agents.exceptions.AgentsException` (which covers `ModelBehaviorError` and `MaxTurnsExceeded`), `openai.OpenAIError`, `pydantic.ValidationError` and `TimeoutError`. It then returns `AgentDecision(output=spec.policy(payload), source="policy_fallback", error=f"{type(e).__name__}: {e}"[:300], ...)`.

Instructions (`mas/instructions.py`, one function per role, hospital instructions templated by hospital): about 15 to 25 lines each covering role, objective, inputs, tools, the decision rules (same as section 5.3, in words), hard legality rules, and "rationale at most 200 chars; reply only with the JSON object".

### 5.3 Offline policies (`policies/`, pure, SDK-free; WP2)

All parameters below are fixed a priori from principle. They must not be tuned on evaluation seeds (section 8.3).

```python
@dataclass(frozen=True)
class PolicyParams:
    red_weight: float = 3.0; yellow_weight: float = 2.0; green_weight: float = 1.0
    window_logistic_scale_min: float = 5.0          # p_window(slack) = 1 / (1 + exp(-slack/scale))
    travel_penalty_per_min: float = 0.01            # λ
    nominal_red_deadline_min: float = 40.0          # midpoint of public 20-60 range
    nominal_cardiac_arrest_deadline_min: float = 10.0
    nominal_yellow_deadline_min: float = 165.0
    nominal_green_deadline_min: float = 480.0
    deteriorated_red_deadline_min: float = 30.0
    scene_overhead_min: float = 5.0                 # load + handling estimate
    no_unit_wait_min: float = 20.0                  # assumed wait when an incident gets no unit
    queue_wait_per_patient_min: float = 15.0
    max_ambulances_pre_triage: int = 3              # shared with B0 (fairness)
    diversion_urgency_margin: int = 20
    diversion_max_eta_min: float = 15.0
```

`policies/estimates.py` (shared by policies, tools and `control/cases.py`):
- `eta_min(origin, dest, unit_class, include_turnout)`, a wrapper over `geo.travel_time` plus turnout
- `rank_units(inp, incident_id, ambulance_type)`
- `rank_hospitals(board, unit)`
- `capacity_check(inp, proposal_id) -> CapacityCheck`
- `triage_weight(triage | "SUSPECTED_RED" | "UNKNOWN", params)`
- `nominal_deadline(triage, condition, params)`
- `p_window(slack, params)`
- `utility(weight, slack, travel, params) = weight * p_window(slack) - λ * travel`

**Incident Assessment** (`assess(inp, params) -> AssessmentOutput`):
- **Before triage.**
  - `suspected_red` = the caller report contains any severity cue from a generic lexicon: "not responding", "not breathing", "CPR", "trapped", "badly burned", "collapsed", "blue", "a lot of blood", "not making sense", "drooping", "rubble".
  - `suspected_cardiac_arrest` = type `cardiac_event` and (`"CPR"` or `"not responding"`).
  - urgency = base by type (cardiac_event 70, stroke 70, pediatric_emergency 65, multi_vehicle_collision / building_fire / structural_collapse 60, assault 55, chemical_spill 50, fall 40) + 25 if suspected_red + `min(10, 2*reported_victims)`, clipped to 0..100.
  - `est_patients = reported_victims`.
  - `needed_capabilities` from a public type profile table.
  - `als_units_needed` = 1 if suspected_red, else 0.
  - `deadline_in_min` = nominal (cardiac arrest 10, RED 40, else 165) minus minutes since report.
- **After triage.** Urgency = 100 if any RED is waiting, 60 if YELLOW only, 30 if GREEN only, 0 if none. `est_patients` = waiting non-BLACK count. `als_units_needed` = RED waiting. The deadline uses the earliest nominal deadline among waiting victims; a deteriorated victim gets 30.

**Dispatch** (`dispatch(inp, params) -> DispatchOutput`):
1. **Transports.** For each loaded unit (sorted by id): the hospitals whose capabilities cover all its victims, ranked by `(beds_free_unreserved == 0, eta, hospital_id)`.
2. **Incident order:** `(-urgency, deadline_in_min, incident_id)`. Free units are those `AVAILABLE`.
   - *Changed during development (see `docs/evaluation.md`, F1):* the ALS and seat passes run first for scenes that are ready and then for scenes that are still blocked. A scene is ready when it is accessible, or when `access_wait_min` is known and no longer than the ETA of the nearest free ambulance. Urgency order is kept within each group.
3. **ALS pass.** For incidents with `als_needed - als_en_route > 0`, compute each one's nearest free ALS.
   - If `allow_escalation` and two or more incidents share the same nearest free ALS: emit `ContentionFlag(unit, incidents)` and withhold that unit from greedy assignment. RA decides.
   - Otherwise assign greedily in priority order.
   - If no ALS is free and `allow_diversion`: pick an ALS `EN_ROUTE_SCENE` to an incident with urgency at most `this - diversion_urgency_margin`, `suspected_red == False` and `red_waiting == 0`, with ETA from its current position at most `diversion_max_eta_min`. Emit `DiversionRequest`.
4. **Seat pass.** `seats_short = est_patients_remaining - seats_en_route`. Before triage, cap total ambulances at `max_ambulances_pre_triage`. Fill nearest BLS first. Use an ALS only if no incident in this round still has an unmet ALS need.
5. **Responder pass.** For each missing responder capability, send the nearest free team that has it (one team may cover several). *Changed during development (D2, see `docs/evaluation.md`):* incidents whose missing capabilities can all be covered by free teams are served first, in priority order, because sending those teams opens the scene.

B1 runs the same function with both flags false, which gives pure greedy-by-urgency.

**Hospital Coordination** (`hospital(inp, params) -> HospitalOutput`). Proposals are processed in `proposal_id` order with a running bed count:
```
n = len(victims); has_red = any RED
if any victim capability not in capabilities: REJECT NOT_CAPABLE
elif free < n: REJECT NO_BEDS
elif not has_red and free - n < critical_reserve:
     COUNTER RESERVE_RELEASE if allowed (and staff ok) else REJECT RESERVE_HELD
elif a needed doctor skill has 0 eligible slots: COUNTER FLOAT_DOCTOR(skill) if allowed else REJECT NO_DOCTOR
elif nurse slots < n: COUNTER FLOAT_NURSE if allowed else REJECT NO_NURSE
else ACCEPT (free -= n)
```
Doctor slots = Σ over on-shift (or holdover) doctors with that skill of `(max_patients - load)`.

**Resource Allocation** (`allocate(inp, params) -> AllocationOutput`):
- Cases are processed by descending best-option utility, then `case_id`.
- Pick the option with the highest `utility`, ties broken by `option_id`.
- For `FLOAT_STAFF`: skip `GRANT` options whose staff was already granted in this output, falling back to `DECLINE`. GREEN-only requests are always declined.
- `RESERVE_RELEASE`: approve iff `beds_free_after - pending_red_demand(h) >= 0`. The coordinator precomputes `pending_red_demand` into the options' utilities: approve utility = `w * p_window(slack_here)`; decline utility = `w * p_window(slack_alternative) + red_weight * min(1, demand)`.

### 5.4 Coordinator protocol per decision point (`control/coordinator.py`; WP3)

`AgentCoordinator(backend: DecisionBackend, config: NegotiationConfig, sink: EventSink, params: PolicyParams)` implements `Controller`.

Belief state:
- `_board: dict[incident_id, PriorityReport]`
- `MessageBus` (message ids `M-0001`, threads `NEG-0001`, cases `CASE-0001`)

Every agent call is logged as `AgentCallRecord`. Every message gets `t_min = point.t_min`.

**Phase A: assessment.**
- Run only if triggers include `NEW_INCIDENT`, `SCENE_TRIAGED`, `DETERIORATION` or (changed during development, F4) `PATIENTS_LOADED` for open incidents.
- One call carries all such incidents.
- Validate (item level: missing, duplicate or unknown incident, urgency outside 0..100, negative counts → replace with the policy's item).
- Update `_board`. Send `PRIORITY_REPORT` (incident_assessment → dispatch) per incident.

**Phase B: dispatch.**
- Call only if (some open incident has an unmet need and a free unit exists, or diversion is possible) or some unit is `AWAITING_DESTINATION`.
- Validate (section level). Sections are orders, diversions, contentions and transports.
  - Any violation in a section (busy or unknown unit, duplicate unit, flag disallowed by config, ranking with a non-capable hospital, missing or extra loaded unit) replaces that section with the policy's section.
  - Re-check cross-section unit conflicts. If any remain, use the policy's whole output.
- Apply orders through `port.apply(DispatchUnit)` with actor `dispatch`. Send `INFORM` (dispatch → all, `DispatchAction`) per incident.
- If contentions or diversions are allowed (Full), `cases.unit_contention_cases(...)` builds `UNIT_CONTENTION` cases.
  - Each option is a complete assignment, for example "AMB-03 → INC-09; INC-07 keeps none / gets AMB-04".
  - Option utility = Σ over affected incidents of `utility(w_i, slack_i, eta_i)`. An incident left without a unit uses `no_unit_wait_min`.
  - *Changed during development (F2):* `slack_i = deadline_i − max(eta_i, access_wait_i) − overhead`. Travel and responder work overlap. When the access wait is unknown, it is `no_unit_wait_min`.
  - Send `ESCALATE` (dispatch → resource_allocation, `ResolveAction`), then run RA, then `COMMIT` (RA → dispatch, `CommitAction` with the option's actions, `RA_COMMIT` or `KEPT_ASSIGNMENT`), then apply.

**Phase C: transport negotiation**, rounds `r = 1..config.max_rounds` (3). Each loaded unit owns one thread and a pointer into its ranking.
1. Group pending units by `ranking[pointer]`. For each hospital (sorted):
   - Send `PROPOSE` (dispatch → hospital_Hx, `AdmitAction` with ETA).
   - Make one call per hospital per round with all its requests. Validate per item: unknown or missing proposal, response or condition not in `allowed_*`, `counter_skill` inconsistent → policy item.
   - **ACCEPT:** `port.apply(AssignTransport(reserve_beds=True))` with actor `hospital_Hx`. If the engine rejects (beds raced), log `ENGINE_REJECTED`, send `INFORM`, and advance the pointer. Otherwise send `ACCEPT` (hospital → dispatch), resolution `ACCEPTED`, or `ACCEPTED_AFTER_REJECT` if `pointer > 0`.
   - **REJECT:** send `REJECT` and advance the pointer.
   - **COUNTER** (Full only): send `COUNTER` (hospital → dispatch, `ConditionalAdmitAction`). Then for FLOAT_* send `REQUEST` (hospital → resource_allocation, `StaffRequestAction`); for RESERVE_RELEASE send `ESCALATE` (hospital → resource_allocation). This creates a case.
   - Refresh `port.view()` after each hospital, because reservations change.
2. **Resolve cases** (Full) with one RA call per round:
   - Cases with only one feasible option (for example no float staff with that skill) are auto-resolved without a call and logged with `rationale="no feasible alternative"`.
   - **Grant:** apply `LendStaff` (if float) and `AssignTransport(reserve_beds=True)`. Send `COMMIT` (RA → hospital and RA → dispatch), `COUNTER_GRANTED`.
   - **Decline:** send `REJECT` (RA → hospital), `COUNTER_DECLINED`, and advance the pointer.
   - If the engine rejects a commit, treat it as declined and log it.
3. **After the rounds, units still without a destination:**
   - **Full** (`allow_escalation`): open a `NO_ACCEPTING_HOSPITAL` case. Options are the capable hospitals, with utility from slack = deadline − (ETA + estimated wait). Estimated wait is 0 if beds and staff are free, else `queue_wait_per_patient_min * (queue + 1)`. RA picks; `COMMIT` with `AssignTransport(reserve_beds = beds available)`, `RA_COMMIT`.
   - **B1:** `AssignTransport(ranking[0], reserve_beds=False)` plus `INFORM` (dispatch → hospital), `FORCED_ADMISSION`.

`HOSPITAL_FULL_ON_ARRIVAL` is ignored by the agent conditions (only forced admissions can hit it); those patients queue.

Configs (`control/conditions.py`):
```python
@dataclass(frozen=True)
class NegotiationConfig: allow_counter: bool; allow_float_staff: bool; allow_escalation: bool; allow_diversion: bool; max_rounds: int = 3
NEGOTIATION: Final[Mapping[Condition, NegotiationConfig]] = MappingProxyType({
  Condition.B1: NegotiationConfig(False, False, False, False),
  Condition.FULL: NegotiationConfig(True, True, True, True),
  Condition.FULL_NO_FLOAT: NegotiationConfig(True, False, True, True),
  Condition.FULL_NO_DIVERSION: NegotiationConfig(True, True, True, False),
  Condition.FULL_NO_ESCALATION: NegotiationConfig(True, True, False, False)})
```
`allowed_responses` is `["ACCEPT", "REJECT", "COUNTER"]` if `allow_counter`, else `["ACCEPT", "REJECT"]`. `allowed_counter_conditions` contains FLOAT_* only if `allow_float_staff`, plus RESERVE_RELEASE if `allow_counter`.

`control/needs.py` (shared by B0 and the observations builder so accounting never diverges): `incident_needs(view, board | None, params) -> list[IncidentNeed]`.
- Seats en route = Σ capacity of assigned ambulances in `EN_ROUTE_SCENE` or `ON_SCENE`.
- `als_en_route` likewise for ALS.
- `red_waiting` = triaged RED `WAITING`.
- `als_needed` = `red_waiting` if triaged, else the board's `als_units_needed` (B0: 0 before triage).
- Responder caps missing = required − present − en route.
- `access_wait_min` (added during development, F1) is the estimated wait until loading may start:
  - 0 if the scene is accessible;
  - None if a required capability is neither present nor en route;
  - otherwise the latest ETA of the en-route teams, plus `heavy_access_min` when a heavy capability is required.

### 5.5 B0 greedy baseline (`control/greedy.py`; competent and on the same information)

`GreedyController(params: PolicyParams)`. Actor is `b0_greedy`; there are no messages and no agent calls. On every decision point:
1. **Incidents in FCFS order** `(t_report, id)`:
   - responders: the nearest free team per missing capability;
   - before triage: send the nearest free ambulances of any type until `min(ceil(reported/2), max_ambulances_pre_triage)` are assigned;
   - after triage: for each RED waiting without an ALS en route, the nearest free ALS; then fill seats with the nearest free unit of any type.
2. **Loaded units:** the nearest hospital capable of all victims with `beds_free_unreserved > 0`, else the nearest capable one; `reserve_beds=False`.
3. **`AT_HOSPITAL_FULL` units:** redirect to the next nearest capable hospital with free beds. If there is none, do nothing, so the patients queue.

B0 never diverts, never reserves, never lends staff, ignores severity cues and never sees staff loads.

### 5.6 Where gains are expected to come from (hypotheses, not promises)

**B0 → B1** (communication and coordination):
- Urgency-ordered dispatch instead of FCFS.
- ALS matched to suspected RED and BLS to the rest, so fewer BLS trips are wasted on RED scenes.
- Pre-alert and bed reservation, so arrival-to-full does not happen.
- Staff-aware admission: the hospital rejects when it has no doctor slot for that skill, which sends trauma patients to H2 when H1's single trauma doctor is saturated.
- Critical reserve keeps capable beds for RED.

**B1 → Full** (negotiation):
- Contested ALS resolved by window-feasibility utility rather than raw urgency. A cardiac arrest 15 minutes away cannot be saved, while a 40-minute trauma RED can.
- Diversion of an en-route ALS from low-acuity calls.
- Float staff lent through COUNTER → REQUEST → COMMIT, which adds treatment capacity where queues form (cluster near H1).
- Reserve release when no RED is expected, which avoids needless long transports for YELLOW.
- RA picks the destination for no-accept cases by expected care start rather than defaulting to the nearest.

**Possible costs (must be reported):**
- Holding reserve or diverting can hurt YELLOW and GREEN times and the mean response time.
- B1 rejections can lengthen transports.

Expect the largest differences in `surge` and `constrained`, and small or no differences in `normal`. That is an honest finding, not a failure.

## 6. Metrics (`evaluation/metrics.py`; WP4). Exact formulas

Let `P` = non-BLACK victims, `H` = horizon, `R = {v in P : final_triage == RED}` (includes deteriorated).

- `victims_served = |{v in P : care_start <= H}|`; `served_rate = victims_served / |P|`.
- `red_total = |R|`; `red_in_window = |{v in R : care_start <= deadline_v}|`; `red_in_window_rate = red_in_window / red_total` (None if 0).
- `unserved_critical = |{v in R : care_start is None or care_start > H}|`.
- `response_time` per incident = `first_ambulance_on_scene - t_report`, over reached incidents.
  - `response_time_mean_min` = mean.
  - `response_time_p90_min = numpy.percentile(values, 90, method="linear")`.
  - `incidents_reached` = count.
- `red_response_mean_min` = mean of `als_on_scene - t_report` over initially-RED victims with ALS arrival.
- `door_to_doctor_mean_min` = mean of `treatment_start - arrived_hospital`.
- `survival_weighted_score = Σ_{v in P} W(final_triage) * 1[care_start <= deadline_v] / Σ_{v in P} W(final_triage)`, where `W = {RED: 3, YELLOW: 2, GREEN: 1}`. These scoring weights are defined in `metrics.py`, independent of `PolicyParams`.
- `ambulance_utilisation = Σ_a busy_min(a) / (ambulances_in_service * H)`; `responder_utilisation` likewise over the 6 teams.
- `er_utilisation` = mean over snapshots `k = 0..H-1` of `Σ_h beds_occupied / Σ_h er_capacity`.
- `doctor_utilisation` = mean over snapshots of `Σ load / Σ max_patients` over doctors eligible at `k` (on shift or holdover, at a hospital). Same for `nurse_utilisation`.
- `hospital_full_on_arrival` and `controller_omissions` are counts of those `SimEventType`s. `diversions` = count of `UNIT_DIVERTED`; `float_staff_lent` = count of `STAFF_LENT`.
- `messages_by_type` counts per `MessageType`.
- `conflicts` = number of distinct `conversation_id`s containing at least one REJECT, COUNTER or ESCALATE.
  - `conflicts_by_kind` keys: `BED_REJECTION` (REJECT only), `FLOAT_STAFF` / `RESERVE_RELEASE` / `UNIT_CONTENTION` / `NO_ACCEPTING_HOSPITAL` (case kinds), and `FLOAT_STAFF_COMPETITION` (at least two FLOAT_STAFF requests in the same round sharing a candidate staff member).
  - `resolutions` counts the `Resolution` of each thread's closing message.
- `agent_calls` = count of `AgentCallRecord`. `fallbacks` = calls with `source == "policy_fallback"` or non-empty `fallbacks`.

API:
```python
def compute_metrics(outcomes: RunOutcomes, records: Sequence[LogRecord]) -> RunMetrics
class KpiSnapshot(FrozenModel): t_min: float; served: int; red_in_window: int; red_missed: int; unserved_critical_overdue: int; conflicts: int; messages: int; mean_response_min: float | None
def kpis_at(outcomes: RunOutcomes, records: Sequence[LogRecord], t: float) -> KpiSnapshot   # thresholds every time at t; used by dashboard
```

## 7. Evaluation runner, report, CLI (WP4)

```python
# evaluation/runner.py
BackendKind = Literal["offline", "llm"]
@dataclass(frozen=True)
class RunSpec: scenario: str; seed: int; condition: Condition; backend: BackendKind = "offline"; model: str | None = None; repeat: int = 0
@dataclass(frozen=True)
class RunResult: spec: RunSpec; metrics: RunMetrics; log_path: Path | None
def build_controller(condition: Condition, backend: DecisionBackend | None, sink: EventSink, params: PolicyParams) -> Controller
def run_simulation(spec: RunSpec, *, sim_params: SimParams = SimParams(), policy_params: PolicyParams = PolicyParams(), backend: DecisionBackend | None = None, log_path: Path | None = None) -> RunResult
def run_grid(specs: Sequence[RunSpec], out_dir: Path, *, keep_logs_for_seeds: frozenset[int]) -> list[RunResult]
def run_dir(out_dir: Path, spec: RunSpec, backend_label: str) -> Path   # out_dir/runs/{scenario}/{condition}/{backend_label}/seed-{seed}_r{repeat}/
```

`run_simulation` steps:
1. `scenario_for`, then `MemoryEventSink`.
2. `RunStartRecord` with deterministic `run_id = f"{scenario}-{condition}-{label}-s{seed}-r{repeat}"`.
3. `Engine`, then the controller.
4. `while (p := engine.advance()) is not None: controller.handle(p, engine)`.
5. `compute_metrics`, then `RunEndRecord`.
6. Write `metrics.json` always, and `events.jsonl` if `log_path`.

```python
# evaluation/report.py
HIGHER_IS_BETTER: Final = frozenset({"served_rate","victims_served","red_in_window","red_in_window_rate","survival_weighted_score"})
def summarise(results: Sequence[RunResult]) -> pd.DataFrame        # scenario, condition, backend, metric, n, mean, std(ddof=1), min, max
def paired_differences(results, pairs=((FULL,B0),(FULL,B1),(B1,B0))) -> pd.DataFrame   # per scenario & metric: n, mean_diff, std_diff, wins, ties, losses, sign_test_p (two-sided binomial via math.comb)
def write_report(results, out_dir: Path) -> None                     # summary.csv, paired.csv, summary.md (headline table)
def load_results(out_dir: Path) -> list[RunResult]                   # from metrics.json files
```

CLI (`python -m emergency_sim.cli`, argparse, exits non-zero on error):
```
run    --scenario {normal,surge,constrained} --condition {b0,b1,full,full-no-float,full-no-diversion,full-no-escalation} [--seed 7380] [--backend offline|llm] [--model NAME] [--repeat 0] [--out results]
grid   [--scenarios ...] [--conditions b0 b1 full] [--seeds 7380 1001-1020] [--backend offline] [--model NAME] [--repeats 1] [--keep-logs-for 7380] [--out results]
report [--results results]
```

## 8. Work packages (parallel), ownership, acceptance

Order: **WP0** (alone, about 45 to 60 min) → **WP1 through WP5 in parallel** → **WP6** integration.

No two packages edit the same file. A contract change goes through the lead (WP0/WP6 owner), who edits `domain/*` and notifies everyone.

All commands run from the project root with `.venv/bin/python`.

### WP0: Contracts and foundations (lead, first)

- **Files:** `pyproject.toml`, `requirements.txt`, `emergency_sim/domain/*`, `emergency_sim/scenarios.py`, `emergency_sim/eventlog.py`, `tests/conftest.py`, `tests/factories.py`, `tests/test_contracts.py`, `tests/test_scenarios.py`, `tests/test_architecture.py`.
- **`tests/factories.py`:** tiny-scenario builders used by everyone:
  - `make_scenario(*, hospitals, ambulances, staff, responders, incidents, horizon=180.0)`
  - `hospital(id="H1", x=5.0, y=5.0, caps=("general",), er_capacity=4, occupied=0, reserve=1)`
  - `ambulance(id, type="ALS", x, y, status="available", busy_until=0)`
  - `doctor(id, home, skills, load=0, shift_end=300)`, `nurse(...)`, `responder(id, caps, x, y)`
  - `victim(vid, triage, capability="general", window=None, deteriorates=None, condition="lacerations")`
  - `incident(iid, t, x, y, victims, reported=None, responders=(), caller_report=...)`
- **Acceptance:**
  - `.venv/bin/python -m pytest tests/test_contracts.py tests/test_scenarios.py tests/test_architecture.py -q` passes.
  - Tests check that all three scenarios load with counts matching `summary.json` (normal 11/21, surge 14/39, constrained 14/39).
  - `scenario_for("surge", 1)` is deterministic across two calls.
  - Every `LogRecord` variant round-trips through `write_jsonl`/`read_log`.
  - `Message` requires all fields.
  - `.venv/bin/python -m emergency_sim.validate` still prints 56/56. It rewrites `data/validation_report.json` with identical content.

### WP1: Engine (strongest implementer; largest risk)

- **Files:** `emergency_sim/engine/*`, `tests/test_engine_*.py`, `tests/test_public_view.py`.
- **Depends on:** WP0.
- **Tests** (hermetic, tiny factory scenarios plus real data):
  - dispatch arrival time = turnout + `geo.travel_time`;
  - BLS refuses RED;
  - ALS takes 1 RED plus 1 non-RED;
  - mixed load only if a common capable hospital exists;
  - cardiac arrest field care at ALS arrival;
  - responder access gating (+10) and triage by the first unit;
  - reservation legality and conversion;
  - full-on-arrival trigger, then guard queueing;
  - queue priority RED first;
  - doctor skill matching and holdover rule;
  - T0 bed and staff releases;
  - deterioration (visible vs hidden);
  - BLACK never loaded;
  - illegal actions leave state unchanged (deep compare of `view()`);
  - `check_invariants()` empty after every `advance()` in a full surge run with a trivial controller (nearest-unit stub defined in the test);
  - horizon accounting;
  - `test_public_view.py`: the serialized view never contains the hidden keys and untriaged incidents have `victims == ()`;
  - determinism: two runs with the same controller give byte-identical JSONL.
- **Acceptance:** `.venv/bin/python -m pytest tests/test_engine_lifecycle.py tests/test_engine_hospital.py tests/test_engine_rules.py tests/test_public_view.py tests/test_engine_determinism.py -q`

### WP2: Policies and SDK agents

- **Files:** `emergency_sim/policies/*`, `emergency_sim/mas/*`, `tests/test_policies.py`, `tests/test_estimates.py`, `tests/test_strict_schemas.py`, `tests/test_mas_offline_runner.py`, `tests/test_mas_backend_fallback.py`.
- **Depends on:** WP0.
- **Tests:**
  - **Policy unit cases:** severity keywords; post-triage urgency; Dispatch ALS-to-RED; contention flag only when `allow_escalation`; diversion only when `allow_diversion`; seat cap; hospital ACCEPT / REJECT RESERVE_HELD in B1 vs COUNTER RESERVE_RELEASE in Full; COUNTER FLOAT_DOCTOR(skill); running bed count across two proposals; RA picks max utility with id tie-break and never double-grants staff.
  - **`test_strict_schemas.py`:** `AgentOutputSchema(T)` (strict) builds for all four output types.
  - **`test_mas_offline_runner.py`:** `monkeypatch.delenv("OPENAI_API_KEY", raising=False)`. For each of the 6 agents run `make_offline_backend(...).decide(name, payload)`. Assert the output equals `policy(payload)`, `source == "model"`, exactly one tool trace with the expected tool name, and `usage.requests == 2`. Also assert directly through `Runner.run_sync` that `result.new_items` types are `[ToolCallItem, ToolCallOutputItem, MessageOutputItem]` and `len(result.raw_responses) == 2`.
  - **`test_mas_backend_fallback.py`:** build an `SdkBackend` with `fallback_on_error=True` where an agent's model is `agents.testing.ScriptedModel` returning, in turn:
    - (a) `[assistant_message("not json")]` → `source == "policy_fallback"`, `"ModelBehaviorError" in error`;
    - (b) a raised `openai.APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))` → fallback;
    - (c) valid JSON → `source == "model"`.
  - Also test `make_llm_backend` raises `BackendUnavailableError` without a key.
- **Acceptance:** `env -u OPENAI_API_KEY .venv/bin/python -m pytest tests/test_policies.py tests/test_estimates.py tests/test_strict_schemas.py tests/test_mas_offline_runner.py tests/test_mas_backend_fallback.py -q`

### WP3: Control (B0, coordinator, negotiation, validation)

- **Files:** `emergency_sim/control/*`, `tests/test_needs.py`, `tests/test_greedy.py`, `tests/test_validation.py`, `tests/test_coordinator.py`.
- **Depends on:** WP0. Imports WP2's `policies` (signatures fixed in 5.3). Start with `needs.py` and `greedy.py`; they need only the domain.
- **Test doubles:** `FakePort` (in-memory, implements `EnginePort` over a hand-built `PublicView` and records applied actions; can be told to reject) and `PolicyBackend` (implements `DecisionBackend` by calling policies directly; no SDK).
- **Tests:**
  - B0 emits no messages, uses FCFS and nearest unit, redirects on full arrival, never reserves;
  - B1 messages contain no COUNTER, ESCALATE, REQUEST or COMMIT;
  - Full: hospital at reserve → COUNTER → ESCALATE → COMMIT with `COUNTER_GRANTED`;
  - float REQUEST → `LendStaff` + `AssignTransport`;
  - two hospitals competing for one float doctor → one granted, one declined;
  - max 3 rounds respected;
  - all-reject → B1 `FORCED_ADMISSION` vs Full `RA_COMMIT`;
  - message ids sequential and every message has `in_reply_to` set for replies;
  - validation replaces an illegal Dispatch section (busy unit) with the policy section and records a `FallbackNote`;
  - an engine-rejected ACCEPT gives `ENGINE_REJECTED` and the next hospital.
- **Acceptance:** `.venv/bin/python -m pytest tests/test_needs.py tests/test_greedy.py tests/test_validation.py tests/test_coordinator.py -q`

### WP4: Evaluation, metrics, CLI

- **Files:** `emergency_sim/evaluation/*`, `emergency_sim/cli.py`, `tests/test_metrics.py`, `tests/test_report.py`, `tests/test_cli.py`.
- **Depends on:** WP0. The runner wiring compiles against the WP1/WP2/WP3 public names in this plan.
- **Tests:**
  - `compute_metrics` on hand-built outcomes and records gives exact values for every formula (including p90, deteriorated deadlines, BLACK exclusion and conflict and resolution counting);
  - `kpis_at` is monotone in `t`;
  - `summarise`/`paired_differences` on synthetic `RunResult`s (std ddof=1, sign test p for 5/0/0 = 0.0625);
  - CLI parses `--seeds 7380 1001-1003` into `[7380, 1001, 1002, 1003]`;
  - `run` and `grid` are tested with a monkeypatched `run_simulation`, so no engine is needed.
- **Acceptance:** `.venv/bin/python -m pytest tests/test_metrics.py tests/test_report.py tests/test_cli.py -q`

### WP5: Dashboard (Streamlit + Plotly)

- **Files:** `dashboard/__init__.py`, `dashboard/replay.py`, `dashboard/figures.py`, `dashboard/app.py`, `tests/test_dashboard_*.py`.
- **Depends on:** WP0 (log contract). Uses `evaluation.metrics.kpis_at` and `evaluation.report.load_results` (signatures fixed above); stub them locally in tests until WP4 lands.

`replay.py` (pure, cached in the app with `st.cache_data`):
```python
@dataclass(frozen=True)
class RunRef: scenario: str; condition: str; backend: str; seed: int; repeat: int; path: Path
@dataclass(frozen=True)
class RunLog: start: RunStartRecord; snapshots: tuple[SnapshotRecord, ...]; events: tuple[SimEventRecord, ...]; messages: tuple[MessageRecord, ...]; agent_calls: tuple[AgentCallRecord, ...]; actions: tuple[ActionRecord, ...]; end: RunEndRecord | None
def discover_runs(results_dir: Path) -> list[RunRef]
def load_run(path: Path) -> RunLog
def snapshot_at(run: RunLog, t: float) -> SnapshotRecord          # latest with t_min <= t
def messages_until(run: RunLog, t: float) -> list[Message]
@dataclass(frozen=True)
class ThreadSummary: conversation_id: str; incident_id: str; kind: str; contested_resource: str; messages: tuple[Message, ...]; resolution: str | None; t_min: float
def negotiation_threads(run: RunLog) -> list[ThreadSummary]
@dataclass(frozen=True)
class AgentStatus: agent: str; role: str; calls: int; last_call_t: float | None; last_triggers: tuple[str, ...]; last_decision: str; fallbacks: int; pending: tuple[str, ...]
def agent_status_at(run: RunLog, t: float) -> list[AgentStatus]
```
`pending` lists operational items at `t`:
- Dispatch: incidents with unmet needs and loaded units without a destination.
- Hospital: inbound reserved units and queued patients.
- RA: open cases.

Protocol-level pending (unanswered PROPOSE, REQUEST or ESCALATE) is shown in a "step through decision point" view ordered by `seq`.

`figures.py` (pure, returns `go.Figure`):
- `incident_map(snapshot, start)`: schematic 20×20 km grid with avenue and street ticks from `geo`.
  - Hospitals: squares coloured by bed use. Stations: triangles.
  - Incidents: grey if untriaged, else worst known triage colour; size = reported victims; `x` marker when cleared.
  - Units: ALS/BLS/responder symbols, with lines to `destination_id`.
- `ambulance_timeline(run)`: `px.timeline` of unit statuses.
- `er_occupancy(snapshot)`: stacked occupied / reserved / free / queue per hospital.
- `staff_load(snapshot)`: doctor and nurse load vs capacity per hospital, plus lent float staff.
- `resource_timeseries(run)`
- `kpi_comparison(summary_df, metric)`: bars with error bars (std) by condition per scenario.
- `cumulative_served(runs_by_condition)`
- `resolution_breakdown(threads)`

`app.py` layout:
- **Sidebar:** results dir, scenario, seed, backend; condition A plus optional condition B (compare mode); time slider 0..horizon with step 1; optional autoplay via `st.fragment(run_every=...)`.
- **Tabs:**
  1. Map and incidents
  2. Resources (ambulances, responders, doctors, nurses, ER)
  3. Agent activity (status cards, recent messages, pending, last decisions with rationale, tool calls, fallbacks)
  4. Negotiation history (thread table with contested resource, PROPOSE/ACCEPT/REJECT/COUNTER/ESCALATE/COMMIT sequence and resolution, plus a resolution breakdown chart)
  5. KPIs (cards at `t` and final; side-by-side conditions; cross-seed comparison from `results/summary.csv`)
- Graceful empty state if no runs are found.

Tests:
- Build a small fixture log in-test from domain models (no engine).
- `snapshot_at`, `negotiation_threads` and `agent_status_at` on the fixture.
- Each figure builder returns a `go.Figure` with the expected trace counts.
- `AppTest.from_file("dashboard/app.py")` with `RESULTS_DIR` env pointing to a `tmp_path` holding the fixture runs: `.run()` has no exceptions and the tabs render.

Acceptance: `.venv/bin/python -m pytest tests/test_dashboard_replay.py tests/test_dashboard_figures.py tests/test_dashboard_app.py -q`

### WP6: Integration, results, docs (lead, after WP1 to WP5)

1. Wire and fix mismatches. Add `tests/test_end_to_end.py`:
   - all 3 scenarios × {b0, b1, full}, seed 7380, offline;
   - every run finishes with `check_invariants` clean, `fallbacks == 0` and `controller_omissions == 0` (or explained);
   - B0 has 0 messages and 0 agent calls; B1 has no COUNTER or ESCALATE;
   - every message type appears at least once across the Full runs;
   - two identical runs give byte-identical `events.jsonl`;
   - `read_log` parses everything.
   Mark the 21-seed grid test `@pytest.mark.slow`.
2. Add `tests/test_llm_fallback_integration.py`: a Full run on a tiny factory scenario where the dispatch agent uses a `ScriptedModel` producing a schema-valid but illegal order (a busy unit) and the hospital uses malformed JSON. Assert the run completes, the engine state is consistent, the `AgentCallRecord`s show `FallbackNote`s / `policy_fallback`, and there are no double dispatches.
3. Run the evaluation:
   ```
   env -u OPENAI_API_KEY .venv/bin/python -m emergency_sim.cli grid --scenarios normal surge constrained --conditions b0 b1 full full-no-float full-no-diversion full-no-escalation --seeds 7380 1001-1020 --backend offline --keep-logs-for 7380 --out results
   .venv/bin/python -m emergency_sim.cli report --results results
   ```
   Check determinism with a second run into `results_check/`, then `diff -r results/runs results_check/runs` (expect no output). Delete `results_check/` afterwards.
4. Dashboard smoke: the AppTest suite, plus a manual `.venv/bin/streamlit run dashboard/app.py --server.headless true --server.port 8599` started as a background process, checked with `curl -s localhost:8599/_stcore/health` (expect `ok`), then stopped. macOS has no `timeout` command.
5. Docs:
   - `README.md`: quickstart, architecture diagram, how to run offline and LLM (`OPENAI_API_KEY`, `EMS_LLM_MODEL`, `--backend llm --repeats 3`), results headline table from `results/summary.md`, honesty notes.
   - `docs/simulation_design.md`: a "v2 changes" section with section 1 of this plan.
   - `docs/evaluation.md`: method, dev vs eval seeds, mean ± std, paired W/T/L, sign-test p, ablation attribution, and an honest discussion including where Full loses.
   - Update `docs/system_flow.mmd`; append to `prompts.md`.
6. Final: `.venv/bin/python -m pytest -q` (all, excluding `slow`), then `.venv/bin/python -m pytest -q -m slow`, then `.venv/bin/python -m emergency_sim.validate`.

### 8.1 Cut line if time runs short (priority order)

1. WP0, WP1, `greedy.py`, metrics and runner, giving B0 numbers.
2. WP2 and the coordinator, giving B1/Full offline.
3. Grid and report.
4. Dashboard core: map, resources, KPIs.
5. Dashboard agent and negotiation tabs.
6. Ablation conditions and docs polish.

### 8.2 Expected size

| Package | Approx. LOC |
|---|---|
| engine | 900 |
| policies | 450 |
| mas | 400 |
| control | 700 |
| evaluation + cli | 400 |
| dashboard | 600 |
| tests | 1,500 |

### 8.3 Honesty protocol (must be in the docs)

- **Dev seeds** 1 to 5: the only seeds allowed for debugging and any parameter change; every change is logged in `docs/evaluation.md`.
- **Eval seeds** 7380 and 1001 to 1020: fixed now, before any results.
- Report all scenarios, all eval seeds and all metrics, including the ones that get worse. Use paired comparisons on identical worlds.
- Offline results measure the coordination protocol with rule-based agents inside real SDK `Agent`/`Runner`. The LLM backend is implemented and tested with fake models only; it was not run because there was no key. Say this explicitly.

## 9. Testing strategy (summary)

- pytest, hermetic: no network, no API key (tests `delenv OPENAI_API_KEY`), `tmp_path` for all files.
- Deterministic: no wall clock; offline `latency_ms=None`; sorted iteration.
- **Layers:**
  - contract round-trips;
  - architecture import rules (AST);
  - engine invariants, lifecycle and rules on tiny factory scenarios;
  - information-boundary tests (no hidden keys in `PublicView`);
  - pure policy tests;
  - strict-schema tests;
  - offline backend through `Runner` (tool loop and structured output verified);
  - LLM-fallback tests with `agents.testing.ScriptedModel` (malformed JSON, raised `OpenAIError`, legal-schema but illegal decision → coordinator validation fallback and engine rejection);
  - coordinator protocol tests with `FakePort` and `PolicyBackend`;
  - exact metric formulas;
  - dashboard pure functions plus `AppTest`;
  - end-to-end determinism; slow-marked grid.

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Full shows little or no gain, or B1 < B0 | Mechanisms are principled, not seed-tuned. Report honestly with paired stats and ablations. Changes only on dev seeds 1 to 5. A null result in `normal` is expected and explained. |
| Engine bugs or deadlocks (units stuck, victims lost) | `check_invariants()` (beds occupied + reserved <= capacity; loads <= max; each victim in exactly one place; unit ↔ victim consistency) asserted after every `advance()` in tests. Liveness guard for unassigned loaded units. Holdover rule prevents capability loss. Assign the strongest implementer to WP1. |
| Contract drift between parallel agents | WP0 lands first. Contracts are frozen in `domain/`, single owner, changes via the lead. Architecture test. Fakes (`FakePort`, `PolicyBackend`) let WP3/WP5 progress independently. |
| SDK API drift or private API use | Use only public `Agent`, `Runner`, `RunConfig`, `function_tool`, `Model`, `ModelResponse`, `Usage`, `AgentOutputSchema`, `set_tracing_disabled`, and `openai.types.responses` types. `agents.testing` in tests only. Pins in `requirements.txt`. |
| Strict schema rejected by the OpenAI API later | No dicts, defaults, numeric constraints or root unions in outputs. `test_strict_schemas.py`. Range checks in semantic validation. |
| Accidental network or tracing export offline | `set_tracing_disabled(True)` + `RunConfig(tracing_disabled=True)` in the offline factory. Offline agents use `PolicyModel` instances (never a model-name string). Tests run with the key unset; any provider use would fail loudly. |
| LLM returns illegal decisions | Two-level validation: SDK schema (→ call-level fallback) and coordinator semantics (→ item/section fallback), with the engine as the final gate. Everything logged and counted in `fallbacks`. |
| Baseline judged a strawman | B0 retries hospitals, matches ALS after triage, dispatches responders, sees the bed board, and shares the same ambulance cap and crew rules. Differences are documented. |
| Log volume | Snapshots are about 10 to 15 KB each, so roughly 2.5 MB per run. Keep `events.jsonl` only for seed 7380 (`--keep-logs-for`); `metrics.json` for all. |
| Python 3.14 or Streamlit quirks | `AppTest` in CI-style tests. `run_sync` already handles the 3.14 loop change. No new installs. |
| LLM cost later | Event-driven calls: one per hospital per round and batched dispatch. `AgentCallRecord.usage` logged. README states expected calls per run (about 100 to 250) and suggests `--repeats 3`. |

### Critical Files for Implementation

- `/Users/suhas/Suhas/WorkWithMe/CSYE-7380-Agentic AI/Intelligent Agents and Multi-Agent Systems for Emergency Response Coordination/Shubham_Bagwe_Emergency_Response_MAS/emergency_sim/domain/agent_io.py` (plus the sibling `domain/*.py` contracts: views, messages, log_records, ports)
- `/Users/suhas/Suhas/WorkWithMe/CSYE-7380-Agentic AI/Intelligent Agents and Multi-Agent Systems for Emergency Response Coordination/Shubham_Bagwe_Emergency_Response_MAS/emergency_sim/engine/engine.py` (new; DES engine, the single source of truth; reuses `emergency_sim/geo.py`)
- `/Users/suhas/Suhas/WorkWithMe/CSYE-7380-Agentic AI/Intelligent Agents and Multi-Agent Systems for Emergency Response Coordination/Shubham_Bagwe_Emergency_Response_MAS/emergency_sim/control/coordinator.py` (new; contract-net protocol, B1/Full)
- `/Users/suhas/Suhas/WorkWithMe/CSYE-7380-Agentic AI/Intelligent Agents and Multi-Agent Systems for Emergency Response Coordination/Shubham_Bagwe_Emergency_Response_MAS/emergency_sim/mas/policy_model.py` (new; offline `Model` subclass driving `Runner`, modelled on the verified `ModelResponse`/`ResponseOutputMessage` path in `.venv/lib/python3.14/site-packages/agents/run_internal/turn_resolution.py`)
- `/Users/suhas/Suhas/WorkWithMe/CSYE-7380-Agentic AI/Intelligent Agents and Multi-Agent Systems for Emergency Response Coordination/Shubham_Bagwe_Emergency_Response_MAS/emergency_sim/datagen.py` (existing; reused unchanged via `scenarios.scenario_for` for multi-seed evaluation)