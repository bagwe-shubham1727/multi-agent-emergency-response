# Emergency Response Coordination with Multi-Agent AI (Mid-Term Project)

CSYE7380 Theory & Prac App AI Gen Model · Prof. Subrata Das · Shubham Bagwe · due 4 Nov 2026, 18:00

Status (26 Oct 2026): phase 2 complete, code review fixes applied. Engine, agents, negotiation,
evaluation and dashboard are implemented; 788 tests pass; the held-out evaluation (441 runs) is
in `results/`. It replaces the first evaluation (378 runs), which a code review showed was
confounded by caller reports that revealed hidden triage (see "Honest notes").

## Overview

A city has three hospitals, six ambulances, six responder teams and a stream of incidents,
including a mass-casualty cluster. Six agents built with the OpenAI Agents SDK (Incident
Assessment, Dispatch, one Hospital Coordination agent per hospital, Resource Allocation) decide
where units and patients go. They exchange typed messages and negotiate over contested
resources: ambulances, ER beds, critical-reserve beds and float staff. A deterministic
discrete-event engine owns the world state and applies only validated actions. The same
synthetic worlds are run under a greedy baseline (B0), the same baseline with the agents'
caller-report assessment (B0-cues), agents without negotiation (B1), the full negotiating system
(Full) and three ablations. Results are compared pairwise on identical worlds.
A Streamlit dashboard replays every run from its JSONL event log.

## Quickstart

The virtual environment already exists at `.venv` (Python 3.14.5, pins in `requirements.txt`).
Run every command from the project root.

```bash
# Tests (788 tests, about 40 s; hermetic, no network, no API key)
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest -q -m "not slow"     # skip the 21-seed grid test

# Synthetic data (phase 1, standard library only)
.venv/bin/python -m emergency_sim.datagen       # writes data/normal, data/surge, data/constrained (seed 7380)
.venv/bin/python -m emergency_sim.validate      # 57/57 checks; exits non-zero on any failure

# One simulation (offline backend; writes metrics.json and events.jsonl)
.venv/bin/python -m emergency_sim.cli run --scenario surge --condition full --seed 7380 --out results_demo

# The held-out evaluation exactly as run (441 runs)
env -u OPENAI_API_KEY .venv/bin/python -m emergency_sim.cli grid \
    --scenarios normal surge constrained \
    --conditions b0 b0-cues b1 full full-no-float full-no-diversion full-no-escalation \
    --seeds 7380 1001-1020 --backend offline --keep-logs-for 7380 --out results

# Rebuild summary.csv, paired.csv and summary.md from the metrics.json files
.venv/bin/python -m emergency_sim.cli report --results results

# Dashboard (replays logs; never calls a model)
RESULTS_DIR=results .venv/bin/streamlit run dashboard/app.py
```

CLI reference (`python -m emergency_sim.cli`):

| Command | Main options | Defaults |
|---|---|---|
| `run` | `--scenario {normal,surge,constrained}` `--condition {b0,b0-cues,b1,full,full-no-float,full-no-diversion,full-no-escalation}` `--seed` `--backend {offline,llm}` `--model` `--repeat` `--out` | seed 7380, backend offline, repeat 0, out `results` |
| `grid` | `--scenarios` `--conditions` `--seeds` (ranges allowed, e.g. `7380 1001-1020`) `--backend` `--model` `--repeats` `--keep-logs-for` `--out` | all scenarios; b0 b1 full; seeds 7380 1001-1020; 1 repeat; logs kept for 7380 |
| `report` | `--results` | `results` |

The global flag `-v` / `--verbose`, placed before the command (`python -m emergency_sim.cli -v run ...`),
logs progress and prints the traceback on an error.
Exit status is 0 on success, 1 on a runtime error and 2 on a usage error. To recreate the
environment elsewhere: `python3.14 -m venv .venv && .venv/bin/pip install -r requirements.txt`.

## Running with a real LLM

The LLM backend uses the same agents, tools, instructions, input payloads and output schemas as the
offline backend. Only the model changes.

```bash
export OPENAI_API_KEY=sk-...
export EMS_LLM_MODEL=gpt-4.1-mini          # optional; this is the default
.venv/bin/python -m emergency_sim.cli run --scenario surge --condition full --backend llm --model gpt-4.1-mini
.venv/bin/python -m emergency_sim.cli grid --scenarios surge constrained --conditions b1 full \
    --seeds 7380 1001-1004 --backend llm --model gpt-4.1-mini --repeats 3 --out results_llm
```

- Without `OPENAI_API_KEY` the LLM backend raises `BackendUnavailableError`.
- LLM output is not seeded, so use `--repeats 3` (or more) and report mean ± std.
- Expect about 100 to 250 model calls per run. The offline runs on the eval seeds made 55 to 178
  agent calls per run (means: 75.4 to 76.5 in normal, 117.5 to 127.5 in surge and constrained).
  An LLM agent may use more than one model request per call (tool round trip; at most 6 turns).
- Each model request has a 60 s timeout; a timeout falls back to the policy.
- A failed or invalid model reply (bad JSON, API error, schema error) falls back to the rule-based
  policy for that call. Semantically illegal decisions (for example a busy unit) are replaced per
  item or per section by the coordinator. Every fallback is logged and counted in `fallbacks`.
- Logs go to `runs/{scenario}/{condition}/llm-<model>/seed-{seed}_r{repeat}/`. Token usage and
  latency are recorded in each `agent_call` record.

## Architecture

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
              emergency_sim/geo.py, emergency_sim/datagen.py
```

One decision point, step by step:

```
Engine.advance() ──► DecisionPoint(t, triggers) ──► Controller.handle(point, EnginePort)
                                                      │
     B0, B0-cues: GreedyController (no messages)      │  B1 / Full / ablations: AgentCoordinator
                                                      │    Phase A  Assessment  → PRIORITY_REPORT
                                                      │    Phase B  Dispatch    → orders, (Full) ESCALATE/COMMIT
                                                      │    Phase C  Transport negotiation, ≤ 3 rounds
                                                      │             PROPOSE → ACCEPT / REJECT / COUNTER
                                                      │             → REQUEST / ESCALATE → COMMIT
                                                      ▼
            each agent call: DecisionBackend.decide(agent, payload) → SDK Runner → typed output
            coordinator validates the output → engine Action → EnginePort.apply (engine re-checks)
                                                      │
      every record (snapshots, events, messages, agent calls, actions) → events.jsonl → dashboard
```

Layer rules (enforced by `tests/test_architecture.py`, which parses imports with `ast`):

| Package | May import |
|---|---|
| `domain` | stdlib, pydantic, `emergency_sim.geo` |
| `engine` | `domain`, `geo`. Not `policies`, `control`, `mas` or the `agents` SDK |
| `policies` | `domain`, `geo`. Not the `agents` SDK |
| `mas` | `domain`, `policies`, `agents`, `openai`. The only package that imports the SDK |
| `control` | `domain`, `policies`, `geo`. Sees the engine only through `EnginePort` and agents only through `DecisionBackend` |
| `evaluation`, `cli` | anything in `emergency_sim` |
| `dashboard` | `domain`, `eventlog`, `evaluation`. Not `engine`, `control`, `mas` or `policies` |

Design rules that follow from this:

- The engine is the only source of truth. Agents never mutate state.
- Controllers get an `EnginePortAdapter` (`engine/port.py`), not the engine. It exposes only the
  four `EnginePort` members, so a controller cannot reach engine internals or hidden truth.
- Agent tools are read-only. Proposals and decisions are structured outputs (`output_type`), not
  tool calls. Only the coordinator applies engine `Action`s, and the engine validates each one
  again. An illegal action is logged with a reason and changes nothing. When the engine refuses a
  negotiated outcome, the coordinator sends an `INFORM` with resolution `ENGINE_REJECTED` and the
  attempted actions.
- Agents see only `PublicView`. It has no hidden truth (no golden windows, no true victim counts,
  no victims of untriaged incidents). A test checks this.
- The offline backend runs the rule-based policies inside real SDK `Agent`/`Runner` calls through
  a custom `Model` (`mas/policy_model.py`), including one real tool round trip per call.

## Agents

| Agent | Role | Objective | Read-only tools | Output type |
|---|---|---|---|---|
| `incident_assessment` | Incident Assessment | Sickest incidents seen first: urgency 0–100, suspected RED / cardiac arrest, patients, capabilities, ALS units, deadline | `incident_type_profile`, `nominal_deadline` | `AssessmentOutput` |
| `dispatch` | Dispatch | Send help now, minimise time to care: unit orders, diversion requests, contention flags, hospital rankings for loaded units | `rank_units_for_incident`, `rank_hospitals_for_unit` | `DispatchOutput` |
| `hospital_H1`, `hospital_H2`, `hospital_H3` | Hospital Coordination (one per hospital) | Protect `critical_reserve` beds for RED; accept only patients it can treat on arrival | `capacity_check` | `HospitalOutput` |
| `resource_allocation` | Resource Allocation | Maximise the survival-weighted outcome across contested resources (units, float staff, reserve beds, destinations) | `score_case_options` | `AllocationOutput` |

Agents are stateless per call. All context is in the JSON input, which is identical for the
offline and LLM backends. Instructions are in `emergency_sim/mas/instructions.py`.

## Messages and negotiation protocol

Every message is a frozen Pydantic model (`emergency_sim/domain/messages.py`) and is logged.

| Field | Meaning |
|---|---|
| `message_id` | `M-0001`, sequential per run |
| `t_min` | Simulation time of the decision point |
| `sender`, `recipient` | Agent names, or `all` for broadcast |
| `type` | `PRIORITY_REPORT`, `PROPOSE`, `ACCEPT`, `REJECT`, `COUNTER`, `REQUEST`, `ESCALATE`, `COMMIT`, `INFORM` |
| `incident_id` | The triggering incident |
| `proposed_action` | Typed action: `assess`, `dispatch`, `admit`, `admit_if`, `decline`, `lend_staff`, `resolve`, `commit` |
| `rationale` | Short reason |
| `in_reply_to` | Message this one answers, or null |
| `conversation_id` | `NEG-0001`: one per negotiation thread |
| `round` | 0 outside transport negotiation, else 1 to 3 |

Protocol at each decision point (contract-net style):

1. **Phase A, assessment.** On a new incident, triage, deterioration or loading, Incident Assessment
   sends a `PRIORITY_REPORT` per incident to Dispatch.
2. **Phase B, dispatch.** Dispatch orders units (`INFORM`). In Full, if two incidents want the same
   nearest ALS, or an en-route ALS could be diverted, Dispatch sends `ESCALATE` with a
   `UNIT_CONTENTION` case. Resource Allocation picks an option by utility and sends `COMMIT`.
3. **Phase C, transport negotiation (at most 3 rounds).** Dispatch sends `PROPOSE` (admit) to the
   first hospital in the unit's ranking. The hospital answers:
   - `ACCEPT`: beds are reserved and the unit departs;
   - `REJECT` with a reason (`NO_BEDS`, `RESERVE_HELD`, `NO_DOCTOR`, `NO_NURSE`, `NOT_CAPABLE`):
     Dispatch tries the next hospital;
   - `COUNTER` (Full only), a conditional acceptance: "accept if lent a float doctor/nurse"
     (`REQUEST` to Resource Allocation) or "accept by using my critical reserve"
     (`ESCALATE`, `RESERVE_RELEASE`). Resource Allocation grants or declines with `COMMIT` or
     `REJECT`.
4. **No hospital accepted after 3 rounds.** Full escalates a `NO_ACCEPTING_HOSPITAL` case and
   Resource Allocation chooses the destination. B1 forces admission at the first-ranked hospital.

Resource Allocation scores options with `utility = triage_weight × p_window(slack) − 0.01 ×
travel_min`, where `p_window(slack) = 1 / (1 + exp(−slack / 5))`. Ties are broken by id.

Real example: thread `NEG-0015` from `results/runs/constrained/full/offline/seed-7380_r0/events.jsonl`
at t = 68.6. Dispatch proposes a patient to H1; H1 has no free trauma doctor slot and counters;
Resource Allocation lends a float doctor. The `COUNTER` message, copied from the log:

```json
{"message_id": "M-0040", "t_min": 68.6, "sender": "hospital_H1", "recipient": "dispatch",
 "type": "COUNTER", "incident_id": "INC-04",
 "proposed_action": {"action": "admit_if", "hospital_id": "H1", "unit_id": "AMB-01",
                     "victim_ids": ["INC-04-V4"], "condition": "FLOAT_DOCTOR", "skill": "trauma"},
 "rationale": "no doctor slot with skill trauma; accept with a float doctor",
 "in_reply_to": "M-0039", "conversation_id": "NEG-0015", "round": 1}
```

The thread is `PROPOSE` (M-0039) → `COUNTER` (M-0040) → `REQUEST` (M-0041, CASE-0003) →
`COMMIT` (M-0042 to H1 and M-0043 to Dispatch: `lend_staff DR-FLOAT-1` and
`assign_transport AMB-01 → H1, reserve_beds=true`, resolution `COUNTER_GRANTED`, rationale "lend
DR-FLOAT-1: highest utility 2.630").

## Conditions

| Condition | Description |
|---|---|
| `b0` | Greedy baseline, no agents and no messages. FCFS incidents, nearest free unit, nearest capable hospital with a free bed, redirect if full. Ignores caller-report wording. Same public view, same pre-triage ambulance cap and crew rules |
| `b0-cues` | B0 plus the agents' public caller-report assessment rule: incidents in assessed priority order, ALS first to a suspected RED. Still no messages, agents, bed reservations, staff information or negotiation |
| `b1` | Agents, no negotiation: hospitals may only ACCEPT or REJECT; no COUNTER, escalation, float staff or diversion; forced admission after 3 rounds |
| `full` | All mechanisms: COUNTER (float staff and reserve release), escalation to Resource Allocation (unit contention, no accepting hospital, reserve release) and ALS diversion |
| `full-no-float` | Full without float-staff counters |
| `full-no-diversion` | Full without ALS diversion |
| `full-no-escalation` | Full without any escalation: no ESCALATE messages, so no unit contention, no-accepting-hospital or `RESERVE_RELEASE` cases, and no diversion (diversion requires escalation). Hospitals still COUNTER with float-staff conditions, which are staff `REQUEST`s |

The conditions decompose the gain of Full over B0 into three steps on identical worlds:

| Step | Pair | What it adds |
|---|---|---|
| Information | B0-cues vs B0 | Reading the caller report: priority order and ALS to suspected RED |
| Coordination | B1 vs B0-cues | Agents and messages: hospital agents with staff information, bed reservation on ACCEPT |
| Negotiation | Full vs B1 | COUNTER, escalation to Resource Allocation, diversion |

## Dashboard

`dashboard/app.py` replays any run in `results/runs/` minute by minute (time slider, optional
autoplay, optional side-by-side comparison of two conditions on the same world).

| Tab | Shows |
|---|---|
| Map & incidents | Schematic 20 × 20 km city: hospitals coloured by bed use, stations, incidents by worst known triage and size, ALS/BLS/responder units with lines to their destination |
| Resources | Ambulance status timeline, ER occupancy (occupied / reserved / free / queue) per hospital, doctor and nurse load vs capacity, lent float staff, resource time series |
| Agent activity | Status card per agent: calls, last triggers, last decision with rationale, tool calls, fallbacks, pending items; recent messages; step-through view of a decision point |
| Negotiation history | One row per thread: contested resource, PROPOSE / ACCEPT / REJECT / COUNTER / ESCALATE / COMMIT sequence and resolution; resolution breakdown chart |
| KPIs | KPI cards at time t and at the end; conditions side by side; cross-seed comparison from `results/summary.csv` |

Event logs are kept for seed 7380 (21 runs, 2.6 to 3.9 MB each); all 441 runs keep `metrics.json`.

## Headline results

Held-out evaluation: 3 scenarios × 7 conditions × 21 eval seeds (7380 and 1001–1020), offline
backend, data generator 1.2.0 (noisy caller wording), 441 runs, 0 fallbacks, 0 controller
omissions. It was run once, after development and the review fixes were frozen. Cells are mean ±
std (ddof = 1) over 21 runs. Copied from `results/summary.md`.

| Scenario | Condition | Served rate | RED in window | Survival score | Response (min) |
|---|---|---:|---:|---:|---:|
| normal | B0 | 0.894 ± 0.125 | 0.471 ± 0.203 | 0.685 ± 0.130 | 17.0 ± 7.0 |
| normal | B0-cues | 0.865 ± 0.157 | 0.486 ± 0.206 | 0.671 ± 0.158 | 16.8 ± 5.6 |
| normal | B1 | 0.894 ± 0.128 | 0.505 ± 0.215 | 0.694 ± 0.158 | 16.2 ± 4.6 |
| normal | Full | 0.905 ± 0.118 | 0.565 ± 0.185 | 0.727 ± 0.148 | 15.1 ± 3.8 |
| surge | B0 | 0.852 ± 0.124 | 0.319 ± 0.161 | 0.585 ± 0.143 | 32.3 ± 12.5 |
| surge | B0-cues | 0.783 ± 0.171 | 0.312 ± 0.157 | 0.535 ± 0.175 | 30.5 ± 11.9 |
| surge | B1 | 0.807 ± 0.151 | 0.350 ± 0.205 | 0.571 ± 0.179 | 27.8 ± 9.1 |
| surge | Full | 0.796 ± 0.168 | 0.357 ± 0.189 | 0.571 ± 0.177 | 25.6 ± 7.4 |
| constrained | B0 | 0.772 ± 0.132 | 0.271 ± 0.114 | 0.506 ± 0.134 | 38.9 ± 11.6 |
| constrained | B0-cues | 0.714 ± 0.155 | 0.260 ± 0.141 | 0.459 ± 0.155 | 34.0 ± 10.0 |
| constrained | B1 | 0.734 ± 0.160 | 0.306 ± 0.166 | 0.502 ± 0.164 | 33.5 ± 8.7 |
| constrained | Full | 0.739 ± 0.175 | 0.344 ± 0.170 | 0.524 ± 0.173 | 30.2 ± 7.6 |

Key paired comparisons (same scenario and seed, n = 21; W/T/L = runs where the first condition is
better / equal / worse; exact two-sided sign test, ties excluded):

| Scenario | Comparison | Metric | Mean diff | W/T/L | Sign-test p |
|---|---|---|---:|---:|---:|
| surge | B0-cues vs B0 | Served rate | -0.069 | 1/9/11 | 0.0063 |
| surge | B0-cues vs B0 | Survival score | -0.049 | 3/3/15 | 0.0075 |
| constrained | B0-cues vs B0 | Served rate | -0.058 | 3/5/13 | 0.0213 |
| constrained | B0-cues vs B0 | Survival score | -0.046 | 5/4/12 | 0.1435 |
| constrained | B1 vs B0-cues | Door-to-doctor (min) | -1.846 | 16/1/4 | 0.0118 |
| constrained | Full vs B1 | Response (min) | -3.363 | 18/0/3 | 0.0015 |
| surge | Full vs B0-cues | Survival score | +0.036 | 15/2/4 | 0.0192 |
| constrained | Full vs B0-cues | Survival score | +0.065 | 17/1/3 | 0.0026 |
| constrained | Full vs B0-cues | RED in window (rate) | +0.084 | 15/3/3 | 0.0075 |
| normal | Full vs B0 | RED in window (rate) | +0.095 | 11/7/3 | 0.0574 |
| surge | Full vs B0 | RED in window (rate) | +0.039 | 10/6/5 | 0.3018 |
| surge | Full vs B0 | Served rate | -0.056 | 5/6/10 | 0.3018 |
| surge | Full vs B0 | Response (min) | -6.727 | 15/0/6 | 0.0784 |
| constrained | Full vs B0 | RED in window (rate) | +0.073 | 13/1/7 | 0.2632 |
| constrained | Full vs B0 | Served rate | -0.034 | 7/5/9 | 0.8036 |
| constrained | Full vs B0 | Response (min) | -8.703 | 18/0/3 | 0.0015 |

All metrics for all conditions are in `results/summary.csv` and `results/paired.csv` (pairs:
Full–B0, Full–B0-cues, Full–B1, B1–B0-cues, B0-cues–B0, B1–B0). The method, all paired
comparisons, the ablations and the discussion are in [`docs/evaluation.md`](docs/evaluation.md).

## Honest notes

- **The first evaluation was confounded and is superseded.** A code review found that caller
  wording was a deterministic function of hidden truth: the severe phrase appeared exactly when
  the worst triage was RED. The Incident Assessment agent's keyword cues therefore recovered RED
  with 100% recall and 92% precision, an information oracle that B0 did not use. With cues
  disabled, the agents' gain over B0 vanished (reviewer's ablation, normal: Full RED in window
  0.581 → 0.450, B0 0.471). Fixes: seeded caller-wording noise (generator 1.2.0; cue recall ≈
  0.875, precision ≈ 0.864), the `b0-cues` baseline, and corrected ablation semantics. The v1
  results (378 runs) are archived outside the repository. Their headline claims (for example
  normal Full vs B0 RED in window +0.111, p = 0.0002; survival +0.050, p = 0.0074) no longer hold.
  `docs/evaluation.md`, sections 6 and 7, has the details and the old numbers.
- **Information alone hurts.** B0-cues is below B0: surge survival -0.049 (3/3/15, p = 0.0075),
  surge served -0.069 (1/9/11, p = 0.0063), constrained served -0.058 (p = 0.0213). It reaches
  scenes faster (constrained -4.848 min, p = 0.0414) but serves fewer patients.
- **Coordination and negotiation help on the same information, more under pressure.** Full vs
  B0-cues: constrained survival +0.065 (17/1/3, p = 0.0026), constrained RED in window +0.084
  (15/3/3, p = 0.0075), surge survival +0.036 (15/2/4, p = 0.0192); normal is not significant.
- **Against plain B0, Full is faster but not clearly better on outcomes.** Response time falls in
  constrained by 8.7 min (-8.703, 18/0/3, p = 0.0015). RED-in-window gains are +0.039 to +0.095
  and not significant. Served rate is slightly lower in surge (-0.056) and constrained (-0.034),
  also not significant.
- **Why plain B0 is a strong baseline.** Urgency-first dispatch serves suspected-RED scenes first,
  so older YELLOW-only scenes wait and their patients deteriorate to RED. B0's FCFS order bounds
  every scene's wait. Deteriorated victims, B0-cues vs B0: +0.905 per run in surge (p = 0.0654) and
  +0.762 in constrained (p = 0.0574); Full vs B0: +0.714 (p = 0.0391) and +0.905 (p = 0.0490). This
  is the open design issue from the dev-seed diagnosis. One fix (D1, expected-loss ordering) was
  tried on dev seeds and rejected by the pre-registered rule.
- **Ablations.** Float staff is the one mechanism with a clear effect: constrained Full vs
  full-no-float survival +0.026 (13/8/0, p = 0.0002), served +0.020 (p = 0.0117). Diversion and
  escalation have no significant outcome effect. In normal, Full and full-no-float give identical
  outcome metrics in all 21 runs.
- **`full-no-escalation` now means no escalation at all.** `allow_escalation` governs every
  escalation, including `RESERVE_RELEASE` counters, so this condition has no ESCALATE messages.
  Hospitals still COUNTER with float-staff conditions (staff `REQUEST`s).
- **Rule-based agents, not an LLM.** The offline results measure the coordination protocol with
  rule-based agents running inside real SDK `Agent`/`Runner` calls. The LLM backend is implemented
  and tested with fake (scripted) models only. It was not run, because no API key was available.
  No claim is made about LLM agent quality.
- **Multiple comparisons.** `results/summary.md` shows 126 paired tests. With a Bonferroni
  correction (α = 0.05 / 126), only the conflict counts, constrained Full vs B0-cues door-to-doctor
  time and constrained B1 vs B0 response time remain significant. No served-rate, RED-in-window or
  survival-score difference survives.
- **Caller-report assumptions.** The cue lexicon was written against the generator's own
  phrasings, and the noise rates (0.20 under-triage, 0.10 over-triage) are documented assumptions,
  not measurements. Other rates would change the value of the information.
- **Eval seeds were seen before this run**: seed 7380 during development (the control-layer
  implementer ran it with seeds 1–3 and reported a Full vs full-no-escalation gap, which prompted
  the unit-contention diagnosis), and all eval seeds by the v1 evaluation and the reviewer's
  ablation. No agent rule, B0 rule or parameter was tuned on them. Excluding 7380 leaves every
  direction unchanged; see `docs/evaluation.md`, threats to validity.
- **Not yet done** (from the review): public game rules are still defined in two places (engine and
  `control` / `policies` constants); the dashboard keeps its own KPI fallback alongside
  `evaluation.metrics.kpis_at`; line length is not enforced by a linter (no formatter installed).
- **Synthetic data** from one schematic city layout. No real dispatch data was used.

## Repository layout

| Path | What |
|---|---|
| `emergency_sim/domain/` | Contracts: enums, scenario specs, actions, `PublicView`, messages, agent I/O models, outcomes, log records, Protocols (`EnginePort`, `Controller`, `DecisionBackend`, `EventSink`) |
| `emergency_sim/engine/` | Deterministic discrete-event engine: state, event queue, legality rules, hospital care (beds, queue, staff, holdover), public view, invariants; `port.py` is the narrow `EnginePort` adapter handed to controllers |
| `emergency_sim/policies/` | Pure, SDK-free decision rules for the four roles and shared estimates (ETA, rankings, capacity check, utility) |
| `emergency_sim/mas/` | OpenAI Agents SDK layer: agent specs, instructions, read-only `@function_tool`s, offline `PolicyModel`, `SdkBackend` (offline and LLM) |
| `emergency_sim/control/` | B0 / B0-cues greedy controller, `AgentCoordinator` (phases A/B/C), negotiation configs, escalation cases, output validation, message bus, shared need accounting |
| `emergency_sim/evaluation/` | Runner, exact metric formulas, report (summary, paired differences, sign test) |
| `emergency_sim/cli.py` | `run`, `grid`, `report` |
| `emergency_sim/scenarios.py`, `eventlog.py` | Scenario loading (committed data or regenerated per seed); JSONL event sink and reader |
| `emergency_sim/datagen.py`, `validate.py`, `geo.py` | Phase 1: seeded synthetic data generator, validation, map and travel times |
| `dashboard/` | Streamlit app (`app.py`), pure replay functions (`replay.py`), Plotly figures (`figures.py`) |
| `data/<scenario>/` | Seed-7380 data: hospitals, stations, ambulances, staff, responders, incidents (JSON), victims and travel times (CSV), summary, SHA-256 manifest |
| `results/` | `summary.md`, `summary.csv`, `paired.csv`; `runs/<scenario>/<condition>/offline/seed-<s>_r0/` with `metrics.json` (all runs) and `events.jsonl` (seed 7380) |
| `tests/` | 788 pytest tests (one marked `slow`) |
| `docs/` | Design, plan, evaluation and system-flow diagram (see below) |
| `slides/` | Phase-1 progress deck and its build script |

## Reproducibility

- **Seeds.** Data seed 7380 is committed in `data/`. Other seeds are regenerated in memory by
  `datagen` through `scenarios.scenario_for`. Dev seeds 1–5 were the only seeds used for debugging.
  Eval seeds 7380 and 1001–1020 were fixed in the plan before any result.
- **Determinism.** The engine uses an event heap with deterministic tie-breaking, sorted iteration,
  no wall clock and no UUIDs. Offline agent calls record `latency_ms = null`. Two runs of the same
  spec give byte-identical `events.jsonl` and `metrics.json`. This is tested in
  `tests/test_end_to_end.py` and was re-checked on 26 Oct 2026 by re-running
  `run --scenario constrained --condition full --seed 7380` and comparing both files with
  `cmp` against `results/` (identical).
- **Environment.** Python 3.14.5 on macOS with the pins in `requirements.txt`
  (openai-agents 0.24.0, openai 3.37.0, pydantic 2.14.1, streamlit 1.66.1, plotly 7.1.0,
  pandas 3.1.0, numpy 2.5.4, pytest 9.1.1). Another Python version may draw different random
  data; the SHA-256 manifest in `data/` flags that.
- **No network in tests.** Tests unset `OPENAI_API_KEY`. SDK tracing is off for offline runs; it is set per run through the SDK `RunConfig`, with no process-global switch.

## Files

| File | What |
|---|---|
| `README.md` | This file |
| `docs/simulation_design.md` | Phase-1 design (architecture, agents, messages, protocol, data) plus the v2 changes |
| `docs/implementation_plan.md` | Agreed phase-2 plan: contracts, engine semantics, policies, protocol, metric formulas, work packages, honesty protocol |
| `docs/evaluation.md` | Held-out evaluation (method, results, ablations, interpretation, threats to validity), the code review and information-oracle fix, the superseded v1 results, and the dev-seed change log (fixes F1–F5, round 2) |
| `docs/system_flow.mmd` | Mermaid diagram of the implemented architecture and decision-point flow |
| `results/summary.md` | Held-out results: per-scenario tables and paired comparisons |
| `results/summary.csv`, `results/paired.csv` | All metrics per scenario and condition; all paired differences |
| `prompts.md` | Prompt log: the prompts given and the key responses, per course requirement |
| `pyproject.toml`, `requirements.txt` | Project metadata, pytest config, pinned versions |
| `data/validation_report.json` | Result of the last data validation run |
| `slides/Shubham_Bagwe_Emergency_Response_Progress.pptx` | Phase-1 progress deck (`slides/build_slides.js` builds it from `data/`) |
