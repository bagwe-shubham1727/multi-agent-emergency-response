# Simulation Design and Data Plan

CSYE7380 Theory & Prac App AI Gen Model · Prof. Subrata Das · Shubham Bagwe
Status as of 25 Oct 2026: phase 2 implemented and evaluated (engine, agents, negotiation, dashboard,
378-run held-out evaluation). Sections 1 to 7 are the phase-1 design as agreed on 21 Oct 2026.
Section 8 lists where the implementation differs and why. Due 4 Nov 2026, 18:00.

## 1. What the project has to show

The brief asks for more than agents talking. Communication, coordination and negotiation must produce
**measurable** gains in emergency outcomes. So the design is built around a controlled comparison: the
same synthetic scenarios run under a rule-based baseline, under agents without negotiation, and under
the full negotiating system.

## 2. Architecture

```
synthetic data (seeded JSON/CSV)
        │
        ▼
simulation engine  ── owns world state, 1-minute ticks, applies only validated actions
        ▲   │ tools (read state / propose / commit)
        │   ▼
 ┌───────────── message bus (typed messages, full log) ─────────────┐
 │ Incident Assessment │ Dispatch │ Hospital Coord. (×3) │ Resource Allocation │
 └──────────────────────────────────────────────────────────────────┘
        │
        ▼
event log (JSONL)  ──►  Streamlit + Plotly dashboard (replay with a time slider)
```

* **The engine is deterministic and is the single source of truth.** Agents never edit state
  directly; they call function tools, and the engine rejects any action that breaks a rule (unit
  already busy, bed already taken, staff over workload). An LLM mistake therefore cannot corrupt
  the simulation.
* **Agents are built with the OpenAI Agents SDK** (`Agent`, `function_tool`, `Runner`), each with its
  own instructions, tools and a Pydantic `output_type`, so every reply is a structured message.
* **Agents run on events, not on every tick:** a new incident, a unit becoming free, a rejection or
  a YELLOW patient turning RED. Between events the engine just advances the clock. This keeps the
  number of LLM calls per run in the low hundreds instead of one per agent per minute.
* **Every message and transcript is logged.** The dashboard replays runs from the log, so it never
  calls a model.
* **Dashboard views** (all five the brief requires): incident map with severity and assigned units;
  resource status (ambulances, responder teams, doctors, nurses, ER beds); agent activity (status,
  recent messages, pending requests); allocation and negotiation history (proposals, accepts,
  rejects, counters, resolutions); KPIs (response time, victims served, unserved critical,
  utilisation, conflicts).

## 3. Agents

| Agent | Sees | Decides | Its own priority |
|---|---|---|---|
| Incident Assessment | caller report, reported victim count, location | urgency score 0–100, estimated triage mix, deadline, needed capabilities | get the sickest incidents seen first |
| Dispatch | units, positions, ETAs, priority reports | which ambulance / responder team goes where | send help now, minimise response time |
| Hospital Coordination (one per hospital) | own ER beds, ICU, staff roster and load | accept / reject / counter each admission; ask for float staff | keep `critical_reserve` beds free for RED patients |
| Resource Allocation | open conflicts and proposals | final allocation when others disagree; float staff placement | maximise survival-weighted victims served |

The Hospital agent's wish to hold beds and the Dispatch agent's wish to place patients now are the
built-in conflict the brief describes.

## 4. Messages

Every message is a Pydantic model and is logged. Required fields: `sender`, `recipient`, `type`,
`incident_id`, `proposed_action`; plus `message_id`, `t_min`, `rationale` and `in_reply_to`.

```json
{"message_id": "M-0142", "t_min": 42, "sender": "dispatch", "recipient": "hospital_H1",
 "type": "PROPOSE", "incident_id": "INC-04",
 "proposed_action": {"action": "admit", "hospital_id": "H1", "victims": ["INC-04-V1", "INC-04-V3"], "unit_id": "AMB-01"},
 "rationale": "2 RED trauma, H1 is the only Level I trauma centre, ETA 6.8 min",
 "in_reply_to": null}
```

Types: `PRIORITY_REPORT`, `PROPOSE`, `ACCEPT`, `REJECT`, `COUNTER`, `REQUEST`, `ESCALATE`,
`COMMIT`, `INFORM`.

## 5. Negotiation protocol (contract-net style)

1. Assessment posts a `PRIORITY_REPORT` for each new incident.
2. Dispatch `PROPOSE`s unit and hospital assignments, highest urgency first.
3. Each hospital `ACCEPT`s, `REJECT`s with a reason, or `COUNTER`s (for example "take the 2 RED,
   send the YELLOW to H2", or "accept if I get one float nurse").
4. Any clash (two incidents want one ambulance, a reject with no alternative, two hospitals asking
   for the same float doctor) is `ESCALATE`d to Resource Allocation.
5. At most 3 rounds. Resource Allocation then decides by a utility score
   (triage weight × chance of meeting the golden window − travel time penalty), ties broken by
   incident id, and sends `COMMIT`. The engine applies it.

## 6. Evaluation

Same scenarios for all conditions, run in the same engine under the same rules:

| Condition | What it is |
|---|---|
| B0 greedy baseline | no messages. Nearest free unit; nearest capable hospital with a free bed, retrying the next nearest if full; first come, first served |
| B1 agents, no negotiation | agents decide, but there are no counter-offers, escalation or staff sharing; a rejection sends the patient to the next hospital on Dispatch's list |
| Full system | agents with counter-offers, escalation and float-staff sharing |

Fairness rules:

* B0 sees only what the agents see (`reported_victims` and the caller report), never the hidden
  true victim list.
* B0 is a competent baseline (it retries other hospitals), so gains are not inflated by a
  strawman.
* BLACK (expectant) victims are not transported. They are reported separately and left out of
  "victims served" and "RED within golden window".
* LLM output is not seeded. Each LLM condition is repeated over several runs and several data
  seeds (`--seed`); results are reported as mean ± spread.

Metrics: response time (mean and 90th percentile), RED patients reached within their golden window,
victims served, unserved critical victims, ambulance / ER / staff utilisation, number of conflicts
and how each was resolved.

Cost: event-driven calls plus a small model keep each run to roughly 100–200 calls. The full grid
(2 LLM conditions × 3 scenarios × ~5 repeats) is a few thousand calls, so B0 runs start as soon as
the engine works and LLM runs start by 30 Oct.

## 7. Synthetic data (done)

`python -m emergency_sim.datagen` then `python -m emergency_sim.validate`. Standard library only.

The numbers are seeded rather than written by an LLM so that every condition sees exactly the same
world and the evaluation can be repeated. The caller reports are templated text with seeded
wording noise (step 8), so no LLM is involved in the data either.

| Step | What |
|---|---|
| 1 Schema | 7 entity types: hospitals, stations, ambulances, staff, responder teams, incidents, victims |
| 2 Fixed infrastructure | 20 × 20 km grid city, 3 hospitals with different capabilities, 3 stations |
| 3 Seeded random draws | 11 Poisson arrivals over 150 min, START triage per victim, ER/ICU occupancy, staff rosters, load and shift end |
| 4 Contention | `surge` and `constrained` add a mass-casualty cluster: 3 incidents within 8 min, 13–21 victims |
| 5 Partial information | caller report (severe or mild wording, with seeded noise, see step 8) and a noisy `reported_victims` (±2) for the agents; true victims kept for scoring |
| 6 Travel times | Manhattan distance with a downtown congestion factor, in `geo.py`, which the engine will import |
| 7 Validation | 17–18 checks per scenario plus 4 cross-scenario checks, including regeneration with a SHA-256 comparison and identical caller reports for incidents shared by two scenarios; the report also lists severe wording versus true RED per scenario (information only) |
| 8 Noisy caller reports (done, generator 1.2.0) | Implemented as seeded wording noise rather than the planned LLM rewrite: a caller with a RED victim uses the mild wording with probability 0.20, and a caller with no RED victim uses the severe wording with probability 0.10. Before this, the wording was severe exactly when the worst triage was RED, so a cue lexicon could read RED off the report. Seeded noise keeps the data reproducible with no model calls or cache |

### Controlled scenarios

Randomness comes from one stream per entity type (`<seed>-incidents`, `<seed>-staff`, ...), shared
by all scenarios, so scenarios differ only in their settings:

* `surge` = `normal`'s 11 incidents + the cluster (cluster drawn from its own stream).
* `constrained` = exactly `surge`'s incidents, with fewer resources.
* Caller-report wording takes one draw per incident from its own streams (`<seed>-reports` for
  base incidents, `<seed>-cluster-reports` for the cluster), so it changes no other draw and a
  shared incident gets the same report in every scenario.
* A smaller staff roster is a subset of the larger one; each hospital's occupancy uses the same
  random draw mapped into that scenario's range.

`validate.py` checks all of these properties.

### Scenarios (seed 7380)

| Scenario | Incidents | Victims (R/Y/G/B) | ER beds free at t=0 | Ambulances ready at t=0 | Cluster |
|---|---|---|---|---|---|
| normal | 11 | 21 (8/9/4/0) | 18 / 32 | 5 / 6 | none |
| surge | 14 | 39 (12/17/9/1) | 14 / 32 | 4 / 6 | 17 patients (4 RED) in 6 min vs 12 seats, 3 ALS |
| constrained | 14 | 39 (12/17/9/1) | 10 / 32 | 3 / 6 | same 17 patients vs 10 seats, 3 ALS |

### Assumptions

* Map is schematic: 20 km × 20 km, Manhattan routing, speeds 0.9 km/min (ambulance) and 0.8 km/min
  (responders), congestion up to ×1.5 within 6 km of downtown.
* `travel_times.csv` holds driving time only. The engine adds turnout time (1 min ambulance,
  2 min responders) when a unit is alerted.
* Triage follows START.
  * RED needs care within 20–60 min and an ALS ambulance. A cardiac arrest needs it within
    8–12 min, because survival falls quickly without CPR and defibrillation.
  * YELLOW needs care within 90–240 min.
  * GREEN needs care within 480 min.
  * BLACK is expectant and is not transported.
* Deterioration: an untreated YELLOW turns RED at 60% of its window. From then on it needs ALS and
  has 30 minutes left.
* An ambulance carries up to 2 patients, at most 1 RED.
* Hospital capabilities are deliberately uneven, so some patients have only one valid destination:
  * H1 is the only burns centre and the only Level I trauma centre.
  * H3 is the only paediatric unit.
* Each hospital wants to hold `critical_reserve` ER beds for RED patients.
* Staff:
  * Every ER doctor is an emergency physician. Specialties are spread so that every capability a
    hospital offers has at least one doctor with that skill.
  * Nurse skills follow the hospital's capabilities.
  * Doctors handle up to 4 patients and nurses up to 3.
  * A small float pool can be lent to any hospital.
* Caller reports are imperfect telephone triage. These are documented assumptions, not tuned to
  any result: sensitivity for critical calls is well below 100%, and some calls are over-triaged.
  * `REPORT_UNDERTRIAGE_P = 0.20`: a caller with a RED victim describes the scene mildly.
  * `REPORT_OVERTRIAGE_P = 0.10`: a caller with no RED victim describes it severely.
  * `stroke` uses the same wording ("one side of the face is drooping") either way. Every stroke
    victim is RED, so the incident type already gives this away and the noise cannot hide it.
  * Seed 7380: of the RED incidents with distinguishable wording, 3 of 7 in `normal` and 3 of 9 in
    `surge`/`constrained` are reported mildly (one is a cardiac arrest); no non-RED incident is
    reported severely. Over seeds 1–40 (`surge`) the observed rates are 0.20 and 0.09.
* Units busy at t=0 are free by minute 35. Incidents are reported between minute 1 and 150 of a
  180-minute horizon.
* Reproducibility: same seed and same Python version give byte-identical files (tested on Python
  3.14, macOS). Another Python version may draw differently; the SHA-256 manifest will flag that.

## 8. v2 changes (phase 2)

The implementation follows sections 1 to 7 except for the changes below. Source:
`docs/implementation_plan.md`, section 1 (design changes agreed before coding),
`docs/evaluation.md`, Part B (fixes found on dev seeds 1 to 5 before the evaluation) and
`docs/evaluation.md`, Part A, section 6 (code review fixes).

### 8.1 Design changes agreed before implementation

| # | Phase-1 design | Implemented (v2) | Why |
|---|---|---|---|
| 1 | Fixed 1-minute ticks | Discrete-event engine (priority queue, deterministic tie-breaks) plus a snapshot every minute | Travel times are 0.1-min floats; events are exact and cheaper; snapshots still give the dashboard a per-minute timeline |
| 2 | Agents call tools to read state, propose and commit | Agent tools are read-only. Proposals and decisions are structured outputs (`output_type`). Only the coordinator applies engine `Action`s, and the engine validates each one | One validation point; no side effects hidden in LLM tool calls; the same contract for the offline and LLM backends |
| 3 | COUNTER such as "take the 2 RED, send the YELLOW to H2" | COUNTER is conditional acceptance only: "accept if lent a float doctor/nurse" (`FLOAT_DOCTOR`, `FLOAT_NURSE`) or "accept by using my critical reserve" (`RESERVE_RELEASE`, needs approval). No partial-load splitting | An ambulance carries at most 1 RED and goes to one destination, so the phase-1 example is physically impossible |
| 4 | Hospital state private | The bed board (free unreserved beds, queue length) is public to every condition, including B0. Staff loads stay private to each hospital agent | B0's rule "nearest capable hospital with a free bed" needs it. Revealing staff loads is the value of communication |
| 5 | Shift ends are hard | Staff holdover: a staff member past shift end takes no new patients unless no on-shift colleague of the same role at that hospital has the needed skill | Otherwise H1 loses all cardiac and burns cover at t = 120 in every scenario, patients are stuck, and the comparison is unfair |
| 6 | "Reached within golden window" | Care start = start of ER treatment at a capable hospital; for cardiac arrest, arrival of an ALS unit on scene (field defibrillation) | A precise, testable definition |
| 7 | Not specified | Responder teams gate scene access: a scene opens when every required capability is on scene, 10 min later if extrication, heavy rescue, fire suppression or hazmat is required. The first unit on scene performs START triage, which reveals the victims to every controller | Makes rescue teams a real bottleneck and keeps hidden truth hidden until triage |
| 8 | Planned | Not modelled: ICU dynamics, negotiation latency (decisions take zero simulated time) and the LLM rewrite of caller reports. Data step 8 was done as seeded wording noise instead (section 7; 8.3) | Scope and time |

Other implementation details that refine the design:

- Six SDK agents: `incident_assessment`, `dispatch`, `hospital_H1`, `hospital_H2`, `hospital_H3`,
  `resource_allocation`. Each has read-only `@function_tool`s and a strict Pydantic output type.
- Two backends behind one `DecisionBackend` interface. Offline: the rule-based policy runs inside
  a custom SDK `Model`, through `Runner`, with one real tool round trip per call. LLM: an OpenAI
  model (`EMS_LLM_MODEL`, default `gpt-4.1-mini`) with policy fallback on any model error.
- Messages carry two extra fields: `conversation_id` (one per negotiation thread) and `round`.
- Three ablation conditions were added: `full-no-float`, `full-no-diversion`,
  `full-no-escalation`.
- Escalation cases: `UNIT_CONTENTION`, `NO_ACCEPTING_HOSPITAL`, `FLOAT_STAFF`,
  `RESERVE_RELEASE`. Resource Allocation scores options with
  `utility = triage_weight × p_window(slack) − 0.01 × travel_min`,
  `p_window(slack) = 1 / (1 + exp(−slack / 5))`.
- B1 forces admission at the first-ranked hospital after 3 rounds without acceptance; Full asks
  Resource Allocation instead.
- Metrics are defined exactly in `docs/implementation_plan.md`, section 6. The survival-weighted
  score uses weights RED 3, YELLOW 2, GREEN 1.
- Evaluation protocol: dev seeds 1 to 5 for all debugging; eval seeds 7380 and 1001 to 1020 fixed
  in advance; paired comparisons on identical worlds with an exact sign test.

### 8.2 Fixes found on dev seeds (before the evaluation)

The first end-to-end comparison on dev seeds 1 to 5 showed the agents losing to B0 in surge and
constrained. A per-victim differential diagnosis found these causes. Details, evidence and
effects are in [`docs/evaluation.md`](evaluation.md), Part B.

| Fix | Problem | Change |
|---|---|---|
| F1 | Dispatch sent ambulances to scenes that could not be loaded yet (no access) while accessible scenes waited | `IncidentNeed.access_wait_min`; ALS and seat passes serve ready scenes first, then blocked scenes |
| F2 | Resource Allocation added access delay to travel time | Slack = deadline − max(ETA, access wait) − overhead |
| F3 | A deteriorated RED got an already-expired deadline | `TriagedVictim.minutes_since_deterioration`; 30-minute window counted from deterioration |
| F4 | Priority reports went stale after loading | `PATIENTS_LOADED` is an assessment trigger |
| F5 | The ALS pass could exceed the pre-triage ambulance cap that B0 obeys | The cap applies to every ambulance |
| D1 | Pre-registered: deadline-aware (expected-loss) dispatch order | Tried and rejected: lowered dev-seed scores (B1 0.5848 → 0.5352). Reverted |
| D2 | Pre-registered: send responders first where they complete scene access | Kept (very small effect) |

No numeric constant was tuned, and B0 was not changed.

### 8.3 Code review fixes (after the first evaluation)

A code review found that the caller-report wording was a deterministic function of hidden truth
(severe wording if and only if the worst triage was RED). The Incident Assessment agent's cue
lexicon recovered RED with 100% recall and 92% precision, and B0 did not use that information.
The first evaluation (378 runs) is superseded. Details: `docs/evaluation.md`, Part A, sections 6
and 7.

| Change | What | Where |
|---|---|---|
| Caller-wording noise | Generator 1.2.0: seeded under- and over-triage of the wording (`REPORT_UNDERTRIAGE_P = 0.20`, `REPORT_OVERTRIAGE_P = 0.10`), from separate random streams; only `caller_report` changes | Section 7 (step 8, assumptions); `emergency_sim/datagen.py` |
| `b0-cues` baseline | B0 plus the agents' public caller-report assessment rule: assessed priority order, ALS first to suspected RED. No messages, agents, reservations or negotiation. Decomposition: B0 → B0-cues (information) → B1 (coordination) → Full (negotiation) | `emergency_sim/control/greedy.py` (`use_caller_cues=True`) |
| Escalation semantics | `allow_escalation` governs every escalation to Resource Allocation, including `RESERVE_RELEASE` counters. `full-no-escalation` has no ESCALATE messages; float-staff counters remain (they are staff REQUESTs). Diversion requires escalation (checked when a configuration is built) | `emergency_sim/control/conditions.py` |
| Engine port adapter | Controllers get `EnginePortAdapter`, which exposes only the four `EnginePort` members, not the `Engine` object | `emergency_sim/engine/port.py` |
| Smaller fixes | `bls_for_red_only` dispatch validation; engine-refused outcomes always logged as INFORM with `ENGINE_REJECTED` and the attempted actions; LLM per-request timeout 60 s; tracing set per run, no process-global switch | `control/validation.py`, `control/coordinator.py`, `mas/backend.py` |

Conditions after the review (section 6 lists the phase-1 three):

| Condition | What it is |
|---|---|
| `b0` | Greedy baseline, FCFS, ignores caller wording |
| `b0-cues` | B0 with the caller-report assessment rule |
| `b1` | Agents, ACCEPT / REJECT only |
| `full` | Agents with COUNTER, escalation and diversion |
| `full-no-float`, `full-no-diversion`, `full-no-escalation` | Full without one mechanism |

Not yet done: public game rules are still defined in two places (engine and `control` /
`policies` constants); the dashboard keeps its own KPI fallback alongside
`evaluation.metrics.kpis_at`; line length is not enforced by a linter.

### 8.4 Results in one line

On the 21 held-out seeds (441 runs), the caller-report information alone hurts (B0-cues serves
fewer patients than B0 in surge and constrained). On the same information, coordination and
negotiation help, more under pressure (Full vs B0-cues survival score +0.065 in constrained,
p = 0.0026). Against plain B0, Full is faster (constrained response −8.703 min, p = 0.0015) but its
outcome gains are not significant. See `docs/evaluation.md`, Part A.
