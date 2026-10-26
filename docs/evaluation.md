# Evaluation

CSYE7380 Theory & Prac App AI Gen Model · Prof. Subrata Das · Shubham Bagwe

This document has two parts.

- **Part A: held-out evaluation** (sections 1 to 8): method, results on the evaluation seeds,
  ablations, interpretation, threats to validity, the code review that changed the evaluation,
  the superseded first results, and next steps. The evaluation was run once, after development
  and the review fixes were frozen.
- **Part B: change log during development** (dev seeds 1 to 5 only): how the agent conditions
  were debugged and what changed before the first evaluation. It is kept as written by the
  debugging agent. Its numbers come from the data before the caller-wording fix (section 6), so
  they are not comparable with Part A.

All numbers in Part A come from `results/summary.md`, `results/summary.csv` and
`results/paired.csv`. The ablation comparisons in section 3 and the sensitivity check in
section 5 are not in those files. They were computed from the 441 `metrics.json` files with the
project's own `emergency_sim.evaluation.report.paired_differences` on `load_results("results")`
(same sign test). A check confirmed that this function reproduces `paired.csv` exactly (936 of
936 rows).

**Read section 6 first if you saw the earlier results.** A code review found that the caller
reports leaked hidden truth to the agents. The data generator, the baselines and the ablation
semantics were fixed, and the evaluation was re-run. The first results (378 runs) are superseded;
section 7 lists them.

# Part A: held-out evaluation

## 1. Method

### 1.1 Conditions

| Condition | Agents | Reads caller reports (assessment rule) | Hospital replies | Float staff | Escalation to Resource Allocation | ALS diversion |
|---|---|---|---|---|---|---|
| `b0` | none (greedy controller) | no | none | no | no | no |
| `b0-cues` | none (greedy controller) | yes | none | no | no | no |
| `b1` | yes | yes | ACCEPT / REJECT | no | no | no |
| `full` | yes | yes | ACCEPT / REJECT / COUNTER | yes | yes | yes |
| `full-no-float` | yes | yes | ACCEPT / REJECT / COUNTER (reserve release only) | no | yes | yes |
| `full-no-diversion` | yes | yes | ACCEPT / REJECT / COUNTER | yes | yes | no |
| `full-no-escalation` | yes | yes | ACCEPT / REJECT / COUNTER (float staff only) | yes | no | no |

- **B0** is a competent baseline. It sees the same public view (including the bed board), obeys
  the same pre-triage ambulance cap and crew rules, sends ALS to triaged REDs, dispatches
  responder teams and redirects when a hospital is full. It serves incidents first come, first
  served and ignores the wording of caller reports. It never reserves beds, diverts or lends
  staff, and never sees staff loads.
- **B0-cues** is B0 plus the public caller-report assessment rule that the Incident Assessment
  agent uses (`policies.assessment`). It serves incidents in assessed priority order and sends an
  ALS first to a scene with a suspected RED. It has no messages, no agents, no bed reservations,
  no staff information and no negotiation.
- **Escalation semantics.** `allow_escalation` governs every escalation to Resource Allocation:
  unit contention, no accepting hospital, diversion and `RESERVE_RELEASE` counters. A float-staff
  counter is a staff `REQUEST`, not an escalation. So `full-no-escalation` has no `ESCALATE`
  message at all, but hospitals still COUNTER with float-staff conditions. Diversion requires
  escalation; a configuration with diversion but no escalation is rejected at construction.
- B1 sends a patient to its first-ranked hospital after 3 rounds without acceptance
  (`FORCED_ADMISSION`). Full asks Resource Allocation instead.

**Decomposition.** Each step adds one thing to the step before, on identical worlds:

| Step | Pair (treatment vs baseline) | What it adds |
|---|---|---|
| Information | B0-cues vs B0 | Caller-report assessment: priority order and ALS to suspected RED |
| Coordination | B1 vs B0-cues | Agents and messages: priority reports, hospital agents with staff information, bed reservation on ACCEPT, rankings |
| Negotiation | Full vs B1 | COUNTER (float staff, reserve release), escalation to Resource Allocation, ALS diversion |

The report also gives Full vs B0-cues (coordination + negotiation on the same information), Full
vs B0 and B1 vs B0.

### 1.2 Metrics

`P` = non-BLACK victims, `H` = 180-minute horizon, `R` = victims in `P` whose final triage is RED
(includes YELLOWs that deteriorated). Care start is the start of ER treatment at a capable
hospital; for cardiac arrest it is the arrival of an ALS unit on scene. Full formulas:
`docs/implementation_plan.md`, section 6.

| Metric | Definition | Better |
|---|---|---|
| Served rate | victims in `P` with care start ≤ H, divided by \|P\| | higher |
| RED in window (rate) | victims in `R` with care start ≤ their deadline, divided by \|R\| | higher |
| Survival score | Σ over `P` of W(final triage) × 1[care start ≤ deadline] / Σ W, with W = RED 3, YELLOW 2, GREEN 1 | higher |
| Unserved RED (`unserved_critical`) | victims in `R` with no care start by H | lower |
| Response (min) | mean over reached incidents of first ambulance on scene − report time | lower |
| Door-to-doctor (min) | mean of treatment start − hospital arrival | lower |
| RED response (min) | mean of first ALS on scene − report time, over initially-RED victims | lower |
| Deteriorated | YELLOWs that turned RED before care | lower |
| Conflicts | negotiation threads with at least one REJECT, COUNTER or ESCALATE | descriptive (lower is "better" in W/T/L) |

Deadlines: RED = report time + hidden golden window (20–60 min; 8–12 min for cardiac arrest);
deteriorated = deterioration time + 30 min; YELLOW = report + window; GREEN = report + 480 min.
The hidden windows are used only for scoring. Agents and baselines see only public nominal
windows.

### 1.3 Seeds and protocol

- **Data.** Generator 1.2.0: caller-report wording has seeded noise (`docs/simulation_design.md`,
  section 7). A caller with a RED victim uses mild wording with probability 0.20; a caller with no
  RED victim uses severe wording with probability 0.10.
- **Dev seeds 1 to 5**: the only seeds used for debugging and for every change (Part B).
- **Eval seeds 7380 and 1001 to 1020** (21 worlds per scenario): fixed in the plan
  (`docs/implementation_plan.md`, 8.3) before any result.
- Grid: 3 scenarios × 7 conditions × 21 seeds × 1 repeat = **441 runs**, offline backend, run once
  with the command in the README. 0 fallbacks and 0 controller omissions in every run.
- One repeat suffices offline because the offline backend is deterministic: the same spec gives
  byte-identical logs.
- **Paired comparisons.** For each scenario and seed, both conditions run on the identical world
  (same incidents, victims, caller reports, hidden windows, resources). The difference is
  treatment − baseline. W/T/L counts runs where the treatment is better / equal / worse. The
  p-value is the exact two-sided sign test (binomial, p = 0.5, ties excluded).
- Cells in the condition tables are mean ± std (ddof = 1) over the 21 runs.

### 1.4 What the offline results measure

The offline backend runs the rule-based policies inside real OpenAI Agents SDK `Agent`/`Runner`
calls (custom `Model`, one real tool round trip per call, strict structured outputs). The results
therefore measure the **coordination protocol and information sharing with rule-based agents**.
The LLM backend is implemented and tested with scripted fake models only. It was not run, because
no API key was available.

## 2. Held-out results

### 2.1 Conditions per scenario (copied from `results/summary.md`)

**normal** (mean 25.10 patients per run, 8.95 to 9.14 RED)

| Condition | n | Served rate | RED in window | Survival score | Unserved RED | Response (min) | Door-to-doctor (min) | Conflicts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| b0 | 21 | 0.894 ± 0.125 | 0.471 ± 0.203 | 0.685 ± 0.130 | 1.33 ± 1.62 | 17.0 ± 7.0 | 0.5 ± 1.3 | 0.0 ± 0.0 |
| b0-cues | 21 | 0.865 ± 0.157 | 0.486 ± 0.206 | 0.671 ± 0.158 | 1.43 ± 1.75 | 16.8 ± 5.6 | 0.5 ± 1.3 | 0.0 ± 0.0 |
| b1 | 21 | 0.894 ± 0.128 | 0.505 ± 0.215 | 0.694 ± 0.158 | 1.24 ± 1.67 | 16.2 ± 4.6 | 0.4 ± 1.1 | 0.3 ± 0.6 |
| full | 21 | 0.905 ± 0.118 | 0.565 ± 0.185 | 0.727 ± 0.148 | 1.00 ± 1.45 | 15.1 ± 3.8 | 0.4 ± 1.1 | 2.1 ± 2.0 |
| full-no-float | 21 | 0.905 ± 0.118 | 0.565 ± 0.185 | 0.727 ± 0.148 | 1.00 ± 1.45 | 15.1 ± 3.8 | 0.4 ± 1.1 | 2.0 ± 2.0 |
| full-no-diversion | 21 | 0.877 ± 0.148 | 0.518 ± 0.215 | 0.692 ± 0.171 | 1.43 ± 2.11 | 16.0 ± 4.6 | 0.4 ± 1.1 | 1.4 ± 1.5 |
| full-no-escalation | 21 | 0.893 ± 0.129 | 0.505 ± 0.215 | 0.693 ± 0.159 | 1.24 ± 1.67 | 16.2 ± 4.6 | 0.4 ± 1.1 | 0.4 ± 0.7 |

**surge** (mean 40.76 patients per run, 14.67 to 15.62 RED)

| Condition | n | Served rate | RED in window | Survival score | Unserved RED | Response (min) | Door-to-doctor (min) | Conflicts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| b0 | 21 | 0.852 ± 0.124 | 0.319 ± 0.161 | 0.585 ± 0.143 | 2.76 ± 2.57 | 32.3 ± 12.5 | 0.9 ± 1.3 | 0.0 ± 0.0 |
| b0-cues | 21 | 0.783 ± 0.171 | 0.312 ± 0.157 | 0.535 ± 0.175 | 4.00 ± 4.18 | 30.5 ± 11.9 | 1.2 ± 1.6 | 0.0 ± 0.0 |
| b1 | 21 | 0.807 ± 0.151 | 0.350 ± 0.205 | 0.571 ± 0.179 | 3.52 ± 3.54 | 27.8 ± 9.1 | 0.4 ± 0.6 | 3.3 ± 2.4 |
| full | 21 | 0.796 ± 0.168 | 0.357 ± 0.189 | 0.571 ± 0.177 | 3.81 ± 4.12 | 25.6 ± 7.4 | 0.4 ± 0.5 | 9.6 ± 3.1 |
| full-no-float | 21 | 0.786 ± 0.165 | 0.357 ± 0.185 | 0.563 ± 0.177 | 4.05 ± 4.13 | 25.2 ± 7.4 | 0.5 ± 0.7 | 9.1 ± 3.0 |
| full-no-diversion | 21 | 0.793 ± 0.165 | 0.365 ± 0.190 | 0.573 ± 0.175 | 3.86 ± 4.11 | 25.9 ± 7.4 | 0.4 ± 0.5 | 8.1 ± 3.1 |
| full-no-escalation | 21 | 0.809 ± 0.145 | 0.346 ± 0.214 | 0.573 ± 0.179 | 3.38 ± 3.43 | 27.6 ± 9.2 | 0.3 ± 0.5 | 3.0 ± 2.2 |

**constrained** (same incidents as surge, fewer resources; 15.76 to 16.67 RED)

| Condition | n | Served rate | RED in window | Survival score | Unserved RED | Response (min) | Door-to-doctor (min) | Conflicts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| b0 | 21 | 0.772 ± 0.132 | 0.271 ± 0.114 | 0.506 ± 0.134 | 4.19 ± 3.03 | 38.9 ± 11.6 | 2.7 ± 2.5 | 0.0 ± 0.0 |
| b0-cues | 21 | 0.714 ± 0.155 | 0.260 ± 0.141 | 0.459 ± 0.155 | 5.14 ± 3.75 | 34.0 ± 10.0 | 3.2 ± 2.6 | 0.0 ± 0.0 |
| b1 | 21 | 0.734 ± 0.160 | 0.306 ± 0.166 | 0.502 ± 0.164 | 4.90 ± 3.83 | 33.5 ± 8.7 | 1.4 ± 1.3 | 5.1 ± 2.4 |
| full | 21 | 0.739 ± 0.175 | 0.344 ± 0.170 | 0.524 ± 0.173 | 5.00 ± 3.97 | 30.2 ± 7.6 | 1.0 ± 1.1 | 11.9 ± 2.4 |
| full-no-float | 21 | 0.719 ± 0.168 | 0.313 ± 0.163 | 0.498 ± 0.168 | 5.24 ± 3.83 | 30.5 ± 7.8 | 1.6 ± 1.3 | 12.3 ± 2.6 |
| full-no-diversion | 21 | 0.754 ± 0.179 | 0.347 ± 0.171 | 0.533 ± 0.178 | 4.71 ± 4.05 | 30.5 ± 7.3 | 1.1 ± 1.0 | 10.6 ± 2.6 |
| full-no-escalation | 21 | 0.744 ± 0.166 | 0.315 ± 0.149 | 0.514 ± 0.163 | 4.81 ± 3.67 | 32.6 ± 9.6 | 0.9 ± 0.9 | 5.0 ± 2.3 |

The number of REDs differs between conditions because YELLOWs that deteriorate become RED.

### 2.2 Paired comparisons (from `results/paired.csv`)

Each cell is **mean difference · W/T/L · sign-test p**, n = 21 runs. Columns follow the
decomposition (information, coordination, negotiation), then the combined pairs. A p of 0.0000
means p < 0.00005. Conflicts rise in every agent comparison by construction (B0 and B0-cues have
no messages); they are in `summary.md` and are omitted here. "RED response" and "Deteriorated"
are in `paired.csv` only.

**normal**

| Metric | B0-cues vs B0 | B1 vs B0-cues | Full vs B1 | Full vs B0-cues | Full vs B0 | B1 vs B0 |
|---|---:|---:|---:|---:|---:|---:|
| Served rate | -0.030 · 1/15/5 · 0.2188 | +0.029 · 6/14/1 · 0.1250 | +0.011 · 3/16/2 · 1.0000 | +0.040 · 7/13/1 · 0.0703 | +0.011 · 7/11/3 · 0.3438 | -0.000 · 7/11/3 · 0.3438 |
| RED in window (rate) | +0.015 · 4/12/5 · 1.0000 | +0.019 · 5/12/4 · 1.0000 | +0.060 · 5/16/0 · 0.0625 | +0.079 · 8/11/2 · 0.1094 | +0.095 · 11/7/3 · 0.0574 | +0.034 · 8/8/5 · 0.5811 |
| Survival score | -0.015 · 4/11/6 · 0.7539 | +0.023 · 6/11/4 · 0.7539 | +0.033 · 6/13/2 · 0.2891 | +0.056 · 9/9/3 · 0.1460 | +0.042 · 11/5/5 · 0.2101 | +0.008 · 9/7/5 · 0.4240 |
| Unserved RED | +0.095 · 1/17/3 · 0.6250 | -0.190 · 5/15/1 · 0.2188 | -0.238 · 2/18/1 · 1.0000 | -0.429 · 5/16/0 · 0.0625 | -0.333 · 6/13/2 · 0.2891 | -0.095 · 5/14/2 · 0.4531 |
| Response (min) | -0.181 · 7/4/10 · 0.6291 | -0.574 · 10/3/8 · 0.8145 | -1.086 · 7/11/3 · 0.3438 | -1.660 · 12/2/7 · 0.3593 | -1.841 · 12/0/9 · 0.6636 | -0.755 · 10/1/10 · 1.0000 |
| Door-to-doctor (min) | +0.001 · 1/19/1 · 1.0000 | -0.119 · 4/15/2 · 0.6875 | +0.045 · 2/15/4 · 0.6875 | -0.074 · 4/14/3 · 1.0000 | -0.073 · 4/14/3 · 1.0000 | -0.118 · 4/14/3 · 1.0000 |
| RED response (min) | -1.156 · 9/4/8 · 1.0000 | -1.234 · 13/3/5 · 0.0963 | -3.134 · 7/12/2 · 0.1797 | -4.368 · 16/2/3 · 0.0044 | -5.524 · 18/0/3 · 0.0015 | -2.390 · 14/1/6 · 0.1153 |
| Deteriorated | +0.095 · 0/19/2 · 0.5000 | +0.048 · 0/20/1 · 1.0000 | -0.048 · 1/20/0 · 1.0000 | +0.000 · 0/21/0 · 1.0000 | +0.095 · 0/19/2 · 0.5000 | +0.143 · 0/18/3 · 0.2500 |

**surge**

| Metric | B0-cues vs B0 | B1 vs B0-cues | Full vs B1 | Full vs B0-cues | Full vs B0 | B1 vs B0 |
|---|---:|---:|---:|---:|---:|---:|
| Served rate | -0.069 · 1/9/11 · 0.0063 | +0.024 · 11/5/5 · 0.2101 | -0.011 · 5/8/8 · 0.5811 | +0.012 · 10/5/6 · 0.4545 | -0.056 · 5/6/10 · 0.3018 | -0.045 · 5/6/10 · 0.3018 |
| RED in window (rate) | -0.007 · 7/6/8 · 1.0000 | +0.038 · 10/5/6 · 0.4545 | +0.008 · 9/7/5 · 0.4240 | +0.045 · 13/3/5 · 0.0963 | +0.039 · 10/6/5 · 0.3018 | +0.031 · 8/5/8 · 1.0000 |
| Survival score | -0.049 · 3/3/15 · 0.0075 | +0.035 · 11/4/6 · 0.3323 | +0.000 · 10/4/7 · 0.6291 | +0.036 · 15/2/4 · 0.0192 | -0.013 · 9/2/10 · 1.0000 | -0.014 · 6/4/11 · 0.3323 |
| Unserved RED | +1.238 · 3/11/7 · 0.3438 | -0.476 · 7/9/5 · 0.7744 | +0.286 · 3/12/6 · 0.5078 | -0.190 · 9/8/4 · 0.2668 | +1.048 · 4/9/8 · 0.3877 | +0.762 · 4/9/8 · 0.3877 |
| Response (min) | -1.823 · 14/1/6 · 0.1153 | -2.740 · 15/0/6 · 0.0784 | -2.164 · 13/0/8 · 0.3833 | -4.903 · 12/0/9 · 0.6636 | -6.727 · 15/0/6 · 0.0784 | -4.563 · 14/0/7 · 0.1892 |
| Door-to-doctor (min) | +0.320 · 5/5/11 · 0.2101 | -0.781 · 15/5/1 · 0.0005 | -0.086 · 6/10/5 · 1.0000 | -0.867 · 15/4/2 · 0.0023 | -0.548 · 12/4/5 · 0.1435 | -0.461 · 13/5/3 · 0.0213 |
| RED response (min) | -2.908 · 14/1/6 · 0.1153 | +0.375 · 10/0/11 · 1.0000 | -3.814 · 16/0/5 · 0.0266 | -3.439 · 16/0/5 · 0.0266 | -6.347 · 17/0/4 · 0.0072 | -2.533 · 15/0/6 · 0.0784 |
| Deteriorated | +0.905 · 2/10/9 · 0.0654 | -0.333 · 7/8/6 · 1.0000 | +0.143 · 2/14/5 · 0.4531 | -0.190 · 7/9/5 · 0.7744 | +0.714 · 1/12/8 · 0.0391 | +0.571 · 1/12/8 · 0.0391 |

**constrained**

| Metric | B0-cues vs B0 | B1 vs B0-cues | Full vs B1 | Full vs B0-cues | Full vs B0 | B1 vs B0 |
|---|---:|---:|---:|---:|---:|---:|
| Served rate | -0.058 · 3/5/13 · 0.0213 | +0.020 · 11/3/7 · 0.4807 | +0.005 · 9/5/7 · 0.8036 | +0.025 · 12/5/4 · 0.0768 | -0.034 · 7/5/9 · 0.8036 | -0.038 · 5/3/13 · 0.0963 |
| RED in window (rate) | -0.011 · 6/4/11 · 0.3323 | +0.046 · 13/2/6 · 0.1671 | +0.038 · 11/6/4 · 0.1185 | +0.084 · 15/3/3 · 0.0075 | +0.073 · 13/1/7 · 0.2632 | +0.035 · 10/3/8 · 0.8145 |
| Survival score | -0.046 · 5/4/12 · 0.1435 | +0.042 · 15/0/6 · 0.0784 | +0.022 · 14/2/5 · 0.0636 | +0.065 · 17/1/3 · 0.0026 | +0.018 · 14/0/7 · 0.1892 | -0.004 · 10/0/11 · 1.0000 |
| Unserved RED | +0.952 · 3/7/11 · 0.0574 | -0.238 · 11/3/7 · 0.4807 | +0.095 · 6/7/8 · 0.7905 | -0.143 · 11/4/6 · 0.3323 | +0.810 · 4/8/9 · 0.2668 | +0.714 · 7/4/10 · 0.6291 |
| Response (min) | -4.848 · 15/1/5 · 0.0414 | -0.491 · 12/0/9 · 0.6636 | -3.363 · 18/0/3 · 0.0015 | -3.854 · 14/0/7 · 0.1892 | -8.703 · 18/0/3 · 0.0015 | -5.339 · 19/0/2 · 0.0002 |
| Door-to-doctor (min) | +0.570 · 5/2/14 · 0.0636 | -1.846 · 16/1/4 · 0.0118 | -0.382 · 11/1/9 · 0.8238 | -2.228 · 19/1/1 · 0.0000 | -1.658 · 17/0/4 · 0.0072 | -1.276 · 15/0/6 · 0.0784 |
| RED response (min) | -2.991 · 12/1/8 · 0.5034 | +0.198 · 9/0/12 · 0.6636 | -3.890 · 13/0/8 · 0.3833 | -3.692 · 14/0/7 · 0.1892 | -6.683 · 16/0/5 · 0.0266 | -2.793 · 12/0/9 · 0.6636 |
| Deteriorated | +0.762 · 3/7/11 · 0.0574 | -0.143 · 9/6/6 · 0.6072 | +0.286 · 2/12/7 · 0.1797 | +0.143 · 5/7/9 · 0.4240 | +0.905 · 4/4/13 · 0.0490 | +0.619 · 5/4/12 · 0.1435 |

## 3. Ablations

Full against each ablation on identical worlds (computed with `paired_differences`; see the note
at the top). Same cell format as section 2.2. `full-no-escalation vs b1` isolates the float-staff
counters: with the new escalation semantics, `full-no-escalation` differs from B1 only in
allowing COUNTER with float-staff conditions.

**normal**

| Metric | Full vs no-float | Full vs no-diversion | Full vs no-escalation | no-escalation vs B1 |
|---|---:|---:|---:|---:|
| Served rate | +0.000 · 0/21/0 · 1.0000 | +0.028 · 4/17/0 · 0.1250 | +0.012 · 3/16/2 · 1.0000 | -0.001 · 0/20/1 · 1.0000 |
| RED in window (rate) | +0.000 · 0/21/0 · 1.0000 | +0.048 · 3/18/0 · 0.2500 | +0.060 · 5/16/0 · 0.0625 | +0.000 · 0/21/0 · 1.0000 |
| Survival score | +0.000 · 0/21/0 · 1.0000 | +0.035 · 4/17/0 · 0.1250 | +0.034 · 6/13/2 · 0.2891 | -0.001 · 0/20/1 · 1.0000 |
| Unserved RED | +0.000 · 0/21/0 · 1.0000 | -0.429 · 3/18/0 · 0.2500 | -0.238 · 2/18/1 · 1.0000 | +0.000 · 0/21/0 · 1.0000 |
| Response (min) | +0.000 · 0/21/0 · 1.0000 | -0.891 · 4/15/2 · 0.6875 | -1.086 · 7/11/3 · 0.3438 | +0.000 · 0/21/0 · 1.0000 |

**surge**

| Metric | Full vs no-float | Full vs no-diversion | Full vs no-escalation | no-escalation vs B1 |
|---|---:|---:|---:|---:|
| Served rate | +0.010 · 5/15/1 · 0.2188 | +0.003 · 3/17/1 · 0.6250 | -0.014 · 6/7/8 · 0.7905 | +0.002 · 3/15/3 · 1.0000 |
| RED in window (rate) | -0.000 · 4/15/2 · 0.6875 | -0.008 · 0/18/3 · 0.2500 | +0.011 · 10/6/5 · 0.3018 | -0.004 · 2/17/2 · 1.0000 |
| Survival score | +0.008 · 5/14/2 · 0.4531 | -0.002 · 1/17/3 · 0.6250 | -0.001 · 11/2/8 · 0.6476 | +0.002 · 3/16/2 · 1.0000 |
| Unserved RED | -0.238 · 4/16/1 · 0.3750 | -0.048 · 2/18/1 · 1.0000 | +0.429 · 3/11/7 · 0.3438 | -0.143 · 3/18/0 · 0.2500 |
| Response (min) | +0.357 · 3/15/3 · 1.0000 | -0.316 · 5/13/3 · 0.7266 | -1.972 · 14/0/7 · 0.1892 | -0.192 · 5/13/3 · 0.7266 |

**constrained**

| Metric | Full vs no-float | Full vs no-diversion | Full vs no-escalation | no-escalation vs B1 |
|---|---:|---:|---:|---:|
| Served rate | +0.020 · 10/10/1 · 0.0117 | -0.015 · 1/14/6 · 0.1250 | -0.005 · 6/6/9 · 0.6072 | +0.010 · 5/14/2 · 0.4531 |
| RED in window (rate) | +0.031 · 7/13/1 · 0.0703 | -0.003 · 3/15/3 · 1.0000 | +0.029 · 11/6/4 · 0.1185 | +0.009 · 5/13/3 · 0.7266 |
| Survival score | +0.026 · 13/8/0 · 0.0002 | -0.009 · 3/12/6 · 0.5078 | +0.010 · 12/1/8 · 0.5034 | +0.013 · 6/12/3 · 0.5078 |
| Unserved RED | -0.238 · 6/11/4 · 0.7539 | +0.286 · 1/16/4 · 0.3750 | +0.190 · 6/4/11 · 0.3323 | -0.095 · 4/14/3 · 1.0000 |
| Response (min) | -0.310 · 6/9/6 · 1.0000 | -0.309 · 4/11/6 · 0.7539 | -2.423 · 15/0/6 · 0.0784 | -0.941 · 8/11/2 · 0.1094 |

In normal, Full and full-no-float give identical outcome metrics in all 21 runs. Full lent a
float staff member in one run there (mean 0.05), with no effect on the metrics.

### 3.1 How often each mechanism is used (means per run, from `results/summary.csv`)

| Scenario | Condition | Agent calls | Messages | ESCALATE msgs | Bed rejections | Float-staff cases | Float-staff competition | Reserve-release cases | Unit-contention cases | No-accepting-hospital cases | Forced admissions | Float staff lent | Diversions |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| normal | b1 | 75.4 | 77.2 | 0.00 | 0.33 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| normal | full | 76.5 | 80.1 | 2.00 | 0.05 | 0.05 | 0.00 | 0.24 | 1.76 | 0.00 | 0.00 | 0.05 | 0.52 |
| normal | full-no-float | 76.4 | 80.0 | 2.00 | 0.05 | 0.00 | 0.00 | 0.24 | 1.76 | 0.00 | 0.00 | 0.00 | 0.52 |
| normal | full-no-diversion | 76.3 | 78.2 | 1.29 | 0.05 | 0.05 | 0.00 | 0.19 | 1.10 | 0.00 | 0.00 | 0.05 | 0.00 |
| normal | full-no-escalation | 75.6 | 77.8 | 0.00 | 0.24 | 0.19 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.10 | 0.00 |
| surge | b1 | 118.5 | 125.0 | 0.00 | 3.29 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.29 | 0.00 | 0.00 |
| surge | full | 123.8 | 132.6 | 8.00 | 0.76 | 0.90 | 0.19 | 1.52 | 6.38 | 0.10 | 0.00 | 0.48 | 0.57 |
| surge | full-no-float | 124.3 | 130.0 | 7.48 | 1.67 | 0.00 | 0.00 | 1.24 | 6.19 | 0.05 | 0.00 | 0.00 | 0.62 |
| surge | full-no-diversion | 121.0 | 129.5 | 6.62 | 0.67 | 0.86 | 0.19 | 1.48 | 5.05 | 0.10 | 0.00 | 0.43 | 0.00 |
| surge | full-no-escalation | 117.6 | 124.9 | 0.00 | 2.24 | 0.71 | 0.00 | 0.00 | 0.00 | 0.00 | 0.19 | 0.43 | 0.00 |
| constrained | b1 | 117.5 | 121.6 | 0.00 | 5.14 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.43 | 0.00 | 0.00 |
| constrained | full | 127.5 | 138.0 | 8.90 | 0.90 | 2.38 | 0.10 | 1.76 | 6.67 | 0.43 | 0.00 | 0.67 | 0.67 |
| constrained | full-no-float | 126.7 | 134.1 | 8.81 | 3.52 | 0.00 | 0.00 | 1.52 | 6.62 | 0.62 | 0.00 | 0.00 | 0.81 |
| constrained | full-no-diversion | 124.7 | 135.5 | 7.81 | 0.86 | 2.29 | 0.00 | 1.90 | 5.43 | 0.43 | 0.00 | 0.76 | 0.00 |
| constrained | full-no-escalation | 118.3 | 126.5 | 0.00 | 2.62 | 2.38 | 0.00 | 0.00 | 0.00 | 0.00 | 0.29 | 0.62 | 0.00 |

Case columns are `conflicts_by_kind`; "Bed rejections" are threads with a REJECT only. B0 and
B0-cues have no agent calls or messages. `full-no-escalation` has 0 ESCALATE messages and 0
reserve-release cases, as the new semantics require.

## 4. Interpretation

### 4.1 Decomposition: what each step adds

| Step | Main effects (W/T/L, p) | Reading |
|---|---|---|
| Information (B0-cues vs B0) | Surge: served −0.069 (1/9/11, p = 0.0063), survival −0.049 (3/3/15, p = 0.0075). Constrained: served −0.058 (3/5/13, p = 0.0213), survival −0.046 (n.s.). Response time falls (constrained −4.848 min, p = 0.0414) | **Information alone hurts.** Priority order by noisy caller cues reaches some scenes faster but serves fewer patients |
| Coordination (B1 vs B0-cues) | Door-to-doctor −0.781 min in surge (15/5/1, p = 0.0005) and −1.846 min in constrained (16/1/4, p = 0.0118). Survival +0.035 (surge) and +0.042 (constrained), n.s. | Hospital agents with staff information and bed reservation get patients to a doctor faster. Outcome gains point the right way but are not significant |
| Negotiation (Full vs B1) | Constrained: response −3.363 min (18/0/3, p = 0.0015), survival +0.022 (p = 0.0636). Normal: RED in window +0.060 (5/16/0, p = 0.0625) | Counters replace rejections (bed-rejection threads 5.14 → 0.90 per run in constrained); response time falls |
| Coordination + negotiation (Full vs B0-cues) | Constrained: survival +0.065 (17/1/3, p = 0.0026), RED in window +0.084 (15/3/3, p = 0.0075), door-to-doctor −2.228 (19/1/1). Surge: survival +0.036 (15/2/4, p = 0.0192), door-to-doctor −0.867 (p = 0.0023). Normal: all n.s. | **On the same information, coordination and negotiation help, more under pressure** |

### 4.2 Why information alone hurts

1. **Urgency-first order starves YELLOW-only scenes.** B0-cues serves suspected-RED scenes first.
   Older YELLOW-only scenes wait, and an untreated YELLOW turns RED at 60% of its window, with only
   30 minutes left. B0-cues has more deteriorated victims than B0: +0.905 per run in surge
   (2/10/9, p = 0.0654) and +0.762 in constrained (3/7/11, p = 0.0574). Unserved RED rises by
   +1.238 (surge, p = 0.3438) and +0.952 (constrained, p = 0.0574).
2. **The cue is noisy.** About 1 in 8 RED incidents is reported mildly, and about 1 in 7 severe
   reports has no RED (cue recall ≈ 0.875, precision ≈ 0.864 on surge seeds 1–40). Wrongly
   prioritised scenes take units from the rest.
3. **B0's first-come-first-served order bounds the wait** of every scene, which protects YELLOWs.
4. **The agents inherit the problem.** Full has more deteriorated victims than B0: +0.714 in surge
   (1/12/8, p = 0.0391) and +0.905 in constrained (4/4/13, p = 0.0490). Against B0-cues, Full's
   deterioration difference is small and not significant (−0.190 and +0.143).
5. **This is the open design issue from the dev-seed diagnosis.** Part B, "Remaining", item 1
   found the same mechanism on dev seeds. The pre-registered fix (D1, expected-loss ordering)
   lowered dev-seed scores and was rejected by the pre-registered rule, so urgency order stayed.

### 4.3 Full vs plain B0

| Metric | normal | surge | constrained |
|---|---|---|---|
| Response (min) | −1.841 (12/0/9, p = 0.6636) | −6.727 (15/0/6, p = 0.0784) | −8.703 (18/0/3, p = 0.0015) |
| RED response (min) | −5.524 (18/0/3, p = 0.0015) | −6.347 (17/0/4, p = 0.0072) | −6.683 (16/0/5, p = 0.0266) |
| RED in window (rate) | +0.095 (11/7/3, p = 0.0574) | +0.039 (10/6/5, p = 0.3018) | +0.073 (13/1/7, p = 0.2632) |
| Survival score | +0.042 (11/5/5, p = 0.2101) | −0.013 (9/2/10, p = 1.0000) | +0.018 (14/0/7, p = 0.1892) |
| Served rate | +0.011 (7/11/3, p = 0.3438) | −0.056 (5/6/10, p = 0.3018) | −0.034 (7/5/9, p = 0.8036) |
| Door-to-doctor (min) | −0.073 (n.s.) | −0.548 (n.s.) | −1.658 (17/0/4, p = 0.0072) |

- Full reaches scenes and REDs faster than B0 and gets patients to a doctor faster in
  constrained.
- RED-in-window gains (+0.039 to +0.095) are not significant.
- Full serves slightly fewer patients than B0 in surge and constrained (not significant).
- **Plain B0 is a strong baseline**, for the reason in section 4.2: its FCFS order protects
  YELLOW-only scenes. Full recovers the loss that information causes (Full vs B0-cues is
  positive), but not enough to beat B0 clearly on outcomes.

### 4.4 Ablations

| Mechanism | Evidence | Reading |
|---|---|---|
| Float staff | Constrained, Full vs full-no-float: survival +0.026 (13/8/0, p = 0.0002), served +0.020 (10/10/1, p = 0.0117). Bed-rejection threads 3.52 → 0.90 per run | The one mechanism with a clear outcome effect. A lent doctor or nurse turns a rejection into an admission (example: thread NEG-0015, H1 trauma doctor) |
| Diversion | No significant effect in any scenario. Normal survival +0.035 (4/17/0, p = 0.1250); constrained −0.009 (n.s.) | Rarely used (0.52 to 0.67 diversions per run) |
| Escalation | Full vs full-no-escalation: no significant outcome effect. Response −1.972 min (surge, p = 0.1892) and −2.423 min (constrained, p = 0.0784); normal RED in window +0.060 (5/16/0, p = 0.0625) | Unit-contention choices often fall between two blocked scenes (Part B, "Remaining", item 3) |
| Float-staff counters alone | full-no-escalation vs B1: all differences small and n.s. (constrained survival +0.013, p = 0.5078) | Used as often as in Full (2.38 float-staff cases per run in constrained), but the gain over B1 is small. The float-staff effect shows only together with the other Full mechanisms; the cause was not analysed |

### 4.5 Dev seeds vs held-out seeds

Part B's dev-seed numbers were produced with the earlier data, in which caller wording revealed
the worst triage. They are not comparable with Part A and are kept only as the record of how
the agents were debugged. Only the 21 held-out seeds are used for claims.

## 5. Threats to validity

| Threat | Effect | What was done |
|---|---|---|
| Synthetic data | Effects may not transfer to real dispatch data | Seeded generator with documented assumptions; numbers not written by an LLM; 57 validation checks (57/57 pass) |
| One city layout | 3 hospitals, 3 stations and capabilities are fixed; seeds vary incidents, victims, occupancy, staff and caller wording only | Results are claims about this layout. Other layouts are future work |
| Caller-report lexicon | The severity-cue lexicon (`policies.assessment.SEVERITY_CUES`) was written against the generator's own phrasings. Real caller language is more varied | The wording noise limits what the lexicon can recover (recall ≈ 0.875, precision ≈ 0.864). B0-cues uses the same lexicon, so the information is matched across conditions |
| Noise rates are assumptions | `REPORT_UNDERTRIAGE_P = 0.20` and `REPORT_OVERTRIAGE_P = 0.10` are documented assumptions about telephone triage, not measured values. Other rates would change the value of the information | They were set before the re-run and not tuned to any result. The B0 → B0-cues → B1 → Full decomposition shows which effects depend on the information |
| Rule-based agents, not an LLM | Results measure the protocol and information sharing, not LLM reasoning | Stated everywhere. LLM backend tested with scripted fake models only; not run (no key) |
| Multiple comparisons | `summary.md` has 126 paired tests (6 pairs × 7 metrics × 3 scenarios); 31 have p < 0.05 | With Bonferroni (α = 0.05 / 126 ≈ 0.0004), only these remain: the 13 conflict-count tests, constrained Full vs B0-cues door-to-doctor (−2.228 min, 19/1/1) and constrained B1 vs B0 response time (−5.339 min, 19/0/2, p = 0.0002). **No served-rate, RED-in-window or survival-score difference survives.** Outside this family, the float-staff ablation (constrained survival, 13/8/0, p = 0.0002) would also pass. Other p-values are descriptive |
| Sign test | Ignores the size of differences and drops ties; low power with many ties | W/T/L and mean differences reported next to every p |
| Eval seed 7380 seen during development | 7380 is the seed of the committed data. Before the debugging rounds, the control-layer implementer ran all conditions on seeds 1–3 **and 7380** and reported outcome numbers, e.g. constrained served 22 (Full) vs 34 (full-no-escalation) at 7380. That report triggered the unit-contention diagnosis. The end-to-end tests also run on 7380 but assert only structural properties | No rule or parameter was changed on it; debugging used dev seeds only (Part B). Sensitivity check below: excluding 7380 gives the same directions |
| Eval seeds used twice | The v1 evaluation (section 7) and the reviewer's ablation ran on the same eval seeds before this re-run | No agent policy, B0 rule or parameter was tuned on them. The fixes change the data (wording noise), add a baseline and correct ablation semantics; the noise rates are documented assumptions |
| Hidden-truth scoring | Golden windows are drawn by the generator; the survival score is a proxy for survival | Windows follow the documented ranges (`docs/simulation_design.md`, assumptions); agents never see them |
| Simplifications | No ICU dynamics; negotiation takes zero simulated time; caller reports are templated with seeded noise, not LLM-rewritten | Documented in `docs/simulation_design.md` (v2 changes) |
| Same author for baseline and agents | B0 could be a strawman | B0 shares the public view, the pre-triage cap and crew rules; F5 removed an agent advantage over B0; B0 was never changed. B0-cues gives a baseline with the agents' information |

**Sensitivity check: eval seeds 1001–1020 only (n = 20, seed 7380 excluded).**

| Scenario | Comparison | Metric | Mean diff | W/T/L | Sign-test p |
|---|---|---|---:|---:|---:|
| surge | b0-cues vs b0 | Served rate | -0.071 | 1/9/10 | 0.0117 |
| surge | b0-cues vs b0 | Survival score | -0.050 | 3/3/14 | 0.0127 |
| constrained | b0-cues vs b0 | Served rate | -0.057 | 3/5/12 | 0.0352 |
| surge | full vs b0-cues | Survival score | +0.035 | 14/2/4 | 0.0309 |
| constrained | full vs b0-cues | Survival score | +0.054 | 16/1/3 | 0.0044 |
| constrained | full vs b0-cues | RED in window | +0.071 | 14/3/3 | 0.0127 |
| normal | full vs b0 | RED in window | +0.106 | 11/7/2 | 0.0225 |
| surge | full vs b0 | Served rate | -0.060 | 4/6/10 | 0.1796 |
| constrained | full vs b0 | RED in window | +0.064 | 12/1/7 | 0.3593 |
| constrained | full vs b0 | Response (min) | -8.645 | 17/0/3 | 0.0026 |
| constrained | full vs b1 | Response (min) | -3.345 | 17/0/3 | 0.0026 |

All directions are unchanged.

## 6. Code review and the information-oracle fix

### 6.1 Finding

| Item | Detail |
|---|---|
| Problem | Caller-report wording was a deterministic function of hidden truth: the severe phrase appeared if and only if the worst triage at the incident was RED |
| Consequence | The Incident Assessment agent's keyword cues recovered RED with 100% recall and 92% precision before triage. B0 ignored the wording. The agents' gain over B0 therefore mixed coordination with an information oracle that B0 did not use |
| Reviewer's ablation (v1 data, eval seeds, normal) | With cues disabled, Full RED in window fell from 0.581 to 0.450, against B0 at 0.471. The gain over B0 vanished |

### 6.2 Fixes

| Fix | What changed |
|---|---|
| Caller-wording noise (`datagen` 1.2.0) | Seeded noise from separate streams: `REPORT_UNDERTRIAGE_P = 0.20` (RED caller uses mild wording), `REPORT_OVERTRIAGE_P = 0.10` (non-RED caller uses severe wording). Documented assumptions, not tuned. Only `caller_report` changed; every other draw is identical. Validator 57/57. Cue recall ≈ 0.875, precision ≈ 0.864 (surge, seeds 1–40). See `docs/simulation_design.md`, section 7 |
| New baseline `b0-cues` | B0 plus the same public caller-report assessment rule as the agents: assessed priority order, ALS first to suspected RED. No messages, agents, reservations or negotiation. Gives the decomposition B0 → B0-cues (information) → B1 (coordination) → Full (negotiation) |
| Ablation semantics | `allow_escalation` now governs every escalation, including `RESERVE_RELEASE` counters. `full-no-escalation` has no ESCALATE messages; float-staff counters remain (they are REQUESTs). Diversion requires escalation (configuration check) |

Other review fixes (no effect on the decision rules):

| Fix | What changed |
|---|---|
| Engine port adapter | `engine/port.py` (`EnginePortAdapter`) exposes only the four `EnginePort` members, so controllers cannot reach engine internals or hidden truth |
| Dispatch validation | New check `bls_for_red_only`: a BLS order to an incident whose only waiting victims are RED is a violation, and the orders section falls back to the policy |
| Engine-refused outcomes | Always logged as an `INFORM` with resolution `ENGINE_REJECTED` and the attempted actions |
| LLM backend | Per-request timeout 60 s; a timeout falls back to the policy. No process-global tracing switch: tracing is set per run through `RunConfig` |
| Tests and CLI | SDK deprecation warning filtered in pytest; `--verbose` prints tracebacks on errors |
| Dashboard | Fixes found by replaying real logs |

### 6.3 Not yet done

- Public game rules are still defined in two places: the engine and the constants in
  `control` / `policies`.
- The dashboard keeps its own KPI fallback alongside `evaluation.metrics.kpis_at`.
- Line length is not enforced by a linter (no formatter is installed).

## 7. v1 results superseded

The first held-out evaluation (378 runs = 3 scenarios × 6 conditions × 21 seeds, data before
generator 1.2.0) is superseded by this one. It was run on data in which caller wording revealed
the worst triage (section 6.1), and it had no B0-cues baseline, so its Full vs B0 gains mixed
coordination with an information oracle. Its files are archived outside the repository. Its
headline numbers, for the record:

| Scenario | Comparison | Metric | v1 (378 runs) | v2 (this evaluation) |
|---|---|---|---|---|
| normal | Full vs B0 | RED in window (rate) | +0.111 · 13/8/0 · 0.0002 | +0.095 · 11/7/3 · 0.0574 |
| normal | Full vs B0 | Survival score | +0.050 · 13/6/2 · 0.0074 | +0.042 · 11/5/5 · 0.2101 |
| surge | Full vs B0 | Response (min) | -6.720 · 17/0/4 · 0.0072 | -6.727 · 15/0/6 · 0.0784 |
| surge | Full vs B0 | Served rate | -0.056 · 4/6/11 · 0.1185 | -0.056 · 5/6/10 · 0.3018 |
| constrained | Full vs B0 | Survival score | +0.035 · 17/0/4 · 0.0072 | +0.018 · 14/0/7 · 0.1892 |
| constrained | Full vs B0 | Response (min) | -7.034 · 18/0/3 · 0.0015 | -8.703 · 18/0/3 · 0.0015 |
| constrained | Full vs B0 | RED in window (rate) | +0.091 · 14/2/5 · 0.0636 | +0.073 · 13/1/7 · 0.2632 |
| constrained | Full vs B1 | Survival score | +0.024 · 15/2/4 · 0.0192 | +0.022 · 14/2/5 · 0.0636 |

In v1, Full's normal RED in window was 0.581 (v2: 0.565). The v1 claims "Full beats B0 on RED in
window and survival score in normal (p = 0.0002, 0.0074) and on survival score in constrained
(p = 0.0072)" no longer hold.

## 8. Next steps

1. **Protect YELLOW-only scenes from starvation.** This is now the main open design issue,
   because it makes information hurt (section 4.2). For example, an ageing term in the dispatch
   order. Pre-register the rule, evaluate it on dev seeds only, then on new held-out seeds.
2. **Run the LLM backend** (`--backend llm --repeats 3`) on surge and constrained for B0-cues,
   B1 and Full, and compare with the offline policies: outcome metrics, fallback rate, and how
   often the LLM decision differs from the policy on the same input.
3. **Caller-report noise sensitivity.** Re-run with other under- and over-triage rates to show
   how the value of the information depends on them.
4. **Responder bottleneck.** Prefer scenes where one more team completes access (Part B,
   "Remaining", item 4).
5. **Escalation.** Skip unit-contention cases when both scenes are blocked, since the choice then
   changes nothing.
6. **Generalisation and statistics.** More city layouts and hospital mixes; a Wilcoxon
   signed-rank test or bootstrap intervals next to the sign test; a Holm correction across the
   headline tests.
7. **Review leftovers** (section 6.3).

# Part B: change log during development

This part was written by the debugging agent during development. Its protocol notes:

- Every diagnosis and every check of a fix used **dev seeds 1 to 5 only**. Results for the
  evaluation seeds (7380 and 1001 to 1020) were not run or looked at during this work.
  (Disclosure added later: seed 7380 is the committed-data seed and was seen during
  implementation; see Part A, section 5.)
- Every number below is from the **offline backend**: the rule-based policies running inside real
  SDK `Agent`/`Runner` calls. The scratch harness called the same policies directly, without the
  SDK. Its numbers match `evaluation.runner.run_simulation` exactly; this was checked on all
  90 dev runs.
- No numeric constant was tuned. One constant was added (`PolicyParams.heavy_access_min = 10.0`).
  It copies a public engine rule (plan 4.3) so that the policies can estimate scene access. It is
  not a tuning knob, and a test checks that it equals `SimParams().heavy_access_min`.
- B0 was not changed.

## Change log during development (dev seeds 1–5 only)

### Starting point and method

Means over dev seeds 1 to 5, offline backend, before any change:

| Scenario | Condition | served | RED in window | score | resp. mean | unserved crit. |
|---|---|---|---|---|---|---|
| surge | b0 | 0.799 | 0.347 | 0.592 | 32.6 | 3.0 |
| surge | b1 | 0.791 | 0.314 | 0.565 | 29.2 | 3.6 |
| surge | full | 0.759 | 0.298 | 0.542 | 28.3 | 4.2 |
| surge | full-no-escalation | 0.811 | 0.363 | 0.594 | 31.4 | 3.2 |
| constrained | b0 | 0.768 | 0.365 | 0.557 | 37.1 | 3.6 |
| constrained | b1 | 0.708 | 0.213 | 0.479 | 30.4 | 5.4 |
| constrained | full | 0.629 | 0.244 | 0.447 | 29.7 | 7.0 |
| constrained | full-no-escalation | 0.732 | 0.227 | 0.493 | 31.5 | 4.8 |

There were two failures:

1. B1 was below B0, although B1 reached scenes faster.
2. Full was below full-no-escalation.

**Method: per-victim differential diagnosis.**

- For every victim, the event log gives the victim's timeline: report, first ambulance on scene,
  access granted, loaded (and by which unit type), hospital, arrival, bedded, and treatment
  start.
- For each pair of conditions on the same world, the diagnosis compared victims that one
  condition served within their window and the other did not. Each lost victim was attributed to
  the phase with the largest delay.
- Two budgets were added on top of that, built from the per-minute snapshots:
  - an ambulance-minute budget by status, which separates time staged at a scene that is not yet
    accessible;
  - responder-minutes held at scenes that are already accessible while other scenes still wait
    for access.

**Where the time was lost, before any fix.** Across surge and constrained, seeds 1 to 5, B1
lost 42 victims that B0 served in their window (survival weight 99) and won 27 (weight 52).

- 36 of the 42 losses were before the hospital:
  - 15 never loaded;
  - 8 never reached;
  - 8 lost the most time in the response phase;
  - 2 lost it at the scene.
- Only 6 losses were in hospital phases (queue or staff).

Most of the lost REDs were YELLOWs that had deteriorated while their incident waited. The
budgets showed that pre-hospital time went to idle units:

| Constrained, per run | b0 | b1 | full |
|---|---|---|---|
| Ambulance-minutes staged at an inaccessible scene | 120 | 165 | 276 |
| Responder-minutes held at an accessible scene while other scenes waited for access | 138 | 227 | 280 |
| Ambulance loads | 22.8 | 20.8 | 16.8 |

Comparing Full with full-no-escalation found 30 victims lost, mostly never loaded. They traced
back to UNIT_CONTENTION choices that sent ALS units to scenes that could not be loaded.

### F1: Dispatch ignored when a scene becomes accessible

- **Symptom.** Ambulances waited at blocked scenes while patients at accessible scenes waited for
  an ambulance.
  - Example: constrained seed 3, t=97.4. The only free ALS went to INC-07, an untriaged
    suspected-RED collision whose police team was not even dispatched yet. It waited there from
    t=105.9 until access at t=145.3. Meanwhile one GREEN patient at INC-04, which had been
    accessible since t=68.8, was never picked up. INC-04 therefore never cleared, and its fire
    engine and one of the two police teams were held there until the end of the run: 224
    responder-minutes during which other scenes waited for access.
- **Evidence.** The staged-ambulance and held-responder minutes in the table above. B1 loaded
  fewer patients than B0 (20.8 vs 22.8 loads per run in constrained).
- **Root cause.** The Dispatch objective is to minimise time to care. But its ALS and seat passes
  ranked incidents only by clinical urgency. An ambulance that reaches a scene before access is
  granted can do nothing there, so it gives its patients no earlier care than an ambulance sent
  later would. Suspected-RED scenes (collisions, fires, collapses) usually need rescue teams
  first, so urgency order kept sending ambulances to exactly these scenes.
- **Fix.**
  1. `control/needs.py` computes `IncidentNeed.access_wait_min`, an access estimate from the
     public view:
     - 0 if the scene is accessible;
     - None if a required responder is neither on scene nor en route;
     - otherwise the latest arrival among the teams still en route, plus the public 10-minute
       heavy-rescue delay when that rule applies.
  2. The Dispatch policy splits incidents into scenes that are ready (the nearest free
     ambulance would not arrive before access) and scenes still blocked. It runs the ALS and seat
     passes for ready scenes first, then for blocked scenes. Urgency order is kept within each
     group, and responders still go by urgency.
  3. The rule "use an ALS for seats only if no incident lacks an ALS" now looks only at the
     incidents served so far in the call. A blocked scene's ALS need can therefore no longer keep
     an ALS idle while a ready scene needs a seat.
  4. The LLM instructions say the same.
- **Files.**
  - Code: `domain/agent_io.py` (additive field), `control/needs.py`, `policies/dispatch.py`,
    `policies/params.py`, `mas/instructions.py`.
  - Tests: `tests/test_needs.py`, `tests/test_policies.py`.
- **Effect on dev seeds.** Staged minutes per run in constrained fell to 112 (b1) and 142 (full).
  Loads per run rose to 22.2 (b1) and 22.0 (full). Metric effects are in the attribution table
  below.

### F2: Resource Allocation added access delay to travel time instead of overlapping them

- **Symptom.** Full was below full-no-escalation. Resource Allocation (RA) kept giving contested
  ALS units to fresh suspected-RED scenes that had no access, while triaged REDs at accessible
  scenes waited.
  - Example: surge seed 2, t=56.7, CASE-0001. The option "AMB-03 to INC-08" (untriaged, heavy
    rescue and police both missing) scored 4.022. The option "AMB-03 to INC-04" (2 RED waiting,
    accessible) scored 3.987.
- **Root cause.** `cases.incident_utility` computed the slack as deadline − ETA − overhead − access
  delay. Its own docstring says why access matters: an ambulance cannot load before access. But
  the ambulance drives while the responders work, so loading starts at max(ETA, access), not at
  the sum.
  - The additive form charges an unserved blocked scene twice: no-unit wait plus access.
  - As a result, "give the unit to the blocked scene" looked like a large gain when in fact it
    changed nothing.
  - The old code also used a 20-minute access delay only when a responder was missing, and zero
    when responders were en route but still far away.
- **Fix.** The slack is now deadline − max(ETA, access wait) − overhead.
  - The access wait is the need's `access_wait_min`.
  - When the access wait is unknown, it is `no_unit_wait_min`, which keeps the old meaning of
    "unknown access delay".
- **Files.** Code: `control/cases.py`. Tests: `tests/test_cases.py`.
- **Effect on dev seeds (on top of F1).**
  - Surge, full: served 0.808→0.820, RED in window 0.381→0.400, score 0.613→0.632.
  - Constrained, full: mixed. Served 0.717→0.731, RED in window 0.266→0.228, score 0.499→0.491.
    This is within seed noise; see "Remaining".

### F3: Assessment gave a deteriorated RED an expired window

- **Symptom and evidence.** In B1 and Full, all dev seeds, 54 of 54 assessments run on a
  DETERIORATION trigger with a RED waiting reported a negative deadline. For example, a YELLOW
  that turned RED 100 minutes after its report got 40 − 100 = −60 minutes, as if it could no
  longer be saved, though its true window is 30 minutes from the deterioration. After the fix,
  15 of 56 were negative. In those, another RED at the same incident really had an expired
  window.
  - Plan 5.3 says "a deteriorated victim gets 30", but `_assess_triaged` called `nominal_deadline`
    without the `deteriorated` flag and subtracted the minutes since report.
  - RA utilities therefore treated these patients as lost (p_window ≈ 0).
- **Root cause.** `TriagedVictim` did not tell the Assessment agent which victims had
  deteriorated, or since when.
- **Fix.**
  - `TriagedVictim.minutes_since_deterioration` was added (additive field).
  - The coordinator keeps a `DeteriorationLog` of when it first saw each deterioration: either at
    the DETERIORATION trigger, or at the triage that revealed it. The view has no deterioration
    times, so this is the agents' own knowledge.
  - The assessment uses the 30-minute window counted from that time.
- **Files.**
  - Code: `domain/agent_io.py`, `control/observations.py`, `control/coordinator.py`,
    `policies/assessment.py`, `mas/instructions.py`.
  - Tests: `tests/test_policies.py`, `tests/test_coordinator.py`.
- **Effect on dev seeds.** Small on its own: about ±0.01. The deadline is only a secondary sort
  key in Dispatch, and RA cases rarely involve such patients. It is a correctness fix.

### F4: Priority reports went stale when patients were loaded

- **Symptom.** After the only RED at a scene was loaded, the incident kept "urgency 100,
  suspected_red = true". As a result:
  - the GREEN left behind outranked other scenes;
  - RA weighted the leftover GREEN as RED (`incident_weight` reads `suspected_red`).
  - Before the fix, 42 of 1,013 triaged incident needs in B1 dispatch inputs (surge and
    constrained, seeds 1 to 5) showed urgency 100 with no RED waiting. Example: surge seed 3,
    t=87, INC-05. After the fix there were 2; both were transient, because a RED is "loading" for
    3 minutes before the PATIENTS_LOADED trigger.
- **Root cause.** A report describes the incident's waiting victims. Loading changes that set, but
  the coordinator re-assessed only on new incident, triage and deterioration.
- **Fix.** PATIENTS_LOADED is now an assessment trigger for the loaded unit's incident.
- **Files.** Code: `control/coordinator.py`. Tests: `tests/test_coordinator.py`.
- **Evidence that F1 had to come first.** Applied alone, before F1, this fix made things worse:
  B1 constrained served fell 0.708→0.639. The stale urgency 100 had been hiding the leftover
  victims at accessible scenes. Fresh urgency let them starve, and their responders stayed held.
  With F1, those scenes are "ready" and get served first.
- **Effect on dev seeds (on top of F1 to F3).** Constrained b1 served 0.726→0.743, RED in window
  0.243→0.288, score 0.483→0.507. Normal and surge did not change.

### F5: The ALS pass ignored the pre-triage ambulance cap shared with B0

- **Symptom.** In surge seed 5 (t=118.7, INC-11), the dispatch validator flagged the policy's own
  output. A suspected-RED incident with three ambulances already assigned got a fourth (an ALS).
- **Root cause.** Only the seat pass checked `max_ambulances_pre_triage`, while the validator
  and B0 apply the cap to every ambulance. This gave the agents a small unfair advantage over B0,
  and an LLM that followed the instructions would have behaved differently from the offline
  policy.
- **Fix.** An untriaged incident at the cap has no ALS gap. As a result it gets no ALS order, no
  contention flag and no diversion.
- **Files.** Code: `policies/dispatch.py`. Tests: `tests/test_policies.py`.
- **Effect on dev seeds.** Metrics unchanged. Validator fallbacks: 1 run before any fix, 3 runs
  after F1 to F4, 0 after F5.

### Attribution (cumulative; cells are served rate / survival-weighted score)

| Scenario | Condition | before | +F3 | +F1 | +F2 | +F4 (+F5, all) |
|---|---|---|---|---|---|---|
| normal | b1 | 0.837 / 0.653 | 0.837 / 0.653 | 0.848 / 0.666 | 0.848 / 0.666 | 0.848 / 0.666 |
| normal | full | 0.814 / 0.632 | 0.814 / 0.632 | 0.856 / 0.666 | 0.856 / 0.666 | 0.856 / 0.666 |
| normal | full-no-float | 0.814 / 0.632 | 0.814 / 0.632 | 0.848 / 0.658 | 0.848 / 0.658 | 0.848 / 0.658 |
| normal | full-no-diversion | 0.814 / 0.632 | 0.814 / 0.632 | 0.851 / 0.666 | 0.856 / 0.666 | 0.856 / 0.666 |
| normal | full-no-escalation | 0.837 / 0.653 | 0.837 / 0.653 | 0.856 / 0.673 | 0.856 / 0.673 | 0.856 / 0.673 |
| surge | b1 | 0.791 / 0.565 | 0.795 / 0.561 | 0.799 / 0.582 | 0.799 / 0.582 | 0.799 / 0.582 |
| surge | full | 0.759 / 0.542 | 0.755 / 0.538 | 0.808 / 0.613 | 0.820 / 0.632 | 0.820 / 0.632 |
| surge | full-no-float | 0.730 / 0.525 | 0.722 / 0.521 | 0.789 / 0.585 | 0.796 / 0.587 | 0.796 / 0.587 |
| surge | full-no-diversion | 0.783 / 0.562 | 0.780 / 0.558 | 0.808 / 0.618 | 0.820 / 0.632 | 0.820 / 0.632 |
| surge | full-no-escalation | 0.811 / 0.594 | 0.815 / 0.590 | 0.838 / 0.632 | 0.838 / 0.632 | 0.838 / 0.632 |
| constrained | b1 | 0.708 / 0.479 | 0.717 / 0.481 | 0.726 / 0.483 | 0.726 / 0.483 | 0.743 / 0.507 |
| constrained | full | 0.629 / 0.447 | 0.629 / 0.447 | 0.717 / 0.499 | 0.731 / 0.491 | 0.739 / 0.495 |
| constrained | full-no-float | 0.656 / 0.443 | 0.642 / 0.430 | 0.710 / 0.513 | 0.698 / 0.484 | 0.706 / 0.487 |
| constrained | full-no-diversion | 0.616 / 0.442 | 0.616 / 0.442 | 0.718 / 0.477 | 0.708 / 0.464 | 0.716 / 0.468 |
| constrained | full-no-escalation | 0.732 / 0.493 | 0.744 / 0.495 | 0.760 / 0.510 | 0.760 / 0.510 | 0.772 / 0.519 |

To produce each column, the later fixes were reverted by monkeypatching. With every fix
reverted, the harness reproduces the "before" table exactly. Each regression test fails when its
own fix is reverted.

### Before and after round 1 (dev seeds 1–5, means of 5 runs; the final table is after round 2)

| Scenario | Condition | served | RED in window | score | resp. mean (min) | unserved crit. |
|---|---|---|---|---|---|---|
| normal | b0 | 0.859 → 0.859 | 0.320 → 0.320 | 0.617 → 0.617 | 17.0 → 17.0 | 1.0 → 1.0 |
| normal | b1 | 0.837 → 0.848 | 0.480 → 0.493 | 0.653 → 0.666 | 15.7 → 17.6 | 1.0 → 0.8 |
| normal | full | 0.814 → 0.856 | 0.453 → 0.480 | 0.632 → 0.666 | 15.2 → 16.7 | 1.4 → 0.8 |
| normal | full-no-float | 0.814 → 0.848 | 0.453 → 0.480 | 0.632 → 0.658 | 15.2 → 16.7 | 1.4 → 0.8 |
| normal | full-no-diversion | 0.814 → 0.856 | 0.453 → 0.480 | 0.632 → 0.666 | 15.2 → 17.0 | 1.4 → 0.8 |
| normal | full-no-escalation | 0.837 → 0.856 | 0.480 → 0.493 | 0.653 → 0.673 | 15.7 → 17.6 | 1.0 → 0.8 |
| surge | b0 | 0.799 → 0.799 | 0.347 → 0.347 | 0.592 → 0.592 | 32.6 → 32.6 | 3.0 → 3.0 |
| surge | b1 | 0.791 → 0.799 | 0.314 → 0.313 | 0.565 → 0.582 | 29.2 → 28.0 | 3.6 → 3.6 |
| surge | full | 0.759 → 0.820 | 0.298 → 0.400 | 0.542 → 0.632 | 28.3 → 27.8 | 4.2 → 3.6 |
| surge | full-no-float | 0.730 → 0.796 | 0.267 → 0.313 | 0.525 → 0.587 | 26.6 → 27.6 | 5.2 → 4.0 |
| surge | full-no-diversion | 0.783 → 0.820 | 0.316 → 0.400 | 0.562 → 0.632 | 28.5 → 27.8 | 4.0 → 3.6 |
| surge | full-no-escalation | 0.811 → 0.838 | 0.363 → 0.400 | 0.594 → 0.632 | 31.4 → 27.9 | 3.2 → 2.8 |
| constrained | b0 | 0.768 → 0.768 | 0.365 → 0.365 | 0.557 → 0.557 | 37.1 → 37.1 | 3.6 → 3.6 |
| constrained | b1 | 0.708 → 0.743 | 0.213 → 0.288 | 0.479 → 0.507 | 30.4 → 30.5 | 5.4 → 4.2 |
| constrained | full | 0.629 → 0.739 | 0.244 → 0.228 | 0.447 → 0.495 | 29.7 → 31.7 | 7.0 → 4.6 |
| constrained | full-no-float | 0.656 → 0.706 | 0.201 → 0.249 | 0.443 → 0.487 | 31.4 → 31.4 | 6.8 → 5.6 |
| constrained | full-no-diversion | 0.616 → 0.716 | 0.220 → 0.200 | 0.442 → 0.468 | 29.3 → 30.4 | 7.6 → 5.0 |
| constrained | full-no-escalation | 0.732 → 0.772 | 0.227 → 0.253 | 0.493 → 0.519 | 31.5 → 30.9 | 4.8 → 4.0 |

The mean response time of the agent conditions rose by 1 to 2 minutes in normal. This is
expected: ambulances are no longer sent early to scenes they could not load.

### Round 2: pre-registered design changes (written before any result was run)

Two remaining design issues were found above, where an agent acts against its own stated
objective. Each gets one change, with the formulation fixed in advance:

- **D1, deadline-aware dispatch priority.**
  - Dispatch's objective is care within each patient's window. Flat urgency lets older YELLOW-only
    scenes deteriorate.
  - Incidents are ordered by the **expected loss of delaying them**:
    `loss = U(eta) - U(eta + no_unit_wait_min)`, where:
    - `U(t) = w * p_window(deadline_in_min - max(t, access wait) - overhead)` is the slack model
      Resource Allocation already uses (F2);
    - `eta` is the ETA of the nearest free ambulance;
    - `no_unit_wait_min` is the existing "assumed extra wait when an incident gets no unit".
  - Ties go by incident id. Ids follow report order, so all-zero ties fall back to first come,
    first served.
  - The ready/blocked split of F1 is kept; this ordering is used within each group and for
    responders.
  - Waiting YELLOWs get the latest care time that keeps them in window under the public rules
    (simulation design: "an untreated YELLOW turns RED at 60% of its window, from then on it has
    30 minutes"):
    `min(nominal YELLOW window, 0.6 * nominal YELLOW window + 30) = min(165, 129) = 129` minutes
    from the report.
  - Only the public nominal window is used, never a victim's hidden window. 0.6 enters as a new
    `PolicyParams` constant that copies the public rule.
- **D2, responder access completion.**
  - A responder team only helps once the scene's whole required set is present.
  - The responder pass therefore first serves, in priority order, the incidents whose every
    missing capability can be covered by teams that are free now (sending them completes access).
  - Then it serves the remaining incidents as before.
  - This is an ordering rule; it has no weights.

**Selection rule (pre-registered).**
- D1 is evaluated against the configuration above (F1 to F5). D2 is evaluated on top of whatever
  was kept.
- A change is kept only if it does not lower the mean dev-seed `survival_weighted_score`
  (averaged over normal, surge and constrained) for B1 or for Full. Otherwise it is reverted and
  recorded here as "tried, rejected", with its numbers.
- Reference values before round 2: B1 0.5848, Full 0.5976.

### Round 2 results

**D1 (expected-loss dispatch priority, with the 129-minute YELLOW care deadline): tried, rejected.**

Mean score over the three scenarios fell for both B1 and Full, which breaks the selection rule.

| Condition | mean score (ref → D1) | normal | surge | constrained |
|---|---|---|---|---|
| B1 | 0.5848 → 0.5352 | 0.666 → 0.658 | 0.582 → 0.534 | 0.507 → 0.414 |
| Full | 0.5976 → 0.5690 | 0.666 → 0.666 | 0.632 → 0.551 | 0.495 → 0.490 |

Served rate also fell (B1 surge 0.799 → 0.703, constrained 0.743 → 0.597).

Why it failed. B1 dispatch inputs on surge and constrained, seeds 1 to 5, had 2,574 incident
needs with waiting patients:
- 70% of them had an expected delay loss below 0.01, because 20 minutes changes p_window very
  little when the slack is far from 0 either way.
- 428 of those near-zero needs were RED or suspected-RED incidents past the nominal 40-minute
  window. Their true windows are 20 to 60 minutes, and their patients still count as served.

So the ordering fell back to incident id (first come, first served) for most scenes, and pushed
REDs past the nominal window to the back. The nominal window is too coarse to rank by
"already lost".

All D1 code was reverted:
- the dispatch priority;
- the YELLOW care deadline and the `yellow_deterioration_fraction` parameter;
- the move of `incident_utility` into `policies/estimates.py`.

After the revert, the harness reproduced the reference table exactly. As pre-registered, no
second formulation was tried.

**D2 (responder access completion): kept.**

- **Code.** `policies/dispatch.py` (`_team_plan`, `_send_teams`, `_responder_pass`) and the
  matching line in `mas/instructions.py`.
- **Tests.** `tests/test_policies.py::test_responder_goes_where_it_completes_access` fails with
  the old pass. `test_partial_teams_still_go_by_priority_when_no_scene_can_be_completed` checks
  the fallback.
- **Selection rule.** B1 mean score 0.5848 → 0.5848 (unchanged); Full 0.5976 → 0.5982.
- **Effect size.** The effect is very small. Replaying the recorded dispatch inputs of the final
  B1 and Full dev runs, the new pass changed the responder orders in only 3 of 259 calls that sent
  any team. Under this bottleneck there is rarely more than one free team.
- **What it does not cover.** The constrained seed 4 example from "Remaining", item 4, is not
  changed: at t=93.9 both INC-05 and INC-06 could be fully opened by the two free teams, so
  priority order still decides between them.

**Final table (dev seeds 1–5, means of 5 runs; before → after rounds 1 and 2).** Produced with the
official runner (offline SDK backend), and identical to the harness:

| Scenario | Condition | served | RED in window | score | resp. mean (min) | unserved crit. |
|---|---|---|---|---|---|---|
| normal | b0 | 0.859 → 0.859 | 0.320 → 0.320 | 0.617 → 0.617 | 17.0 → 17.0 | 1.0 → 1.0 |
| normal | b1 | 0.837 → 0.848 | 0.480 → 0.493 | 0.653 → 0.666 | 15.7 → 17.6 | 1.0 → 0.8 |
| normal | full | 0.814 → 0.856 | 0.453 → 0.480 | 0.632 → 0.666 | 15.2 → 16.7 | 1.4 → 0.8 |
| normal | full-no-float | 0.814 → 0.848 | 0.453 → 0.480 | 0.632 → 0.658 | 15.2 → 16.7 | 1.4 → 0.8 |
| normal | full-no-diversion | 0.814 → 0.856 | 0.453 → 0.480 | 0.632 → 0.666 | 15.2 → 17.0 | 1.4 → 0.8 |
| normal | full-no-escalation | 0.837 → 0.856 | 0.480 → 0.493 | 0.653 → 0.673 | 15.7 → 17.6 | 1.0 → 0.8 |
| surge | b0 | 0.799 → 0.799 | 0.347 → 0.347 | 0.592 → 0.592 | 32.6 → 32.6 | 3.0 → 3.0 |
| surge | b1 | 0.791 → 0.799 | 0.314 → 0.313 | 0.565 → 0.582 | 29.2 → 28.0 | 3.6 → 3.6 |
| surge | full | 0.759 → 0.820 | 0.298 → 0.400 | 0.542 → 0.632 | 28.3 → 27.8 | 4.2 → 3.6 |
| surge | full-no-float | 0.730 → 0.796 | 0.267 → 0.313 | 0.525 → 0.587 | 26.6 → 27.6 | 5.2 → 4.0 |
| surge | full-no-diversion | 0.783 → 0.820 | 0.316 → 0.400 | 0.562 → 0.632 | 28.5 → 27.8 | 4.0 → 3.6 |
| surge | full-no-escalation | 0.811 → 0.838 | 0.363 → 0.400 | 0.594 → 0.632 | 31.4 → 27.9 | 3.2 → 2.8 |
| constrained | b0 | 0.768 → 0.768 | 0.365 → 0.365 | 0.557 → 0.557 | 37.1 → 37.1 | 3.6 → 3.6 |
| constrained | b1 | 0.708 → 0.743 | 0.213 → 0.288 | 0.479 → 0.507 | 30.4 → 30.5 | 5.4 → 4.2 |
| constrained | full | 0.629 → 0.751 | 0.244 → 0.228 | 0.447 → 0.497 | 29.7 → 31.7 | 7.0 → 4.2 |
| constrained | full-no-float | 0.656 → 0.713 | 0.201 → 0.249 | 0.443 → 0.489 | 31.4 → 31.4 | 6.8 → 5.4 |
| constrained | full-no-diversion | 0.616 → 0.728 | 0.220 → 0.200 | 0.442 → 0.470 | 29.3 → 30.4 | 7.6 → 4.6 |
| constrained | full-no-escalation | 0.732 → 0.772 | 0.227 → 0.253 | 0.493 → 0.519 | 31.5 → 30.9 | 4.8 → 4.0 |


### Suspects examined and ruled out (dev-seed evidence)

- **Hospital queue and staff waits.** Only 6 of the 42 victims B1 lost (before the fixes) were
  lost in hospital phases. Door-to-doctor time was shorter for B1 than for B0 (constrained 0.8
  vs 4.1 minutes).
- **BLS sent to scenes whose victims turned out RED.** Before the fixes, wasted BLS trips
  (dispatched, then cleared without loading) per run were:
  - surge: 3.2 (b0), 2.0 (b1), 2.8 (full);
  - constrained: 1.4 (b0), 1.4 (b1), 2.2 (full).

  That is at most about one trip per run more than B0, so this is not a main mechanism.
- **Units not re-dispatched after UNIT_FREED.** Ambulance-minutes idle while some scene had an
  open need were the same for B0 and B1. For example, 62 vs 62 in constrained seed 3, and
  52 vs 54 in constrained seed 4.
- **Need accounting.** `control/needs.py` is shared by B0 and the agents. No divergence was found.
- **Transport ranking.** Pruning hospital rankings to options within the agents' own expected
  wait (15 min per queued patient) changed nothing measurable, so the ranking was not changed.
- **B0 fairness.** B0 sees the same public view and follows the same pre-triage cap and crew
  protocol. No hidden advantage was found. F5 was the reverse case: the agents could break the
  shared cap.

### Remaining: where the agents still lose, and why

The system is now consistent with its own objectives, but the agents do not beat B0 everywhere.
With 5 dev seeds, per-seed spread is large: for example, constrained served counts per seed were
B0 [34, 32, 33, 35, 29] and B1 [31, 34, 28, 34, 31]. Mean differences below about 0.03 in served
rate are within noise.

1. **RED in window in surge and constrained: B1 0.313 / 0.288 vs B0 0.347 / 0.365.** The gap is
   mostly deteriorated YELLOWs. In constrained, B0 got 3.0 deteriorated REDs per run into their
   window and B1 got 2.4.
   - Urgency-first dispatch serves suspected-RED scenes before older YELLOW-only scenes.
   - Under overload, the YELLOW-only scenes wait until their patients deteriorate (public rule:
     at 60% of the window), and then only 30 minutes are left.
   - B0's first-come-first-served order bounds that wait.
   - This is the design hypothesis "urgency-ordered dispatch" not paying off when there are 5
     ambulances and 2 police teams.
   - The pre-registered expected-loss ordering (round 2, D1) was tried and rejected: it lowered
     scores. Urgency order stays.
2. **Hospital rejections in B1.** Per run in constrained, about 10.6 patients were redirected after
   a hospital rejection (mostly NO_DOCTOR, then RESERVE_HELD), each adding about 6.5 transport
   minutes. 13 of the 17 REDs among them (over 5 seeds) missed their window.
   - Each rejection follows the hospital's stated rule ("accept only patients you can treat on
     arrival; keep the reserve for RED").
   - B1 has no negotiation to weigh it. In Full, counters and RA remove most of these.
   - This is a reported cost of the B1 design (plan 5.6), not a bug.
3. **Full vs full-no-escalation.** UNIT_CONTENTION, NO_ACCEPTING_HOSPITAL and diversion add
   nothing measurable on dev seeds:
   - Surge: identical RED-in-window outcomes, served 0.820 vs 0.838 (one seed differs by 3
     patients).
   - Constrained: 0.751 vs 0.772 served (0.739 before D2), 0.228 vs 0.253 RED in window.
   - Observed mechanism (example cases, not an aggregate): contention between two scenes that
     are both blocked, where any ALS choice waits for police. In constrained seed 5, CASE-0003,
     the utilities were 2.545 vs 2.516, so the choice is close to arbitrary.
4. **Responder bottleneck.** Constrained has 2 police teams, and nearly every scene needs scene
   control. Round 2 (D2) now sends teams first where they complete access, but this rarely
   applies: 3 of 259 calls changed. Among scenes that can all be completed, the responder pass
   still goes by urgency.
   - Example: constrained seed 4. The rescue squad waited at INC-06 from t=64 for a police team.
     B0 sent one at t=93.9; B1 sent both free teams to INC-05 instead.
   - Responder-minutes held at accessible scenes while other scenes are blocked are still higher
     than B0's: constrained 210 for B1 vs 138 for B0.
   - Still open: preferring the completable scene that needs fewer teams (or that has teams
     already waiting) would be a further rule. It was not tried, because round 2 was limited to
     one formulation per issue.

**Where the agents now win.**

- Normal: every agent condition beats B0 on RED in window (0.48 to 0.49 vs 0.32) and on the
  survival-weighted score (0.66 to 0.67 vs 0.62). Served rate is level: 0.848 to 0.856 vs 0.859.
- Surge:
  - Full, full-no-diversion and full-no-escalation beat B0 on served rate (0.820 to 0.838 vs
    0.799), RED in window (0.400 vs 0.347) and score (0.632 vs 0.592).
  - B1 and full-no-float do not: their RED in window is 0.313 vs 0.347.
- Constrained: B0 is still best on RED in window (0.365) and score (0.557). Only
  full-no-escalation matches it on served rate (0.772 vs 0.768, within noise). Full reaches 0.751. The agents' mean
  response time is lower (about 31 vs 37 minutes), but this does not turn into better outcomes.
