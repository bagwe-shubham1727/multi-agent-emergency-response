"""Synthetic data generator for the emergency-response multi-agent simulation.

    python -m emergency_sim.datagen                      # all scenarios, seed 7380
    python -m emergency_sim.datagen --seed 42 --out data_alt

Standard library only. Randomness comes from one random.Random stream per
kind of entity ("<seed>-incidents", "<seed>-staff", ...), shared by every
scenario. Scenarios differ only in their knobs, so they form a controlled
comparison: normal has the base incidents, surge adds a mass-casualty
cluster, and constrained has exactly the surge incidents with fewer
resources. The same seed and Python version always write byte-identical
files (validate.py regenerates them and compares SHA-256 hashes).

Caller-report severity wording is noisy: it takes one draw per incident from
its own streams ("<seed>-reports" for base incidents, "<seed>-cluster-reports"
for the cluster), so the noise changes no other draw and an incident shared by
two scenarios gets the same wording in both.

Fixed by design (documented in docs/simulation_design.md): the city map,
hospital sites and capabilities, ambulance stations and responder bases.
Drawn at random: incident times, places, types and victims; hospital
occupancy at t=0; staff rosters and workloads; units busy at t=0.
"""

import argparse
import csv
import hashlib
import json
import platform
import random
from pathlib import Path

from . import geo

GENERATOR_VERSION = "1.2.0"
DEFAULT_SEED = 7380
HORIZON_MIN = 180
BASE_INCIDENTS = 11
LAST_CALL_MIN = 150

# START triage: RED immediate, YELLOW delayed, GREEN minor, BLACK expectant
TRIAGE_LEVELS = ["RED", "YELLOW", "GREEN", "BLACK"]

# condition -> (hospital capability needed, triage levels it can present with)
CONDITIONS = {
    "cardiac arrest": ("cardiac", ["RED"]),
    "chest pain (suspected MI)": ("cardiac", ["RED", "YELLOW"]),
    "stroke symptoms": ("neuro", ["RED"]),
    "head injury (TBI)": ("neuro", ["RED", "YELLOW"]),
    "blunt chest trauma": ("trauma", ["RED", "YELLOW"]),
    "internal bleeding": ("trauma", ["RED"]),
    "crush injury": ("trauma", ["RED", "YELLOW"]),
    "stab wound": ("trauma", ["RED", "YELLOW"]),
    "long-bone fracture": ("trauma", ["YELLOW"]),
    "hip fracture": ("trauma", ["YELLOW"]),
    "major burns (>20% TBSA)": ("burns", ["RED"]),
    "minor burns": ("general", ["YELLOW", "GREEN"]),
    "smoke inhalation": ("general", ["RED", "YELLOW", "GREEN"]),
    "chemical exposure": ("general", ["RED", "YELLOW", "GREEN"]),
    "pediatric respiratory distress": ("pediatric", ["RED", "YELLOW"]),
    "lacerations": ("general", ["YELLOW", "GREEN"]),
    "sprain / whiplash": ("general", ["GREEN"]),
}

# minutes until care must start; a cardiac arrest needs CPR and a
# defibrillator within about 10 minutes, far sooner than other RED cases
RED_WINDOW = (20, 60)
CONDITION_RED_WINDOW = {"cardiac arrest": (8, 12)}

# type -> victims range, condition mix, responder capabilities needed,
# share of random arrivals, chance a victim is BLACK, caller phrasing.
# "hazard" is the severe wording and "hazard_mild" the mild one. A caller with a
# RED victim usually uses the severe wording and otherwise the mild one, but
# not always (see REPORT_UNDERTRIAGE_P), so the report hints at severity
# without giving away the triage. "stroke" uses the same wording for both; every
# stroke victim is RED, so the incident type already says as much.
INCIDENT_TYPES = {
    "multi_vehicle_collision": dict(
        victims=(3, 7), weight=3, p_black=0.05,
        conditions={"blunt chest trauma": 3, "head injury (TBI)": 2, "long-bone fracture": 3,
                    "internal bleeding": 1, "crush injury": 1, "lacerations": 3, "sprain / whiplash": 3},
        responders=["extrication", "scene_control"],
        phrase="a multi-car pile-up", hazard="vehicles are smoking and people are trapped",
        hazard_mild="people are hurt but out of the cars and talking"),
    "building_fire": dict(
        victims=(2, 6), weight=2, p_black=0.04,
        conditions={"smoke inhalation": 5, "major burns (>20% TBSA)": 2, "minor burns": 3,
                    "lacerations": 1},
        responders=["fire_suppression", "scene_control"],
        phrase="a fire in an apartment building", hazard="someone was pulled out badly burned or not breathing well",
        hazard_mild="people got out but are coughing"),
    "chemical_spill": dict(
        victims=(2, 5), weight=1, p_black=0.0,
        conditions={"chemical exposure": 1},
        responders=["hazmat", "scene_control"],
        phrase="a chemical leak at a warehouse", hazard="someone has collapsed and others are coughing",
        hazard_mild="there is a strong smell and people's eyes are burning"),
    "structural_collapse": dict(
        victims=(2, 5), weight=1, p_black=0.08,
        conditions={"crush injury": 3, "head injury (TBI)": 2, "long-bone fracture": 2, "lacerations": 2},
        responders=["heavy_rescue", "scene_control"],
        phrase="part of a building has collapsed", hazard="people are trapped under the rubble",
        hazard_mild="people got out with cuts and injuries"),
    "cardiac_event": dict(
        victims=(1, 1), weight=3, p_black=0.0,
        conditions={"cardiac arrest": 1, "chest pain (suspected MI)": 2},
        responders=[],
        phrase="a person who collapsed clutching their chest", hazard="they are not responding and a bystander is starting CPR",
        hazard_mild="they are awake but sweating and in pain"),
    "stroke": dict(
        victims=(1, 1), weight=2, p_black=0.0,
        conditions={"stroke symptoms": 1},
        responders=[],
        phrase="an older adult with slurred speech", hazard="one side of the face is drooping",
        hazard_mild="one side of the face is drooping"),
    "fall": dict(
        victims=(1, 1), weight=2, p_black=0.0,
        conditions={"hip fracture": 2, "head injury (TBI)": 1, "lacerations": 1},
        responders=[],
        phrase="an elderly person who fell down stairs", hazard="they hit their head and are not making sense",
        hazard_mild="they cannot get up"),
    "assault": dict(
        victims=(1, 2), weight=1, p_black=0.0,
        conditions={"stab wound": 2, "lacerations": 2, "head injury (TBI)": 1},
        responders=["scene_control"],
        phrase="a fight with a person stabbed", hazard="there is a lot of blood and the attacker may be nearby",
        hazard_mild="the victim is bleeding but talking"),
    "pediatric_emergency": dict(
        victims=(1, 1), weight=1, p_black=0.0,
        conditions={"pediatric respiratory distress": 1},
        responders=[],
        phrase="a child having trouble breathing", hazard="the child's lips look blue",
        hazard_mild="the child is breathing fast but can talk"),
}

# Caller-report noise. Documented assumptions, not tuned to any result:
# telephone triage is imperfect, with sensitivity for critical calls well below
# 100% and some over-triage. A caller with a RED victim describes the scene
# mildly with probability REPORT_UNDERTRIAGE_P; a caller with no RED victim
# describes it severely with probability REPORT_OVERTRIAGE_P.
REPORT_UNDERTRIAGE_P = 0.20
REPORT_OVERTRIAGE_P = 0.10

# victims per incident inside the mass-casualty cluster (13-21 in total)
CLUSTER_VICTIMS = {"multi_vehicle_collision": (5, 8), "building_fire": (4, 7), "structural_collapse": (4, 6)}

# Fixed infrastructure. Capabilities differ on purpose so that some patients
# can only go to one hospital, which creates the contention the agents resolve.
HOSPITALS = [
    dict(id="H1", name="Riverside General Hospital", x=6.0, y=13.5, trauma_level=1,
         capabilities=["trauma", "burns", "neuro", "cardiac", "general"],
         er_capacity=14, icu_beds=6, critical_reserve=2),
    dict(id="H2", name="St. Agnes Medical Center", x=14.5, y=12.0, trauma_level=2,
         capabilities=["cardiac", "neuro", "trauma", "general"],
         er_capacity=10, icu_beds=4, critical_reserve=1),
    dict(id="H3", name="Eastgate Community Hospital", x=12.0, y=4.0, trauma_level=3,
         capabilities=["pediatric", "general"],
         er_capacity=8, icu_beds=2, critical_reserve=1),
]

STATIONS = [
    dict(id="S1", name="Station 1 (North)", x=4.0, y=17.0),
    dict(id="S2", name="Station 2 (Central)", x=10.0, y=9.0),
    dict(id="S3", name="Station 3 (South-East)", x=16.0, y=5.0),
]

RESPONDER_TEAMS = [
    dict(id="FR-ENG1", type="fire_engine", base="S1", capabilities=["fire_suppression", "extrication"]),
    dict(id="FR-ENG2", type="fire_engine", base="S3", capabilities=["fire_suppression", "extrication"]),
    dict(id="FR-RES1", type="rescue_squad", base="S2", capabilities=["heavy_rescue", "extrication"]),
    dict(id="FR-HAZ1", type="hazmat_unit", base="S2", capabilities=["hazmat", "decontamination"]),
    dict(id="FR-POL1", type="police_patrol", base="S2", capabilities=["scene_control"]),
    dict(id="FR-POL2", type="police_patrol", base="S1", capabilities=["scene_control"]),
]

# Every ER doctor is an emergency physician; specialty skills say who can
# lead specialist care. Nurse skills follow what the hospital treats.
NURSE_SKILL_FOR = {"trauma": "trauma", "burns": "burns", "pediatric": "pediatric",
                   "cardiac": "icu", "neuro": "icu"}
FLOAT_DOCTOR_SPECIALTIES = ["trauma", "cardiac", "pediatric"]
FLOAT_NURSE_SKILLS = ["trauma", "icu", "pediatric", "burns"]
# full roster drawn in every scenario; scenarios keep the first n per site
ROSTER_MAX = dict(doctors=3, nurses=6, float_doctors=2, float_nurses=3)

# Scenario knobs. Incidents come from shared streams, so constrained has the
# same incidents as surge and normal has surge's incidents minus the cluster.
SCENARIOS = {
    "normal": dict(
        description="The 11 base incidents, independent of each other, with full resources.",
        mci_cluster=False, occupancy=(0.35, 0.60),
        doctors_per_hospital=3, nurses_per_hospital=6, float_doctors=2, float_nurses=3,
        out_of_service=[], busy_at_start=1),
    "surge": dict(
        description="The base incidents plus a 3-incident mass-casualty cluster around minute 40, with fuller ERs.",
        mci_cluster=True, occupancy=(0.45, 0.70),
        doctors_per_hospital=3, nurses_per_hospital=6, float_doctors=2, float_nurses=3,
        out_of_service=[], busy_at_start=2),
    "constrained": dict(
        description="Exactly the surge incidents with one ambulance out of service, fuller ERs and fewer staff.",
        mci_cluster=True, occupancy=(0.60, 0.85),
        doctors_per_hospital=2, nurses_per_hospital=5, float_doctors=1, float_nurses=2,
        out_of_service=["AMB-06"], busy_at_start=2),
}
STREAMS = ["hospitals", "staff", "ambulances", "responders", "incidents", "cluster",
           "reports", "cluster-reports"]


def weighted_choice(rng, weights):
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys])[0]


def specialties(h):
    return [c for c in h["capabilities"] if c != "general"]


def nurse_pool(h):
    return sorted({NURSE_SKILL_FOR[c] for c in specialties(h)} | {"icu"})


def make_hospitals(rng, cfg):
    out = []
    lo, hi = cfg["occupancy"]
    for h in HOSPITALS:
        # one uniform draw per hospital, mapped into this scenario's range
        er_occ = round(h["er_capacity"] * (lo + (hi - lo) * rng.random()))
        icu_occ = round(h["icu_beds"] * (lo + (hi - lo) * rng.random()))
        out.append(dict(h, er_occupied_t0=er_occ, icu_occupied_t0=icu_occ,
                        er_free_t0=h["er_capacity"] - er_occ,
                        intersection=geo.nearest_intersection((h["x"], h["y"]))))
    return out


def make_staff(rng, cfg):
    """Draw the full roster, then keep the first n per site.

    Specialties are dealt round-robin over the first two doctors, so even a
    two-doctor ER covers every capability its hospital offers.
    """
    roster = []
    for h in HOSPITALS:
        spec = specialties(h)
        for i in range(ROSTER_MAX["doctors"]):
            extra = spec[i::2] if i < 2 else rng.sample(spec, k=min(2, len(spec)))
            roster.append(dict(id=f"DR-{h['id']}-{i + 1}", role="doctor", home=h["id"],
                               skills=sorted({"emergency", *extra}), max_patients=4))
        pool = nurse_pool(h)
        for i in range(ROSTER_MAX["nurses"]):
            skill = pool[i] if i < len(pool) else rng.choice(pool)
            roster.append(dict(id=f"RN-{h['id']}-{i + 1}", role="nurse", home=h["id"],
                               skills=sorted({"emergency", skill}), max_patients=3))
    for i in range(ROSTER_MAX["float_doctors"]):
        extra = rng.sample(FLOAT_DOCTOR_SPECIALTIES, k=2)
        roster.append(dict(id=f"DR-FLOAT-{i + 1}", role="doctor", home="FLOAT",
                           skills=sorted({"emergency", *extra}), max_patients=4))
    for i in range(ROSTER_MAX["float_nurses"]):
        roster.append(dict(id=f"RN-FLOAT-{i + 1}", role="nurse", home="FLOAT",
                           skills=sorted({"emergency", rng.choice(FLOAT_NURSE_SKILLS)}), max_patients=3))
    for s in roster:
        s["load_t0"] = rng.randint(0, 2)
        s["shift_end_min"] = rng.choice([120, 180, 240, 300])

    keep = {"doctor": cfg["doctors_per_hospital"], "nurse": cfg["nurses_per_hospital"]}
    keep_float = {"doctor": cfg["float_doctors"], "nurse": cfg["float_nurses"]}
    out = []
    for s in roster:
        n = int(s["id"].rsplit("-", 1)[1])
        limit = keep_float[s["role"]] if s["home"] == "FLOAT" else keep[s["role"]]
        if n <= limit:
            out.append(s)
    return out


def make_ambulances(rng, cfg):
    ambs = []
    n = 0
    for st in STATIONS:
        for kind in ("ALS", "BLS"):
            n += 1
            ambs.append(dict(id=f"AMB-{n:02d}", type=kind, station=st["id"],
                             x=st["x"], y=st["y"], capacity=2, max_red_per_trip=1,
                             status="available", busy_until_min=0))
    # same draws in every scenario: who is busy first at t=0, and for how long
    order = rng.sample([a["id"] for a in ambs], k=len(ambs))
    busy_for = {a["id"]: rng.randint(10, 35) for a in ambs}
    for a in ambs:
        if a["id"] in cfg["out_of_service"]:
            a["status"] = "out_of_service"
    by_id = {a["id"]: a for a in ambs}
    busy = [i for i in order if by_id[i]["status"] == "available"][:cfg["busy_at_start"]]
    for i in busy:
        by_id[i]["status"] = "busy"
        by_id[i]["busy_until_min"] = busy_for[i]
    return ambs


def make_responders(rng):
    teams = []
    bases = {s["id"]: s for s in STATIONS}
    for t in RESPONDER_TEAMS:
        b = bases[t["base"]]
        teams.append(dict(t, x=b["x"], y=b["y"], crew=rng.randint(3, 5),
                          status="available", busy_until_min=0))
    return teams


def make_victims(rng, spec):
    victims = []
    for _ in range(rng.randint(*spec["victims"])):
        cond = weighted_choice(rng, spec["conditions"])
        capability, allowed = CONDITIONS[cond]
        triage = "BLACK" if rng.random() < spec["p_black"] else rng.choice(allowed)
        if triage == "RED":
            window = rng.randint(*CONDITION_RED_WINDOW.get(cond, RED_WINDOW))
        elif triage == "YELLOW":
            window = rng.randint(90, 240)
        elif triage == "GREEN":
            window = 480
        else:
            window = 0
        victims.append(dict(
            condition=cond, triage=triage,
            required_capability=capability,
            needs_als=triage == "RED",
            needs_decon=cond == "chemical exposure",
            golden_window_min=window,
            # an untreated YELLOW turns RED at 60% of its window
            deteriorates_to_red_at_min=round(window * 0.6) if triage == "YELLOW" else None))
    return victims


def incident_point(rng, centre=None, spread=1.5):
    if centre is None:
        return round(rng.uniform(1, 19), 1), round(rng.uniform(1, 19), 1)
    x = min(19.5, max(0.5, rng.gauss(centre[0], spread)))
    y = min(19.5, max(0.5, rng.gauss(centre[1], spread)))
    return round(x, 1), round(y, 1)


def caller_hazard(report_rng, spec, worst_triage):
    """Picks the caller's severity wording for one incident.

    Always takes exactly one draw from ``report_rng``, whatever the triage, so
    the k-th incident built from a stream always gets the k-th draw.

    Args:
        report_rng: The incident's report stream (``random.Random``).
        spec: The incident type's entry in ``INCIDENT_TYPES``.
        worst_triage: The incident's true worst triage level.

    Returns:
        ``spec["hazard"]`` (severe) or ``spec["hazard_mild"]`` (mild).
    """
    u = report_rng.random()
    if worst_triage == "RED":
        severe = u >= REPORT_UNDERTRIAGE_P
    else:
        severe = u < REPORT_OVERTRIAGE_P
    return spec["hazard"] if severe else spec["hazard_mild"]


def build_incident(rng, report_rng, itype, t, point, spec=None):
    spec = spec or INCIDENT_TYPES[itype]
    victims = make_victims(rng, spec)
    true_n = len(victims)
    reported = true_n if true_n == 1 else max(1, true_n + rng.randint(-2, 2))
    place = geo.nearest_intersection(point)
    worst = min((TRIAGE_LEVELS.index(v["triage"]) for v in victims if v["triage"] != "BLACK"),
                default=TRIAGE_LEVELS.index("BLACK"))
    hazard = caller_hazard(report_rng, spec, TRIAGE_LEVELS[worst])
    return dict(
        type=itype, t_report_min=t,
        x=point[0], y=point[1], intersection=place,
        congestion=round(geo.congestion(point), 2),
        # what the caller says; the assessment agent only sees this side
        reported_victims=reported,
        caller_report=(f"Caller at {place} reports {spec['phrase']}; "
                       f"about {reported} {'person' if reported == 1 else 'people'} hurt; "
                       f"{hazard}."),
        required_responder_capabilities=spec["responders"],
        # ground truth used for scoring, hidden from the agents
        true_victims=true_n,
        worst_triage=TRIAGE_LEVELS[worst],
        time_sensitivity_min=min((v["golden_window_min"] for v in victims if v["triage"] != "BLACK"),
                                 default=0),
        mci_cluster=False,
        victims=victims)


def make_incidents(streams, cfg):
    rng = streams["incidents"]
    types = list(INCIDENT_TYPES)
    weights = [INCIDENT_TYPES[t]["weight"] for t in types]
    # A Poisson process conditioned on n arrivals in (0, T] places them as n
    # sorted uniform draws, so this is Poisson arrivals with a fixed count.
    times = sorted(max(1, round(rng.uniform(0, LAST_CALL_MIN))) for _ in range(BASE_INCIDENTS))
    incidents = []
    for t in times:
        itype = rng.choices(types, weights=weights)[0]
        incidents.append(build_incident(rng, streams["reports"], itype, t, incident_point(rng)))

    if cfg["mci_cluster"]:
        # three large incidents within 8 minutes of each other in the
        # north-west, closest to H1 and Station 1, so they compete for the
        # same units and the same trauma beds
        crng = streams["cluster"]
        centre = (crng.uniform(5, 8), crng.uniform(12, 16))
        t0 = crng.randint(36, 42)
        for dt, itype in [(0, "multi_vehicle_collision"), (crng.randint(2, 5), "building_fire"),
                          (crng.randint(5, 8), "structural_collapse")]:
            # a mass-casualty incident is one that outstrips local resources,
            # so cluster incidents take victim counts from the top of the range
            spec = dict(INCIDENT_TYPES[itype], victims=CLUSTER_VICTIMS[itype])
            inc = build_incident(crng, streams["cluster-reports"], itype, t0 + dt,
                                 incident_point(crng, centre), spec)
            inc["mci_cluster"] = True
            incidents.append(inc)

    # stable sort keeps base incidents before cluster ones at equal times;
    # ids are given after sorting so they run in time order
    incidents.sort(key=lambda i: i["t_report_min"])
    out = []
    for k, inc in enumerate(incidents, start=1):
        iid = f"INC-{k:02d}"
        victims = [dict(victim_id=f"{iid}-V{j}", **v) for j, v in enumerate(inc["victims"], start=1)]
        out.append(dict(incident_id=iid, **{**inc, "victims": victims}))
    return out


def travel_rows(hospitals, incidents):
    nodes = [(s["id"], "station", (s["x"], s["y"])) for s in STATIONS]
    nodes += [(h["id"], "hospital", (h["x"], h["y"])) for h in hospitals]
    nodes += [(i["incident_id"], "incident", (i["x"], i["y"])) for i in incidents]
    rows = []
    for a_id, a_kind, a in nodes:
        for b_id, b_kind, b in nodes:
            if a_id == b_id:
                continue
            # driving time only; the engine adds turnout time when a unit is alerted
            rows.append(dict(from_id=a_id, from_kind=a_kind, to_id=b_id, to_kind=b_kind,
                             distance_km=round(geo.manhattan(a, b), 2),
                             ambulance_min=geo.travel_time(a, b, "ambulance"),
                             responder_min=geo.travel_time(a, b, "responder")))
    return rows


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def cluster_contention(incidents, ambulances):
    """Patients in the cluster versus what one wave of in-service units can carry."""
    cluster = [i for i in incidents if i["mci_cluster"]]
    if not cluster:
        return None
    in_service = [a for a in ambulances if a["status"] != "out_of_service"]
    cv = [v for i in cluster for v in i["victims"]]
    return dict(
        incidents=len(cluster),
        span_min=max(i["t_report_min"] for i in cluster) - min(i["t_report_min"] for i in cluster),
        victims=len(cv),
        # BLACK (expectant) victims are not transported
        patients=sum(v["triage"] != "BLACK" for v in cv),
        red=sum(v["triage"] == "RED" for v in cv),
        seats=sum(a["capacity"] for a in in_service),
        als_units=sum(a["type"] == "ALS" for a in in_service))


def summarise(name, cfg, hospitals, ambulances, staff, responders, incidents):
    victims = [v for i in incidents for v in i["victims"]]
    by_triage = {lvl: sum(v["triage"] == lvl for v in victims) for lvl in TRIAGE_LEVELS}
    by_cap = {}
    for v in victims:
        if v["triage"] != "BLACK":
            by_cap[v["required_capability"]] = by_cap.get(v["required_capability"], 0) + 1
    return dict(
        scenario=name, description=cfg["description"],
        incidents=len(incidents), victims=len(victims), victims_by_triage=by_triage,
        victims_by_capability=dict(sorted(by_cap.items())),
        reported_victims_total=sum(i["reported_victims"] for i in incidents),
        mci_cluster=cluster_contention(incidents, ambulances),
        hospitals=len(hospitals),
        er_beds_total=sum(h["er_capacity"] for h in hospitals),
        er_beds_free_t0=sum(h["er_free_t0"] for h in hospitals),
        ambulances=len(ambulances),
        ambulances_available_t0=sum(a["status"] == "available" for a in ambulances),
        als_available_t0=sum(a["status"] == "available" and a["type"] == "ALS" for a in ambulances),
        doctors=sum(s["role"] == "doctor" for s in staff),
        nurses=sum(s["role"] == "nurse" for s in staff),
        float_pool=sum(s["home"] == "FLOAT" for s in staff),
        responder_teams=len(responders),
        # >1 means more RED patients than free ER beds at t=0: agents must
        # negotiate (reserve beds, divert, share staff) rather than just assign
        red_to_free_er_ratio=round(by_triage["RED"] / max(1, sum(h["er_free_t0"] for h in hospitals)), 2),
    )


def make_streams(seed):
    """One independent ``random.Random`` per entry of STREAMS, seeded "<seed>-<name>"."""
    return {k: random.Random(f"{seed}-{k}") for k in STREAMS}


def generate_scenario(name, seed, out_dir):
    cfg = SCENARIOS[name]
    streams = make_streams(seed)
    hospitals = make_hospitals(streams["hospitals"], cfg)
    staff = make_staff(streams["staff"], cfg)
    ambulances = make_ambulances(streams["ambulances"], cfg)
    responders = make_responders(streams["responders"])
    incidents = make_incidents(streams, cfg)

    d = out_dir / name
    d.mkdir(parents=True, exist_ok=True)
    write_json(d / "hospitals.json", hospitals)
    write_json(d / "stations.json", STATIONS)
    write_json(d / "ambulances.json", ambulances)
    write_json(d / "staff.json", staff)
    write_json(d / "responders.json", responders)
    write_json(d / "incidents.json", incidents)
    write_csv(d / "victims.csv", [
        dict(incident_id=i["incident_id"], t_report_min=i["t_report_min"], **vic)
        for i in incidents for vic in i["victims"]])
    write_csv(d / "travel_times.csv", travel_rows(hospitals, incidents))
    summary = summarise(name, cfg, hospitals, ambulances, staff, responders, incidents)
    write_json(d / "summary.json", summary)

    files = sorted(p for p in d.iterdir() if p.name != "manifest.json")
    write_json(d / "manifest.json", dict(
        scenario=name, seed=seed, generator_version=GENERATOR_VERSION,
        horizon_min=HORIZON_MIN, python=platform.python_version(),
        sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--scenario", choices=list(SCENARIOS), action="append",
                    help="repeatable; default is all scenarios")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    args = ap.parse_args(argv)
    for name in args.scenario or list(SCENARIOS):
        s = generate_scenario(name, args.seed, args.out)
        t = s["victims_by_triage"]
        c = s["mci_cluster"]
        print(f"{name:12s} {s['incidents']:2d} incidents  {s['victims']:2d} victims "
              f"(R{t['RED']} Y{t['YELLOW']} G{t['GREEN']} B{t['BLACK']})  "
              f"ER free {s['er_beds_free_t0']}/{s['er_beds_total']}  "
              f"ambulances {s['ambulances_available_t0']}/{s['ambulances']}  "
              + (f"cluster {c['patients']} patients ({c['red']} RED) in {c['span_min']} min "
                 f"vs {c['seats']} seats / {c['als_units']} ALS" if c else "no cluster"))


if __name__ == "__main__":
    main()
