"""Check the generated data before any agent touches it.

    python -m emergency_sim.validate

For every scenario in data/: regenerate it from the seed in its manifest into
a temporary folder and compare SHA-256 hashes (reproducibility), then run
integrity checks on the files. Writes data/validation_report.json and exits
non-zero if any check fails.

The regeneration check also covers the caller-report wording noise, which is
drawn from seeded streams. The report adds, for information only, how often the
severe caller wording goes with a true RED incident in each scenario.
"""

import csv
import hashlib
import json
import sys
import tempfile
from pathlib import Path

from . import datagen, geo

DATA = Path(__file__).resolve().parent.parent / "data"


def load(d, name):
    return json.loads((d / name).read_text())


def check_scenario(d):
    manifest = load(d, "manifest.json")
    hospitals = load(d, "hospitals.json")
    ambulances = load(d, "ambulances.json")
    staff = load(d, "staff.json")
    responders = load(d, "responders.json")
    incidents = load(d, "incidents.json")
    victims = [v for i in incidents for v in i["victims"]]
    checks = []

    def check(name, ok, detail=""):
        checks.append(dict(check=name, passed=bool(ok), detail=detail))

    with tempfile.TemporaryDirectory() as tmp:
        datagen.generate_scenario(manifest["scenario"], manifest["seed"], Path(tmp))
        regen = load(Path(tmp) / manifest["scenario"], "manifest.json")["sha256"]
    diff = [f for f in manifest["sha256"] if manifest["sha256"][f] != regen.get(f)]
    check("reproducible from seed", not diff, f"seed {manifest['seed']}; differing: {diff or 'none'}")

    on_disk = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in d.iterdir() if p.name in manifest["sha256"]}
    check("files match manifest", on_disk == manifest["sha256"])

    ids = [i["incident_id"] for i in incidents]
    check("incident ids unique and time-ordered",
          len(set(ids)) == len(ids) and
          [i["t_report_min"] for i in incidents] == sorted(i["t_report_min"] for i in incidents))
    check("incidents inside map and horizon",
          all(0 <= i["x"] <= geo.MAP_SIZE_KM and 0 <= i["y"] <= geo.MAP_SIZE_KM and
              0 < i["t_report_min"] < datagen.HORIZON_MIN for i in incidents))
    check("true_victims matches victim list",
          all(i["true_victims"] == len(i["victims"]) for i in incidents))

    caps = {c for h in hospitals for c in h["capabilities"]}
    missing = sorted({v["required_capability"] for v in victims} - caps)
    check("every victim's need is treatable somewhere", not missing, f"missing: {missing or 'none'}")
    check("RED victims need ALS; only RED do",
          all(v["needs_als"] == (v["triage"] == "RED") for v in victims))
    check("golden windows consistent with triage",
          all((v["triage"] == "RED" and
               datagen.CONDITION_RED_WINDOW.get(v["condition"], datagen.RED_WINDOW)[0]
               <= v["golden_window_min"] <=
               datagen.CONDITION_RED_WINDOW.get(v["condition"], datagen.RED_WINDOW)[1]) or
              (v["triage"] == "YELLOW" and 90 <= v["golden_window_min"] <= 240) or
              (v["triage"] == "GREEN" and v["golden_window_min"] == 480) or
              (v["triage"] == "BLACK" and v["golden_window_min"] == 0) for v in victims))
    check("hospital occupancy within capacity",
          all(0 <= h["er_occupied_t0"] <= h["er_capacity"] and
              0 <= h["icu_occupied_t0"] <= h["icu_beds"] for h in hospitals))
    homes = {h["id"] for h in hospitals} | {"FLOAT"}
    check("staff assigned to real hospitals or float pool",
          all(s["home"] in homes for s in staff))
    check("every hospital has an emergency physician",
          all(any(s["home"] == h["id"] and s["role"] == "doctor" and "emergency" in s["skills"]
                  for s in staff) for h in hospitals))
    gaps = [f"{h['id']}:{c}" for h in hospitals for c in datagen.specialties(h)
            if not any(s["home"] == h["id"] and s["role"] == "doctor" and c in s["skills"] for s in staff)]
    check("every hospital capability has a doctor with that skill", not gaps, f"missing: {gaps or 'none'}")
    pools = {h["id"]: set(datagen.nurse_pool(h)) | {"emergency"} for h in hospitals}
    check("nurse skills match their hospital",
          all(set(s["skills"]) <= pools[s["home"]] for s in staff if s["role"] == "nurse" and s["home"] != "FLOAT"))
    team_caps = {c for t in responders for c in t["capabilities"]}
    need = sorted({c for i in incidents for c in i["required_responder_capabilities"]} - team_caps)
    check("every incident's responder need exists", not need, f"missing: {need or 'none'}")
    check("at least one ALS ambulance available at t=0",
          any(a["type"] == "ALS" and a["status"] == "available" for a in ambulances))

    with (d / "travel_times.csv").open() as f:
        rows = list(csv.DictReader(f))
    n_nodes = len(datagen.STATIONS) + len(hospitals) + len(incidents)
    check("travel matrix complete and positive",
          len(rows) == n_nodes * (n_nodes - 1) and
          all(float(r["ambulance_min"]) > 0 for r in rows if float(r["distance_km"]) > 0),
          f"{len(rows)} rows for {n_nodes} nodes")
    sym = {(r["from_id"], r["to_id"]): r["ambulance_min"] for r in rows}
    check("travel times symmetric", all(sym[(a, b)] == sym[(b, a)] for a, b in sym))

    c = datagen.cluster_contention(incidents, ambulances)
    if c:
        # units busy at t=0 are free again by minute 35, before the cluster.
        # One wave of every in-service unit cannot clear the scene: either more
        # patients than seats, or more RED patients than ALS units (1 RED per trip)
        check("MCI cluster creates contention",
              c["incidents"] == 3 and c["span_min"] <= 8 and
              (c["patients"] > c["seats"] or c["red"] > c["als_units"]),
              f"{c['patients']} patients in {c['span_min']} min vs {c['seats']} seats; "
              f"{c['red']} RED vs {c['als_units']} ALS units")
    return checks


def scenario_fingerprint(d):
    """Incidents without ids, so the same incident matches across scenarios."""
    return [(i["t_report_min"], i["type"], i["x"], i["y"],
             tuple((v["condition"], v["triage"], v["golden_window_min"]) for v in i["victims"]))
            for i in load(d, "incidents.json")]


def wording_confusion(incidents):
    """Counts severe or mild caller wording against a true worst triage of RED.

    Incident types whose severe and mild wording are the same ("stroke") cannot
    be told apart from the text, so they are counted under ``same_wording``.

    Args:
        incidents: Incident dicts as written to ``incidents.json``.

    Returns:
        Counts keyed ``severe_red``, ``severe_not_red``, ``mild_red``,
        ``mild_not_red`` and ``same_wording``.
    """
    counts = dict(severe_red=0, severe_not_red=0, mild_red=0, mild_not_red=0, same_wording=0)
    for i in incidents:
        spec = datagen.INCIDENT_TYPES[i["type"]]
        if spec["hazard"] == spec["hazard_mild"]:
            counts["same_wording"] += 1
            continue
        wording = "severe" if i["caller_report"].endswith(f"; {spec['hazard']}.") else "mild"
        counts[f"{wording}_{'red' if i['worst_triage'] == 'RED' else 'not_red'}"] += 1
    return counts


def check_controlled(report):
    """Scenarios must differ only in their knobs, so results can be compared."""
    checks = []
    normal, surge, constrained = (scenario_fingerprint(DATA / n) for n in ("normal", "surge", "constrained"))
    surge_base = [f for f, i in zip(surge, load(DATA / "surge", "incidents.json")) if not i["mci_cluster"]]
    checks.append(dict(check="constrained has exactly the surge incidents",
                       passed=constrained == surge, detail=""))
    checks.append(dict(check="surge = normal incidents + cluster",
                       passed=surge_base == normal, detail=f"{len(normal)} base incidents"))
    staff = {n: {s["id"] for s in load(DATA / n, "staff.json")} for n in ("normal", "constrained")}
    checks.append(dict(check="constrained staff is a subset of normal staff",
                       passed=staff["constrained"] <= staff["normal"], detail=""))
    # the wording noise must not break the controlled comparison: an incident
    # present in two scenarios has to reach the agents with the same report
    reports = {n: {f: i["caller_report"] for f, i in zip(fp, load(DATA / n, "incidents.json"))}
               for n, fp in (("normal", normal), ("surge", surge), ("constrained", constrained))}
    pairs = [("normal", "surge"), ("normal", "constrained"), ("surge", "constrained")]
    shared = {p: reports[p[0]].keys() & reports[p[1]].keys() for p in pairs}
    differing = sorted({f"{a}/{b}@t{f[0]}" for (a, b), fs in shared.items()
                        for f in fs if reports[a][f] != reports[b][f]})
    checks.append(dict(check="shared incidents have identical caller reports",
                       passed=all(shared.values()) and not differing,
                       detail=", ".join(f"{a}/{b}: {len(shared[(a, b)])} shared" for a, b in pairs)
                       + f"; differing: {differing or 'none'}"))
    report["cross_scenario"] = checks
    return checks


def main():
    report, failed = {}, 0
    for d in sorted(p for p in DATA.iterdir() if (p / "manifest.json").exists()):
        checks = check_scenario(d)
        report[d.name] = checks
        bad = [c for c in checks if not c["passed"]]
        failed += len(bad)
        print(f"{d.name:12s} {len(checks) - len(bad)}/{len(checks)} checks passed")
        for c in bad:
            print(f"   FAIL {c['check']}: {c['detail']}")
    cross = check_controlled(report)
    bad = [c for c in cross if not c["passed"]]
    failed += len(bad)
    print(f"{'cross':12s} {len(cross) - len(bad)}/{len(cross)} checks passed")
    for c in bad:
        print(f"   FAIL {c['check']}: {c['detail']}")
    report["caller_wording_vs_red"] = {}
    for d in sorted(p for p in DATA.iterdir() if (p / "manifest.json").exists()):
        c = wording_confusion(load(d, "incidents.json"))
        report["caller_wording_vs_red"][d.name] = c
        print(f"{d.name:12s} wording vs RED (info): severe {c['severe_red']} RED / "
              f"{c['severe_not_red']} not; mild {c['mild_red']} RED / {c['mild_not_red']} not; "
              f"same wording {c['same_wording']}")
    (DATA / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
