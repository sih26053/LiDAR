"""Phase 1: inventory all existing recorded data (read-only).

Produces results/laya_dataset_inventory.{json,csv,md}:
runs, frames, scenarios, per-scenario/​per-run counts, action
distributions (live/raw/constrained), emergencies, overrides,
collisions, revisions, sources, missing fields, duplicates.
Never modifies source data.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> dict:
    from backend.services import store

    inv: dict = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                time.gmtime()),
                 "db_path": str(store.db_path())}
    runs = store.list_runs(limit=1000)
    inv["n_runs"] = len(runs)
    inv["runs"] = [
        {"run_id": r["run_id"], "mode": r.get("mode"),
         "scenario": r.get("scenario"), "status": r.get("status"),
         "frame_count": r.get("frame_count"),
         "decision_model": r.get("decision_model") or r.get("jev_model")}
        for r in runs]
    con = store._connect()
    n_frames = con.execute("SELECT COUNT(*) c FROM frames").fetchone()["c"]
    inv["n_frames"] = n_frames
    scen = con.execute(
        "SELECT scenario, COUNT(*) c FROM frames GROUP BY scenario"
        ).fetchall()
    inv["frames_per_scenario"] = {r["scenario"]: r["c"] for r in scen}
    runframes = con.execute(
        "SELECT run_id, COUNT(*) c FROM frames GROUP BY run_id").fetchall()
    inv["frames_per_run"] = {r["run_id"]: r["c"] for r in runframes}

    dec = con.execute(
        "SELECT source, proposed_action, COUNT(*) c FROM decisions "
        "GROUP BY source, proposed_action").fetchall()
    inv["decisions_by_source_action"] = [
        {"source": r["source"], "action": r["proposed_action"],
         "count": r["c"]} for r in dec]
    raw = con.execute(
        "SELECT raw_action, COUNT(*) c FROM decisions WHERE raw_action "
        "IS NOT NULL GROUP BY raw_action").fetchall()
    inv["raw_laya_action_distribution"] = {r["raw_action"]: r["c"]
                                           for r in raw}
    conact = con.execute(
        "SELECT constrained_action, COUNT(*) c FROM decisions WHERE "
        "constrained_action IS NOT NULL GROUP BY constrained_action"
        ).fetchall()
    inv["constrained_action_distribution"] = {
        r["constrained_action"]: r["c"] for r in conact}
    modes = con.execute(
        "SELECT mode, COUNT(*) c FROM decisions WHERE mode IS NOT NULL "
        "GROUP BY mode").fetchall()
    inv["decisions_by_mode"] = {r["mode"]: r["c"] for r in modes}
    emer = con.execute(
        "SELECT COUNT(*) c FROM safety_events WHERE emergency_flag=1"
        ).fetchone()["c"]
    over = con.execute(
        "SELECT COUNT(*) c FROM safety_events WHERE override=1").fetchone()["c"]
    coll = con.execute(
        "SELECT COUNT(*) c FROM executions WHERE collision=1").fetchone()["c"]
    inv["emergency_events"] = emer
    inv["safety_overrides"] = over
    inv["collisions"] = coll
    revs = con.execute(
        "SELECT checkpoint_revision, COUNT(*) c FROM decisions WHERE "
        "checkpoint_revision IS NOT NULL GROUP BY checkpoint_revision"
        ).fetchall()
    inv["checkpoint_revisions"] = {r["checkpoint_revision"]: r["c"]
                                   for r in revs}
    srcs = con.execute(
        "SELECT source, COUNT(*) c FROM decisions GROUP BY source").fetchall()
    inv["sources"] = {r["source"]: r["c"] for r in srcs}
    # Missing-field audit on newest-schema columns.
    miss = {}
    for col in ("answer_confidence", "mode", "eligible_json", "raw_action",
                "constrained_action", "checkpoint_revision", "gate_threshold"):
        try:
            m = con.execute(
                f"SELECT COUNT(*) c FROM decisions WHERE {col} IS NULL"
                ).fetchone()["c"]
            miss[col] = m
        except Exception as exc:  # noqa: BLE001
            miss[col] = f"unavailable: {exc}"
    inv["missing_new_fields"] = miss
    dup = con.execute(
        "SELECT points_path, COUNT(*) c FROM frames GROUP BY points_path "
        "HAVING c > 1").fetchall()
    inv["duplicate_points_paths"] = [
        {"points_path": r["points_path"], "count": r["c"]} for r in dup]
    bad = con.execute(
        "SELECT frame_id FROM frames WHERE point_count IS NULL OR "
        "point_count <= 0").fetchall()
    inv["invalid_lidar_frames"] = [r["frame_id"] for r in bad]
    fids = con.execute(
        "SELECT frame_id, run_id, scenario FROM frames ORDER BY run_id,"
        " sequence").fetchall()
    inv["frame_ids"] = [r["frame_id"] for r in fids]

    out = PROJECT_ROOT / "results" / "laya_dataset_inventory.json"
    out.write_text(json.dumps(inv, indent=2, default=str))
    csvp = PROJECT_ROOT / "results" / "laya_dataset_inventory.csv"
    with open(csvp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dimension", "key", "value"])
        for r in inv["runs"]:
            w.writerow(["run", r["run_id"],
                        f"frames={r['frame_count']} scenario={r['scenario']} "
                        f"mode={r['mode']}"])
        for k, v in inv["frames_per_scenario"].items():
            w.writerow(["scenario", k, v])
        for d in inv["decisions_by_source_action"]:
            w.writerow(["decision", f"{d['source']}:{d['action']}", d["count"]])
    md = PROJECT_ROOT / "results" / "laya_dataset_inventory.md"
    lines = ["# Laya dataset inventory (measured, read-only)",
             f"Generated: {inv['generated_utc']}",
             f"DB: {inv['db_path']}",
             f"Runs: {inv['n_runs']}, Frames: {inv['n_frames']}",
             f"Emergencies: {emer}, Overrides: {over}, Collisions: {coll}",
             "", "## Frames per scenario"]
    lines += [f"- {k}: {v}" for k, v in inv["frames_per_scenario"].items()]
    lines += ["", "## Decisions by source/action"]
    lines += [f"- {d['source']} {d['action']}: {d['count']}"
              for d in inv["decisions_by_source_action"]]
    lines += ["", "## Missing new-schema fields (NULL count)"]
    lines += [f"- {k}: {v}" for k, v in miss.items()]
    md.write_text("\n".join(lines) + "\n")
    print(json.dumps({k: inv[k] for k in
                      ("n_runs", "n_frames", "frames_per_scenario",
                       "sources", "emergency_events", "safety_overrides",
                       "collisions", "checkpoint_revisions")}, indent=2))
    print(f"wrote {out}, {csvp}, {md}")
    store.close()
    return inv


if __name__ == "__main__":
    main()
