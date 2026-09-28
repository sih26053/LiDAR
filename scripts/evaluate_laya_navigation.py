"""Forward-bias diagnostics: run Laya on recorded Stage-6 states.

Measures (never assumes) corridor behaviour. Two explicit modes per frame:
    unconstrained_laya : model sees all four choices (raw preference)
    constrained_laya   : model sees ONLY eligible actions (production)

State sources:
    pybullet : deterministic scenarios from config/pybullet_scenarios.json
               re-run through Stages 1-6 (same code as the live path).
    recorded : live-recorded frames from SQLite (state_vector; meter
               derivation falls back to normalized x 30 m — documented in
               the adapter; oracle geometry uses the same values the
               production path would see).

Per category (A..G) the report measures: frame count, proposed-action
distribution, probability/confidence summaries, safety result, executed
action, eligibility, unsafe proposals, overrides. Outputs:
    results/laya_navigation_evaluation.json
    results/laya_navigation_evaluation.csv

A behaviour is only called "bias" when the data shows systematic
preference beyond what the navigation objective permits (oracle
preferred/admissible comparison).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EXEC_TO_CHOICE = {"forward": "forward", "turn_left": "left",
                  "turn_right": "right", "stop": "stop"}


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def collect_pybullet_states() -> list:
    """Deterministic scenario states via Stages 1-6 (live-path code)."""
    from src.decision.jev_state_adapter import enrich_for_laya, to_named_state
    from src.simulation.pybullet_env import PyBulletEnv
    from src.simulation.pybullet_lidar import PyBulletLidar, to_lidar_frame
    from src.simulation.stages_runner import run_stages_1_to_6

    scenarios = json.loads(
        (PROJECT_ROOT / "config" / "pybullet_scenarios.json").read_text()
    )["scenarios"]
    states = []
    for sc in scenarios:
        env = PyBulletEnv()
        env.connect()
        env.reset(seed=42, scenario={"obstacles": sc["obstacles"]})
        try:
            lidar = PyBulletLidar()
            pts = lidar.to_points(lidar.scan(env))
            frame = to_lidar_frame(pts, f"eval-{sc['id']}", 0.0)
            res = run_stages_1_to_6(frame.points, frame.frame_id,
                                    frame.timestamp)
            st = res["state"]
            named = to_named_state(st["state_vector"], st["sector_ranges_m"])
            named = enrich_for_laya(named, st["safety"])
            states.append({
                "frame_id": f"pybullet:{sc['id']}",
                "origin": "pybullet-scenario",
                "scenario": sc["id"],
                "scenario_goal": sc.get("goal"),
                "named_state": named,
                "safety": st["safety"],
            })
        finally:
            env.close()
    return states


SYNTHETIC_CORRIDOR = [
    # (id, fwd_m, left_m, right_m, emergency, goal) — constructed named
    # states (documented synthetic benchmark input, NOT recorded frames).
    ("open_corridor", 20.0, 8.0, 8.0, False, "drive straight"),
    ("fwd_blocked_left_escape", 2.0, 9.0, 1.0, False, "escape left"),
    ("fwd_blocked_right_escape", 2.0, 1.0, 9.0, False, "escape right"),
    ("fwd_left_blocked_right_open", 3.0, 1.0, 8.0, False, "escape right"),
    ("fwd_right_blocked_left_open", 3.0, 8.0, 1.0, False, "escape left"),
    ("narrow_corridor", 6.0, 3.0, 3.0, False, "proceed carefully"),
    ("symmetric_corridor", 7.0, 7.0, 7.0, False, "drive straight"),
    ("centerline_obstacle", 4.0, 6.0, 6.0, False, "pass obstacle"),
    ("offcenter_obstacle", 4.0, 3.0, 7.0, False, "pass on the open side"),
    ("emergency_corridor", 1.0, 1.0, 1.0, True, "immediate safety STOP"),
    ("no_safe_action", 2.0, 1.0, 1.0, False, "stop, no escape"),
    ("emergency_flag_open", 15.0, 8.0, 8.0, True, "flag dominates geometry"),
]


def collect_synthetic_states() -> list:
    """Balanced corridor benchmark (constructed, deterministic)."""
    from src.decision.jev_state_adapter import (
        FALLBACK_RADIUS_M, enrich_for_laya, to_named_state)

    def norm(m: float) -> float:
        return max(0.0, min(1.0, m / FALLBACK_RADIUS_M))

    states = []
    for sid, fwd, left, right, emergency, goal in SYNTHETIC_CORRIDOR:
        rear = 15.0
        vec = [norm(fwd), norm(left), norm(left), norm(rear), norm(rear),
               norm(right), norm(right), norm(fwd),
               0.30, 0.0, 0.30, 0.35, 0.25]
        ranges_m = [fwd, left, left, rear, rear, right, right, fwd]
        named = to_named_state(vec, ranges_m)
        safety = {"emergency_stop": bool(emergency),
                  "nearest_forward_obstacle_m": round(min(fwd, 30.0), 2)}
        named = enrich_for_laya(named, safety)
        states.append({
            "frame_id": f"synthetic:{sid}",
            "origin": "synthetic-corridor",
            "scenario": sid,
            "scenario_goal": goal,
            "named_state": named,
            "safety": safety,
        })
    return states


def collect_recorded_states(run_id: str | None = None,
                            limit: int | None = None) -> list:
    """Recorded-frame states from SQLite (state_vector + fallback meters)."""
    from src.decision.jev_state_adapter import enrich_for_laya, to_named_state

    from backend.services import store

    frames = store.list_frames(run_id=run_id,
                               limit=min(int(limit or 500), 5000))
    try:
        goals = {s["id"]: s.get("goal") for s in json.loads(
            (PROJECT_ROOT / "config" / "pybullet_scenarios.json"
             ).read_text())["scenarios"]}
    except (OSError, ValueError):
        goals = {}
    states = []
    for f in frames:
        try:
            vec = json.loads(f.get("state_vector_json") or "[]")
            named = to_named_state([float(v) for v in vec], None)
            safety = {"emergency_stop": False,
                      "nearest_forward_obstacle_m": None}
            named = enrich_for_laya(named, safety)
            states.append({
                "frame_id": f["frame_id"],
                "origin": "live-recorded",
                "scenario": f.get("scenario"),
                "scenario_goal": goals.get(f.get("scenario")),
                "named_state": named,
                "safety": safety,
            })
        except (ValueError, TypeError, KeyError):
            continue
    return states


def evaluate(states: list) -> dict:
    from src.decision.action_eligibility import navigation_oracle
    from src.decision.policy_interface import decide_with_safety, get_policy

    policy = get_policy("laya")
    rows = []
    for s in states:
        named, safety = s["named_state"], s["safety"]
        nearest = (safety or {}).get("nearest_forward_obstacle_m")
        emergency = bool((safety or {}).get("emergency_stop", False))
        oracle = navigation_oracle(named, safety,
                                   scenario_goal=s.get("scenario_goal"))
        for mode in ("unconstrained", "constrained"):
            dec = decide_with_safety(
                policy, named, nearest, True, emergency,
                frame_id=s["frame_id"], run_id="laya-nav-eval",
                source="laya-unconstrained-diagnostic"
                if mode == "unconstrained" else "laya",
                mode=mode, safety_dict=safety)
            prop_choice = EXEC_TO_CHOICE.get(
                str(dec.get("proposed_action") or ""), None)
            exec_choice = EXEC_TO_CHOICE.get(
                str(dec.get("executed_action") or ""), None)
            rows.append({
                "frame_id": s["frame_id"],
                "origin": s["origin"],
                "scenario": s.get("scenario"),
                "category": dec.get("category"),
                "oracle_preferred": oracle["preferred"],
                "oracle_admissible": oracle["admissible"],
                "mode": dec.get("mode"),
                "eligible_actions": dec.get("eligible_actions"),
                "proposed_action": (prop_choice or dec.get("proposed_action")),
                "answer_confidence": dec.get("answer_confidence"),
                "raw_confidence": dec.get("engine_confidence_raw"),
                "probabilities": dec.get("probabilities"),
                "raw_laya_action": dec.get("raw_laya_action"),
                "constrained_action": dec.get("constrained_action"),
                "laya_skipped": dec.get("laya_skipped"),
                "gate": dec.get("gate"),
                "safety_status": dec.get("safety_status"),
                "safety_override": dec.get("safety_override"),
                "executed_action": (exec_choice or dec.get("executed_action")),
                "proposed_eligible": (
                    prop_choice in (dec.get("eligible_actions") or []))
                if prop_choice else None,
                "proposed_unsafe": (
                    prop_choice not in (oracle["admissible"] or []))
                if prop_choice else None,
                "executed_matches_oracle": (
                    exec_choice == oracle["preferred"]),
                "policy_latency_ms": dec.get("policy_latency_ms"),
            })
    return {"generated_utc": _utc(), "n_states": len(states), "rows": rows}


def summarize(evaluation: dict, eval_set: str) -> dict:
    rows = evaluation["rows"]
    by_mode: dict = {}
    for mode in ("unconstrained_laya", "constrained_laya"):
        mrows = [r for r in rows if r["mode"] == mode]
        n = len(mrows)
        dist: dict = {}
        for r in mrows:
            dist[r["proposed_action"]] = dist.get(r["proposed_action"], 0) + 1
        exec_dist: dict = {}
        for r in mrows:
            exec_dist[r["executed_action"]] = exec_dist.get(
                r["executed_action"], 0) + 1
        aconfs = [r["answer_confidence"] for r in mrows
                  if r["answer_confidence"] is not None]
        by_mode[mode] = {
            "n": n,
            "action_distribution": dist,
            "executed_distribution": exec_dist,
            "raw_forward_rate": (round(dist.get("forward", 0) / n, 4)
                                 if n else 0.0),
            "executed_forward_rate": (round(exec_dist.get("forward", 0) / n, 4)
                                      if n else 0.0),
            "unsafe_proposals": sum(1 for r in mrows if r["proposed_unsafe"]),
            "unsafe_proposal_rate": (round(sum(1 for r in mrows
                                               if r["proposed_unsafe"]) / n, 4)
                                     if n else 0.0),
            "emergency_raw_forward": sum(
                1 for r in mrows if r["category"] == "F_EMERGENCY"
                and (r["raw_laya_action"] or r["proposed_action"])
                in ("forward", "FORWARD")),
            "emergency_executed_forward": sum(
                1 for r in mrows if r["category"] == "F_EMERGENCY"
                and r["executed_action"] == "forward"),
            "oracle_preferred_accuracy": (
                round(sum(1 for r in mrows if r["executed_matches_oracle"])
                      / n, 4) if n else 0.0),
            "safety_overrides": sum(1 for r in mrows if r["safety_override"]),
            "stop_rate": (round(exec_dist.get("stop", 0) / n, 4) if n else 0.0),
            "mean_answer_confidence": (round(sum(aconfs) / len(aconfs), 4)
                                       if aconfs else None),
        }
    cats: dict = {}
    for r in rows:
        key = (r["category"], r["mode"])
        cats.setdefault(key, {"n": 0, "proposed": {}, "executed": {},
                              "unsafe": 0, "overrides": 0})
        c = cats[key]
        c["n"] += 1
        c["proposed"][r["proposed_action"]] = c["proposed"].get(
            r["proposed_action"], 0) + 1
        c["executed"][r["executed_action"]] = c["executed"].get(
            r["executed_action"], 0) + 1
        c["unsafe"] += 1 if r["proposed_unsafe"] else 0
        c["overrides"] += 1 if r["safety_override"] else 0
    # Confusion matrix: oracle preferred x executed (per mode).
    labels = ["forward", "left", "right", "stop"]
    confusion = {}
    for mode in ("unconstrained_laya", "constrained_laya"):
        m = {t: {p: 0 for p in labels} for t in labels}
        for r in rows:
            if r["mode"] == mode and r["oracle_preferred"] in m \
                    and r["executed_action"] in labels:
                m[r["oracle_preferred"]][r["executed_action"]] += 1
        confusion[mode] = m
    return {"generated_utc": evaluation["generated_utc"],
            "evaluation_set": eval_set,
            "n_states": evaluation["n_states"],
            "by_mode": by_mode,
            "by_category_mode": {f"{k[0]}:{k[1]}": v
                                 for k, v in sorted(cats.items())},
            "confusion_oracle_preferred_x_executed": confusion}


def main() -> dict:
    ap = argparse.ArgumentParser(description="Laya navigation diagnostics")
    ap.add_argument("--source", default="balanced",
                    choices=("pybullet", "recorded", "synthetic", "balanced",
                             "both"))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--out", default="results/laya_navigation_evaluation.json")
    args = ap.parse_args()

    states = []
    if args.source in ("pybullet", "both", "balanced"):
        states += collect_pybullet_states()
    if args.source in ("synthetic", "balanced"):
        states += collect_synthetic_states()
    if args.source in ("recorded", "both"):
        states += collect_recorded_states(args.run_id, args.limit)
    if not states:
        raise SystemExit("no states collected (no scenarios / no recorded frames)")
    evaluation = evaluate(states)
    summary = summarize(evaluation, eval_set=args.source)
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary,
                                    "evaluation": evaluation}, indent=2))
    csv_path = out_path.with_suffix(".csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "frame_id", "origin", "scenario", "category",
            "oracle_preferred", "mode", "eligible_actions",
            "proposed_action", "answer_confidence", "raw_confidence",
            "raw_laya_action", "constrained_action", "laya_skipped",
            "gate", "safety_status", "safety_override", "executed_action",
            "proposed_eligible", "proposed_unsafe",
            "executed_matches_oracle", "policy_latency_ms"])
        w.writeheader()
        for r in evaluation["rows"]:
            row = {k: (json.dumps(v, default=str)
                       if isinstance(v, (dict, list)) else v)
                   for k, v in r.items()
                   if k in w.fieldnames or k == "probabilities"}
            w.writerow({k: row.get(k) for k in w.fieldnames})
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path} + {csv_path}")
    return summary


if __name__ == "__main__":
    main()
