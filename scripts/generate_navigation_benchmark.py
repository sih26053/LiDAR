"""Generate deterministic navigation-benchmark frames via PyBullet.

Targeted families fill the gaps in recorded data (B/C/E/F) plus held-out
OOD families. NO Laya inference here (fast, pure perception): each sample
runs scan -> Stages 1-6 -> oracle label, then persists through the
existing frame recorder + SQLite (frames/maps/safety; NO decision rows).
Evaluation with Laya happens separately (source=benchmark-laya).

Run/family separation is built in: one run_id per family, so splits can
separate by run (no adjacent-frame leakage across splits by construction:
every sample is an independent env reset).

Tags: runs carry mode='benchmark', scenario=<family>;
recorder jev source='benchmark-generated' (never live autonomy metrics).

Run:
    python scripts/generate_navigation_benchmark.py [--targets B:100 C:100 ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FAMILIES_IND = ("ind_open", "ind_fwd_blocked", "ind_side_blocked", "ind_trap",
                "ind_emergency")
FAMILIES_OOD = ("ood_narrow", "ood_wide", "ood_big", "ood_dense",
                "ood_pose", "ood_noisy", "ood_dropout", "ood_sparse")


def _rng(seed: int):
    import numpy as np
    return np.random.default_rng(seed)


def make_candidate(family: str, rng, idx: int) -> tuple:
    """Return (env_config, obstacles, perturb) for one independent sample.

    The default vehicle start (-50, 0, yaw 0 from simulation_config.json)
    is kept for IND families (variation comes from obstacle layouts);
    OOD pose/width families override config (env recreated per sample).
    Every sample is an independent env reset -> genuinely distinct state.
    """
    sx, sy, yaw = -50.0, 0.0, 0.0
    cfg: dict = {}
    perturb = None
    obs = []

    def box(dx, dy, hx=1.5, hy=1.0, hz=1.0, h=1.0):
        return {"xyz": [sx + dx, dy, h], "half": [hx, hy, hz]}

    if family == "ind_open":
        if idx % 3 == 0:
            obs = [box(float(rng.uniform(15, 30)), float(rng.uniform(-4, 4)))]
    elif family == "ind_fwd_blocked":
        # Measured recipe (probe_log): a CLOSE slightly-off-center box
        # blocks forward via the forward sectors while staying outside
        # the safety cone (no emergency); a second near box on the same
        # side blocks that side. Yields B/C without emergency.
        side = "left" if idx % 2 == 0 else "right"
        sgn = 1.0 if side == "right" else -1.0
        obs = [box(float(rng.uniform(1.5, 3.0)), sgn * float(
            rng.uniform(0.2, 0.9)))]
        obs.append(box(float(rng.uniform(2.0, 4.0)),
                       sgn * float(rng.uniform(1.5, 3.0))))
    elif family == "ind_side_blocked":
        # Measured recipe (probe_A_geometry): single box almost ABEAM
        # (dx 0.3-0.8, lateral 1.8-2.5) blocks one side while forward
        # stays open and the safety cone misses it (category A).
        sgn = 1.0 if idx % 2 == 0 else -1.0
        obs = [box(float(rng.uniform(0.3, 0.8)),
                   sgn * float(rng.uniform(1.8, 2.5)),
                   hx=1.2, hy=1.2)]
    elif family == "ind_trap":
        # Measured E recipe (probe_log G hits): diagonal pair CLOSE
        # (dx 1.0-1.8, lateral 1.2-2.6) + forward box 7.5-10.5 m offset.
        import math
        dx = float(rng.uniform(1.0, 1.8))
        dy = float(rng.uniform(1.2, 2.6))
        hx = float(rng.uniform(1.0, 1.6))
        sgn = 1.0 if idx % 2 == 0 else -1.0
        obs = [box(dx, dy, hx, hx),
               box(dx, -dy, hx, hx),
               box(float(rng.uniform(7.5, 10.5)), sgn * float(
                   rng.uniform(0.2, 0.8)), 2.0, 0.8)]
    elif family == "ind_emergency":
        obs = [box(float(rng.uniform(1.5, 3.0)), float(rng.uniform(-0.5, 0.5)))]
    elif family == "ood_narrow":
        cfg["road"] = {"width_m": 8.0}
        if idx % 2 == 0:
            obs = [box(float(rng.uniform(5, 9)), 0.0)]
    elif family == "ood_wide":
        cfg["road"] = {"width_m": 18.0}
        obs = [box(float(rng.uniform(4, 8)), float(rng.uniform(-6, 6)),
                   2.5, 1.5)]
    elif family == "ood_big":
        obs = [box(float(rng.uniform(5, 9)), float(rng.uniform(-2, 2)),
                   3.0, 2.0, 2.0, 2.0)]
    elif family == "ood_dense":
        obs = [box(float(rng.uniform(4, 14)), float(rng.uniform(-5, 5)),
                   0.6, 0.6, 0.6, 0.6) for _ in range(int(rng.integers(4, 8)))]
    elif family == "ood_pose":
        sy = float(rng.uniform(-4, 4))
        yaw = float(rng.uniform(-25, 25))
        cfg = {"vehicle": {"start_xy": [sx, sy], "start_yaw_deg": yaw}}
        obs = [box(float(rng.uniform(5, 12)), float(rng.uniform(-4, 4)))]
    elif family in ("ood_noisy", "ood_dropout", "ood_sparse"):
        obs = [box(float(rng.uniform(5, 10)), float(rng.uniform(-3, 3)))]
        perturb = family
    return cfg, obs, perturb


def perturb_points(pts, kind: str, rng):
    import numpy as np
    pts = np.asarray(pts, dtype=np.float64)
    if kind == "ood_noisy":
        pts[:, :3] += rng.normal(0, 0.15, pts[:, :3].shape)
    elif kind == "ood_dropout":
        keep = rng.random(len(pts)) > 0.30
        pts = pts[keep]
    elif kind == "ood_sparse":
        keep = rng.random(len(pts)) < 0.25
        pts = pts[keep]
    return np.ascontiguousarray(pts)


def main() -> dict:
    ap = argparse.ArgumentParser(description="Generate nav benchmark")
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--per-category", type=int, default=100,
                    help="target B/C/E/F samples (IND families)")
    ap.add_argument("--ood-total", type=int, default=200)
    ap.add_argument("--max-attempts", type=int, default=4000)
    ap.add_argument("--probe-log", type=str, default=None,
                    help="if set: log EVERY attempt (config, clearances, "
                         "category) to this JSONL and persist nothing")
    args = ap.parse_args()

    from src.decision.action_eligibility import navigation_oracle
    from src.decision.jev_state_adapter import enrich_for_laya, to_named_state
    from src.simulation.pybullet_env import PyBulletEnv
    from src.simulation.pybullet_lidar import PyBulletLidar, to_lidar_frame
    from src.simulation.stages_runner import run_stages_1_to_6
    from backend.services import frame_recorder, store

    rng = _rng(args.seed)
    ts = int(time.time())
    # E = no-safe-direction WITHOUT emergency flag (G category,
    # non-emergency). Tracked as E_NO_SAFE_DIRECTION.
    want = {"A_OPEN_FORWARD": args.per_category,
            "B_FORWARD_BLOCKED_LEFT_OPEN": args.per_category,
            "C_FORWARD_BLOCKED_RIGHT_OPEN": args.per_category,
            "E_NO_SAFE_DIRECTION": args.per_category,
            "F_EMERGENCY": args.per_category}
    got: dict = {k: 0 for k in want}
    ood_got = 0
    sidecar: list = []
    attempts = 0
    fams = list(FAMILIES_IND) + list(FAMILIES_OOD)
    # One run per family (run-group separation for splits).
    run_ids = {f: f"benchmark-{f}-{ts}" for f in fams}
    for f in fams:
        store.create_run(run_ids[f], "benchmark", scenario=f,
                         decision_model="laya",
                         metadata={"family": f, "seed": args.seed,
                                   "generator": "generate_navigation_benchmark.py"})
        frame_recorder.write_run_metadata(
            run_ids[f], {"mode": "benchmark", "scenario": f,
                         "started_utc": time.strftime(
                             "%Y-%m-%dT%H:%M:%SZ", time.gmtime())})

    env = PyBulletEnv(gui=False)
    env.connect()
    try:
        seq = {f: 0 for f in fams}
        # Need-based family selection: focus attempts on unfilled targets.
        sources = {"A_OPEN_FORWARD": ["ind_side_blocked", "ind_open"],
                   "B_FORWARD_BLOCKED_LEFT_OPEN": ["ind_fwd_blocked"],
                   "C_FORWARD_BLOCKED_RIGHT_OPEN": ["ind_fwd_blocked"],
                   "E_NO_SAFE_DIRECTION": ["ind_trap"],
                   "F_EMERGENCY": ["ind_emergency"]}
        ood_cycle = 0
        while attempts < args.max_attempts and (
                any(got[k] < want[k] for k in want) or ood_got < args.ood_total):
            pool = []
            for cat, fams_ in sources.items():
                if got[cat] < want[cat]:
                    pool += fams_
            if ood_got < args.ood_total:
                pool += [FAMILIES_OOD[ood_cycle % len(FAMILIES_OOD)]]
                ood_cycle += 1
            if not pool:
                break
            family = pool[rng.integers(len(pool))]
            attempts += 1
            is_ood = family in FAMILIES_OOD
            cfg, obs, perturb = make_candidate(family, rng, attempts)
            # Apply custom config via reconnect when needed.
            if cfg.get("road") or cfg.get("vehicle"):
                env.close()
                env = PyBulletEnv(gui=False, config=cfg)
                env.connect()
                env.reset(seed=args.seed + attempts,
                          scenario={"obstacles": obs})
            else:
                try:
                    env.reset(seed=args.seed + attempts,
                              scenario={"obstacles": obs})
                except Exception:
                    # Config-carrying env from a previous OOD sample:
                    # rebuild default before reuse.
                    env.close()
                    env = PyBulletEnv(gui=False)
                    env.connect()
                    env.reset(seed=args.seed + attempts,
                              scenario={"obstacles": obs})
            lidar = PyBulletLidar()
            pts = lidar.to_points(lidar.scan(env))
            if perturb:
                pts = perturb_points(pts, perturb, rng)
                if len(pts) < 10:
                    continue
            frame = to_lidar_frame(pts, f"gen-{attempts:05d}", 0.0)
            try:
                res = run_stages_1_to_6(frame.points, frame.frame_id,
                                        frame.timestamp)
            except Exception:
                continue
            st = res["state"]
            try:
                named = to_named_state(st["state_vector"],
                                       st["sector_ranges_m"])
                named = enrich_for_laya(named, st["safety"])
                oracle = navigation_oracle(named, st["safety"])
            except (ValueError, TypeError):
                continue
            cat = oracle["category"]
            emergency = bool(st["safety"]["emergency_stop"])
            is_E = (cat == "G_NO_SAFE_ACTION" and not emergency)
            if args.probe_log:
                with open(args.probe_log, "a") as fh:
                    fh.write(json.dumps({
                        "family": family, "obstacles": obs,
                        "clearances": {
                            "forward": named.get("forward_clearance_m"),
                            "left": named.get("left_clearance_m"),
                            "right": named.get("right_clearance_m")},
                        "emergency": emergency, "category": cat,
                    }) + "\n")
                continue
            accept = False
            if is_ood:
                accept = ood_got < args.ood_total
            elif is_E and got["E_NO_SAFE_DIRECTION"] < want[
                    "E_NO_SAFE_DIRECTION"]:
                accept = True
            elif cat in want and got[cat] < want[cat]:
                accept = True
            if not accept:
                continue
            # Persist via the existing recorder + SQLite.
            run_id = run_ids[family]
            i = seq[family]
            veh = env.get_vehicle_state()
            safety_rec = {
                "verdict": "BENCHMARK-NO-DECISION",
                "reason": "benchmark generation: perception only, no Laya call",
                "safety_override": False,
                "emergency_stop": emergency,
                "nearest_forward_obstacle_m": st["safety"][
                    "nearest_forward_obstacle_m"],
                "forward_clearance_m": named.get("forward_clearance_m"),
                "left_clearance_m": named.get("left_clearance_m"),
                "right_clearance_m": named.get("right_clearance_m"),
            }
            rec = frame_recorder.record_frame(
                run_id=run_id, sequence=i,
                loop_frame_id=frame.frame_id, scenario=family,
                timestamp=frame.timestamp, lidar_points=pts,
                pipeline_result=res["result"],
                state_vector=st["state_vector"],
                jev={"source": "benchmark-generated",
                     "decision_model": "laya",
                     "request_id": f"{run_id}-q{i:05d}"},
                safety=safety_rec,
                execution={"executed_action": None,
                           "source": "benchmark-generated"},
                vehicle=veh)
            fid = rec["frame_id"]
            prefix = f"results/recordings/{run_id}"
            store.create_frame(
                fid, run_id, frame.timestamp, i, family,
                f"{prefix}/{rec['points_path']}", int(pts.shape[0]),
                [float(v) for v in st["state_vector"]],
                {k: veh.get(k) for k in
                 ("x", "y", "z", "yaw_deg", "speed_ms", "yaw_rate")},
                dict(res["result"].get("timing") or {}),
                f"{prefix}/{rec['metadata_path']}")
            store.create_map_cells(fid, rec["map_cells"])
            store.record_safety_event(
                fid, "BENCHMARK-NO-DECISION", override=False,
                reason=safety_rec["reason"], emergency_flag=emergency,
                nearest_obstacle_m=st["safety"]["nearest_forward_obstacle_m"],
                forward_clearance_m=named.get("forward_clearance_m"),
                left_clearance_m=named.get("left_clearance_m"),
                right_clearance_m=named.get("right_clearance_m"))
            store.finish_run(run_id, total_steps=seq[family] + 1,
                             status="BENCHMARK")
            sidecar.append({
                "frame_id": fid, "run_id": run_id, "family": family,
                "ood": is_ood, "sequence": i,
                "category": ("E_NO_SAFE_DIRECTION" if is_E else cat),
                "oracle_admissible": oracle["admissible"],
                "oracle_preferred": oracle["preferred"],
                "oracle_rule": oracle["rule"],
                "emergency": emergency,
                "clearances_m": {
                    "forward": named.get("forward_clearance_m"),
                    "left": named.get("left_clearance_m"),
                    "right": named.get("right_clearance_m")},
                "state_vector": [float(v) for v in st["state_vector"]],
            })
            seq[family] += 1
            if is_ood:
                ood_got += 1
            elif is_E:
                got["E_NO_SAFE_DIRECTION"] += 1
            else:
                got[cat] += 1
    finally:
        env.close()
    for f in fams:
        try:
            store.finish_run(run_ids[f], total_steps=seq[f], status="BENCHMARK")
        except Exception:
            pass
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                             time.gmtime()),
              "seed": args.seed, "attempts": attempts,
              "targets": want, "got": got, "ood_got": ood_got,
              "run_ids": run_ids, "n_frames": len(sidecar),
              "note": ("E_NO_SAFE_DIRECTION = G_NO_SAFE_ACTION without "
                       "emergency flag (all directions unsafe, no flag).")}
    (PROJECT_ROOT / "results" / "benchmark_runs.json").write_text(
        json.dumps(report, indent=2))
    (PROJECT_ROOT / "results" / "benchmark_labels.json").write_text(
        json.dumps({"generated_utc": report["generated_utc"], "rows": sidecar},
                   indent=2))
    print(json.dumps({k: report[k] for k in
                      ("targets", "got", "ood_got", "n_frames",
                       "attempts")}, indent=2))
    store.close()
    return report


if __name__ == "__main__":
    main()
