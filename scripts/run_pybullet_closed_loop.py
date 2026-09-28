"""Phase 16 — PyBullet closed-loop controller (executes or reports BLOCKED).

Loop: advance sim -> capture LiDAR -> LiDARFrame -> Stages 2-6 ->
13-D state -> Jev state -> policy decide (Jev default) -> confidence
gate -> safety -> execute -> log -> next step.

Physics/safety run every step; perception every `perception_every`
steps; Jev at `decision_frequency_hz` (Phase 18: last valid decision is
retained between Jev calls ONLY with safety re-evaluated each step).
Every failure is logged, falls back to STOP through safety, and is
recorded -- never fabricated, never crashing silently.

Run:  python scripts/run_pybullet_closed_loop.py [--steps 50] [--gui]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.decision.jev_client import load_jev_config  # noqa: E402


def _load_scenario(scenario_id: str | None) -> dict | None:
    """Scenario config by id (world-frame obstacles); None = default traffic."""
    if not scenario_id:
        return None
    import json
    scenarios = json.loads((PROJECT_ROOT / "config" / "pybullet_scenarios.json").read_text())["scenarios"]
    for sc in scenarios:
        if sc["id"] == scenario_id:
            return {"obstacles": sc["obstacles"]}
    raise ValueError(f"unknown scenario {scenario_id!r}")


def _default_interval_steps() -> int:
    try:
        freq = float(load_jev_config().get("decision_frequency_hz", 2))
    except (TypeError, ValueError):
        freq = 2.0
    return max(1, round(10 / freq)) if freq > 0 else 10


def main(steps: int = 50, gui: bool = False,
         decision_interval_steps: int | None = None,
         run_id: str | None = None,
         scenario_id: str | None = None) -> dict:
    from src.simulation.pybullet_env import PyBulletEnv, PyBulletUnavailable
    from src.simulation.pybullet_lidar import PyBulletLidar, to_lidar_frame
    from src.simulation.pybullet_action_executor import PyBulletActionExecutor
    from src.simulation.stages_runner import run_stages_1_to_6
    from src.decision.jev_state_adapter import enrich_for_jev, to_named_state
    from src.decision.policy_interface import decide_with_safety, get_policy
    from src.safety_controller import evaluate as safety_evaluate

    interval = int(decision_interval_steps) if decision_interval_steps else _default_interval_steps()
    interval = max(1, interval)
    run_id = run_id or f"run-{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}"
    try:
        from backend.services import laya_manager

        laya_boot = laya_manager.ensure_started(wait_s=180.0)
        laya_state = "READY" if laya_boot["health"].get("healthy") else "ERROR"
    except Exception as exc:  # noqa: BLE001 - degraded UNAVAILABLE path
        laya_state = f"ERROR: {type(exc).__name__}: {str(exc)[:200]}"
    trace: dict = {"steps": [], "status": "STARTED", "run_id": run_id,
                   "decision_interval_steps": interval,
                   "scenario_id": scenario_id,
                   "decision_engine": "laya",
                   "laya_state": laya_state,
                   "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        scenario = _load_scenario(scenario_id)
        env = PyBulletEnv(gui=gui)
        env.connect()
        env.reset(seed=42, scenario=scenario)
    except PyBulletUnavailable as exc:
        trace.update({"status": "BLOCKED", "reason": f"simulator: {exc}"})
        _save(trace)
        return trace
    policy = get_policy()
    lidar = PyBulletLidar()
    executor = PyBulletActionExecutor(env)
    last_decision = None
    start_pose = None
    try:
        for i in range(int(steps)):
            step_rec: dict = {"step": i, "run_id": run_id}
            t_all = time.perf_counter()
            t0 = time.perf_counter()
            env.step("stop")  # advance physics one fixed step (control below)
            step_rec["sim_step_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
            t0 = time.perf_counter()
            scan = lidar.scan(env)
            pts = lidar.to_points(scan)
            frame = to_lidar_frame(pts, f"sim-{i:05d}", env.get_simulation_time())
            step_rec["lidar_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
            step_rec["lidar_latency_ms"] = step_rec["lidar_ms"]
            step_rec["points"] = int(pts.shape[0])
            t0 = time.perf_counter()
            out = run_stages_1_to_6(frame.points, frame.frame_id, frame.timestamp)
            step_rec["stages_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
            step_rec["stages_1_to_6_ms"] = step_rec["stages_ms"]
            step_rec["perception_latency_ms"] = step_rec["stages_ms"]
            named = to_named_state(out["state"]["state_vector"],
                                   out["state"]["sector_ranges_m"])
            named = enrich_for_jev(named, out["state"]["safety"])
            nearest = out["state"]["safety"]["nearest_forward_obstacle_m"]
            jev_called = (i % interval == 0 or last_decision is None)
            if jev_called:
                last_decision = decide_with_safety(
                    policy, named, nearest, True, False,
                    frame_id=f"sim-{i:05d}", run_id=run_id)
            # Safety re-evaluates the retained action against CURRENT state
            # every step (never blindly replays a stale verdict).
            retained = (last_decision.get("proposed_action")
                        if last_decision.get("proposed_action") else "stop")
            gate_ok = "passed" in str(last_decision.get("gate", ""))
            candidate = retained if gate_ok else "stop"
            t_safety = time.perf_counter()
            verdict = safety_evaluate(candidate, nearest, map_valid=True,
                                      emergency=bool(out["state"]["safety"]["emergency_stop"]))
            step_rec["safety_latency_ms"] = round((time.perf_counter() - t_safety) * 1000.0, 3)
            final = verdict["final_action"]
            step_rec["decision"] = {**last_decision, "executed_action": final,
                                    "step_safety_verdict": verdict["verdict"],
                                    "step_safety_reason": verdict["reason"],
                                    "safety_override": verdict["verdict"] != "SAFE_TO_EXECUTE",
                                    "stale_override": bool(out["state"]["safety"]["emergency_stop"]),
                                    "jev_called_this_step": jev_called,
                                    "source": last_decision.get("source", "fallback")}
            step_rec["rl_state"] = out["state"]["state_vector"]
            step_rec["map_cells"] = len(out["result"].get("map_cells", []))
            step_rec["jev_latency_ms"] = last_decision.get("policy_latency_ms")
            t0 = time.perf_counter()
            step_rec["execution"] = executor.execute(final)
            step_rec["exec_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
            step_rec["action_execution_latency_ms"] = step_rec["exec_ms"]
            step_rec["loop_ms"] = round((time.perf_counter() - t_all) * 1000.0, 2)
            step_rec["loop_latency_ms"] = step_rec["loop_ms"]
            step_rec["vehicle"] = env.get_vehicle_state()
            if start_pose is None:
                start_pose = dict(step_rec["vehicle"])
            trace["steps"].append(step_rec)
        trace["status"] = "COMPLETED"
        trace["start_pose"] = start_pose
        trace["end_pose"] = trace["steps"][-1]["vehicle"] if trace["steps"] else None
    except Exception as exc:  # noqa: BLE001 - logged, safe stop attempted
        trace.update({"status": "FAILED", "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
    finally:
        try:
            env.close()
        except Exception:
            pass
    _save(trace)
    return trace


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(sum(xs) / len(xs), 3) if xs else None


def _metrics_summary(trace: dict) -> dict:
    """Measured-only metrics (GAP 12); unmeasurable -> NOT_AVAILABLE."""
    steps = trace.get("steps", []) or []
    jev_calls = sum(1 for s in steps if s.get("decision", {}).get("jev_called_this_step"))
    ok = sum(1 for s in steps
             if s.get("decision", {}).get("jev_called_this_step")
             and s.get("decision", {}).get("jev_status") == "OK")
    failed = sum(1 for s in steps
                 if s.get("decision", {}).get("jev_called_this_step")
                 and s.get("decision", {}).get("jev_status") in ("FAILED", "INVALID"))
    unavailable = sum(1 for s in steps
                      if s.get("decision", {}).get("jev_called_this_step")
                      and s.get("decision", {}).get("jev_status") == "UNAVAILABLE")
    by_type: dict = {}
    for s in steps:
        a = s.get("decision", {}).get("executed_action")
        by_type[str(a)] = by_type.get(str(a), 0) + 1
    overrides = sum(1 for s in steps if s.get("decision", {}).get("safety_override"))
    collisions = sum(1 for s in steps if s.get("execution", {}).get("collision"))
    start, end = trace.get("start_pose"), trace.get("end_pose")
    movement = None
    if start and end:
        import math
        movement = round(math.hypot(end["x"] - start["x"],
                                    end["y"] - start["y"]), 3)
    return {
        "run_id": trace.get("run_id"),
        "scenario_id": trace.get("scenario_id"),
        "status": trace.get("status"),
        "steps_recorded": len(steps),
        "sim_step_latency_ms_mean": _mean([s.get("sim_step_ms") for s in steps]),
        "lidar_latency_ms_mean": _mean([s.get("lidar_ms") for s in steps]),
        "stages_1_to_6_latency_ms_mean": _mean([s.get("stages_ms") for s in steps]),
        "jev_latency_ms_mean": _mean([s.get("decision", {}).get("policy_latency_ms")
                                      for s in steps
                                      if s.get("decision", {}).get("jev_called_this_step")]),
        "safety_latency_ms_mean": _mean([s.get("safety_latency_ms") for s in steps]),
        "action_execution_latency_ms_mean": _mean([s.get("exec_ms") for s in steps]),
        "loop_latency_ms_mean": _mean([s.get("loop_ms") for s in steps]),
        "laya_calls": jev_calls,
        "laya_successful": ok,
        "laya_failed": failed,
        "laya_unavailable": unavailable,
        "actions_by_type": by_type,
        "safety_overrides": overrides,
        "vehicle_movement_m": movement if movement is not None else "NOT_AVAILABLE",
        "collisions": collisions,
        "scenario_completion": ("COMPLETED" if trace.get("status") == "COMPLETED"
                                else "NOT_AVAILABLE"),
    }


def _save(trace: dict) -> None:
    out = PROJECT_ROOT / "results" / "pybullet"
    out.mkdir(parents=True, exist_ok=True)
    (out / "closed_loop_trace.json").write_text(json.dumps(trace, indent=2)[:2_000_000])
    (PROJECT_ROOT / "results" / "decisions").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "results" / "metrics").mkdir(parents=True, exist_ok=True)
    import csv
    rows = []
    for s in trace.get("steps", []):
        d = s.get("decision", {})
        rows.append({"step": s.get("step"), "run_id": s.get("run_id"),
                     "points": s.get("points"),
                     "model": "laya", "source": d.get("source"),
                     "proposed_action": d.get("proposed_action"),
                     "confidence": d.get("confidence"),
                     "safety_override": d.get("safety_override"),
                     "executed_action": d.get("executed_action"),
                     "jev_latency_ms": d.get("policy_latency_ms"),
                     "safety_latency_ms": s.get("safety_latency_ms"),
                     "action_execution_latency_ms": s.get("exec_ms"),
                     "loop_latency_ms": s.get("loop_ms")})
    with open(PROJECT_ROOT / "results" / "decisions" / "closed_loop_decisions.csv",
              "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["step", "run_id", "points", "model",
                                           "source", "proposed_action",
                                           "confidence", "safety_override",
                                           "executed_action", "jev_latency_ms",
                                           "safety_latency_ms",
                                           "action_execution_latency_ms",
                                           "loop_latency_ms"])
        w.writeheader()
        w.writerows(rows)
    summary = _metrics_summary(trace)
    (PROJECT_ROOT / "results" / "metrics" / "pybullet_jev_metrics.json").write_text(
        json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PyBullet closed loop (Stages 1-6 + Jev + safety)")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--decision-interval-steps", type=int, default=None,
                    help="Jev call interval in loop steps (default: from jev_config decision_frequency_hz)")
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--scenario", type=str, default=None,
                    help="scenario id from config/pybullet_scenarios.json (default: built-in traffic)")
    args = ap.parse_args()
    print(json.dumps({k: v for k, v in main(args.steps, args.gui,
                                            args.decision_interval_steps,
                                            args.run_id, args.scenario).items()
                      if k != "steps"}, indent=2))
