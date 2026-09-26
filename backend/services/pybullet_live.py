"""In-process PyBullet + Jev live runtime (GAP 7/8/18).

Background-thread control loop driving the REAL chain per step:
PyBullet -> LiDAR -> LiDARFrame -> Stages 1-6 -> 13-D -> named Jev
state -> Jev (autonomous mode) -> confidence gate -> safety (every
step) -> executor -> vehicle.

Modes:
  idle        - no loop running.
  autonomous  - loop calls Jev every `decision_interval_steps`, safety
                re-evaluated every step, last valid action retained only
                while safety keeps passing.
  manual      - loop does NOT call Jev; each POST /simulation/action/*
                executes exactly one safety-checked manual step.

Every action record carries `source` ("jev" | "fallback" | "manual")
so manual controls never contaminate Jev metrics. Snapshots served by
GET /simulation/state and /ws/live contain live values only -- never
credentials, never fabricated numbers.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List

from backend.services import decision_store

_lock = threading.RLock()
_thread: threading.Thread | None = None
_stop_flag = False

_runtime: Dict[str, Any] = {
    "mode": "idle",  # idle | autonomous | manual
    "run_id": None,
    "step": 0,
    "started_utc": None,
    "decision_interval_steps": 5,
    "jev_calls": 0,
    "jev_ok": 0,
    "jev_failed": 0,
    "manual_calls": 0,
    "safety_overrides": 0,
    "collisions": 0,
    "directional_actions": 0,
    "stop_actions": 0,
    "distance_m": 0.0,
    "last_step_utc": None,
    "loop_fps": None,
    "error": None,
}

_live: Dict[str, Any] = {
    "env": None,
    "policy": None,
    "lidar": None,
    "executor": None,
    "last_decision": None,
    "snapshot": None,
}

MANUAL_ACTIONS = ("forward", "left", "right", "stop")


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _build_runtime():
    from src.decision.policy_interface import get_policy
    from src.simulation.pybullet_action_executor import PyBulletActionExecutor
    from src.simulation.pybullet_env import PyBulletEnv
    from src.simulation.pybullet_lidar import PyBulletLidar

    env = PyBulletEnv(gui=False)
    env.connect()
    env.reset()
    _live["env"] = env
    _live["policy"] = get_policy("jev")
    _live["lidar"] = PyBulletLidar()
    _live["executor"] = PyBulletActionExecutor(env)


def _close_runtime():
    env = _live.get("env")
    if env is not None:
        try:
            env.close()
        except Exception:
            pass
    for k in ("env", "policy", "lidar", "executor", "last_decision", "last_pose"):
        _live[k] = None


def _snapshot(**fields) -> Dict[str, Any]:
    """Flat (compat) + nested §2 runtime contract. All values measured.

    pipeline stages: LIVE while this loop executes them each step
    (1-6 validated frozen code, still executed live); stage7 follows
    the Jev status; stage8 ACTIVE; stage9 RUNNING. verified=true marks
    stages covered by the validation suite (not a live claim by itself).
    """
    veh = fields.get("vehicle_state") or {}
    safety_status = fields.get("safety_status")
    jev_status = fields.get("jev_status")
    pres = fields.get("pipeline_result") or {}
    timing = pres.get("timing") or {}
    imp = pres.get("importance") or {}
    res = pres.get("resolution") or {}
    cells = pres.get("map_cells") or []
    sems = sorted({str(c.get("semantic_class", "?")) for c in cells})
    stage7 = {"OK": "LIVE", "FAILED": "ERROR", "INVALID": "ERROR",
              "UNAVAILABLE": "UNAVAILABLE", "NOT_CALLED": "IDLE"}.get(
                  str(jev_status), "IDLE" if jev_status is None else "ERROR")
    snap = {
        "frame_id": fields.get("frame_id"),
        "timestamp": fields.get("timestamp"),
        "run_id": _runtime["run_id"],
        "mode": _runtime["mode"],
        "simulator": "pybullet",
        # --- flat keys (backward compatible) ---
        "lidar_points": fields.get("lidar_points"),
        "map_cells": fields.get("map_cells"),
        "rl_state": fields.get("rl_state"),
        "jev_action": fields.get("jev_action"),
        "jev_confidence": fields.get("jev_confidence"),
        "jev_probabilities": fields.get("jev_probabilities"),
        "jev_status": jev_status,
        "safety_status": safety_status,
        "executed_action": fields.get("executed_action"),
        "source": fields.get("source"),
        "vehicle_position": fields.get("vehicle_position"),
        "vehicle_heading": fields.get("vehicle_heading"),
        "collision": fields.get("collision"),
        "jev_latency_ms": fields.get("jev_latency_ms"),
        "loop_latency_ms": fields.get("loop_latency_ms"),
        "simulation_time": fields.get("simulation_time"),
        "system_status": fields.get("system_status", "RUNNING"),
        # --- nested §2 contract ---
        "pipeline": {
            "stage1": {"state": "LIVE", "verified": True},
            "stage2": {"state": "LIVE", "verified": True},
            "stage3": {"state": "LIVE", "verified": True},
            "stage4": {"state": "LIVE", "verified": True},
            "stage5": {"state": "LIVE", "verified": True},
            "stage6": {"state": "LIVE", "verified": True},
            "stage7": {"state": stage7, "verified": True},
            "stage8": {"state": "ACTIVE", "verified": True},
            "stage9": {"state": "RUNNING", "verified": True},
        },
        "lidar": {
            "point_count": fields.get("lidar_points"),
            "frame_count": _runtime["step"],
            "fps": _runtime["loop_fps"],
            "status": "STREAMING",
        },
        "map": {
            "cell_count": fields.get("map_cells"),
            "cells": [{k: c.get(k) for k in
                       ("x", "y", "elevation", "occupancy", "resolution",
                        "importance", "semantic_class", "semantic_source",
                        "confidence", "point_count")}
                      for c in cells[:1000]],
            "importance": {"mean": imp.get("mean"), "min": imp.get("min"),
                           "max": imp.get("max")},
            "resolution": res if isinstance(res, dict) else {"summary": res},
            "semantic_classes": sems,
            "status": "LIVE",
        },
        "decision": {
            "model": "jev-1.13",
            "action": fields.get("jev_action"),
            "confidence": fields.get("jev_confidence"),
            "probabilities": fields.get("jev_probabilities"),
            "latency_ms": fields.get("jev_latency_ms"),
            "status": stage7,
        },
        "safety": {
            "status": safety_status,
            "override": safety_status != "SAFE_TO_EXECUTE",
            "reason": fields.get("safety_reason"),
        },
        "vehicle": {
            "x": veh.get("x"), "y": veh.get("y"), "z": veh.get("z"),
            "yaw_deg": veh.get("yaw_deg"),
            "speed_mps": veh.get("speed_ms"),
            "yaw_rate": veh.get("yaw_rate"),
        },
        "execution": {
            "proposed_action": fields.get("proposed_action"),
            "executed_action": fields.get("executed_action"),
            "source": fields.get("source"),
        },
        "metrics": {
            "jev_latency_ms": fields.get("jev_latency_ms"),
            "perception_latency_ms": timing.get("perception_latency_ms"),
            "safety_latency_ms": fields.get("safety_latency_ms"),
            "action_execution_latency_ms": fields.get("exec_latency_ms"),
            "loop_latency_ms": fields.get("loop_latency_ms"),
            "vehicle_speed_mps": veh.get("speed_ms"),
            "distance_m": _runtime["distance_m"],
            "collision": fields.get("collision"),
            "collisions_total": _runtime["collisions"],
            "jev_calls": _runtime["jev_calls"],
            "jev_successful": _runtime["jev_ok"],
            "jev_failed": _runtime["jev_failed"],
            "manual_calls": _runtime["manual_calls"],
            "directional_actions": _runtime["directional_actions"],
            "stop_actions": _runtime["stop_actions"],
            "safety_overrides": _runtime["safety_overrides"],
        },
        "pipeline_result": {
            "frame_id": fields.get("frame_id"),
            "scene_id": "pybullet-live",
            "timestamp": fields.get("timestamp"),
            "input_source": "pybullet_sim",
            "status": "success",
            "input_point_count": fields.get("lidar_points"),
            "processed_point_count": fields.get("lidar_points"),
            "map_cells": [{k: c.get(k) for k in
                           ("x", "y", "elevation", "occupancy", "resolution",
                            "importance", "semantic_class", "semantic_source",
                            "confidence", "point_count", "region_id")}
                          for c in cells[:1000]],
            "map_cell_count": fields.get("map_cells"),
            "importance": pres.get("importance"),
            "resolution": pres.get("resolution"),
            "semantic": pres.get("semantic"),
            "timing": timing,
        },
    }
    _live["snapshot"] = snap
    return snap


def _loop_step(manual_action: str | None = None) -> Dict[str, Any]:
    """Execute one full loop iteration; returns the step snapshot."""
    from src.decision.jev_state_adapter import enrich_for_jev, to_named_state
    from src.decision.policy_interface import decide_with_safety
    from src.safety_controller import evaluate as safety_evaluate
    from src.simulation.pybullet_lidar import to_lidar_frame
    from src.simulation.stages_runner import run_stages_1_to_6

    env = _live["env"]
    t_all = time.perf_counter()
    i = _runtime["step"]
    env.step("stop")
    scan = _live["lidar"].scan(env)
    pts = _live["lidar"].to_points(scan)
    frame = to_lidar_frame(pts, f"live-{i:05d}", env.get_simulation_time())
    out = run_stages_1_to_6(frame.points, frame.frame_id, frame.timestamp)
    named = to_named_state(out["state"]["state_vector"],
                           out["state"]["sector_ranges_m"])
    named = enrich_for_jev(named, out["state"]["safety"])
    nearest = out["state"]["safety"]["nearest_forward_obstacle_m"]

    source = "fallback"
    if manual_action is not None:
        candidate_raw = manual_action
        source = "manual"
        gate_note = "manual command (not a Jev decision)"
        jev_action = None
        jev_conf, jev_probs, jev_status = None, None, "NOT_CALLED"
        jev_ms = None
    elif _runtime["mode"] == "autonomous" and (
            i % _runtime["decision_interval_steps"] == 0
            or _live["last_decision"] is None):
        dec = decide_with_safety(_live["policy"], named, nearest, True, False,
                                 frame_id=frame.frame_id,
                                 run_id=_runtime["run_id"])
        _live["last_decision"] = dec
        _runtime["jev_calls"] += 1
        if dec.get("jev_status") == "OK":
            _runtime["jev_ok"] += 1
        elif dec.get("jev_status") in ("FAILED", "INVALID"):
            _runtime["jev_failed"] += 1
        gate_ok = "passed" in str(dec.get("gate", ""))
        candidate_raw = dec["proposed_action"] if gate_ok else "stop"
        source = dec.get("source", "fallback")
        gate_note = dec.get("gate")
        jev_action, jev_conf = dec.get("proposed_action"), dec.get("confidence")
        jev_probs, jev_status = dec.get("probabilities"), dec.get("jev_status")
        jev_ms = dec.get("policy_latency_ms")
    else:
        prev = _live["last_decision"] or {}
        gate_ok = "passed" in str(prev.get("gate", ""))
        candidate_raw = prev.get("proposed_action") if gate_ok else "stop"
        source = prev.get("source", "fallback")
        gate_note = prev.get("gate", "retained")
        jev_action, jev_conf = prev.get("proposed_action"), prev.get("confidence")
        jev_probs, jev_status = prev.get("probabilities"), prev.get("jev_status")
        jev_ms = prev.get("policy_latency_ms")

    t_safety = time.perf_counter()
    verdict = safety_evaluate(candidate_raw or "stop", nearest, map_valid=True,
                              emergency=bool(out["state"]["safety"]["emergency_stop"]))
    safety_ms = round((time.perf_counter() - t_safety) * 1000.0, 3)
    final = verdict["final_action"]
    if verdict["verdict"] != "SAFE_TO_EXECUTE":
        _runtime["safety_overrides"] += 1
    if source == "manual":
        _runtime["manual_calls"] += 1
    if final == "stop":
        _runtime["stop_actions"] += 1
    elif final in ("forward", "turn_left", "turn_right"):
        _runtime["directional_actions"] += 1
    t_exec = time.perf_counter()
    exec_info = _live["executor"].execute(final)
    exec_ms = round((time.perf_counter() - t_exec) * 1000.0, 2)
    if exec_info.get("collision"):
        _runtime["collisions"] += 1
    veh = env.get_vehicle_state()
    prev_pose = _live.get("last_pose")
    if prev_pose is not None:
        try:
            dx = float(veh["x"]) - float(prev_pose["x"])
            dy = float(veh["y"]) - float(prev_pose["y"])
            _runtime["distance_m"] = round(
                _runtime["distance_m"] + (dx ** 2 + dy ** 2) ** 0.5, 4)
        except (TypeError, ValueError):
            pass
    _live["last_pose"] = {"x": veh["x"], "y": veh["y"]}
    now = time.time()
    if _runtime["last_step_utc"] is not None:
        try:
            dt = now - _runtime["last_step_utc"]
            if dt > 0:
                fps = 1.0 / dt
                prev = _runtime["loop_fps"]
                _runtime["loop_fps"] = round(fps if prev is None else 0.9 * prev + 0.1 * fps, 2)
        except TypeError:
            pass
    _runtime["last_step_utc"] = now
    loop_ms = round((time.perf_counter() - t_all) * 1000.0, 2)
    _runtime["step"] = i + 1

    decision_store.record({
        "frame_id": frame.frame_id,
        "run_id": _runtime["run_id"],
        "decision_model": "jev" if source in ("jev", "fallback") else source,
        "source": source,
        "proposed_action": jev_action if source != "manual" else manual_action,
        "confidence": jev_conf,
        "probabilities": jev_probs,
        "jev_status": jev_status,
        "gate": gate_note,
        "safety_status": verdict["verdict"],
        "executed_action": final,
        "decision_latency_ms": jev_ms,
        "loop_latency_ms": loop_ms,
    })
    return _snapshot(
        frame_id=frame.frame_id, timestamp=frame.timestamp,
        lidar_points=int(pts.shape[0]),
        map_cells=len(out["result"].get("map_cells", [])),
        rl_state=out["state"]["state_vector"],
        jev_action=jev_action, jev_confidence=jev_conf,
        jev_probabilities=jev_probs, jev_status=jev_status,
        safety_status=verdict["verdict"], safety_reason=verdict.get("reason"),
        proposed_action=(jev_action if source != "manual" else manual_action),
        executed_action=final,
        source=source,
        vehicle_state=veh,
        vehicle_position={"x": round(veh["x"], 3), "y": round(veh["y"], 3)},
        vehicle_heading=round(veh["yaw_deg"], 2),
        collision=bool(exec_info.get("collision")),
        jev_latency_ms=jev_ms, loop_latency_ms=loop_ms,
        safety_latency_ms=safety_ms, exec_latency_ms=exec_ms,
        pipeline_result=out["result"],
        simulation_time=env.get_simulation_time(),
        system_status="RUNNING")


def _loop():
    global _stop_flag
    while not _stop_flag:
        try:
            with _lock:
                if _runtime["mode"] not in ("autonomous", "manual"):
                    break
                snap = _loop_step()
                _runtime["error"] = None
        except Exception as exc:  # noqa: BLE001 - recorded, loop keeps safe state
            with _lock:
                _runtime["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            time.sleep(0.5)


def start(mode: str = "autonomous",
          decision_interval_steps: int = 5) -> Dict[str, Any]:
    """Start the background live loop (real env or honest BLOCKED)."""
    global _thread, _stop_flag
    mode = str(mode or "autonomous").lower()
    if mode not in ("autonomous", "manual"):
        raise ValueError("mode must be 'autonomous' or 'manual'.")
    with _lock:
        if _thread is not None and _thread.is_alive():
            _runtime["mode"] = mode
            return status()
        try:
            _build_runtime()
        except Exception as exc:  # noqa: BLE001 - e.g. PyBulletUnavailable
            return {"active": False, "status": "BLOCKED",
                    "reason": f"{type(exc).__name__}: {str(exc)[:300]}",
                    "simulator": "pybullet"}
        _runtime.update({"mode": mode, "run_id": f"live-{int(time.time())}",
                         "step": 0, "started_utc": _now_utc(),
                         "decision_interval_steps": max(1, int(decision_interval_steps)),
                         "jev_calls": 0, "jev_ok": 0, "jev_failed": 0,
                         "manual_calls": 0, "safety_overrides": 0,
                         "collisions": 0, "directional_actions": 0,
                         "stop_actions": 0, "distance_m": 0.0,
                         "last_step_utc": None, "loop_fps": None,
                         "error": None})
        decision_store.clear()
        _stop_flag = False
        _thread = threading.Thread(target=_loop, daemon=True)
        _thread.start()
        return status()


def stop() -> Dict[str, Any]:
    global _thread, _stop_flag
    _stop_flag = True
    th, _thread = _thread, None
    if th is not None:
        th.join(timeout=10)
    with _lock:
        _runtime["mode"] = "idle"
        _close_runtime()
        return status()


def reset() -> Dict[str, Any]:
    mode = _runtime.get("mode") if _runtime.get("mode") in ("autonomous", "manual") else "autonomous"
    interval = _runtime.get("decision_interval_steps", 5)
    stop()
    return start(mode=mode, decision_interval_steps=interval)


def manual(action: str) -> Dict[str, Any]:
    """Execute exactly one safety-checked manual step (source=manual)."""
    from src.simulation.pybullet_action_executor import canonical

    key = canonical(str(action or "").lower())
    if key == "turn_left":
        key = "left"
    if key == "turn_right":
        key = "right"
    with _lock:
        if _live.get("env") is None:
            raise ValueError("No live simulation; POST /simulation/start first.")
        if _runtime["mode"] != "manual":
            raise ValueError("Manual actions require mode='manual'; POST /simulation/start with mode manual.")
        snap = _loop_step(manual_action=key)
        return snap


def snapshot() -> Dict[str, Any] | None:
    with _lock:
        snap = _live.get("snapshot")
        return dict(snap) if snap else None


def status() -> Dict[str, Any]:
    with _lock:
        return {
            "active": _runtime["mode"] in ("autonomous", "manual"),
            "mode": _runtime["mode"],
            "run_id": _runtime["run_id"],
            "steps": _runtime["step"],
            "started_utc": _runtime["started_utc"],
            "decision_interval_steps": _runtime["decision_interval_steps"],
            "simulator": "pybullet",
            "jev_calls": _runtime["jev_calls"],
            "jev_successful": _runtime["jev_ok"],
            "jev_failed": _runtime["jev_failed"],
            "manual_calls": _runtime["manual_calls"],
            "directional_actions": _runtime["directional_actions"],
            "stop_actions": _runtime["stop_actions"],
            "distance_m": _runtime["distance_m"],
            "loop_fps": _runtime["loop_fps"],
            "safety_overrides": _runtime["safety_overrides"],
            "collisions": _runtime["collisions"],
            "error": _runtime["error"],
        }


def metrics() -> Dict[str, Any]:
    st = status()
    st["note"] = "manual_calls excluded from Jev success/failure counts"
    return st
