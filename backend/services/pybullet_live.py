"""PyBullet + local Laya live runtime.

Background-thread control loop driving the REAL chain per step:
PyBullet -> LiDAR -> LiDARFrame -> Stages 1-6 -> 13-D -> enriched
state -> Laya local engine (autonomous mode) -> confidence gate -> safety (every
step) -> executor -> vehicle.

Modes:
  idle        - no loop running.
  autonomous  - loop calls Laya every `decision_interval_steps`, safety
                re-evaluated every step, last valid action retained only
                while safety keeps passing.
  manual      - loop does NOT call Laya; each POST /simulation/action/*
                executes exactly one safety-checked manual step.

Every action record carries `source` ("laya" | "fallback" | "manual"
+ "replay-laya" / legacy "jev" for non-live rows) so manual controls
never contaminate autonomy metrics. Snapshots served by
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
    "laya_calls": 0,
    "laya_ok": 0,
    "laya_failed": 0,
    "manual_calls": 0,
    "safety_overrides": 0,
    "collisions": 0,
    "directional_actions": 0,
    "stop_actions": 0,
    "distance_m": 0.0,
    "scenario": None,
    "recorder_error": None,
    "recorder_writes": 0,
    "recorder_ms": None,
    "store_ms": None,
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


def _load_scenario_dict(scenario_id: str | None) -> Dict[str, Any] | None:
    """Scenario obstacle config by id (world-frame); None = default traffic."""
    if not scenario_id:
        return None
    import json
    from pathlib import Path as _P

    cfg = json.loads((_P(__file__).resolve().parent.parent.parent
                      / "config" / "pybullet_scenarios.json").read_text())
    for sc in cfg.get("scenarios", []):
        if sc.get("id") == scenario_id:
            return {"obstacles": sc.get("obstacles", [])}
    raise ValueError(f"unknown scenario {scenario_id!r}")


def _build_runtime(scenario: Dict[str, Any] | None = None):
    from src.decision.policy_interface import get_policy
    from src.simulation.pybullet_action_executor import PyBulletActionExecutor
    from src.simulation.pybullet_env import PyBulletEnv
    from src.simulation.pybullet_lidar import PyBulletLidar

    env = PyBulletEnv(gui=False)
    env.connect()
    env.reset(seed=42, scenario=scenario)
    _live["env"] = env
    _live["policy"] = get_policy("laya")
    _live["lidar"] = PyBulletLidar()
    _live["executor"] = PyBulletActionExecutor(env)
    try:
        from src.decision.laya_client import detect_device

        _live["device"] = detect_device()
    except Exception:
        _live["device"] = "UNKNOWN"


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
            "model": fields.get("decision_model") or "laya",
            "backend": "LOCAL",
            "endpoint": fields.get("decision_endpoint"),
            "device": fields.get("decision_device"),
            "action": fields.get("jev_action"),
            "confidence": fields.get("jev_confidence"),
            "answer_confidence": fields.get("answer_confidence"),
            "probabilities": fields.get("jev_probabilities"),
            "latency_ms": fields.get("jev_latency_ms"),
            "status": stage7,
            "mode": fields.get("decision_mode"),
            "gate_metric": fields.get("gate_metric"),
            "gate_threshold": fields.get("gate_threshold"),
            "gate": fields.get("gate_note"),
            "eligible_actions": fields.get("eligible_actions"),
            "category": fields.get("decision_category"),
            "raw_action": fields.get("raw_laya_action"),
            "raw_confidence": fields.get("raw_laya_confidence"),
            "constrained_action": fields.get("constrained_action"),
            "laya_skipped": fields.get("laya_skipped"),
            "checkpoint_revision": fields.get("checkpoint_revision"),
            "calibration_version": fields.get("calibration_version"),
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
            "laya_latency_ms": fields.get("jev_latency_ms"),
            "perception_latency_ms": timing.get("perception_latency_ms"),
            "safety_latency_ms": fields.get("safety_latency_ms"),
            "action_execution_latency_ms": fields.get("exec_latency_ms"),
            "loop_latency_ms": fields.get("loop_latency_ms"),
            "vehicle_speed_mps": veh.get("speed_ms"),
            "distance_m": _runtime["distance_m"],
            "collision": fields.get("collision"),
            "collisions_total": _runtime["collisions"],
            "laya_calls": _runtime["laya_calls"],
            "laya_successful": _runtime["laya_ok"],
            "laya_failed": _runtime["laya_failed"],
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


def _record_step(loop_index: int, loop_frame_id: str, timestamp: float | None,
                 points, pipeline_result: Dict[str, Any],
                 state_vector, named_state: Dict[str, Any],
                 nearest: float | None, dec_this_step: Dict[str, Any] | None,
                 last_decision: Dict[str, Any] | None, source: str,
                 manual_action: str | None, jev_action, jev_conf, jev_probs,
                 jev_status, jev_ms, gate_note, verdict: Dict[str, Any],
                 emergency: bool, final: str, exec_info: Dict[str, Any],
                 veh: Dict[str, Any], step_dx: float, step_dy: float,
                 step_dyaw: float, safety_ms: float, exec_ms: float,
                 loop_ms: float) -> None:
    """Single write-through: files + SQLite + in-memory result (Phase 1/2).

    Uses the loop's exact objects; never recomputed. Any failure is
    logged into _runtime["recorder_error"] and autonomy continues
    (persistence is not a hard fault; flat-file exports are preserved
    by the standalone scripts regardless).
    """
    import logging as _logging

    t0 = time.perf_counter()
    try:
        from backend.services import frame_recorder, result_service, store

        run_id = _runtime["run_id"]
        scenario = _runtime.get("scenario")
        request_id = (f"{run_id}-q{int(loop_index):05d}"
                      if dec_this_step is not None else None)
        jev_base = dict(dec_this_step or last_decision or {})
        jev_rec = {**jev_base, "source": source, "request_id": request_id}
        safety_rec = {
            "verdict": verdict.get("verdict"),
            "reason": verdict.get("reason"),
            "safety_override": verdict.get("verdict") != "SAFE_TO_EXECUTE",
            "emergency_stop": emergency,
            "nearest_forward_obstacle_m": nearest,
            "forward_clearance_m": named_state.get("forward_clearance_m"),
            "left_clearance_m": named_state.get("left_clearance_m"),
            "right_clearance_m": named_state.get("right_clearance_m"),
        }
        exec_rec = {**dict(exec_info or {}), "source": source,
                    "executed_action": final}
        rec = frame_recorder.record_frame(
            run_id=run_id, sequence=int(loop_index),
            loop_frame_id=loop_frame_id, scenario=scenario,
            timestamp=timestamp, lidar_points=points,
            pipeline_result=pipeline_result, state_vector=list(state_vector or []),
            jev=jev_rec, safety=safety_rec, execution=exec_rec, vehicle=veh)
        prefix = f"results/recordings/{run_id}"
        fid = rec["frame_id"]
        cells = rec["map_cells"]
        store.create_frame(
            frame_id=fid, run_id=run_id, timestamp=timestamp,
            sequence=int(loop_index), scenario=scenario,
            points_path=f"{prefix}/{rec['points_path']}",
            point_count=int(points.shape[0]),
            state_vector=[float(v) for v in (state_vector or [])],
            vehicle_pose={k: veh.get(k) for k in
                          ("x", "y", "z", "yaw_deg", "speed_ms", "yaw_rate")},
            pipeline_timing=dict((pipeline_result or {}).get("timing") or {}),
            metadata_path=f"{prefix}/{rec['metadata_path']}")
        store.create_map_cells(fid, cells)
        store.record_decision(
            fid, source, model=jev_rec.get("model"),
            proposed_action=(jev_rec.get("proposed_action")
                             if source != "manual" else manual_action),
            confidence=jev_rec.get("confidence"),
            probabilities=jev_rec.get("probabilities"),
            latency_ms=jev_rec.get("policy_latency_ms", jev_rec.get("latency_ms")),
            request_id=request_id, error=jev_rec.get("error"),
            answer_confidence=jev_rec.get("answer_confidence"),
            mode=jev_rec.get("mode"),
            eligible=jev_rec.get("eligible_actions"),
            raw_action=jev_rec.get("raw_laya_action"),
            constrained_action=jev_rec.get("constrained_action"),
            checkpoint_revision=jev_rec.get("checkpoint_revision"),
            gate_threshold=jev_rec.get("gate_threshold"))
        store.record_safety_event(
            fid, verdict.get("verdict"),
            override=verdict.get("verdict") != "SAFE_TO_EXECUTE",
            reason=verdict.get("reason"), emergency_flag=emergency,
            nearest_obstacle_m=nearest,
            forward_clearance_m=named_state.get("forward_clearance_m"),
            left_clearance_m=named_state.get("left_clearance_m"),
            right_clearance_m=named_state.get("right_clearance_m"))
        store.record_execution(
            fid, source, final, dx=round(step_dx, 4), dy=round(step_dy, 4),
            dyaw=round(step_dyaw, 4),
            collision=bool((exec_info or {}).get("collision")))
        result_service.store_result(fid, pipeline_result, status="success")
        _runtime["recorder_writes"] += 1
        _runtime["recorder_error"] = None
        ms = round((time.perf_counter() - t0) * 1000.0, 2)
        _runtime["recorder_ms"] = ms
        _runtime["store_ms"] = ms
    except Exception as exc:  # noqa: BLE001 - autonomy continues
        _logging.getLogger("paradox.backend.recorder").warning(
            "run=%s step=%s record failed: %s: %s", _runtime.get("run_id"),
            loop_index, type(exc).__name__, str(exc)[:200])
        _runtime["recorder_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"


def _loop_step(manual_action: str | None = None) -> Dict[str, Any]:
    """Execute one full loop iteration; returns the step snapshot."""
    from src.decision.jev_state_adapter import enrich_for_laya, to_named_state
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
    named = enrich_for_laya(named, out["state"]["safety"])
    nearest = out["state"]["safety"]["nearest_forward_obstacle_m"]
    safety_dict = out["state"]["safety"]

    source = "fallback"
    dec_this_step = None
    if manual_action is not None:
        candidate_raw = manual_action
        source = "manual"
        gate_note = "manual command (not a Laya decision)"
        jev_action = None
        jev_conf, jev_probs, jev_status = None, None, "NOT_CALLED"
        jev_ms = None
        dec_model, dec_endpoint, dec_device = None, None, None
    elif _runtime["mode"] == "autonomous" and (
            i % _runtime["decision_interval_steps"] == 0
            or _live["last_decision"] is None):
        dec = decide_with_safety(_live["policy"], named, nearest, True, False,
                                 frame_id=frame.frame_id,
                                 run_id=_runtime["run_id"],
                                 mode="constrained",
                                 safety_dict=safety_dict)
        _live["last_decision"] = dec
        dec_this_step = dec
        _runtime["laya_calls"] += 1
        if dec.get("engine_status", dec.get("jev_status")) == "OK":
            _runtime["laya_ok"] += 1
        elif dec.get("engine_status", dec.get("jev_status")) in ("FAILED", "INVALID"):
            _runtime["laya_failed"] += 1
        gate_ok = "passed" in str(dec.get("gate", ""))
        candidate_raw = dec["proposed_action"] if gate_ok else "stop"
        source = dec.get("source", "fallback")
        gate_note = dec.get("gate")
        jev_action, jev_conf = dec.get("proposed_action"), dec.get("confidence")
        jev_probs, jev_status = dec.get("probabilities"), dec.get("engine_status",
                                                                  dec.get("jev_status"))
        jev_ms = dec.get("policy_latency_ms")
        dec_model = dec.get("model")
        dec_endpoint = dec.get("endpoint")
        dec_device = dec.get("device")
    else:
        prev = _live["last_decision"] or {}
        gate_ok = "passed" in str(prev.get("gate", ""))
        candidate_raw = prev.get("proposed_action") if gate_ok else "stop"
        source = prev.get("source", "fallback")
        gate_note = prev.get("gate", "retained")
        jev_action, jev_conf = prev.get("proposed_action"), prev.get("confidence")
        jev_probs, jev_status = prev.get("probabilities"), prev.get("engine_status",
                                                                    prev.get("jev_status"))
        jev_ms = prev.get("policy_latency_ms")
        dec_model = prev.get("model")
        dec_endpoint = prev.get("endpoint")
        dec_device = prev.get("device")

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
    step_dx, step_dy, step_dyaw = 0.0, 0.0, 0.0
    if prev_pose is not None:
        try:
            dx = float(veh["x"]) - float(prev_pose["x"])
            dy = float(veh["y"]) - float(prev_pose["y"])
            step_dx, step_dy = dx, dy
            step_dyaw = float(veh.get("yaw_deg", 0.0)) - float(prev_pose.get("yaw_deg", 0.0))
            _runtime["distance_m"] = round(
                _runtime["distance_m"] + (dx ** 2 + dy ** 2) ** 0.5, 4)
        except (TypeError, ValueError):
            pass
    _live["last_pose"] = {"x": veh["x"], "y": veh["y"],
                          "yaw_deg": veh.get("yaw_deg", 0.0)}
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

    _dec_full = (dec_this_step or _live.get("last_decision") or {})
    decision_store.record({
        "frame_id": frame.frame_id,
        "run_id": _runtime["run_id"],
        "decision_model": "laya" if source in ("laya", "fallback") else source,
        "source": source,
        "mode": _dec_full.get("mode"),
        "proposed_action": jev_action if source != "manual" else manual_action,
        "confidence": jev_conf,
        "answer_confidence": _dec_full.get("answer_confidence"),
        "probabilities": jev_probs,
        "jev_status": jev_status,
        "gate": gate_note,
        "gate_metric": _dec_full.get("gate_metric"),
        "gate_threshold": _dec_full.get("gate_threshold"),
        "eligible_actions": _dec_full.get("eligible_actions"),
        "category": _dec_full.get("category"),
        "raw_laya_action": _dec_full.get("raw_laya_action"),
        "raw_laya_confidence": _dec_full.get("raw_laya_confidence"),
        "constrained_action": _dec_full.get("constrained_action"),
        "laya_skipped": _dec_full.get("laya_skipped"),
        "checkpoint_revision": _dec_full.get("checkpoint_revision"),
        "calibration_version": _dec_full.get("calibration_version"),
        "safety_status": verdict["verdict"],
        "safety_override": verdict["verdict"] != "SAFE_TO_EXECUTE",
        "safety_reason": verdict.get("reason"),
        "executed_action": final,
        "decision_latency_ms": jev_ms,
        "loop_latency_ms": loop_ms,
    })
    _record_step(
        loop_index=i, loop_frame_id=frame.frame_id,
        timestamp=frame.timestamp, points=pts, pipeline_result=out["result"],
        state_vector=out["state"]["state_vector"],
        named_state=named, nearest=nearest,
        dec_this_step=dec_this_step, last_decision=_live.get("last_decision"),
        source=source, manual_action=manual_action,
        jev_action=jev_action, jev_conf=jev_conf, jev_probs=jev_probs,
        jev_status=jev_status, jev_ms=jev_ms, gate_note=gate_note,
        verdict=verdict, emergency=bool(out["state"]["safety"]["emergency_stop"]),
        final=final, exec_info=exec_info, veh=veh,
        step_dx=step_dx, step_dy=step_dy, step_dyaw=step_dyaw,
        safety_ms=safety_ms, exec_ms=exec_ms, loop_ms=loop_ms)
    _full = dec_this_step or _live.get("last_decision") or {}
    return _snapshot(
        frame_id=frame.frame_id, timestamp=frame.timestamp,
        lidar_points=int(pts.shape[0]),
        map_cells=len(out["result"].get("map_cells", [])),
        rl_state=out["state"]["state_vector"],
        jev_action=jev_action, jev_confidence=jev_conf,
        jev_probabilities=jev_probs, jev_status=jev_status,
        answer_confidence=_full.get("answer_confidence"),
        decision_mode=_full.get("mode"),
        gate_metric=_full.get("gate_metric"),
        gate_threshold=_full.get("gate_threshold"),
        gate_note=(gate_note if source != "manual" else
                   "manual command (not a Laya decision)"),
        eligible_actions=_full.get("eligible_actions"),
        decision_category=_full.get("category"),
        raw_laya_action=_full.get("raw_laya_action"),
        raw_laya_confidence=_full.get("raw_laya_confidence"),
        constrained_action=_full.get("constrained_action"),
        laya_skipped=_full.get("laya_skipped"),
        checkpoint_revision=_full.get("checkpoint_revision"),
        calibration_version=_full.get("calibration_version"),
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
        decision_model=dec_model, decision_endpoint=dec_endpoint,
        decision_device=dec_device or _live.get("device"),
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
          decision_interval_steps: int = 5,
          scenario_id: str | None = None) -> Dict[str, Any]:
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
            scenario = _load_scenario_dict(scenario_id)
            _build_runtime(scenario)
        except Exception as exc:  # noqa: BLE001 - e.g. PyBulletUnavailable
            return {"active": False, "status": "BLOCKED",
                    "reason": f"{type(exc).__name__}: {str(exc)[:300]}",
                    "simulator": "pybullet"}
        _runtime.update({"mode": mode, "run_id": f"live-{int(time.time())}",
                         "step": 0, "started_utc": _now_utc(),
                         "decision_interval_steps": max(1, int(decision_interval_steps)),
                         "laya_calls": 0, "laya_ok": 0, "laya_failed": 0,
                         "manual_calls": 0, "safety_overrides": 0,
                         "collisions": 0, "directional_actions": 0,
                         "stop_actions": 0, "distance_m": 0.0,
                         "scenario": scenario_id,
                         "laya_status": _ensure_laya_best_effort(),
                         "recorder_error": None, "recorder_writes": 0,
                         "recorder_ms": None, "store_ms": None,
                         "last_step_utc": None, "loop_fps": None,
                         "error": None})
        decision_store.clear()
        _open_run_record(mode, scenario_id)
        _stop_flag = False
        _thread = threading.Thread(target=_loop, daemon=True)
        _thread.start()
        return status()


def _ensure_laya_best_effort() -> Dict[str, Any]:
    """Connect/reuse local Laya without blocking autonomy on failure.

    Production: set LAYA_ALLOW_SPAWN=0 so the backend only connects to
    the independent Laya service and never spawns its own child.
    Development (default): a missing server is auto-started as a
    managed child. Returns the manager status dict
    (READY/STARTING/ERROR). If Laya is down, the loop still runs:
    decisions report UNAVAILABLE and the safe fallback applies
    through safety.
    """
    import logging as _logging
    import os as _os

    try:
        from backend.services import laya_manager

        allow_spawn = _os.environ.get("LAYA_ALLOW_SPAWN", "1") not in (
            "0", "false", "no")
        st = laya_manager.ensure_started(
            wait_s=5.0, allow_spawn=allow_spawn)
        return laya_manager.status()
    except Exception as exc:  # noqa: BLE001 - degraded, not fatal
        _logging.getLogger("paradox.backend.laya").warning(
            "laya ensure failed: %s: %s", type(exc).__name__, str(exc)[:200])
        try:
            from backend.services import laya_manager as _lm

            return _lm.status()
        except Exception:
            return {"state": "ERROR",
                    "reason": f"{type(exc).__name__}: {str(exc)[:200]}"}


def _open_run_record(mode: str, scenario_id: str | None) -> None:
    """Best-effort durable run open (DB failure never blocks autonomy)."""
    import logging as _logging

    try:
        from backend.services import frame_recorder, store
        from src.decision.laya_client import resolve_model

        store.create_run(_runtime["run_id"], mode, scenario=scenario_id,
                         decision_model=resolve_model(),
                         metadata={"decision_interval_steps":
                                   _runtime.get("decision_interval_steps"),
                                   "decision_engine": "laya"})
        frame_recorder.write_run_metadata(
            _runtime["run_id"],
            {"mode": mode, "scenario": scenario_id,
             "started_utc": _runtime.get("started_utc")})
    except Exception as exc:  # noqa: BLE001
        _logging.getLogger("paradox.backend.recorder").warning(
            "run open record failed: %s: %s", type(exc).__name__, str(exc)[:200])
        _runtime["recorder_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"


def stop() -> Dict[str, Any]:
    global _thread, _stop_flag
    _stop_flag = True
    th, _thread = _thread, None
    if th is not None:
        th.join(timeout=10)
    with _lock:
        _close_run_record()
        _runtime["mode"] = "idle"
        _close_runtime()
        return status()


def _close_run_record() -> None:
    import logging as _logging

    run_id = _runtime.get("run_id")
    if not run_id:
        return
    try:
        from backend.services import store

        store.finish_run(run_id, total_steps=int(_runtime.get("step", 0)),
                         collisions=int(_runtime.get("collisions", 0)),
                         distance=float(_runtime.get("distance_m", 0.0)),
                         status="COMPLETED")
    except Exception as exc:  # noqa: BLE001
        _logging.getLogger("paradox.backend.recorder").warning(
            "run close record failed: %s: %s", type(exc).__name__, str(exc)[:200])


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
            "decision_engine": "laya",
            "laya": _runtime.get("laya_status"),
            "laya_calls": _runtime["laya_calls"],
            "laya_successful": _runtime["laya_ok"],
            "laya_failed": _runtime["laya_failed"],
            "manual_calls": _runtime["manual_calls"],
            "directional_actions": _runtime["directional_actions"],
            "stop_actions": _runtime["stop_actions"],
            "distance_m": _runtime["distance_m"],
            "loop_fps": _runtime["loop_fps"],
            "scenario": _runtime.get("scenario"),
            "recorder_error": _runtime.get("recorder_error"),
            "recorder_writes": _runtime.get("recorder_writes", 0),
            "recorder_ms": _runtime.get("recorder_ms"),
            "store_ms": _runtime.get("store_ms"),
            "safety_overrides": _runtime["safety_overrides"],
            "collisions": _runtime["collisions"],
            "error": _runtime["error"],
        }


def metrics() -> Dict[str, Any]:
    st = status()
    st["note"] = "manual_calls excluded from autonomy success/failure counts"
    return st
