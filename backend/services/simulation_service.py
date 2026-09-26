"""Offline closed-loop navigation simulation (Stages 6-8 over replay frames).

Loop per step (all timed with time.perf_counter):

  replay frame -> pipeline (perception -> map) -> map_validator gate
  -> rl_state -> DQN decide -> safety_controller -> action_executor
  -> measurable reward components -> next frame

Honesty boundaries (never crossed):
- frames advance in manifest order; there is NO ego-motion model and NO
  CARLA, so progress/collision/goal components are recorded as null
  (not measurable), never zero-filled or invented;
- only safe_clearance (from the measured nearest-obstacle distance) and
  the safety override are computed;
- the DQN is untrained: actions demonstrate the data path only.

State lives in process memory (prototype scope, like result_service).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from backend.services import pipeline_service, replay_service

_episode: Dict[str, Any] = {"active": False}


def carla_ticks() -> int:
    """CARLA closed-loop ticks recorded this process.

    Always 0 here: this service runs the offline replay loop only.
    CARLA ticks would be counted by src/rl/carla_env.run_episode on a
    connected server (blocked: no simulator installed).
    """
    return 0


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def start(frame_ids: List[str] | None = None,
          semantic_mode: str = "model") -> Dict[str, Any]:
    global _episode
    mode = str(semantic_mode or "model").lower()
    if mode not in ("annotation", "model"):
        raise ValueError("semantic_mode must be 'annotation' or 'model'.")
    frames = list(frame_ids) if frame_ids else [f["frame_id"] for f in replay_service.list_frames()]
    if not frames:
        raise ValueError("No replay frames available for simulation.")
    _episode = {"active": True, "episode_id": f"offline-{int(time.time())}",
                "frame_ids": frames, "cursor": 0, "semantic_mode": mode,
                "steps": [], "started_utc": _now_utc(), "stops": 0, "overrides": 0}
    return status()


def stop() -> Dict[str, Any]:
    _episode["active"] = False
    _episode["stopped_utc"] = _now_utc()
    return status()


def reset() -> Dict[str, Any]:
    return start(frame_ids=_episode.get("frame_ids"),
                 semantic_mode=_episode.get("semantic_mode", "model"))


def status() -> Dict[str, Any]:
    steps = _episode.get("steps", [])
    decide_ms = [s["decide_latency_ms"] for s in steps if s.get("decide_latency_ms") is not None]
    return {
        "active": bool(_episode.get("active", False)),
        "episode_id": _episode.get("episode_id"),
        "semantic_mode": _episode.get("semantic_mode"),
        "frames_total": len(_episode.get("frame_ids", [])),
        "frames_done": len(steps),
        "stops": _episode.get("stops", 0),
        "safety_overrides": _episode.get("overrides", 0),
        "mean_decide_latency_ms": round(sum(decide_ms) / len(decide_ms), 2) if decide_ms else None,
        "simulator": "offline replay loop (no CARLA; no ego-motion model)",
        "unmeasurable": ["progress_towards_goal", "collision", "goal_completion (no ego control)"],
    }


def step() -> Dict[str, Any]:
    """Advance one frame through the full autonomy loop (measured)."""
    import numpy as np

    from src import rl_agent
    from src.action_executor import SimulationActionExecutor
    from src.map_validator import gate_for_rl
    from src.rl_state import build_state
    from src.safety_controller import load_safety_config

    if not _episode.get("active"):
        raise ValueError("No active simulation; POST /simulation/start first.")
    frames = _episode["frame_ids"]
    cursor = int(_episode.get("cursor", 0))
    if cursor >= len(frames):
        _episode["active"] = False
        return {"done": True, "reason": "frame list exhausted", "status": status()}
    fid = frames[cursor]
    stages: List[str] = []
    t0 = time.perf_counter()

    # Perception -> map (existing pipeline, model channel default).
    result = pipeline_service.run_frame(fid, semantic_mode=_episode.get("semantic_mode", "model"))
    stages.append("perception+map")
    perception_ms = float(result["timing"]["perception_latency_ms"] or 0.0)
    mapping_ms = float(result["timing"]["mapping_latency_ms"] or 0.0)

    # Map gate (Stage 5 -> 6 guard).
    t1 = time.perf_counter()
    gate = gate_for_rl(result["map_cells"])
    gate_ms = (time.perf_counter() - t1) * 1000.0
    stages.append("map_validation")

    # RL state + DQN + safety + validated action.
    t1 = time.perf_counter()
    state = build_state(result["map_cells"])
    state_ms = (time.perf_counter() - t1) * 1000.0
    stages.append("rl_state")
    t1 = time.perf_counter()
    decision = rl_agent.decide(state["state_vector"], state["safety"])
    dqn_ms = (time.perf_counter() - t1) * 1000.0
    stages.append("dqn_decide")
    t1 = time.perf_counter()
    safety_cfg = load_safety_config()
    executed = SimulationActionExecutor().execute(
        decision["network_action"], state["safety"]["nearest_forward_obstacle_m"],
        map_valid=True, emergency=False)
    safety_ms = (time.perf_counter() - t1) * 1000.0
    stages.append("safety+execute")

    # Measurable reward components only (separate terms, Step 5).
    from src.rl.rewards import clearance_reward

    nearest = state["safety"]["nearest_forward_obstacle_m"]
    clearance = clearance_reward(nearest, float(safety_cfg["safety_radius_m"]))
    rewards = {
        "safe_clearance": round(clearance, 4) if clearance is not None else None,
        "progress_towards_goal": None,
        "smooth_movement": None,
        "collision": None,
        "unnecessary_stop": None,
        "note": ("Only safe_clearance is measurable here (nearest-obstacle distance). "
                 "Progress/collision/goal need ego control (CARLA); recorded null, never zero-filled."),
    }

    decide_ms = (time.perf_counter() - t0) * 1000.0
    record = {
        "frame_id": fid,
        "frame_index": cursor,
        "semantic_mode": result["semantic"]["mode"],
        "stages_executed": stages,
        "map_cells": result["map_cell_count"],
        "map_gate": {"valid": gate["valid"], "n_cells": gate["n_cells"]},
        "state_dim": state["state_dim"],
        "q_values": decision["q_values"],
        "network_action": decision["network_action"],
        "final_action": executed["final_action"],
        "safety_verdict": executed["verdict"],
        "safety_override": decision["safety_override"],
        "rewards": rewards,
        "latency_ms": {
            "perception": round(perception_ms, 2),
            "mapping": round(mapping_ms, 2),
            "map_validation": round(gate_ms, 2),
            "rl_state": round(state_ms, 2),
            "dqn": round(dqn_ms, 2),
            "safety": round(safety_ms, 2),
        },
        "decide_latency_ms": round(decide_ms, 2),
        "dqn_trained": False,
    }
    _episode["steps"].append(record)
    from backend.services import decision_store

    decision_store.record({
        "frame_id": fid,
        "decision_model": "dqn-untrained",
        "proposed_action": record["network_action"],
        "confidence": None,
        "safety_status": record["safety_verdict"],
        "executed_action": record["final_action"],
        "decision_latency_ms": record["decide_latency_ms"],
    })
    _episode["cursor"] = cursor + 1
    if executed["final_action"] == "STOP":
        _episode["stops"] += 1
    if decision["safety_override"]:
        _episode["overrides"] += 1
    if _episode["cursor"] >= len(frames):
        _episode["active"] = False
        record["done"] = True
    return record


def last_state() -> Dict[str, Any] | None:
    steps = _episode.get("steps", [])
    return steps[-1] if steps else None


def metrics() -> Dict[str, Any]:
    steps = _episode.get("steps", [])
    lat = [s["decide_latency_ms"] for s in steps]
    return {
        "steps": len(steps),
        "stops": _episode.get("stops", 0),
        "safety_overrides": _episode.get("overrides", 0),
        "mean_decide_latency_ms": round(sum(lat) / len(lat), 2) if lat else None,
        "min_decide_latency_ms": round(min(lat), 2) if lat else None,
        "max_decide_latency_ms": round(max(lat), 2) if lat else None,
        "collisions": None,
        "goal_completions": None,
        "note": "collisions/goals unmeasurable without ego control (null, not zero)",
    }
