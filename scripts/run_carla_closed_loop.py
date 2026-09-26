"""MODE B entry point — CARLA closed loop (friend's CARLA machine ONLY).

CARLA world.tick() -> LiDAR -> Stages 1-6 (frozen) -> RL state -> DQN
-> Safety Controller -> VehicleControl -> CARLA -> next tick.

Fails clearly when CARLA is unavailable (exit 2). On a no-CARLA machine
use scripts/run_replay.py instead. Run from the project root:

    python scripts/run_carla_closed_loop.py [--episodes 1] [--steps 200]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.carla_lidar import (
    CarlaUnavailable, attach_lidar_sensor, capture_frame, carla_available,
    connect_to_carla, convert_to_project_format, spawn_or_attach_vehicle,
)
from src.simulation import carla_manager as M
from src.simulation.carla_action_executor import apply_control
from src.preprocessing import preprocess_points
from src.semantic_model import predict_points
from src.feature_adapter import build_perception_from_labeled_points, adapt_perception_to_regions
from src.mapper_2_5d import build_adaptive_map
from src.importance_engine import ImportanceEngine
from src.resolution_engine import ResolutionEngine
from src.mapper_2_5d import build_adaptive_map
from src import rl_state, rl_agent
from src.rl.rewards import total as reward_total, progress_reward, clearance_reward, smoothness_reward, collision_penalty, unnecessary_stop_penalty
from src.safety_controller import evaluate as safety_evaluate
from backend.config import load_final_config, load_failsafe_config


def run_episode(world, vehicle, lidar, agent, cfg, failsafe, max_steps: int, out: dict) -> dict:
    prev_action, prev_goal = None, None
    trace = {"steps": 0, "reward_sum": 0.0, "collisions": 0, "safety_overrides": 0, "actions": []}
    for _ in range(max_steps):
        pts = convert_to_project_format(capture_frame(lidar))
        clean, _ = preprocess_points(pts)
        m_labels, _, m_conf, _ = predict_points(clean)
        percep = build_perception_from_labeled_points("carla_tick", clean, m_labels, m_conf)
        regions = adapt_perception_to_regions(percep)
        ie = ImportanceEngine(weights=dict(cfg["importance_weights"]), max_distance=float(cfg["max_distance_m"]),
                              lambda_uncertainty=float(cfg["uncertainty_lambda"]))
        re_ = ResolutionEngine(levels=[(float(t), float(r)) for t, r in cfg["resolution_levels"]])
        _, amap = build_adaptive_map(regions, ie, re_)
        st = rl_state.build_state(amap.to_dict("records"))
        dec = rl_agent.decide(st["state_vector"], st["safety"])
        verdict = safety_evaluate(dec["network_action"], st["safety"]["nearest_forward_obstacle_m"], map_valid=True)
        final = verdict["final_action"]
        if verdict["verdict"] == "OVERRIDE_TO_STOP":
            trace["safety_overrides"] += 1
        apply_control(vehicle, final)
        M.tick(world)
        trace["steps"] += 1
        trace["actions"].append(final)
        prev_action = final
    return trace


def main() -> int:
    ap = argparse.ArgumentParser(description="Paradox Protocol MODE B: CARLA closed loop")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--map", default="Town03")
    args = ap.parse_args()
    if not carla_available():
        print("BLOCKED: CARLA Python API not installed. See CARLA_SETUP.md.")
        return 2
    try:
        client = None
        import carla  # noqa: F401  (presence already probed)
        from src.simulation.carla_lidar import connect_to_carla as _c
        world = _c()
    except Exception as exc:
        print(f"BLOCKED: no CARLA server reachable: {exc}")
        return 2
    cfg = load_final_config()
    failsafe = load_failsafe_config()
    agent = rl_agent.get_agent()
    M.configure_world(world)
    vehicle = M.spawn_vehicle(world)
    lidar = M.spawn_lidar(world, vehicle)
    try:
        episodes = []
        for ep in range(max(1, args.episodes)):
            ep_trace = run_episode(world, vehicle, lidar, agent, cfg, failsafe, args.steps, {})
            episodes.append({"episode": ep, **ep_trace})
            print(f"episode {ep}: steps={ep_trace['steps']} overrides={ep_trace['safety_overrides']}")
            world.reload_world()
        p = PROJECT_ROOT / "results" / "carla" / "closed_loop_trace.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(episodes, indent=2))
        print(f"MODE B COMPLETE. Trace: {p}")
        return 0
    finally:
        M.destroy([lidar, vehicle])


if __name__ == "__main__":
    raise SystemExit(main())
