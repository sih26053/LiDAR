"""Methodology-flow endpoints (mirror the 9-box implementation flow).

GET /flow/status -- per-stage status (LIVE / IMPLEMENTED / BLOCKED) with
  evidence pointers. Blocked stages name the missing dependency instead
  of faking capability.
GET /rl/decide/{frame_id} -- RL state + untrained-DQN action + safety
  override computed from the STORED result of a frame the user already
  ran (404 when the frame has no stored result in this session).
"""

from __future__ import annotations

import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

#: Project taxonomy -> flow-diagram display taxonomy
#: (Road / Vehicle / Pedestrian / Static Obstacle / Others).
IMAGE_TAXONOMY = {
    "road_driveable": "Road",
    "vehicle": "Vehicle",
    "pedestrian_vru": "Pedestrian",
    "static_manmade": "Static Obstacle",
    "vegetation": "Static Obstacle",
    "unknown": "Others",
}


@router.get("/flow/status")
def flow_status():
    """Nine stage statuses COMPUTED from live runtime state (never edited text).

    - Stage 7 is LIVE only with loaded trained weights (agent.trained).
    - Stage 8 is LIVE only with a connected CARLA vehicle that received
      controls and returned frames (tracked tick counter; zero here).
    - Stage 9 is LIVE-CARLA only with recorded CARLA ticks, LIVE-REAL WORLD
      only with attached hardware (neither exists here).
    """
    from src import rl_agent
    from src.semantic_model import model_available
    from src.simulation.carla_lidar import carla_available

    n_params = int(rl_agent.get_agent().q.n_parameters)
    dqn_trained = bool(getattr(rl_agent.get_agent(), "trained", False))
    carla_ticks = int(_carla_closed_loop_ticks())

    has_model = False
    try:
        has_model = bool(model_available())
    except Exception:
        has_model = False
    stages = [
        {"box": 1, "name": "Data Acquisition", "status": "LIVE",
         "detail": "Replayed nuScenes LiDAR (X, Y, Z, Intensity); no CARLA/simulator or physical sensor attached",
         "evidence": "GET /frames; data/raw/nuscenes/samples/LIDAR_TOP"},
        {"box": 2, "name": "Perception (semantic segmentation)", "status": "LIVE" if has_model else "BLOCKED",
         "detail": ("Trained MLP point segmenter (point-wise shared-MLP architecture -- the per-point "
                    "branch of the PointNet family; full PointNet++/sparse-CNN needs a torch/GPU runtime "
                    "absent on this CPU-only offline host). Classes: Road / Vehicle / Pedestrian / "
                    "Static Obstacle / Others (project taxonomy display mapping). Genuine model outputs "
                    "with softmax confidence when the model channel is selected."
                    if has_model else "Weights missing; run scripts/train_segmentation_mlp.py"),
         "evidence": "GET /model/info; src/semantic_model.py"},
        {"box": 3, "name": "Scene Analysis", "status": "LIVE",
         "detail": ("Six factors per region: distance from LiDAR, point density, object density "
                    "(dominant-vote proxy), object importance, elevation variation, prediction "
                    "uncertainty -- each with measured/heuristic provenance"),
         "evidence": "src/scene_analysis.py"},
        {"box": 4, "name": "Adaptive Resolution Engine", "status": "LIVE",
         "detail": ("Scene-aware allocation from the frozen validated config: I>=0.70 -> 0.05 m "
                    "(High, image 5 cm); I>=0.45 -> 0.10 m and I>=0.20 -> 0.20 m (Medium band, "
                    "covers image 20 cm); else 0.50 m (Low; image shows 80 cm -- ours is finer "
                    "per the validated config). Variable-resolution cells form a two-level "
                    "spatial hierarchy over the 2 m integration grid (documented quadtree-equivalent)."),
         "evidence": "results/final/config/final_config.json; GET /map/quadtree/{frame_id}"},
        {"box": 5, "name": "2.5D Semantic Map", "status": "LIVE",
         "detail": ("Each cell stores Elevation (Z), Semantic Class, Occupancy, Confidence, "
                    "Resolution -- exactly the five diagram fields (confidence is measured "
                    "softmax output in the model channel)"),
         "evidence": "PipelineResult.map_cells"},
        {"box": 6, "name": "RL State Generation", "status": "LIVE",
         "detail": ("Local map -> 13-dim state vector (8 sector ranges + obstacle/moving/static/terrain "
                    "shares + mean importance + mean uncertainty)"),
         "evidence": "GET /rl/decide/{frame_id}; src/rl_state.py"},
        {"box": 7, "name": "RL Decision Making (DQN)",
         "status": ("LIVE" if dqn_trained else "IMPLEMENTED (untrained)"),
         "detail": ((f"Trained DQN loaded (models/rl/dqn_weights.pkl); deterministic inference "
                     f"(epsilon=0) on live 13-dim states.")
                    if dqn_trained else
                    (f"DQN 13->64(ReLU)->4 implemented in NumPy ({n_params} params); actions forward / "
                     "turn_left / turn_right / stop. Random init, zero episodes: no simulator "
                     "exists here, so Q-values carry no driving competence (labelled on every output).")),
         "evidence": ("models/rl/dqn_weights.pkl (loaded)" if dqn_trained
                      else f"src/rl_agent.py ({n_params} params); no checkpoint in models/rl/")},
        {"box": 8, "name": "Action Execution",
         "status": ("LIVE" if (carla_available() and carla_ticks > 0) else "PARTIAL (safety rule live; no vehicle)"),
         "detail": (("CARLA vehicle control live: actions converted via config/action_config.json "
                     "and applied; safety override armed.")
                    if (carla_available() and carla_ticks > 0) else
                    ("Genuine safety check runs on every decide call (emergency stop when an occupied "
                     "cell is within 5 m and +/-30 deg of heading); motor controller / toy car absent, "
                     "so no physical action is ever issued")),
         "evidence": (f"closed-loop CARLA ticks this process: {carla_ticks}"
                      if (carla_available() and carla_ticks > 0) else
                      "safety block of GET /rl/decide/{frame_id}; no vehicle attached")},
        {"box": 9, "name": "Real-World Testing",
         "status": ("LIVE — CARLA SIMULATION" if carla_ticks > 0 else "BLOCKED (offline replay evaluation instead)"),
         "detail": ((f"CARLA closed-loop evaluation live ({carla_ticks} ticks this process).")
                    if carla_ticks > 0 else
                    ("No vehicle/hardware: obstacle detection/avoidance are evaluated on replay "
                     "frames (model-vs-annotation agreement, distance-binned segmentation metrics), "
                     "not on a physical course")),
         "evidence": ("results/rl/live_carla_evaluation.csv" if carla_ticks > 0 else
                      "GET /segmentation/metrics; model_eval per frame")},
    ]
    outputs = ["High-detail mapping where needed", "Reduced memory & computation",
               "Real-time decision making (measurement-gated)", "Safe autonomous navigation (rule layer)",
               "Scalable & efficient perception system"]
    return {"stages": stages, "system_outputs": outputs,
            "perception_taxonomy": dict(IMAGE_TAXONOMY)}


def _carla_closed_loop_ticks() -> int:
    """Recorded CARLA closed-loop ticks this process (0: no simulator)."""
    try:
        from backend.services import simulation_service

        return int(simulation_service.carla_ticks())
    except Exception:
        return 0


@router.get("/rl/decide/{frame_id}")
def rl_decide(frame_id: str):
    """State + DQN + safety from a stored frame result (run it first)."""
    import numpy as np

    from backend.services import result_service
    from src import rl_agent
    from src.rl_state import build_state

    stored = result_service.get_result(str(frame_id))
    if not stored:
        return JSONResponse(
            status_code=404,
            content={"message": f"No stored result for {frame_id} in this session; run the frame first."})
    t0 = time.perf_counter()
    try:
        state = build_state(stored.get("map_cells", []))
    except (ValueError, KeyError, TypeError) as exc:
        return JSONResponse(status_code=400, content={"message": f"Cannot build RL state: {exc}"})
    decision = rl_agent.decide(state["state_vector"], state["safety"])
    decide_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "frame_id": str(frame_id),
        "semantic_mode": stored.get("semantic", {}).get("mode"),
        "state": state,
        "decision": decision,
        "decide_latency_ms": round(decide_ms, 2),
    }


@router.get("/map/quadtree/{frame_id}")
def map_quadtree(frame_id: str):
    """Two-level spatial hierarchy derived from a stored adaptive map.

    Level 0 = 2 m integration cells (parents); Level 1 = adaptive map
    cells at 0.05/0.10/0.20/0.50 m (leaves). Genuinely derived from the
    stored cells: parents are the floor(x/2)x floor(y/2) bins, leaves are
    grouped by parent. Documented quadtree-equivalent (regular two-level
    subdivision, not a pointer-based tree).
    """
    from collections import defaultdict

    from backend.services import result_service

    stored = result_service.get_result(str(frame_id))
    if not stored:
        return JSONResponse(
            status_code=404,
            content={"message": f"No stored result for {frame_id} in this session; run the frame first."})
    cells = stored.get("map_cells", [])
    parents: Dict[str, list] = defaultdict(list)
    for c in cells:
        try:
            key = f"{int(float(c['x']) // 2)}:{int(float(c['y']) // 2)}"
        except (KeyError, TypeError, ValueError):
            continue
        parents[key].append(float(c.get("resolution", 0.5)))
    import numpy as np

    leaf_res = np.array([r for v in parents.values() for r in v]) if parents else np.array([0.5])
    return {
        "frame_id": str(frame_id),
        "structure": "two-level hierarchy: 2 m parent cells -> variable-resolution leaf cells",
        "n_parents": len(parents),
        "n_leaves": int(len(leaf_res)),
        "mean_children_per_parent": round(float(len(leaf_res) / max(len(parents), 1)), 2),
        "leaf_resolution_mix": {str(r): int((leaf_res == r).sum()) for r in (0.05, 0.10, 0.20, 0.50)},
        "note": ("Regular two-level subdivision derived from stored cells "
                 "(quadtree-equivalent); the mapper itself assigns per-region "
                 "resolution from the frozen config."),
    }
