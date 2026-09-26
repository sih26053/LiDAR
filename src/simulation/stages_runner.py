"""Phase 8 — Stages 1-6 runner on live/sim N x 4 (adapter, no duplication).

Reuses `pipeline_service._execute(..., live_input=True)` (the same code
path as POST /stream/push) plus `rl_state.build_state`. No perception,
mapping, or state logic is reimplemented here.
"""

from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np


def run_stages_1_to_6(points: np.ndarray, frame_id: str,
                      timestamp: float | None = None,
                      semantic_mode: str = "model",
                      max_map_cells: int | None = 2000) -> Dict[str, Any]:
    """N x 4 sim/live cloud -> perception/map/state record (measured)."""
    from backend.services import pipeline_service
    from src.rl_state import build_state

    pts = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
    if pts.ndim != 2 or pts.shape[1] != 4 or pts.shape[0] == 0:
        raise ValueError(f"points must be non-empty N x 4, got {pts.shape!r}")
    t_wall0 = time.perf_counter()
    meta = {"frame_id": str(frame_id),
            "timestamp": float(timestamp if timestamp is not None else time.time()),
            "source_path": "", "replay_reference": "pybullet-sim"}
    result = pipeline_service._execute(pts, None, meta, str(frame_id),
                                       semantic_mode, max_map_cells,
                                       t_wall0, live_input=True)
    state = build_state(result["map_cells"])
    return {"frame_id": str(frame_id), "result": result, "state": state,
            "stages": ["perception", "scene", "resolution", "map", "state"]}
