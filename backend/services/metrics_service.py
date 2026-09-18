"""Metrics service (Step 0.3): derive DemoMetrics from stored results only."""

from __future__ import annotations

from typing import Any, Dict


def metrics_from_result(result: Dict[str, Any]) -> Dict[str, Any]:
    res = result.get("resolution", {}) or {}
    timing = result.get("timing", {}) or {}
    return {
        "frame_id": result.get("frame_id"),
        "input_point_count": result.get("input_point_count"),
        "processed_point_count": result.get("processed_point_count"),
        "map_cell_count": result.get("map_cell_count"),
        "fine_cells": res.get("fine_cells"),
        "medium_cells": res.get("medium_cells"),
        "coarse_cells": res.get("coarse_cells"),
        "average_resolution": res.get("average_resolution"),
        "latency_ms": timing.get("total_latency_ms"),
        "fps": timing.get("fps"),
    }
