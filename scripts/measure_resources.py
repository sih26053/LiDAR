"""Fair resource comparison: uniform 5cm vs distance-adaptive vs proposed.

Same frame, same input points, same evaluation area, same runtime
environment. Proposed uses the trained MLP path (deployed configuration);
distance/uniform share the annotation-path regions of the same frame
(their resolution rules ignore semantics, so the comparison is exact).

Measured per frame x method (time.perf_counter):
  map_cells, mapping_latency_ms, total_latency_ms, fps,
  map_memory_bytes (pandas deep measurement, documented scope),
  mean_resolution_m, input_points.

Uniform memory is ESTIMATED (exact_cells x measured per-cell bytes --
full per-cell arrays are never materialized) and labelled as such.
Rendering / file I/O excluded everywhere.

Outputs: results/resource/{resource_benchmark.csv, resource_summary.csv,
resource.json}

Run:  python scripts/measure_resources.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import load_final_config  # noqa: E402
from backend.services import replay_service  # noqa: E402
from backend.services.pipeline_service import _engines, _nusc_dataset  # noqa: E402
from src.baselines import (  # noqa: E402
    build_distance_adaptive_map,
    build_uniform_map,
    evaluation_area_from_regions,
)
from src.mapper_2_5d import build_adaptive_map  # noqa: E402
from src.robustness import run_pipeline_condition, safe_fps  # noqa: E402

OUT_DIR = PROJECT_ROOT / "results" / "resource"


def df_bytes(df: pd.DataFrame) -> int:
    """Measured DataFrame memory (pandas deep=True: includes object overhead)."""
    return int(df.memory_usage(index=True, deep=True).sum())


def main() -> None:
    final_cfg = load_final_config()
    failsafe_cfg = __import__("backend.config", fromlist=["load_failsafe_config"]).load_failsafe_config()
    importance_engine, resolution_engine = _engines()
    nusc = _nusc_dataset()
    frames = replay_service.list_frames()
    condition = {"condition": "original", "condition_type": "original",
                 "condition_parameter": None, "seed": int(final_cfg.get("random_seed", 42))}

    rows = []
    for f in frames:
        fid = f["frame_id"]
        raw, _ = replay_service.load_frame_points(fid)
        sample = nusc.get("sample", fid)
        meta = {"frame_id": fid}

        rec_model = run_pipeline_condition(
            np.asarray(raw), meta, sample, nusc, importance_engine,
            resolution_engine, condition,
            cell_size=float(final_cfg["integration_cell_size_m"]),
            failsafe_config=failsafe_cfg, semantic_override={"mode": "model"})
        assert rec_model["success"], rec_model.get("failure_reason")
        rec_ann = run_pipeline_condition(
            np.asarray(raw), meta, sample, nusc, importance_engine,
            resolution_engine, condition,
            cell_size=float(final_cfg["integration_cell_size_m"]),
            failsafe_config=failsafe_cfg)
        assert rec_ann["success"], rec_ann.get("failure_reason")

        regions_ann = rec_ann["_region_features"]
        regions_model = rec_model["_region_features"]
        area = evaluation_area_from_regions(regions_ann)

        # Proposed: authoritative model-path map (timed inside rec_model).
        pmap = rec_model["_map_df"]
        proposed = {
            "method": "proposed", "frame_id": fid,
            "map_cells": int(len(pmap)),
            "mapping_latency_ms": float(rec_model["mapping_latency_ms"]),
            "total_latency_ms": float(rec_model["total_latency_ms"]),
            "map_memory_bytes": df_bytes(pmap),
            "memory_kind": "measured (pandas deep, incl. object overhead)",
            "mean_resolution_m": float(pmap["resolution"].mean()),
            "input_points": int(rec_model["input_points"]),
        }

        # Distance-adaptive on the same regions (timed map build only).
        t0 = time.perf_counter()
        dmap = build_distance_adaptive_map(regions_ann)
        dmap_ms = (time.perf_counter() - t0) * 1000.0
        dtotal = (float(rec_ann["preprocessing_latency_ms"])
                  + float(rec_ann["perception_latency_ms"])
                  + float(rec_ann["feature_extraction_latency_ms"]) + dmap_ms)
        distance = {
            "method": "distance_adaptive", "frame_id": fid,
            "map_cells": int(len(dmap)),
            "mapping_latency_ms": float(dmap_ms),
            "total_latency_ms": float(dtotal),
            "map_memory_bytes": df_bytes(dmap),
            "memory_kind": "measured (pandas deep, incl. object overhead)",
            "mean_resolution_m": float(dmap["resolution"].mean()),
            "input_points": int(rec_ann["input_points"]),
        }

        # Uniform 5cm analytic grid over the SAME area (never materialized).
        t0 = time.perf_counter()
        umap = build_uniform_map(area, resolution_m=0.05, return_arrays=False)
        umap_ms = (time.perf_counter() - t0) * 1000.0
        per_cell = df_bytes(pmap) / max(len(pmap), 1)
        utotal = (float(rec_ann["preprocessing_latency_ms"])
                  + float(rec_ann["perception_latency_ms"])
                  + float(rec_ann["feature_extraction_latency_ms"]) + umap_ms)
        uniform = {
            "method": "uniform_5cm", "frame_id": fid,
            "map_cells": int(umap.exact_cells),
            "mapping_latency_ms": float(umap_ms),
            "total_latency_ms": float(utotal),
            "map_memory_bytes": int(umap.exact_cells * per_cell),
            "memory_kind": "ESTIMATED (exact_cells x measured per-cell bytes; arrays not materialized)",
            "mean_resolution_m": 0.05,
            "input_points": int(rec_ann["input_points"]),
        }
        for r in (proposed, distance, uniform):
            r["fps"] = float(safe_fps(r["total_latency_ms"]))
            r["coverage_area_m2"] = round(float(area["coverage_area"]), 1)
        rows.extend([proposed, distance, uniform])
        print(f"{fid[:8]}: proposed={proposed['map_cells']} cells "
              f"uniform={uniform['map_cells']} cells", flush=True)

    bench = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bench.to_csv(OUT_DIR / "resource_benchmark.csv", index=False)

    summary = []
    for method in ("proposed", "distance_adaptive", "uniform_5cm"):
        sub = bench[bench["method"] == method]
        summary.append({
            "method": method,
            "frames": int(len(sub)),
            "mean_cells": float(sub["map_cells"].mean()),
            "mean_mapping_latency_ms": float(sub["mapping_latency_ms"].mean()),
            "mean_total_latency_ms": float(sub["total_latency_ms"].mean()),
            "mean_fps": float(sub["fps"].mean()),
            "mean_memory_bytes": float(sub["map_memory_bytes"].mean()),
            "mean_resolution_m": float(sub["mean_resolution_m"].mean()),
            "mean_input_points": float(sub["input_points"].mean()),
        })
    pd.DataFrame(summary).to_csv(OUT_DIR / "resource_summary.csv", index=False)

    uni = next(s for s in summary if s["method"] == "uniform_5cm")
    pro = next(s for s in summary if s["method"] == "proposed")
    payload = {
        "provenance": ("measured by scripts/measure_resources.py on this machine; "
                       "same frames/input/area/conditions for all methods"),
        "methods": summary,
        "cell_reduction_proposed_vs_uniform_pct": round(
            100.0 * (uni["mean_cells"] - pro["mean_cells"]) / uni["mean_cells"], 2),
        "memory_reduction_proposed_vs_uniform_pct": round(
            100.0 * (uni["mean_memory_bytes"] - pro["mean_memory_bytes"]) / uni["mean_memory_bytes"], 2),
        "memory_note": ("adaptive memories measured via pandas deep; uniform estimated "
                        "from exact cell count x measured per-cell bytes (arrays not materialized)"),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (OUT_DIR / "resource.json").write_text(json.dumps(payload, indent=2))
    print(f"cell reduction: {payload['cell_reduction_proposed_vs_uniform_pct']}%, "
          f"memory reduction: {payload['memory_reduction_proposed_vs_uniform_pct']}%", flush=True)


if __name__ == "__main__":
    main()
