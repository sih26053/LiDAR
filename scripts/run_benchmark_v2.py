"""Verified benchmark re-run (v2): same frames, same config, recorded hardware.

Methodology (also saved to results/benchmark_v2/benchmark_config_v2.json):
- Dataset: nuScenes Mini; the SAME 10 samples as results/benchmark/
  (frame IDs read from benchmark_results.csv), same LiDAR_TOP input.
- Preprocessing: identical shared prefix per frame (preprocess_points +
  annotation perception + region aggregation, timed once per frame).
- Methods (differ ONLY in the map-resolution stage):
    proposed            importance -> resolution -> adaptive map (CURRENT
                        frozen config results/final/config/final_config.json)
    distance_adaptive   distance -> resolution (same regions)
    uniform_5cm         analytic dense grid over the shared evaluation area
                        (scope: grid sizing only, arrays not materialized)
    uniform_5cm_region  fixed 0.05 m, one row per region (SAME build scope
                        as proposed/distance -> directly comparable latency)
- Timer: time.perf_counter. 1 warm-up run (discarded) + 3 measured runs per
  method per frame; mean/median/std reported. Rendering, file I/O,
  serialization and network excluded (compute stages only).
- Hardware/software detected live at run time and saved alongside.
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import load_final_config  # noqa: E402
from backend.routes.health import environment  # noqa: E402
from backend.services.pipeline_service import _engines, _nusc_dataset  # noqa: E402
from src.baselines import (  # noqa: E402
    build_distance_adaptive_map,
    build_uniform_map,
    build_uniform_region_map,
    evaluation_area_from_regions,
)
from src.lidar_loader import load_sample_lidar  # noqa: E402
from src.mapper_2_5d import build_adaptive_map  # noqa: E402
from src.robustness import run_pipeline_condition  # noqa: E402

N_REPS = 3
OUT = PROJECT_ROOT / "results" / "benchmark_v2"


def load_sample_ids() -> list[str]:
    with open(PROJECT_ROOT / "results" / "benchmark" / "benchmark_results.csv") as fh:
        return sorted({r["frame_id"] for r in csv.DictReader(fh)})


def time_call(fn, reps: int = N_REPS) -> tuple[list[float], object]:
    fn()  # warm-up, discarded
    samples, out = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples, out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_final_config()
    env = environment()
    nusc = _nusc_dataset()
    importance_engine, resolution_engine = _engines()
    sample_ids = load_sample_ids()
    print(f"frames: {len(sample_ids)}; config: {cfg.get('selection')}")

    config_doc = {
        "task": "benchmark v2: verified re-run under the frozen runtime config",
        "frames": sample_ids,
        "n_frames": len(sample_ids),
        "input": "LiDAR_TOP .bin via src/lidar_loader.load_sample_lidar",
        "methods": ["proposed", "distance_adaptive", "uniform_5cm", "uniform_5cm_region"],
        "timer": "time.perf_counter",
        "repetitions_measured": N_REPS,
        "warmup_runs_discarded": 1,
        "excluded": ["rendering", "file I/O", "serialization", "network"],
        "shared_prefix": "preprocessing + annotation perception + region aggregation, identical inputs per frame",
        "end_to_end_definition": "shared prefix + mapping stage (baselines reuse the proposed run's shared prefix of the same frame)",
        "importance_weights": cfg["importance_weights"],
        "max_distance_m": cfg["max_distance_m"],
        "uncertainty_lambda": cfg["uncertainty_lambda"],
        "resolution_levels": cfg["resolution_levels"],
        "integration_cell_size_m": cfg["integration_cell_size_m"],
        "environment": env,
    }
    (OUT / "benchmark_config_v2.json").write_text(json.dumps(config_doc, indent=2))
    (OUT / "environment_v2.json").write_text(json.dumps(env, indent=2))

    per_frame_rows: list[dict] = []
    summary: dict[str, dict] = {}
    for fid in sample_ids:
        raw, info = load_sample_lidar(nusc, fid)
        sample = nusc.get("sample", fid)
        meta = {"frame_id": fid}
        condition = {"condition": "original", "condition_type": "original",
                     "condition_parameter": None,
                     "seed": int(cfg.get("random_seed", 42))}
        # Shared prefix (timed inside run_pipeline_condition too).
        rec = run_pipeline_condition(
            raw, meta, sample, nusc, importance_engine, resolution_engine,
            condition, cell_size=float(cfg["integration_cell_size_m"]),
            failsafe_config=None)
        assert rec.get("success"), f"proposed failed on {fid}: {rec.get('failure_reason')}"
        regions = rec["_region_features"]
        # region_details are not needed for timing/cells/resolution
        # (semantic_source audit only); pass None like a direct caller.
        shared_ms = (rec["preprocessing_latency_ms"] + rec["perception_latency_ms"]
                     + rec["feature_extraction_latency_ms"])
        n_input = int(rec["input_points"])

        prop_times, _ = time_call(lambda: build_adaptive_map(
            regions, importance_engine, resolution_engine))
        dist_times, dist_df = time_call(lambda: build_distance_adaptive_map(regions))
        area = evaluation_area_from_regions(regions)
        uni_times, uni_res = time_call(lambda: build_uniform_map(area, resolution_m=0.05))
        unireg_times, unireg_df = time_call(lambda: build_uniform_region_map(regions))

        prop_res = rec["_map_df"]["resolution"].to_numpy(dtype=float)
        dist_res = dist_df["resolution"].to_numpy(dtype=float)
        methods = {
            "proposed": (prop_times, len(rec["_map_df"]), float(prop_res.mean()),
                         shared_ms + float(statistics.mean(prop_times))),
            "distance_adaptive": (dist_times, len(dist_df), float(dist_res.mean()),
                                  shared_ms + float(statistics.mean(dist_times))),
            "uniform_5cm": (uni_times, int(uni_res.exact_cells), 0.05,
                            shared_ms + float(statistics.mean(uni_times))),
            "uniform_5cm_region": (unireg_times, len(unireg_df), 0.05,
                                   shared_ms + float(statistics.mean(unireg_times))),
        }
        for method, (ts, cells, mean_res, e2e) in methods.items():
            per_frame_rows.append({
                "frame_id": fid,
                "method": method,
                "map_cells": cells,
                "mapping_latency_mean_ms": float(statistics.mean(ts)),
                "mapping_latency_median_ms": float(statistics.median(ts)),
                "mapping_latency_std_ms": float(statistics.pstdev(ts)) if len(ts) > 1 else 0.0,
                "end_to_end_latency_ms": e2e,
                "mean_resolution_m": mean_res,
                "input_points": n_input,
                "status": "pass",
            })
            s = summary.setdefault(method, {"cells": [], "map_mean": [], "e2e": [], "res": []})
            s["cells"].append(cells)
            s["map_mean"].append(float(statistics.mean(ts)))
            s["e2e"].append(e2e)
            s["res"].append(mean_res)
        print(f"{fid[:8]} cells={len(rec['_map_df'])} prop={statistics.mean(prop_times):.1f}ms "
              f"dist={statistics.mean(dist_times):.1f}ms uni={statistics.mean(uni_times):.3f}ms "
              f"unireg={statistics.mean(unireg_times):.1f}ms")

    with open(OUT / "benchmark_results_v2.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_frame_rows[0].keys()))
        w.writeheader()
        w.writerows(per_frame_rows)
    with open(OUT / "benchmark_summary_v2.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "method", "frames", "successful", "mean_mapping_latency_ms",
            "median_mapping_latency_ms", "std_mapping_latency_ms",
            "mean_end_to_end_latency_ms", "mean_cells", "mean_resolution_m"])
        w.writeheader()
        for method, s in summary.items():
            w.writerow({
                "method": method, "frames": len(sample_ids),
                "successful": len(sample_ids),
                "mean_mapping_latency_ms": float(statistics.mean(s["map_mean"])),
                "median_mapping_latency_ms": float(statistics.median(s["map_mean"])),
                "std_mapping_latency_ms": float(statistics.pstdev(s["map_mean"])),
                "mean_end_to_end_latency_ms": float(statistics.mean(s["e2e"])),
                "mean_cells": float(statistics.mean(s["cells"])),
                "mean_resolution_m": float(statistics.mean(s["res"])),
            })
    print("saved to", OUT)


if __name__ == "__main__":
    main()
