"""Clean simulation handoff demo (15 September finalization).

Runs ONE LiDAR frame through the FROZEN pipeline without any development
notebooks:

    LiDAR input -> preprocessing -> perception (annotation semantics) ->
    feature extraction -> Importance Engine -> Resolution Engine ->
    Adaptive 2.5D Mapper -> 2.5D map output (CSV)

Configuration is loaded from config/final_config.json (single source of
truth); engine defaults are asserted equal to it before running.

Requires the nuScenes Mini dataset (for annotation reference) at
--nusc-root (default: ../data/raw/nuscenes relative to this folder).

Usage:
    python run_demo.py [--nusc-root PATH] [--sample-token TOKEN] [--seed 42]

Prints a single JSON summary line to stdout and saves the adaptive map CSV
to sample_outputs/. Exit code 0 on success, 2 when blocked (missing
dataset/sample), 1 on pipeline failure.
"""
import argparse
import json
import sys
import time
from pathlib import Path

HANDOFF_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(HANDOFF_ROOT))

import numpy as np


def fail(message, code):
    print(json.dumps({"status": "FAIL", "error": message}))
    sys.exit(code)


def main():
    ap = argparse.ArgumentParser(description="Frozen-pipeline single-frame demo")
    ap.add_argument("--nusc-root", default=str(HANDOFF_ROOT / ".." / "data" / "raw" / "nuscenes"))
    ap.add_argument("--sample-token", default="5991fad3280c4f84b331536c32001a04")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(HANDOFF_ROOT / "sample_outputs"))
    args = ap.parse_args()

    # 1. Load frozen configuration (single source of truth).
    cfg_path = HANDOFF_ROOT / "config" / "final_config.json"
    if not cfg_path.is_file():
        fail(f"configuration missing: {cfg_path}", 2)
    final_cfg = json.load(open(cfg_path))
    print("config_load: OK (" + str(cfg_path.name) + ")", flush=True)

    # 2. Load frozen modules and bind them to the frozen configuration.
    try:
        from src.importance_engine import ImportanceEngine
        from src.resolution_engine import ResolutionEngine
        from src.robustness import run_pipeline_condition
        from nuscenes.nuscenes import NuScenes
    except Exception as e:  # noqa: BLE001
        fail(f"handoff_import failed: {type(e).__name__}: {e}", 1)
    print("handoff_import: OK", flush=True)

    importance_engine = ImportanceEngine()
    assert importance_engine.weights == final_cfg["importance_weights"], \
        "engine default weights differ from frozen final_config.json"
    assert abs(importance_engine.max_distance - final_cfg["max_distance_m"]) < 1e-9
    assert abs(importance_engine.lambda_uncertainty - final_cfg["uncertainty_lambda"]) < 1e-9
    levels = [(float(t), float(r)) for t, r in final_cfg["resolution_levels"]]
    resolution_engine = ResolutionEngine(levels=levels)
    failsafe_cfg = json.load(open(HANDOFF_ROOT / "config" / "failsafe_config.json"))

    # 3. Model artifact (CASE B: none — config-only prototype).
    models_dir = HANDOFF_ROOT / "models"
    weights = list(models_dir.glob("*.pth")) + list(models_dir.glob("*.pt")) \
        + list(models_dir.glob("*.onnx")) + list(models_dir.glob("*.pkl"))
    if weights:
        fail(f"unexpected weight files in handoff models/: {weights}", 1)
    print("artifact_load: OK (CASE B: no trained weights; config-only)", flush=True)

    # 4. Load sample LiDAR frame (bundled processed .npy).
    sample_path = HANDOFF_ROOT / "sample_data" / f"{args.sample_token}_LIDAR_TOP_xyzi.npy"
    meta_path = HANDOFF_ROOT / "sample_data" / f"{args.sample_token}_metadata.csv"
    if not sample_path.is_file():
        fail(f"sample not bundled: {sample_path}", 2)
    try:
        import pandas as pd
        raw = np.load(sample_path)
        meta = pd.read_csv(meta_path).iloc[0].to_dict()
    except Exception as e:  # noqa: BLE001
        fail(f"sample_load failed: {type(e).__name__}: {e}", 2)
    print(f"sample_load: OK ({args.sample_token}, n={len(raw)})", flush=True)

    # 5. Dataset (annotation reference) + pipeline.
    try:
        nusc = NuScenes(version="v1.0-mini", dataroot=args.nusc_root, verbose=False)
        sample = nusc.get("sample", args.sample_token)
    except Exception as e:  # noqa: BLE001
        fail(f"dataset unavailable at {args.nusc_root}: {type(e).__name__}: {e} "
             f"(nuScenes Mini v1.0-mini is required for the annotation path)", 2)
    condition = {"condition": "original", "condition_type": "original",
                 "condition_parameter": None, "seed": int(args.seed)}
    t0 = time.perf_counter()
    rec = run_pipeline_condition(raw, meta, sample, nusc, importance_engine,
                                 resolution_engine, condition,
                                 cell_size=float(final_cfg["integration_cell_size_m"]),
                                 failsafe_config=failsafe_cfg)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    if not rec["success"]:
        fail(f"pipeline_execution failed: {rec.get('failure_reason')}", 1)
    print("pipeline_execution: OK", flush=True)

    # 6. Save output.
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"adaptive_map_{args.sample_token}.csv"
    rec["_map_df"].to_csv(out_path, index=False)
    print(f"output_generation: OK ({out_path.name}, cells={len(rec['_map_df'])})",
          flush=True)

    res = rec["_map_df"]["resolution"].to_numpy(float)
    summary = {
        "status": "PASS",
        "frame_id": args.sample_token,
        "map_cells": int(len(rec["_map_df"])),
        "mean_importance": round(float(rec["_map_df"]["importance"].mean()), 6),
        "mean_resolution": round(float(res.mean()), 6),
        "resolution_distribution": {str(k): int((res == k).sum())
                                    for k in (0.05, 0.10, 0.20, 0.50)},
        "latency_ms": round(float(rec["total_latency_ms"]), 3),
        "wall_clock_ms": round(wall_ms, 3),
        "fps": round(float(rec["fps"]), 4),
        "semantic_source": "annotation+fallback (no model predictions)",
        "config": "config/final_config.json (W_BASE+THRESH_C)",
        "output": str(out_path),
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
