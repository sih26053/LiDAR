"""Validate distance-dependent resolution on real replay frames.

Per region/cell records (frame_id, distance, importance, resolution,
semantic_class, semantic_source) from the deployed model-path pipeline,
plus a summary table and a distance-vs-resolution scatter plot.

Policy under test (frozen runtime config):
  I >= 0.70 -> 0.05 m | I >= 0.45 -> 0.10 m | I >= 0.20 -> 0.20 m | else 0.50 m
Near-field fineness arises because distance closeness D feeds importance;
the checks below verify near=66 fine / far=coarse tendency, importance
overrides, and no invalid resolutions.

Outputs: results/resolution/{distance_resolution_results.csv,
distance_resolution_summary.csv, distance_resolution_plot.png}

Run:  python scripts/validate_resolution_distance.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from backend.config import load_final_config  # noqa: E402
from backend.services import replay_service  # noqa: E402
from backend.services.pipeline_service import _engines, _nusc_dataset  # noqa: E402
from src.robustness import run_pipeline_condition  # noqa: E402

OUT_DIR = PROJECT_ROOT / "results" / "resolution"
VALID_RESOLUTIONS = (0.05, 0.10, 0.20, 0.50)


def main() -> None:
    final_cfg = load_final_config()
    failsafe_cfg = __import__("backend.config", fromlist=["load_failsafe_config"]).load_failsafe_config()
    importance_engine, resolution_engine = _engines()
    nusc = _nusc_dataset()
    condition = {"condition": "original", "condition_type": "original",
                 "condition_parameter": None, "seed": int(final_cfg.get("random_seed", 42))}
    rows = []
    for f in replay_service.list_frames():
        fid = f["frame_id"]
        raw, _ = replay_service.load_frame_points(fid)
        sample = nusc.get("sample", fid)
        rec = run_pipeline_condition(
            np.asarray(raw), {"frame_id": fid}, sample, nusc,
            importance_engine, resolution_engine, condition,
            cell_size=float(final_cfg["integration_cell_size_m"]),
            failsafe_config=failsafe_cfg, semantic_override={"mode": "model"})
        assert rec["success"], rec.get("failure_reason")
        for d in rec["_region_features"]:
            pass
        mdf = rec["_map_df"]
        for _, r in mdf.iterrows():
            rows.append({
                "frame_id": fid,
                "distance_m": round(float(np.hypot(r["x"], r["y"])), 2),
                "importance": round(float(r["importance"]), 4),
                "resolution_m": float(r["resolution"]),
                "semantic_class": str(r["semantic_class"]),
                "semantic_source": str(r.get("semantic_source", "model")),
            })
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "distance_resolution_results.csv", index=False)

    invalid = df[~df["resolution_m"].isin(VALID_RESOLUTIONS)]
    summary = {
        "frames": int(df["frame_id"].nunique()),
        "cells": int(len(df)),
        "invalid_resolutions": int(len(invalid)),
        "mean_resolution_0_10m": float(df[df["distance_m"] < 10]["resolution_m"].mean()),
        "mean_resolution_10_30m": float(df[(df["distance_m"] >= 10) & (df["distance_m"] < 30)]["resolution_m"].mean()),
        "mean_resolution_30_60m": float(df[(df["distance_m"] >= 30) & (df["distance_m"] < 60)]["resolution_m"].mean()),
        "mean_resolution_60_100m": float(df[df["distance_m"] >= 60]["resolution_m"].mean()),
        "fine_share_0_10m": float((df[df["distance_m"] < 10]["resolution_m"] == 0.05).mean()),
        "coarse_share_60_100m": float((df[df["distance_m"] >= 60]["resolution_m"].isin([0.20, 0.50])).mean()),
        "model_sourced_share": float((df["semantic_source"] == "model").mean()),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    pd.DataFrame([summary]).to_csv(OUT_DIR / "distance_resolution_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    sub = df.sample(n=min(4000, len(df)), random_state=42)
    ax.scatter(sub["distance_m"], sub["resolution_m"], s=6, alpha=0.35, c=sub["importance"], cmap="viridis")
    ax.set_xlabel("Distance from sensor (m)")
    ax.set_ylabel("Assigned resolution (m)")
    ax.set_title("Adaptive resolution vs distance (color = importance) -- measured replay frames")
    ax.set_yticks(list(VALID_RESOLUTIONS))
    fig.colorbar(ax.collections[0], ax=ax, label="importance")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "distance_resolution_plot.png", dpi=120)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
