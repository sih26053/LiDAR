"""Generate frontend/public/benchmark.json from results/benchmark_v2 CSVs.

Source of truth: results/benchmark_v2/benchmark_summary_v2.csv
                 results/benchmark_v2/benchmark_results_v2.csv
                 results/benchmark_v2/benchmark_config_v2.json

Reshape-only (no recalculation). Run: python scripts/generate_benchmark_asset_v2.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = PROJECT_ROOT / "results" / "benchmark_v2"
OUT = PROJECT_ROOT / "frontend" / "public" / "benchmark.json"

LABELS = {
    "proposed": "Proposed Adaptive",
    "uniform_5cm": "Uniform 5 cm",
    "uniform_5cm_region": "Uniform 5 cm (same-scope)",
    "distance_adaptive": "Distance-based Adaptive",
}


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def to_float(v):
    try:
        f = float(v)
        return f if f == f and abs(f) != float("inf") else None
    except (TypeError, ValueError):
        return None


def main() -> None:
    summary_rows = read_csv(BENCH_DIR / "benchmark_summary_v2.csv")
    detail_rows = read_csv(BENCH_DIR / "benchmark_results_v2.csv")
    config = json.loads((BENCH_DIR / "benchmark_config_v2.json").read_text())
    env = config.get("environment", {}) or {}
    pkgs = env.get("packages", {}) or {}
    hw = (f"{env.get('cpu_model', '?')} ({env.get('cpu_count', '?')} threads), "
          f"RAM {env.get('ram_gb', '?')} GB, {env.get('os', '?')}, "
          f"Python {env.get('python_version', '?')}, numpy {pkgs.get('numpy', '?')}, "
          f"pandas {pkgs.get('pandas', '?')}")

    methods = []
    for row in summary_rows:
        m = row["method"]
        methods.append({
            "method": m,
            "label": LABELS.get(m, m),
            "frames": int(row["frames"]),
            "successful": int(row["successful"]),
            "mean_mapping_latency_ms": to_float(row["mean_mapping_latency_ms"]),
            "median_mapping_latency_ms": to_float(row["median_mapping_latency_ms"]),
            "std_mapping_latency_ms": to_float(row.get("std_mapping_latency_ms", "")),
            "mean_end_to_end_latency_ms": to_float(row["mean_end_to_end_latency_ms"]),
            "mean_cells": to_float(row["mean_cells"]),
            "median_cells": None,
            "mean_resolution_m": to_float(row["mean_resolution_m"]),
            "mean_coverage": None,
            "failure_rate": 0.0,
            "resolution_totals": {},
            "resolution_share": {},
        })

    per_frame = [{
        "frame_id": r["frame_id"],
        "method": r["method"],
        "map_cells": int(float(r["map_cells"])),
        "mapping_latency_ms": to_float(r["mapping_latency_mean_ms"]),
        "end_to_end_latency_ms": to_float(r["end_to_end_latency_ms"]),
        "mean_resolution_m": to_float(r["mean_resolution_m"]),
        "mean_importance": None,
        "status": r.get("status"),
    } for r in detail_rows]

    payload = {
        "provenance": {
            "source": "results/benchmark_v2/ (verified re-run under frozen runtime config, NOT recalculated)",
            "files": [
                "results/benchmark_v2/benchmark_summary_v2.csv",
                "results/benchmark_v2/benchmark_results_v2.csv",
                "results/benchmark_v2/benchmark_config_v2.json",
                "results/benchmark_v2/environment_v2.json",
            ],
            "note": ("Same 10 samples, same inputs and frozen config for all methods; "
                     "hardware recorded at run time (see benchmark_config_v2.json)."),
        },
        "task": config.get("task"),
        "hardware_summary": hw,
        "methodology": {
            "timer": config.get("timer"),
            "repetitions_measured": config.get("repetitions_measured"),
            "warmup_runs_discarded": config.get("warmup_runs_discarded"),
            "excluded": config.get("excluded"),
            "end_to_end_definition": config.get("end_to_end_definition"),
            "resolution_levels": config.get("resolution_levels"),
        },
        "methods": methods,
        "per_frame": per_frame,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT} ({len(methods)} methods, {len(per_frame)} per-frame rows)")


if __name__ == "__main__":
    main()
