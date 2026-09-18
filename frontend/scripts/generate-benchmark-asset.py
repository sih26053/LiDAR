"""Generate frontend/public/benchmark.json from stored results/benchmark CSVs.

Source of truth: results/benchmark/benchmark_summary.csv
                 results/benchmark/benchmark_results.csv
                 results/benchmark/resolution_distribution.csv
                 results/benchmark/benchmark_config.json

The frontend NEVER recalculates benchmarks. This script only reshapes
stored, validated rows into a static asset for BenchmarkPanel.
Run: python frontend/scripts/generate-benchmark-asset.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BENCH_DIR = PROJECT_ROOT / "results" / "benchmark"
OUT = PROJECT_ROOT / "frontend" / "public" / "benchmark.json"


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
    summary_rows = read_csv(BENCH_DIR / "benchmark_summary.csv")
    detail_rows = read_csv(BENCH_DIR / "benchmark_results.csv")
    res_rows = read_csv(BENCH_DIR / "resolution_distribution.csv")
    config = json.loads((BENCH_DIR / "benchmark_config.json").read_text())

    methods = []
    for row in summary_rows:
        m = row["method"]
        res_for_method = [r for r in res_rows if r["method"] == m]
        n = len(res_for_method) or 1
        agg = {
            "res_5cm": sum(int(r["res_5cm"]) for r in res_for_method),
            "res_10cm": sum(int(r["res_10cm"]) for r in res_for_method),
            "res_20cm": sum(int(r["res_20cm"]) for r in res_for_method),
            "res_50cm": sum(int(r["res_50cm"]) for r in res_for_method),
        }
        total = sum(agg.values()) or 1
        methods.append(
            {
                "method": m,
                "label": {
                    "proposed": "Proposed Adaptive",
                    "uniform_5cm": "Uniform 5 cm",
                    "distance_adaptive": "Distance-based Adaptive",
                }.get(m, m),
                "frames": int(row["frames"]),
                "successful": int(row["successful"]),
                "mean_mapping_latency_ms": to_float(row["mean_mapping_latency_ms"]),
                "median_mapping_latency_ms": to_float(row["median_mapping_latency_ms"]),
                "mean_end_to_end_latency_ms": to_float(row["mean_end_to_end_latency_ms"]),
                "mean_cells": to_float(row["mean_cells"]),
                "median_cells": to_float(row["median_cells"]),
                "mean_resolution_m": to_float(row["mean_resolution"]),
                "mean_coverage": to_float(row["mean_coverage"]),
                "failure_rate": to_float(row["failure_rate"]),
                "resolution_totals": agg,
                "resolution_share": {k: v / total for k, v in agg.items()},
            }
        )

    per_frame = [
        {
            "frame_id": r["frame_id"],
            "method": r["method"],
            "map_cells": int(float(r["map_cells"])),
            "mapping_latency_ms": to_float(r["mapping_latency_ms"]),
            "end_to_end_latency_ms": to_float(r["end_to_end_latency_ms"]),
            "mean_resolution_m": to_float(r["mean_resolution"]),
            "mean_importance": to_float(r.get("mean_importance") or ""),
            "status": r.get("status"),
        }
        for r in detail_rows
    ]

    payload = {
        "provenance": {
            "source": "results/benchmark/ (stored validated results, NOT recalculated)",
            "files": [
                "results/benchmark/benchmark_summary.csv",
                "results/benchmark/benchmark_results.csv",
                "results/benchmark/resolution_distribution.csv",
                "results/benchmark/benchmark_config.json",
            ],
            "note": "Benchmark values shown here are loaded from the validated benchmark results and are not recalculated by the frontend.",
        },
        "task": config.get("task"),
        "methods": methods,
        "per_frame": per_frame,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT} ({len(methods)} methods, {len(per_frame)} per-frame rows)")


if __name__ == "__main__":
    main()
