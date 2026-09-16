"""Robustness / tuning / fail-safe utilities (14 September 2026 task).

ADDITIVE module. Does not modify any existing file:

- ``src/data_types.py`` (9 September contract)
- ``src/importance_engine.py`` (v1, unchanged)
- ``src/resolution_engine.py`` (v1, unchanged)
- ``src/feature_adapter.py`` ``run_first_handoff`` (10 September, unchanged)
- ``src/semantic_mapping.py`` / ``src/semantic_adapter.py`` (11 Sept, unchanged)
- ``src/mapper_2_5d.py`` (12 September, unchanged)
- ``src/baselines.py`` (13 September, unchanged)

Contents:

1. Controlled degradation transforms (dropout / sparsity / XYZ noise) applied
   to COPIES of real nuScenes frames. They are test conditions only, never
   described as measured sensor error statistics.
2. Fail-safe policy helpers (frame-health check + documented fallback
   behaviour) that wrap -- never replace -- the existing validators.
3. An instrumented pipeline runner that reuses the existing pipeline
   verbatim (``preprocess_points`` -> annotation semantics ->
   ``build_perception_from_semantics`` -> ``aggregate_to_regions`` ->
   ``ImportanceEngine`` / ``ResolutionEngine`` / ``build_adaptive_map``)
   while timing each stage separately with ``time.perf_counter``.
   Compute-only timing; visualization is never inside the timed region.
"""

from __future__ import annotations

import time
import traceback
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Central robustness configuration (mirrors notebooks/06 ROBUSTNESS_CONFIG)
# ---------------------------------------------------------------------------

ROBUSTNESS_DEFAULTS: Dict[str, Any] = {
    "seed": 42,
    "dropout_rates": [0.05, 0.10, 0.20, 0.30],
    "sparsity_levels": [1.00, 0.75, 0.50, 0.25],
    "noise_levels": {"low": 0.01, "medium": 0.03, "high": 0.05},
    "cell_size_integration_m": 2.0,
}

# Fail-safe policy (engineering parameters, chosen after inspecting the
# existing validators/engines -- see notebooks/06 Section 13).
FAILSAFE_DEFAULTS: Dict[str, Any] = {
    "low_point_density_threshold": 200,  # input points below -> safety flag
    "min_regions_for_normal_operation": 3,  # fewer regions -> fallback path
    "high_uncertainty_threshold": 0.8,  # mean region uncertainty above -> flag
    "fallback_resolution": 0.20,  # safety-biased resolution (documented choice)
    "fallback_semantic_label": "unknown",
    "reject_invalid_frame": True,  # NaN/Inf/empty -> reject + log, no map
}

VALID_CONDITION_TYPES = (
    "original",
    "dropout",
    "sparsity",
    "noise",
)


# ---------------------------------------------------------------------------
# Controlled degradation (all operate on copies; source frame never mutated)
# ---------------------------------------------------------------------------

def _require_nx4(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 4:
        raise ValueError(f"points must be N x 4. Received shape {pts.shape!r}")
    return pts


def dropout_points(
    points: np.ndarray, dropout_rate: float, seed: int = 42
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Randomly drop each point with probability ``dropout_rate`` (copy).

    Returns ``(kept_points, stats)`` where stats records original / dropped /
    remaining counts, the rate and the seed. The input array is never modified.
    """
    pts = _require_nx4(points)
    rate = float(dropout_rate)
    if not np.isfinite(rate) or not (0.0 <= rate < 1.0):
        raise ValueError(f"dropout_rate must be in [0,1). Received: {dropout_rate!r}")
    rng = np.random.default_rng(int(seed))
    keep_mask = rng.random(len(pts)) >= rate
    kept = np.ascontiguousarray(pts[keep_mask])
    stats = {
        "original_point_count": int(len(pts)),
        "dropped_point_count": int(len(pts) - int(keep_mask.sum())),
        "remaining_point_count": int(keep_mask.sum()),
        "dropout_rate": rate,
        "seed": int(seed),
    }
    return kept, stats


def sparsity_subsample(
    points: np.ndarray, keep_ratio: float, seed: int = 42
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Deterministically keep ``keep_ratio`` of points (copy).

    Exact-count subsample via a seeded permutation so ``remaining ==
    round(N * keep_ratio)`` reproducibly. Returns ``(kept, stats)``.
    """
    pts = _require_nx4(points)
    ratio = float(keep_ratio)
    if not np.isfinite(ratio) or not (0.0 < ratio <= 1.0):
        raise ValueError(f"keep_ratio must be in (0,1]. Received: {keep_ratio!r}")
    n = len(pts)
    n_keep = int(round(n * ratio))
    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(n)[:n_keep]
    kept = np.ascontiguousarray(pts[np.sort(idx)])
    stats = {
        "original_point_count": n,
        "remaining_point_count": int(n_keep),
        "dropped_point_count": int(n - n_keep),
        "sparsity_level": ratio,
        "seed": int(seed),
    }
    return kept, stats


def add_xyz_noise(
    points: np.ndarray, sigma_m: float, seed: int = 42
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Add zero-mean Gaussian noise (std ``sigma_m``) to XYZ only (copy).

    Intensity column is untouched. Semantic labels are never altered by this
    transform (callers keep the original label arrays). Returns ``(noisy,
    stats)``.
    """
    pts = _require_nx4(points)
    sigma = float(sigma_m)
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError(f"sigma_m must be finite and >= 0. Received: {sigma_m!r}")
    rng = np.random.default_rng(int(seed))
    noisy = pts.copy()
    if sigma > 0 and len(noisy):
        noisy[:, :3] = noisy[:, :3] + rng.normal(0.0, sigma, size=(len(noisy), 3))
    stats = {
        "noise_sigma_m": sigma,
        "seed": int(seed),
        "points_affected": int(len(noisy)),
    }
    return noisy, stats


def apply_condition(
    points: np.ndarray,
    condition_type: str,
    condition_parameter: Any,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply one named robustness condition to a copy of ``points``."""
    if condition_type == "original":
        pts = np.ascontiguousarray(_require_nx4(points).copy())
        return pts, {"original_point_count": len(pts),
                     "remaining_point_count": len(pts), "seed": int(seed)}
    if condition_type == "dropout":
        return dropout_points(points, float(condition_parameter), seed)
    if condition_type == "sparsity":
        return sparsity_subsample(points, float(condition_parameter), seed)
    if condition_type == "noise":
        return add_xyz_noise(points, float(condition_parameter), seed)
    raise ValueError(
        f"Unknown condition_type {condition_type!r}. "
        f"Expected one of {list(VALID_CONDITION_TYPES)}."
    )


def build_condition_table(
    dropout_rates: Sequence[float] = (0.05, 0.10, 0.20, 0.30),
    sparsity_levels: Sequence[float] = (0.75, 0.50, 0.25),
    noise_levels: Dict[str, float] | None = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Structured condition table (one row per test condition)."""
    noise_levels = dict(noise_levels or {"low": 0.01, "medium": 0.03, "high": 0.05})
    table: List[Dict[str, Any]] = [
        {"condition": "original", "condition_type": "original",
         "condition_parameter": None, "noise_name": None, "seed": int(seed)}
    ]
    for r in dropout_rates:
        table.append({"condition": f"dropout_{int(round(float(r) * 100)):d}pct",
                      "condition_type": "dropout",
                      "condition_parameter": float(r),
                      "noise_name": None, "seed": int(seed)})
    for s in sparsity_levels:
        table.append({"condition": f"sparse_{int(round(float(s) * 100)):d}pct",
                      "condition_type": "sparsity",
                      "condition_parameter": float(s),
                      "noise_name": None, "seed": int(seed)})
    for name, sigma in noise_levels.items():
        table.append({"condition": f"noise_{name}",
                      "condition_type": "noise",
                      "condition_parameter": float(sigma),
                      "noise_name": str(name), "seed": int(seed)})
    return table


# ---------------------------------------------------------------------------
# Fail-safe helpers
# ---------------------------------------------------------------------------

def check_frame_health(
    points: np.ndarray,
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Inspect a candidate input frame; never raises on bad data.

    Returns a dict with ``ok`` (bool), ``reject`` (bool), ``reasons`` (list)
    and basic counts. NaN/Inf/empty frames are rejected; very sparse frames
    are flagged (safety path) but not rejected.
    """
    cfg = dict(config or FAILSAFE_DEFAULTS)
    reasons: List[str] = []
    try:
        pts = np.asarray(points, dtype=np.float64)
    except Exception:
        return {"ok": False, "reject": True, "reasons": ["unconvertible input"],
                "n_points": 0, "n_finite": 0}
    if pts.ndim != 2 or pts.shape[1] != 4:
        return {"ok": False, "reject": True,
                "reasons": [f"bad shape {pts.shape!r}, expected N x 4"],
                "n_points": 0, "n_finite": 0}
    n = int(pts.shape[0])
    n_finite = int(np.isfinite(pts).all(axis=1).sum())
    if n == 0:
        reasons.append("empty frame (0 points)")
    if n_finite < n:
        reasons.append(f"{n - n_finite} non-finite rows")
    if n < int(cfg.get("low_point_density_threshold", 200)):
        reasons.append(f"very low point density (n={n})")
    reject = (n == 0) or (n_finite < n) or (n_finite == 0)
    if bool(cfg.get("reject_invalid_frame", True)) and (n == 0 or n_finite < n):
        reject = True
    return {"ok": not reject, "reject": bool(reject), "reasons": reasons,
            "n_points": n, "n_finite": n_finite}


def safe_fps(total_latency_ms: float) -> float:
    """FPS from measured latency; NaN on zero/invalid latency (never invent)."""
    try:
        t = float(total_latency_ms)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(t) or t <= 0:
        return float("nan")
    return 1000.0 / t


# ---------------------------------------------------------------------------
# Instrumented pipeline runner (reuses existing modules verbatim)
# ---------------------------------------------------------------------------

def run_pipeline_condition(
    raw_points: np.ndarray,
    frame_metadata: Dict[str, Any],
    sample,
    nusc,
    importance_engine,
    resolution_engine,
    condition: Dict[str, Any],
    cell_size: float = 2.0,
    failsafe_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run the full existing pipeline for one frame x condition.

    Stages (each timed separately with ``time.perf_counter``)::

        degraded input -> preprocessing -> perception -> feature extraction
        -> importance -> resolution -> mapping

    ``total_latency_ms`` = preprocessing + perception + feature + mapping
    (compute only; visualization excluded). The importance / resolution
    sub-timings are diagnostic passes over the same regions with the same
    engine instances; the authoritative map always comes from
    ``build_adaptive_map``. Never raises: failures are captured into the
    returned record (``success=False`` + ``failure_reason``/``exception_type``
    /``stage_of_failure``).
    """
    # Late imports keep this module importable without the full stack.
    from .preprocessing import preprocess_points
    from .semantic_adapter import (
        aggregate_to_regions,
        annotations_to_sensor_frame,
        assign_annotation_semantics,
        build_perception_from_semantics,
    )
    from .mapper_2_5d import build_adaptive_map, validate_adaptive_map_df

    cfg = dict(failsafe_config or FAILSAFE_DEFAULTS)
    rec: Dict[str, Any] = {
        "condition": condition.get("condition"),
        "condition_type": condition.get("condition_type"),
        "condition_parameter": condition.get("condition_parameter"),
        "seed": condition.get("seed"),
        "success": False,
    }
    try:
        # --- degrade a copy (original never mutated) ---------------------
        degraded, deg_stats = apply_condition(
            raw_points,
            str(condition.get("condition_type")),
            condition.get("condition_parameter"),
            int(condition.get("seed", 42)),
        )
        rec["input_points"] = int(deg_stats.get("original_point_count",
                                                len(np.asarray(raw_points))))
        # --- fail-safe gate ------------------------------------------------
        health = check_frame_health(degraded, cfg)
        if health["reject"]:
            rec.update(success=False, stage_of_failure="failsafe_gate",
                       exception_type="RejectedFrame",
                       failure_reason="; ".join(health["reasons"])[:500],
                       processed_points=0, retained_ratio=float("nan"),
                       total_latency_ms=float("nan"), fps=float("nan"))
            return rec

        # --- preprocessing -------------------------------------------------
        t0 = time.perf_counter()
        clean_points, counts = preprocess_points(degraded)
        preprocessing_ms = (time.perf_counter() - t0) * 1000.0

        # --- perception (annotation semantics + PerceptionResult) -----------
        t0 = time.perf_counter()
        sensor_anns = annotations_to_sensor_frame(nusc, sample)
        sem_labels, sem_source_arr, _ = assign_annotation_semantics(
            clean_points, sensor_anns)
        semantic_labels: List[str] = [
            str(label) for label in np.asarray(sem_labels).reshape(-1).tolist()
        ]
        semantic_source: List[str] = [
            str(source)
            for source in np.asarray(sem_source_arr).reshape(-1).tolist()
        ]
        # NOTE: semantic labels are NOT altered by coordinate noise; the
        # boxes stay in the sensor frame of the same sample.
        perception_result, source_array, _ = build_perception_from_semantics(
            str(frame_metadata["frame_id"]), clean_points,
            semantic_labels, semantic_source)
        perception_ms = (time.perf_counter() - t0) * 1000.0

        # --- feature extraction (regions) ----------------------------------
        t0 = time.perf_counter()
        region_features, region_details = aggregate_to_regions(
            perception_result, source_array, cell_size=float(cell_size))
        feature_ms = (time.perf_counter() - t0) * 1000.0

        # --- diagnostic importance / resolution sub-timings ----------------
        t0 = time.perf_counter()
        importances: List[float] = []
        for region in region_features:
            importances.append(
                float(importance_engine.calculate(region).final_importance))
        importance_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        for value in importances:
            resolution_engine.assign_resolution(float(value))
        resolution_ms = (time.perf_counter() - t0) * 1000.0

        # --- authoritative mapping -----------------------------------------
        t0 = time.perf_counter()
        _, map_df = build_adaptive_map(
            region_features, importance_engine, resolution_engine,
            region_details)
        validate_adaptive_map_df(map_df)
        mapping_ms = (time.perf_counter() - t0) * 1000.0

        total_ms = preprocessing_ms + perception_ms + feature_ms + mapping_ms
        res = map_df["resolution"].to_numpy(dtype=float)
        n_input = int(len(np.asarray(raw_points)))
        n_proc = int(len(clean_points))
        rec.update(
            success=True, stage_of_failure="", exception_type="",
            failure_reason="",
            input_points=n_input, processed_points=n_proc,
            dropped_points=int(n_input - len(degraded))
            if len(degraded) <= n_input else 0,
            retained_ratio=(float(n_proc) / n_input) if n_input > 0 else float("nan"),
            degraded_points=int(len(degraded)),
            map_cells=int(len(map_df)),
            fine_cells=int((res == 0.05).sum()),
            medium_cells=int((res == 0.10).sum()),
            coarse_cells=int(((res == 0.20) | (res == 0.50)).sum()),
            res_20cm_cells=int((res == 0.20).sum()),
            res_50cm_cells=int((res == 0.50).sum()),
            average_resolution=float(res.mean()) if len(res) else float("nan"),
            spatial_coverage=float(
                (map_df["x"].max() - map_df["x"].min())
                * (map_df["y"].max() - map_df["y"].min()))
            if len(map_df) > 1 else 0.0,
            mean_importance=float(map_df["importance"].mean()),
            preprocessing_latency_ms=float(preprocessing_ms),
            perception_latency_ms=float(perception_ms),
            feature_extraction_latency_ms=float(feature_ms),
            importance_latency_ms=float(importance_ms),
            resolution_latency_ms=float(resolution_ms),
            mapping_latency_ms=float(mapping_ms),
            total_latency_ms=float(total_ms),
            fps=float(safe_fps(total_ms)),
            n_regions=int(len(region_features)),
            mean_uncertainty=float(np.mean(
                [float(r.uncertainty) for r in region_features])),
            failsafe_flag=bool(
                n_input < int(cfg.get("low_point_density_threshold", 200))),
        )
        # Keep heavy objects out of the CSV record but available to callers.
        rec["_map_df"] = map_df
        rec["_region_features"] = region_features
        rec["_region_details"] = region_details
        rec["_clean_points"] = clean_points
        return rec
    except Exception as exc:  # noqa: BLE001 -- failures are data, not crashes
        stage = rec.get("stage_of_failure") or "unknown"
        if not rec.get("stage_of_failure"):
            # Best-effort stage attribution from traceback content.
            tb = traceback.format_exc()
            for key in ("preprocess", "perception", "aggregat",
                        "importance", "resolution", "adaptive_map"):
                if key in tb:
                    stage = key
                    break
        rec.update(success=False, stage_of_failure=stage,
                   exception_type=type(exc).__name__,
                   failure_reason=str(exc)[:500])
        for key, default in (("processed_points", 0), ("retained_ratio", float("nan")),
                             ("map_cells", 0), ("total_latency_ms", float("nan")),
                             ("fps", float("nan"))):
            rec.setdefault(key, default)
        return rec


def remap_cached_regions(
    region_features: Sequence[Any],
    region_details: Sequence[Dict[str, Any]],
    importance_engine,
    resolution_engine,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Re-run mapping-only on cached regions (used for tuning comparisons).

    Same engines interface as the baseline path; timed compute-only.
    Returns ``(map_df, timing_dict)``.
    """
    from .mapper_2_5d import build_adaptive_map, validate_adaptive_map_df

    t0 = time.perf_counter()
    importances = [float(importance_engine.calculate(r).final_importance)
                   for r in region_features]
    importance_ms = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    for value in importances:
        resolution_engine.assign_resolution(float(value))
    resolution_ms = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    _, map_df = build_adaptive_map(
        list(region_features), importance_engine, resolution_engine,
        list(region_details))
    validate_adaptive_map_df(map_df)
    mapping_ms = (time.perf_counter() - t0) * 1000.0
    return map_df, {"importance_latency_ms": importance_ms,
                    "resolution_latency_ms": resolution_ms,
                    "mapping_latency_ms": mapping_ms}
