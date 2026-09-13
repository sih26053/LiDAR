"""Importance Engine v1 configuration (10 September 2026 task).

Initial prototype values only. These weights are engineering assumptions
and are NOT claimed to be optimized. They live outside the core algorithm
so future ablation studies can tune them without code changes.

Relationship to the 9 September foundation:
- Values mirror ``src/data_types.py`` (WEIGHTS, MAX_RANGE_M).
- The single deliberate difference is ``UNCERTAINTY_LAMBDA``: the canonical
  Stage-1 core (``src/stage1_engines.py``, LAMBDA=0.5) is preserved untouched;
  this v1 config uses the 10-September prototype safety heuristic 0.15.
"""
from __future__ import annotations

from typing import Dict

WEIGHTS: Dict[str, float] = {
    "distance": 0.30,
    "semantic": 0.30,
    "terrain": 0.15,
    "dynamic": 0.15,
    "uncertainty": 0.10,
}

MAX_DISTANCE: float = 100.0
# Alias kept for compatibility with the 9-September Stage-1 naming.
MAX_RANGE_M: float = MAX_DISTANCE

UNCERTAINTY_LAMBDA: float = 0.15
# Alias kept for compatibility with the 9-September Stage-1 naming.
LAMBDA_UNCERTAINTY: float = UNCERTAINTY_LAMBDA
# Alias matching the 10-September task-spec name (Section 5 / 19).
# Same value: prototype uncertainty safety multiplier lam = 0.15.
UNCERTAINTY_BOOST: float = UNCERTAINTY_LAMBDA

# Tolerance for the "weights sum to 1" check (avoids fragile exact float compare).
WEIGHT_SUM_TOLERANCE: float = 1e-6


def validate_importance_config(
    weights: Dict[str, float] | None = None,
    max_distance: float | None = None,
    uncertainty_lambda: float | None = None,
) -> bool:
    """Validate an importance configuration. Returns True or raises ValueError."""
    import math

    w = dict(weights if weights is not None else WEIGHTS)
    md = MAX_DISTANCE if max_distance is None else max_distance
    lam = UNCERTAINTY_LAMBDA if uncertainty_lambda is None else uncertainty_lambda

    required = {"distance", "semantic", "terrain", "dynamic", "uncertainty"}
    missing = required - set(w.keys())
    if missing:
        raise ValueError(f"WEIGHTS missing keys: {sorted(missing)}")
    for key in sorted(required):
        value = w[key]
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"WEIGHTS[{key!r}] must be finite. Received: {value!r}")
        if float(value) < 0:
            raise ValueError(f"WEIGHTS[{key!r}] must be non-negative. Received: {value!r}")
    total = sum(float(w[k]) for k in sorted(required))
    if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
        raise ValueError(
            f"WEIGHTS must sum to ~1.0 (tol={WEIGHT_SUM_TOLERANCE}). Received sum={total!r}"
        )
    if not isinstance(md, (int, float)) or not math.isfinite(float(md)) or float(md) <= 0:
        raise ValueError(f"MAX_DISTANCE must be finite and > 0. Received: {md!r}")
    if (
        not isinstance(lam, (int, float))
        or not math.isfinite(float(lam))
        or float(lam) < 0
    ):
        raise ValueError(
            f"UNCERTAINTY_LAMBDA must be finite and >= 0. Received: {lam!r}"
        )
    return True


# Validate the shipped defaults at import time so a bad edit fails fast.
validate_importance_config()
