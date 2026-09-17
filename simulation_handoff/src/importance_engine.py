"""Importance Engine v1 (10 September 2026 task).

Deterministic, scalar region-level engine. Consumes the 9-September
``RegionFeatures`` contract (see ``src/data_types.py``) without redesigning it:

- ``distance``: physical distance in metres (>= 0), normalized here via
  ``D = clip01(1 - distance / max_distance)``.
- ``semantic_importance``, ``terrain_complexity`` (``RegionFeatures.roughness``),
  ``dynamic_relevance``, ``uncertainty``, ``confidence``: normalized [0,1].

Formula (initial prototype weights, configurable via ``config/importance_config.py``)::

    I_base = 0.30*D + 0.30*S + 0.15*T + 0.15*M + 0.10*U
    I_safe = min(1, I_base + 0.15*U)

``confidence`` is validated as part of the interface contract and passed
through on the result, but it does not enter the v1 formula; uncertainty
carries the perception-quality term. No heavyweight ML dependencies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping

try:  # Preferred: configurable prototype values live outside the algorithm.
    from config.importance_config import (
        MAX_DISTANCE,
        UNCERTAINTY_LAMBDA,
        WEIGHTS,
        validate_importance_config,
    )
except ImportError:  # Fallback so the engine stays importable standalone.
    WEIGHTS: Dict[str, float] = {
        "distance": 0.30,
        "semantic": 0.30,
        "terrain": 0.15,
        "dynamic": 0.15,
        "uncertainty": 0.10,
    }
    MAX_DISTANCE = 100.0
    UNCERTAINTY_LAMBDA = 0.15

    def validate_importance_config(  # type: ignore[misc]
        weights=None, max_distance=None, uncertainty_lambda=None
    ) -> bool:
        return True


# ---------------------------------------------------------------------------
# Small testable helpers
# ---------------------------------------------------------------------------

def clip01(value: float) -> float:
    """Constrain a value to [0, 1]. Raises on NaN/inf (never silently hide them)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"clip01 expects a number. Received: {value!r}")
    if not math.isfinite(v):
        raise ValueError(f"clip01 received non-finite value: {value!r}")
    return max(0.0, min(1.0, v))


def normalize_distance(distance: float, max_distance: float = MAX_DISTANCE) -> float:
    """Map a physical distance (m) to a closeness score in [0, 1].

    ``D = clip01(1 - distance / max_distance)``: 0 m -> 1.0, 100 m -> 0.0,
    beyond max -> 0.0. Raises on negative, non-finite, or non-positive
    ``max_distance`` (avoids division by zero).
    """
    try:
        d = float(distance)
    except (TypeError, ValueError):
        raise ValueError(f"distance must be a number (metres). Received: {distance!r}")
    try:
        m = float(max_distance)
    except (TypeError, ValueError):
        raise ValueError(f"max_distance must be a number. Received: {max_distance!r}")
    if not math.isfinite(d) or d < 0:
        raise ValueError(f"distance must be finite and >= 0 (metres). Received: {distance!r}")
    if not math.isfinite(m) or m <= 0:
        raise ValueError(f"max_distance must be finite and > 0. Received: {max_distance!r}")
    return max(0.0, min(1.0, 1.0 - d / m))


def _check_normalized(value: Any, name: str) -> float:
    """Validate one normalized [0,1] input. Raises ValueError on violation."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number in [0,1]. Received: {value!r}")
    if not math.isfinite(v) or not (0.0 <= v <= 1.0):
        raise ValueError(f"{name} must be within [0,1]. Received: {value!r}")
    return v


def validate_inputs(
    distance: float,
    semantic_importance: float,
    terrain_complexity: float,
    dynamic_relevance: float,
    uncertainty: float,
    confidence: float,
) -> bool:
    """Validate raw region features before importance calculation.

    ``distance`` is physical (metres, >= 0, finite); every other factor must
    be within [0,1]. NaN/inf are rejected, never silently clipped. When the
    caller passes a ``RegionFeatures`` instance, prefer
    ``validate_region_features`` (single contract owner); this helper covers
    scalar/dict-style callers. Returns True or raises ValueError.
    """
    try:
        d = float(distance)
    except (TypeError, ValueError):
        raise ValueError(f"distance must be a number (metres). Received: {distance!r}")
    if not math.isfinite(d) or d < 0:
        raise ValueError(f"distance must be finite and >= 0 (metres). Received: {distance!r}")
    _check_normalized(semantic_importance, "semantic_importance")
    _check_normalized(terrain_complexity, "terrain_complexity")
    _check_normalized(dynamic_relevance, "dynamic_relevance")
    _check_normalized(uncertainty, "uncertainty")
    _check_normalized(confidence, "confidence")
    return True


def calculate_base_importance(
    distance_score: float,
    semantic_importance: float,
    terrain_complexity: float,
    dynamic_relevance: float,
    uncertainty: float,
    weights: Mapping[str, float] | None = None,
) -> float:
    """Weighted sum of the five normalized factors, constrained to [0,1].

    ``I_base = wd*D + ws*S + wt*T + wm*M + wu*U`` with the configured weights.
    """
    w = dict(weights if weights is not None else WEIGHTS)
    required = ("distance", "semantic", "terrain", "dynamic", "uncertainty")
    missing = [k for k in required if k not in w]
    if missing:
        raise ValueError(f"weights missing keys: {missing}")
    for key in required:
        v = w[key]
        if not isinstance(v, (int, float)) or not math.isfinite(float(v)) or float(v) < 0:
            raise ValueError(f"weights[{key!r}] must be finite and >= 0. Received: {v!r}")
    if abs(sum(float(w[k]) for k in required) - 1.0) > 1e-6:
        raise ValueError(f"weights must sum to ~1.0. Received sum={sum(float(w[k]) for k in required)!r}")
    d = _check_normalized(distance_score, "distance_score")
    s = _check_normalized(semantic_importance, "semantic_importance")
    t = _check_normalized(terrain_complexity, "terrain_complexity")
    m = _check_normalized(dynamic_relevance, "dynamic_relevance")
    u = _check_normalized(uncertainty, "uncertainty")
    base = (
        float(w["distance"]) * d
        + float(w["semantic"]) * s
        + float(w["terrain"]) * t
        + float(w["dynamic"]) * m
        + float(w["uncertainty"]) * u
    )
    return max(0.0, min(1.0, base))


def apply_uncertainty_modifier(
    base_importance: float,
    uncertainty: float,
    lambda_uncertainty: float = UNCERTAINTY_LAMBDA,
) -> float:
    """Prototype safety heuristic: ``I_safe = min(1, I_base + lambda * U)``.

    High uncertainty preserves additional detail instead of allowing
    aggressive simplification. This is a safety-oriented heuristic, NOT a
    formal probabilistic safety guarantee.
    """
    try:
        lam = float(lambda_uncertainty)
    except (TypeError, ValueError):
        raise ValueError(f"lambda_uncertainty must be a number. Received: {lambda_uncertainty!r}")
    if not math.isfinite(lam) or lam < 0:
        raise ValueError(f"lambda_uncertainty must be finite and >= 0. Received: {lambda_uncertainty!r}")
    b = _check_normalized(base_importance, "base_importance")
    u = _check_normalized(uncertainty, "uncertainty")
    return max(0.0, min(1.0, b + lam * u))


# ---------------------------------------------------------------------------
# Region-feature extraction (adapts to the 9-September naming)
# ---------------------------------------------------------------------------

def _get_field(region: Any, *names: str, default: Any = None) -> Any:
    """Read the first available attribute/key from a region object or mapping."""
    for name in names:
        if isinstance(region, Mapping):
            if name in region:
                return region[name]
        elif hasattr(region, name):
            return getattr(region, name)
    return default


def extract_features(region: Any) -> Dict[str, float]:
    """Extract the six validated inputs from a region object or mapping.

    Accepts ``RegionFeatures`` (``roughness`` field), legacy Stage-1 regions
    (``distance_m`` / ``terrain_complexity`` / ``semantic_class``), and plain
    dicts. ``terrain_complexity`` falls back to ``roughness``; ``distance``
    falls back to ``distance_m``.
    """
    distance = _get_field(region, "distance", "distance_m")
    terrain = _get_field(region, "terrain_complexity", "roughness")
    semantic = _get_field(region, "semantic_importance")
    dynamic = _get_field(region, "dynamic_relevance")
    uncertainty = _get_field(region, "uncertainty")
    confidence = _get_field(region, "confidence")
    region_id = _get_field(region, "region_id", default=-1)
    if distance is None or terrain is None or semantic is None or dynamic is None:
        raise ValueError(
            "Region is missing required importance inputs "
            "(distance, semantic_importance, terrain_complexity/roughness, "
            f"dynamic_relevance). Received: {region!r}"
        )
    if uncertainty is None or confidence is None:
        raise ValueError(
            "Region is missing required inputs (uncertainty, confidence). "
            f"Received: {region!r}"
        )
    validate_inputs(distance, semantic, terrain, dynamic, uncertainty, confidence)
    return {
        "region_id": int(region_id) if region_id is not None else -1,
        "distance": float(distance),
        "semantic_importance": float(semantic),
        "terrain_complexity": float(terrain),
        "dynamic_relevance": float(dynamic),
        "uncertainty": float(uncertainty),
        "confidence": float(confidence),
    }


# ---------------------------------------------------------------------------
# Result + engine API
# ---------------------------------------------------------------------------

@dataclass
class ImportanceResult:
    """Output of :meth:`ImportanceEngine.calculate`.

    ``final_importance`` is the value the Resolution Engine consumes.
    ``safe_importance`` is a compatibility alias (same value) for consumers
    written against the Stage-1 ``ImportanceResult`` naming.
    """

    region_id: int
    distance_score: float
    semantic_importance: float
    terrain_complexity: float
    dynamic_relevance: float
    uncertainty: float
    confidence: float
    base_importance: float
    final_importance: float

    @property
    def safe_importance(self) -> float:
        """Alias of ``final_importance`` for Stage-1 consumer compatibility."""
        return self.final_importance

    def to_dict(self) -> Dict[str, float]:
        """Plain-dict view exposing both ``final_importance`` and ``safe_importance``."""
        return {
            "region_id": self.region_id,
            "distance_score": self.distance_score,
            "semantic_importance": self.semantic_importance,
            "terrain_complexity": self.terrain_complexity,
            "dynamic_relevance": self.dynamic_relevance,
            "uncertainty": self.uncertainty,
            "confidence": self.confidence,
            "base_importance": self.base_importance,
            "final_importance": self.final_importance,
            "safe_importance": self.final_importance,
        }


class ImportanceEngine:
    """Importance Engine v1. ``engine.calculate(region_features)`` -> result."""

    def __init__(
        self,
        weights: Mapping[str, float] | None = None,
        max_distance: float = MAX_DISTANCE,
        lambda_uncertainty: float = UNCERTAINTY_LAMBDA,
    ):
        w = dict(weights if weights is not None else WEIGHTS)
        validate_importance_config(
            weights=w, max_distance=max_distance, uncertainty_lambda=lambda_uncertainty
        )
        self.weights: Dict[str, float] = {k: float(w[k]) for k in w}
        self.max_distance: float = float(max_distance)
        self.lambda_uncertainty: float = float(lambda_uncertainty)

    def normalize_distance(self, distance: float) -> float:
        """Instance-scoped distance normalization (uses this engine's max range)."""
        return normalize_distance(distance, self.max_distance)

    def calculate(self, region: Any) -> ImportanceResult:
        """Run raw region features -> normalized inputs -> base -> final importance."""
        feats = extract_features(region)
        distance_score = normalize_distance(feats["distance"], self.max_distance)
        base = calculate_base_importance(
            distance_score,
            feats["semantic_importance"],
            feats["terrain_complexity"],
            feats["dynamic_relevance"],
            feats["uncertainty"],
            weights=self.weights,
        )
        final = apply_uncertainty_modifier(
            base, feats["uncertainty"], self.lambda_uncertainty
        )
        return ImportanceResult(
            region_id=feats["region_id"],
            distance_score=distance_score,
            semantic_importance=feats["semantic_importance"],
            terrain_complexity=feats["terrain_complexity"],
            dynamic_relevance=feats["dynamic_relevance"],
            uncertainty=feats["uncertainty"],
            confidence=feats["confidence"],
            base_importance=base,
            final_importance=final,
        )
