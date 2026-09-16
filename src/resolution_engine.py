"""Resolution Engine v1 (10 September 2026 task).

Maps a final importance score in [0,1] to a 2.5D map cell size in metres.
Thresholds are initial prototype values from ``config/resolution_config.py``
(subject to later tuning and benchmark validation), kept configurable so
policy changes never require code edits.

Higher importance -> equal or finer resolution. Numerically that means the
cell size is non-increasing as importance grows: 0.05 <= 0.10 <= 0.20 <= 0.50.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

try:  # Preferred: policy lives outside the algorithm.
    from config.resolution_config import (
        RESOLUTION_LEVELS,
        validate_resolution_config,
    )
except ImportError:  # Fallback so the engine stays importable standalone.
    RESOLUTION_LEVELS: List[Tuple[float, float]] = [
        (0.75, 0.05),
        (0.50, 0.10),
        (0.25, 0.20),
        (0.00, 0.50),
    ]

    def validate_resolution_config(levels=None) -> bool:  # type: ignore[misc]
        return True


_LEVEL_NAMES = (
    "fine (5 cm)",
    "medium_fine (10 cm)",
    "medium_coarse (20 cm)",
    "coarse (50 cm)",
)


def validate_importance(value: float) -> float:
    """Validate an importance score. Returns it as float or raises ValueError."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"importance must be a number in [0,1]. Received: {value!r}")
    if not math.isfinite(v) or not (0.0 <= v <= 1.0):
        raise ValueError(f"importance must be within [0,1]. Received: {value!r}")
    return v


def assign_resolution(importance: float) -> float:
    """Module-level convenience (spec Section 10) using default policy."""
    return ResolutionEngine().select_resolution(importance)


@dataclass
class ResolutionResult:
    """Structured output of the Resolution Engine (optional, spec Section 23).

    ``importance`` is the validated input score; ``resolution_m`` is the
    assigned cell size in metres.
    """

    importance: float
    resolution_m: float


class ResolutionEngine:
    """Resolution Engine v1. ``select_resolution(importance)`` -> cell size (m)."""

    def __init__(self, levels: List[Tuple[float, float]] | None = None):
        lvls = [(float(t), float(r)) for t, r in (levels if levels is not None else RESOLUTION_LEVELS)]
        validate_resolution_config(lvls)
        # Descending by threshold so the first match wins (boundaries inclusive).
        self.levels: List[Tuple[float, float]] = sorted(lvls, key=lambda p: p[0], reverse=True)

    @property
    def thresholds(self) -> Tuple[float, ...]:
        """Configured importance thresholds, descending."""
        return tuple(t for t, _ in self.levels)

    @property
    def resolutions(self) -> Tuple[float, ...]:
        """Configured cell sizes (m), finest first."""
        return tuple(r for _, r in self.levels)

    def select_resolution(self, importance: float) -> float:
        """Return the map cell size in metres for an importance in [0,1]."""
        v = validate_importance(importance)
        for threshold, resolution in self.levels:
            if v >= threshold:
                return resolution
        # Unreachable: the 0.00 catch-all bin always matches, but kept for safety.
        return self.levels[-1][1]

    def assign_resolution(self, importance: float) -> float:
        """Spec-name alias (Section 10) for :meth:`select_resolution`.

        Same deterministic, boundary-inclusive mapping; kept so callers using
        either name get identical behaviour.
        """
        return self.select_resolution(importance)

    def resolve(self, importance: float) -> ResolutionResult:
        """Structured-result convenience wrapper (spec Section 23, optional)."""
        v = validate_importance(importance)
        return ResolutionResult(importance=v, resolution_m=self.select_resolution(v))

    def select(self, importance: float) -> Tuple[float, str]:
        """Compatibility alias returning ``(resolution_m, level_name)``.

        Mirrors the Stage-1 ``ResolutionEngine.select`` signature so existing
        mapper-side callers work unchanged.
        """
        resolution = self.select_resolution(importance)
        v = float(importance)
        if v >= self.levels[0][0]:
            name = _LEVEL_NAMES[0]
        elif v >= self.levels[1][0]:
            name = _LEVEL_NAMES[1]
        elif v >= self.levels[2][0]:
            name = _LEVEL_NAMES[2]
        else:
            name = _LEVEL_NAMES[3]
        return resolution, name
