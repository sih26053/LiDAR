"""Resolution Engine v1 configuration (10 September 2026 task).

Initial prototype resolution thresholds, subject to later tuning and
benchmark validation. They are NOT claimed to be scientifically optimal.

Policy:
    I >= 0.75 -> 0.05 m (fine / 5 cm)
    I >= 0.50 -> 0.10 m (medium-fine / 10 cm)
    I >= 0.25 -> 0.20 m (medium-coarse / 20 cm)
    else     -> 0.50 m (coarse / 50 cm)

Smaller cell size = finer resolution. Larger cell size = coarser resolution.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# Primary structure: ordered (threshold, resolution_m) pairs, descending.
# Each entry means "importance >= threshold -> resolution_m".
RESOLUTION_LEVELS: List[Tuple[float, float]] = [
    (0.75, 0.05),
    (0.50, 0.10),
    (0.25, 0.20),
    (0.00, 0.50),
]

# Compatibility views matching the 9-September Stage-1 naming in
# ``src/data_types.py`` (RESOLUTION_LEVELS dict + RESOLUTION_THRESHOLDS tuple).
RESOLUTION_LEVELS_DICT: Dict[str, float] = {
    "fine": 0.05,
    "medium_fine": 0.10,
    "medium_coarse": 0.20,
    "coarse": 0.50,
}
RESOLUTION_THRESHOLDS: Tuple[float, float, float] = (0.75, 0.50, 0.25)

# Aliases matching the 10-September task-spec names (Sections 5 / 19).
# Same policy, spec-style naming: thresholds dict + values dict.
# NOTE: the name RESOLUTION_THRESHOLDS is kept as the canonical tuple above
# (consumed by src/stage1_engines.py); the spec-style dict lives here under
# RESOLUTION_THRESHOLDS_MAP so both conventions work without breaking imports.
RESOLUTION_THRESHOLDS_MAP: Dict[str, float] = {
    "very_high": 0.75,
    "high": 0.50,
    "medium": 0.25,
}
RESOLUTION_VALUES: Dict[str, float] = {
    "very_high": 0.05,
    "high": 0.10,
    "medium": 0.20,
    "low": 0.50,
}

LEVEL_NAMES: Tuple[str, str, str, str] = (
    "fine (5 cm)",
    "medium_fine (10 cm)",
    "medium_coarse (20 cm)",
    "coarse (50 cm)",
)


def validate_resolution_config(
    levels: List[Tuple[float, float]] | None = None,
) -> bool:
    """Validate the resolution policy. Returns True or raises ValueError."""
    import math

    lvls = list(levels if levels is not None else RESOLUTION_LEVELS)
    if len(lvls) != 4:
        raise ValueError(f"RESOLUTION_LEVELS must hold 4 entries. Received: {lvls!r}")
    thresholds = [float(t) for t, _ in lvls]
    resolutions = [float(r) for _, r in lvls]
    for t in thresholds:
        if not math.isfinite(t) or not (0.0 <= t <= 1.0):
            raise ValueError(f"Resolution thresholds must be in [0,1]. Received: {t!r}")
    if not (thresholds[0] > thresholds[1] > thresholds[2] >= 0):
        raise ValueError(
            "Resolution thresholds must satisfy t0 > t1 > t2 >= 0 (descending). "
            f"Received: {thresholds!r}"
        )
    if abs(thresholds[-1] - 0.0) > 1e-9:
        raise ValueError(
            f"Last threshold must be 0.00 (catch-all bin). Received: {thresholds[-1]!r}"
        )
    for r in resolutions:
        if not math.isfinite(r) or r <= 0:
            raise ValueError(f"Resolutions must be finite and > 0. Received: {r!r}")
    if not (resolutions[0] < resolutions[1] < resolutions[2] < resolutions[3]):
        raise ValueError(
            "Resolutions must strictly increase as importance falls "
            f"(finer -> coarser). Received: {resolutions!r}"
        )
    return True


# Validate the shipped defaults at import time so a bad edit fails fast.
validate_resolution_config()
