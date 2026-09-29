"""Map quality gate before RL use (flow box 5 -> box 6 guard).

Validates an adaptive 2.5D map (DataFrame or cell list) for:

  cell coordinates, elevation, occupancy, semantic class, confidence,
  resolution -- checking NaN / Inf / invalid cells / invalid resolution /
  empty map / corrupted geometry.

Returns a verdict dict; ``gate_for_rl()`` raises ``InvalidMapError``
so a corrupt map can never reach RL state generation silently.
Reuses (not duplicates) ``src/mapper_2_5d.validate_adaptive_map_df``
for the structural contract, then adds RL-facing checks (confidence
finiteness in model-sourced cells, occupancy range, resolution set).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

VALID_RESOLUTIONS = (0.05, 0.10, 0.20, 0.50)
REQUIRED_FIELDS = ("x", "y", "elevation", "occupancy", "semantic_class",
                   "confidence", "resolution")


class InvalidMapError(ValueError):
    """Raised when a map fails the RL gate."""


def validate_cells(cells: Sequence[Dict[str, Any]],
                   require_model_confidence: bool = False) -> Dict[str, Any]:
    """Cell-list validation -> verdict dict (never raises on bad data)."""
    cells = list(cells or [])
    issues: List[str] = []
    if not cells:
        return {"valid": False, "n_cells": 0, "issues": ["empty map"], "checked_fields": list(REQUIRED_FIELDS)}
    for f in REQUIRED_FIELDS:
        if any(c.get(f, None) is None for c in cells):
            issues.append(f"missing field: {f}")
    for i, c in enumerate(cells):
        try:
            vals = [float(c.get(f)) for f in ("x", "y", "elevation", "occupancy")]
        except (TypeError, ValueError):
            issues.append(f"cell {i}: non-numeric geometry/occupancy")
            break
        if not all(np.isfinite(vals)):
            issues.append(f"cell {i}: NaN/Inf in geometry/occupancy")
            break
        if not (0.0 <= vals[3] <= 1.0):
            issues.append(f"cell {i}: occupancy outside [0,1]")
            break
        try:
            res = float(c.get("resolution"))
        except (TypeError, ValueError):
            issues.append(f"cell {i}: non-numeric resolution")
            break
        if res not in VALID_RESOLUTIONS:
            issues.append(f"cell {i}: invalid resolution {res}")
            break
        if not str(c.get("semantic_class", "")):
            issues.append(f"cell {i}: empty semantic class")
            break
        src = str(c.get("semantic_source", ""))
        conf = c.get("confidence", None)
        if src == "model" or require_model_confidence:
            try:
                cf = float(conf)
            except (TypeError, ValueError):
                issues.append(f"cell {i}: model cell without finite confidence")
                break
            if not (np.isfinite(cf) and 0.0 <= cf <= 1.0):
                issues.append(f"cell {i}: model confidence outside [0,1]")
                break
    return {"valid": not issues, "n_cells": len(cells),
            "issues": issues[:10], "checked_fields": list(REQUIRED_FIELDS)}


def validate_frame(df: pd.DataFrame) -> Dict[str, Any]:
    """DataFrame validation: structural contract + RL-facing checks."""
    from src.mapper_2_5d import validate_adaptive_map_df

    try:
        validate_adaptive_map_df(df)
    except (ValueError, KeyError, TypeError) as exc:
        return {"valid": False, "n_cells": int(len(df)),
                "issues": [f"structural contract: {exc}"], "checked_fields": list(REQUIRED_FIELDS)}
    return validate_cells(df.to_dict(orient="records"))


def gate_for_rl(cells: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """RL gate: returns the verdict or raises InvalidMapError."""
    verdict = validate_cells(cells)
    if not verdict["valid"]:
        raise InvalidMapError(f"Map rejected for RL use: {verdict['issues']}")
    return verdict
