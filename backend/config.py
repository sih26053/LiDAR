"""Backend frozen-configuration loader (17 September task).

Single source of truth: ``results/final/config/final_config.json``
(frozen 15 September, W_BASE+THRESH_C). Falls back to
``results/final/final_config.json`` only if the config/ copy is absent;
both files are byte-compared at startup and a mismatch fails loudly.

No algorithm weights or resolution thresholds are hard-coded here or in
any route file -- everything is read from the frozen JSON.
"""

from __future__ import annotations

import hashlib
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("paradox.backend.config")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRIMARY_CONFIG = PROJECT_ROOT / "results" / "final" / "config" / "final_config.json"
LEGACY_CONFIG = PROJECT_ROOT / "results" / "final" / "final_config.json"
FAILSAFE_CONFIG = PROJECT_ROOT / "results" / "final" / "config" / "failsafe_config.json"
SEMANTIC_MAPPING = PROJECT_ROOT / "results" / "final" / "config" / "semantic_mapping.json"

REQUIRED_KEYS = (
    "importance_weights",
    "max_distance_m",
    "uncertainty_lambda",
    "resolution_levels",
    "integration_cell_size_m",
)

EXPECTED_WEIGHT_KEYS = ("distance", "semantic", "terrain", "dynamic", "uncertainty")


class ConfigNotFoundError(FileNotFoundError):
    pass


class ConfigInvalidError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_config_path() -> Path:
    if PRIMARY_CONFIG.is_file():
        return PRIMARY_CONFIG
    if LEGACY_CONFIG.is_file():
        logger.warning("primary config absent; using legacy %s", LEGACY_CONFIG)
        return LEGACY_CONFIG
    raise ConfigNotFoundError(
        f"Frozen final config not found at {PRIMARY_CONFIG} "
        f"or {LEGACY_CONFIG}. Backend cannot start without it."
    )


def _validate_final_config(cfg: Dict[str, Any]) -> None:
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ConfigInvalidError(f"final_config missing keys: {missing}")
    weights = cfg["importance_weights"]
    if sorted(weights.keys()) != sorted(EXPECTED_WEIGHT_KEYS):
        raise ConfigInvalidError(
            f"importance_weights keys must be {list(EXPECTED_WEIGHT_KEYS)}; got {sorted(weights.keys())}"
        )
    total = sum(float(weights[k]) for k in EXPECTED_WEIGHT_KEYS)
    if abs(total - 1.0) > 1e-6:
        raise ConfigInvalidError(f"importance_weights must sum to ~1.0; got {total!r}")
    if not (isinstance(cfg["max_distance_m"], (int, float)) and cfg["max_distance_m"] > 0):
        raise ConfigInvalidError("max_distance_m must be a positive number")
    levels = cfg["resolution_levels"]
    if not isinstance(levels, list) or len(levels) == 0:
        raise ConfigInvalidError("resolution_levels must be a non-empty list")
    for entry in levels:
        if not (isinstance(entry, (list, tuple)) and len(entry) == 2):
            raise ConfigInvalidError(f"bad resolution_levels entry: {entry!r}")
        t, r = float(entry[0]), float(entry[1])
        if not (0.0 <= t <= 1.0 and r > 0):
            raise ConfigInvalidError(f"bad resolution_levels entry: {entry!r}")
    if not (isinstance(cfg["integration_cell_size_m"], (int, float)) and cfg["integration_cell_size_m"] > 0):
        raise ConfigInvalidError("integration_cell_size_m must be positive")


@lru_cache(maxsize=1)
def load_final_config() -> Dict[str, Any]:
    """Load + validate the frozen final config (cached for process lifetime)."""
    path = resolve_config_path()
    try:
        cfg = json.loads(path.read_text())
    except Exception as exc:
        raise ConfigInvalidError(f"Cannot parse {path}: {exc}") from exc
    _validate_final_config(cfg)
    if PRIMARY_CONFIG.is_file() and LEGACY_CONFIG.is_file():
        if _sha256(PRIMARY_CONFIG) != _sha256(LEGACY_CONFIG):
            logger.warning(
                "results/final/config/final_config.json and "
                "results/final/final_config.json differ; using the config/ copy."
            )
    logger.info("frozen config loaded: %s selection=%s", path, cfg.get("selection"))
    return cfg


@lru_cache(maxsize=1)
def load_failsafe_config() -> Dict[str, Any]:
    if not FAILSAFE_CONFIG.is_file():
        raise ConfigNotFoundError(f"Failsafe config not found: {FAILSAFE_CONFIG}")
    return json.loads(FAILSAFE_CONFIG.read_text())


def load_semantic_mapping() -> Dict[str, Any] | None:
    if SEMANTIC_MAPPING.is_file():
        return json.loads(SEMANTIC_MAPPING.read_text())
    return None


def config_info() -> Dict[str, Any]:
    """Safe, non-sensitive metadata for GET /config."""
    cfg = load_final_config()
    levels = [(float(t), float(r)) for t, r in cfg["resolution_levels"]]
    return {
        "final_version": "prototype-final-v1",
        "selection": cfg.get("selection", ""),
        "resolution_levels": levels,
        "resolution_levels_m": cfg.get("resolution_levels_m", {}),
        "resolution_thresholds": cfg.get("resolution_thresholds", {}),
        "max_mapping_distance_m": float(cfg["max_distance_m"]),
        "integration_cell_size_m": float(cfg["integration_cell_size_m"]),
        "semantic_source_mode": cfg.get("semantic_source", "annotation+fallback (no trained model)"),
        "random_seed": cfg.get("random_seed", 42),
        "config_path": "results/final/config/final_config.json",
    }
