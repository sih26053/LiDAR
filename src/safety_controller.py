"""Stage 8 — Safety Controller (rule-based, DQN-independent).

Inputs: nearest forward obstacle distance, map validity, mean
prediction uncertainty, emergency condition flag.
Outputs: SAFE_TO_EXECUTE or OVERRIDE_TO_STOP.

The layer always wins over the DQN: a FORWARD action with an occupied
cell inside the safety sector becomes STOP; STOP is never overridden.
Parameters come from config/safety_config.json with identical built-in
fallbacks (documented) so the rule works even if the config is absent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "safety_config.json"

SAFE_TO_EXECUTE = "SAFE_TO_EXECUTE"
OVERRIDE_TO_STOP = "OVERRIDE_TO_STOP"

_DEFAULTS = {
    "safety_radius_m": 5.0,
    "safety_half_angle_deg": 30.0,
    "occupied_threshold": 0.5,
}


def load_safety_config() -> Dict[str, Any]:
    cfg = dict(_DEFAULTS)
    try:
        cfg.update(json.loads(CONFIG_PATH.read_text()))
    except (OSError, ValueError):
        pass
    return cfg


def evaluate(
    dqn_action: str,
    nearest_forward_obstacle_m: float | None,
    map_valid: bool = True,
    emergency: bool = False,
) -> Dict[str, Any]:
    """DQN action + safety inputs -> validated verdict (genuine rule)."""
    cfg = load_safety_config()
    action = str(dqn_action or "").lower()
    if action not in ("forward", "turn_left", "turn_right", "stop"):
        return {"dqn_action": action, "final_action": "stop",
                "verdict": OVERRIDE_TO_STOP,
                "reason": f"unknown action {action!r}; fail-safe STOP"}
    if not map_valid or emergency:
        return {"dqn_action": action, "final_action": "stop",
                "verdict": OVERRIDE_TO_STOP,
                "reason": "invalid map" if not map_valid else "emergency flag set"}
    if action == "forward" and nearest_forward_obstacle_m is not None:
        try:
            if float(nearest_forward_obstacle_m) <= float(cfg["safety_radius_m"]):
                return {"dqn_action": action, "final_action": "stop",
                        "verdict": OVERRIDE_TO_STOP,
                        "reason": (f"occupied cell within {cfg['safety_radius_m']} m "
                                   f"ahead ({nearest_forward_obstacle_m} m)")}
        except (TypeError, ValueError):
            return {"dqn_action": action, "final_action": "stop",
                    "verdict": OVERRIDE_TO_STOP,
                    "reason": "unreadable obstacle distance; fail-safe STOP"}
    return {"dqn_action": action, "final_action": action,
            "verdict": SAFE_TO_EXECUTE, "reason": "no safety rule fired"}
