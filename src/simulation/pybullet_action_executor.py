"""Phase 5/15 — PyBullet action executor (existing action contract).

Logical actions use the validated repository mapping (lowercase names,
shared with the DQN/safety layers); index map 0=FORWARD, 1=LEFT,
2=RIGHT, 3=STOP is provided for the closed-loop script. Control values
come from config/simulation_config.json (documented starting points).

`action_to_velocity()` is pure and unit-testable without pybullet.
`execute()` requires a live env and returns measured before/after pose.
"""

from __future__ import annotations

from typing import Any, Dict

ACTION_INDEX = {0: "forward", 1: "left", 2: "right", 3: "stop"}
# Canonical names stay lowercase (repo contract); LEFT/RIGHT are aliases.
ACTION_ALIASES = {"left": "turn_left", "right": "turn_right",
                  "forward": "forward", "stop": "stop",
                  "turn_left": "turn_left", "turn_right": "turn_right"}


def canonical(action: str | int) -> str:
    if isinstance(action, int):
        if action not in ACTION_INDEX:
            raise ValueError(f"action index must be 0-3, got {action}")
        action = ACTION_INDEX[action]
    key = str(action or "").lower()
    if key not in ACTION_ALIASES:
        raise ValueError(f"unknown action {action!r}")
    return ACTION_ALIASES[key]


def action_to_velocity(action: str | int, cfg: Dict[str, Any] | None = None):
    """Logical action -> (forward speed m/s, yaw rate rad/s). Pure function."""
    import math
    cfg = cfg or {}
    ctl = cfg.get("control", {})
    fwd = float(ctl.get("forward_speed_ms", 2.0))
    tspd = float(ctl.get("turn_speed_ms", 1.0))
    yaw = math.radians(float(ctl.get("turn_yaw_deg_s", 25.0)))
    key = canonical(action)
    if key == "forward":
        return fwd, 0.0
    if key == "turn_left":
        return tspd, yaw
    if key == "turn_right":
        return tspd, -yaw
    return 0.0, 0.0  # stop


class PyBulletActionExecutor:
    """Executes validated actions on the live PyBullet vehicle."""

    def __init__(self, env):
        self.env = env

    def execute(self, action: str | int) -> Dict[str, Any]:
        """Apply one action for exactly one env step; return measured status."""
        if self.env is None or getattr(self.env, "_vehicle", None) is None:
            from src.simulation.pybullet_env import PyBulletUnavailable
            raise PyBulletUnavailable("no live vehicle; call after env.reset()")
        key = canonical(action)
        before = self.env.get_vehicle_state()
        step_info = self.env.step(key)
        after = self.env.get_vehicle_state()
        return {"action": key, "executed": True,
                "pose_before": {k: round(before[k], 3) for k in ("x", "y", "yaw_deg")},
                "pose_after": {k: round(after[k], 3) for k in ("x", "y", "yaw_deg")},
                "speed_ms": round(after["speed_ms"], 3),
                "collision": step_info["collision"],
                "frame": step_info["frame"]}
