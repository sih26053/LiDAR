"""Stage 8 — Action execution abstraction (flow box 8).

DQN action -> Safety Validator -> Allowed/Rejected -> Executor.

Implementations:
- SimulationActionExecutor (DEFAULT): validates through the safety
  controller and records the validated action. No hardware touched.
- HardwareActionExecutor: DISABLED by default; instantiation refuses
  unless explicitly enabled (no vehicle is attached in this project).

Hardware execution stays disabled: ``hardware_execution_enabled`` is
False in config/safety_config.json and there is no vehicle interface
to enable.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.safety_controller import evaluate

ACTIONS = ("FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP")


def load_action_config() -> Dict[str, Any]:
    """config/action_config.json with identical built-in fallbacks."""
    import json
    from pathlib import Path

    cfg = {"controls": {
        "forward": {"throttle": 0.4, "steer": 0.0, "brake": 0.0},
        "turn_left": {"throttle": 0.3, "steer": -0.4, "brake": 0.0},
        "turn_right": {"throttle": 0.3, "steer": 0.4, "brake": 0.0},
        "stop": {"throttle": 0.0, "steer": 0.0, "brake": 1.0}},
        "limits": {"max_throttle": 0.6, "max_steer": 0.5}}
    try:
        path = Path(__file__).resolve().parent.parent / "config" / "action_config.json"
        cfg.update(json.loads(path.read_text()))
    except (OSError, ValueError):
        pass
    return cfg


def action_to_control(action: str) -> Dict[str, Any]:
    """DQN action -> CARLA control values (genuine mapping, no CARLA needed
    to compute; clamped to configured limits with the clamp logged)."""
    cfg = load_action_config()
    key = str(action or "").lower()
    if key not in cfg["controls"]:
        raise ValueError(f"Unknown action {action!r}.")
    ctrl = dict(cfg["controls"][key])
    lim = cfg.get("limits", {})
    clamped = []
    if abs(float(ctrl.get("throttle", 0.0))) > float(lim.get("max_throttle", 0.6)):
        ctrl["throttle"] = float(lim["max_throttle"]) * (1 if ctrl["throttle"] >= 0 else -1)
        clamped.append("throttle")
    if abs(float(ctrl.get("steer", 0.0))) > float(lim.get("max_steer", 0.5)):
        ctrl["steer"] = float(lim["max_steer"]) * (1 if ctrl["steer"] >= 0 else -1)
        clamped.append("steer")
    ctrl["clamped"] = clamped
    return ctrl


def apply_control(vehicle, action: str):
    """Apply VehicleControl to a live CARLA vehicle (connected only)."""
    from src.simulation.carla_lidar import _require_carla

    _require_carla()
    import carla  # type: ignore

    ctrl = action_to_control(action)
    vehicle.apply_control(carla.VehicleControl(
        throttle=float(ctrl["throttle"]), steer=float(ctrl["steer"]),
        brake=float(ctrl["brake"]), reverse=bool(ctrl.get("reverse", False))))
    return {"applied": True, "action": str(action).lower(), **{k: ctrl[k] for k in ("throttle", "steer", "brake")}}


class HardwareDisabledError(RuntimeError):
    """Raised whenever hardware execution is requested while disabled."""


class SimulationActionExecutor:
    """Default executor: validate + record (simulation/replay only)."""

    def __init__(self):
        self.history: List[Dict[str, Any]] = []

    def execute(self, dqn_action: str,
                nearest_forward_obstacle_m: float | None,
                map_valid: bool = True,
                emergency: bool = False) -> Dict[str, Any]:
        verdict = evaluate(dqn_action, nearest_forward_obstacle_m,
                           map_valid=map_valid, emergency=emergency)
        record = {
            "executor": "simulation",
            "dqn_action": verdict["dqn_action"],
            "final_action": verdict["final_action"].upper(),
            "verdict": verdict["verdict"],
            "reason": verdict["reason"],
            "hardware_command_issued": False,
        }
        self.history.append(record)
        return record


class HardwareActionExecutor:
    """Hardware executor: disabled (no vehicle interface exists)."""

    def __init__(self, enabled: bool = False):
        if not enabled:
            raise HardwareDisabledError(
                "Hardware execution is disabled by default and no vehicle "
                "interface is attached. Use SimulationActionExecutor.")

    def execute(self, *args, **kwargs) -> Dict[str, Any]:
        raise HardwareDisabledError("No vehicle interface attached.")
