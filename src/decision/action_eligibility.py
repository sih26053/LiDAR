"""Action Eligibility Layer + navigation oracle (system-level mitigation).

System-level mitigation, NOT model-level improvement: the Laya weights are
untouched. This module adds a deterministic pre-decision constraint between
Stage 6 and Laya so the model is only ever asked to choose among actions
that are currently admissible:

    Stage 6 -> decision state -> Action Eligibility Layer -> Laya
        -> Confidence Gate -> Safety Layer -> Executor -> PyBullet

The existing Safety Layer (src/safety_controller.py) remains the FINAL
authority; this layer is an additional pre-decision constraint.

Eligibility rules (deterministic, documented):
    - emergency_flag True            -> eligible = ["stop"] only.
    - forward_clearance_m < forward_min_m -> "forward" removed.
    - left_clearance_m < side_min_m       -> "left" removed.
    - right_clearance_m < side_min_m      -> "right" removed.
    - empty eligible set             -> ["stop"] (fail-safe).

Threshold provenance (NOT silently changed):
    - forward_min_m mirrors config/safety_config.json `safety_radius_m`
      (single source of truth; read at runtime).
    - side_min_m is NEW and explicit: config/laya_config.json
      `eligibility.side_min_m`. There was no prior lateral threshold.

Choice/action naming: the Laya API uses choice names
("forward","left","right","stop"); the executor/safety layer uses
("forward","turn_left","turn_right","stop"). CHOICE_TO_ACTION maps them.

Also contained here (evaluation ONLY, never the production controller):
    - categorize_frame(): corridor categories A..G for diagnostics.
    - navigation_oracle(): deterministic reference policy used to label
      recorded frames for evaluation. Uses only real system fields
      (clearances, nearest_obstacle_m, emergency_flag, obstacle_density,
      scenario goal text when available). The oracle defines ADMISSIBLE
      actions (hard geometry) and a PREFERRED action under the project's
      documented navigation objective: make forward progress along the
      corridor; STOP only when nothing is safe or an emergency is set.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

API_CHOICES = ("forward", "left", "right", "stop")
CHOICE_TO_ACTION = {"forward": "forward", "left": "turn_left",
                    "right": "turn_right", "stop": "stop"}
ACTION_TO_CHOICE = {"forward": "forward", "turn_left": "left",
                    "turn_right": "right", "stop": "stop"}


def eligibility_thresholds() -> Dict[str, float]:
    """Resolve deterministic thresholds with documented provenance."""
    from src.safety_controller import load_safety_config

    forward_min = float(load_safety_config().get("safety_radius_m", 5.0))
    side_min = 2.5
    provenance = "built-in default (no eligibility config present)"
    try:
        from src.decision.laya_client import load_laya_config

        elig = (load_laya_config().get("eligibility", {}) or {})
        if "side_min_m" in elig:
            side_min = float(elig["side_min_m"])
            provenance = "config/laya_config.json eligibility.side_min_m"
        # forward_min_m may be pinned explicitly, but defaults to the
        # live safety radius so the two layers can never disagree.
        # null/missing = mirror safety_radius_m (documented, not silent).
        if elig.get("forward_min_m") is not None:
            forward_min = float(elig["forward_min_m"])
            provenance += " + eligibility.forward_min_m (explicit override)"
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return {"forward_min_m": forward_min, "side_min_m": float(side_min),
            "provenance": provenance}


def _clearances(named: Dict[str, Any]) -> Tuple[float, float, float, bool]:
    fwd = named.get("forward_clearance_m", None)
    left = named.get("left_clearance_m", None)
    right = named.get("right_clearance_m", None)
    if fwd is None or left is None or right is None:
        raise ValueError("named state lacks forward/left/right_clearance_m "
                         "(run enrich_for_jev/enrich_for_laya first)")
    emergency = bool(named.get("emergency_flag", False))
    return float(fwd), float(left), float(right), emergency


def compute_eligible_actions(
    named: Dict[str, Any],
    safety: Dict[str, Any] | None = None,
    thresholds: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """Deterministic eligible-action set for one decision state.

    Returns {"eligible": [...API choices...], "ineligible": {choice: reason},
    "emergency": bool, "thresholds": {...}, "clearances_m": {...}}.
    Never returns an empty eligible list (fail-safe STOP).
    """
    th = dict(thresholds or eligibility_thresholds())
    fwd, left, right, emergency = _clearances(named)
    if safety is not None:
        emergency = emergency or bool(safety.get("emergency_stop", False))
    fwd_min = float(th["forward_min_m"])
    side_min = float(th["side_min_m"])
    eligible: List[str] = []
    ineligible: Dict[str, str] = {}
    if emergency:
        ineligible = {
            "forward": "emergency_flag set; only STOP admissible",
            "left": "emergency_flag set; only STOP admissible",
            "right": "emergency_flag set; only STOP admissible",
        }
        eligible = ["stop"]
    else:
        if fwd >= fwd_min:
            eligible.append("forward")
        else:
            ineligible["forward"] = (
                f"forward_clearance_m {fwd} < forward_min_m {fwd_min}")
        if left >= side_min:
            eligible.append("left")
        else:
            ineligible["left"] = (
                f"left_clearance_m {left} < side_min_m {side_min}")
        if right >= side_min:
            eligible.append("right")
        else:
            ineligible["right"] = (
                f"right_clearance_m {right} < side_min_m {side_min}")
        eligible.append("stop")  # STOP is always admissible
        if not eligible:
            eligible = ["stop"]
    return {"eligible": eligible, "ineligible": ineligible,
            "emergency": bool(emergency), "thresholds": th,
            "clearances_m": {"forward": fwd, "left": left, "right": right}}


def categorize_frame(named: Dict[str, Any],
                     safety: Dict[str, Any] | None = None,
                     thresholds: Dict[str, float] | None = None) -> str:
    """Corridor category A..G (diagnostics only). Disjoint and complete.

    F EMERGENCY: emergency flag set.
    G NO_SAFE_ACTION: no direction meets its minimum.
    B FORWARD_BLOCKED_LEFT_OPEN: forward blocked, only left available.
    C FORWARD_BLOCKED_RIGHT_OPEN: forward blocked, only right available.
    D BOTH_SIDES_AVAILABLE: both sides available (forward open or blocked).
    E BOTH_SIDES_BLOCKED: forward open, both sides blocked.
    A OPEN_FORWARD: forward open, exactly one side available.
    """
    elig = compute_eligible_actions(named, safety, thresholds)
    if elig["emergency"]:
        return "F_EMERGENCY"
    avail = set(elig["eligible"]) - {"stop"}
    fwd_ok = "forward" in avail
    left_ok = "left" in avail
    right_ok = "right" in avail
    if not avail:
        return "G_NO_SAFE_ACTION"
    if not fwd_ok and left_ok and not right_ok:
        return "B_FORWARD_BLOCKED_LEFT_OPEN"
    if not fwd_ok and right_ok and not left_ok:
        return "C_FORWARD_BLOCKED_RIGHT_OPEN"
    if left_ok and right_ok:
        return "D_BOTH_SIDES_AVAILABLE"
    if fwd_ok and not left_ok and not right_ok:
        return "E_BOTH_SIDES_BLOCKED"
    if fwd_ok and (left_ok ^ right_ok):
        return "A_OPEN_FORWARD"
    return "G_NO_SAFE_ACTION"  # unreachable; fail-safe label


# Categories where FORWARD must never be executed.
FORWARD_INADMISSIBLE_CATEGORIES = {
    "F_EMERGENCY", "G_NO_SAFE_ACTION",
    "B_FORWARD_BLOCKED_LEFT_OPEN", "C_FORWARD_BLOCKED_RIGHT_OPEN",
}


def navigation_oracle(
    named: Dict[str, Any],
    safety: Dict[str, Any] | None = None,
    thresholds: Dict[str, float] | None = None,
    scenario_goal: str | None = None,
) -> Dict[str, Any]:
    """Deterministic reference policy for EVALUATION ONLY.

    Admissible set = eligibility minus nothing (same hard geometry as the
    production eligibility layer, so the oracle can never bless an action
    the production path forbids). Preferred action under the documented
    navigation objective ("make forward progress along the corridor;
    STOP only when nothing is safe or an emergency is set"):
        - emergency or nothing safe -> "stop".
        - forward admissible -> "forward" (progress objective; scenario
          goal text is recorded but does not override geometry).
        - forward blocked -> admissible side with the largest clearance
          (deterministic tie-break: left).
    Returns {"admissible": [...], "preferred": choice, "rule": str,
    "category": str, "scenario_goal": ...}.
    """
    elig = compute_eligible_actions(named, safety, thresholds)
    category = categorize_frame(named, safety, thresholds)
    admissible = list(elig["eligible"])
    clear = elig["clearances_m"]
    if elig["emergency"]:
        return {"admissible": admissible, "preferred": "stop",
                "rule": "emergency: only STOP admissible",
                "category": category, "scenario_goal": scenario_goal}
    moving = [c for c in admissible if c != "stop"]
    if not moving:
        return {"admissible": admissible, "preferred": "stop",
                "rule": "no safe direction: STOP is the only admissible action",
                "category": category, "scenario_goal": scenario_goal}
    if "forward" in moving:
        return {"admissible": admissible, "preferred": "forward",
                "rule": ("forward admissible: preferred under the progress "
                         "objective (forward travel when safe)"),
                "category": category, "scenario_goal": scenario_goal}
    # Forward blocked: largest safe side clearance, deterministic left tie-break.
    sides = [c for c in moving if c in ("left", "right")]
    best = max(sides, key=lambda c: (clear[c], 1 if c == "left" else 0))
    return {"admissible": admissible, "preferred": best,
            "rule": ("forward blocked: preferred = admissible side with "
                     "largest clearance (left wins ties)"),
            "category": category, "scenario_goal": scenario_goal}
