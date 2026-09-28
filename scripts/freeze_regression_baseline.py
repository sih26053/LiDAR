"""Freeze the regression baseline (Phase 31) from measured artifacts.

Copies final-test / emergency / sweep numbers into
results/laya_regression_baseline.json with explicit tolerances.
Future runs compare against it; a regression fails tests. Unsafe
actions and emergency FORWARD have ZERO tolerance: a "better"
accuracy never hides them.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TOLERANCES = {
    "accuracy_min_drop": 0.05,
    "balanced_accuracy_min_drop": 0.10,
    "ece_max_rise": 0.10,
    "brier_max_rise": 0.10,
    "unsafe_max_rise": 0.0,
    "emergency_forward_must_be": 0,
    "coverage_min_drop": 0.10,
    "latency_p95_max_factor": 2.0,
}


def main() -> dict:
    final = json.loads((PROJECT_ROOT / "results" / "laya_final_test.json"
                        ).read_text())
    emer = json.loads((PROJECT_ROOT / "results" / "laya_emergency_validation.json"
                       ).read_text())
    sweep = json.loads((PROJECT_ROOT / "results" / "laya_threshold_sweep.json"
                        ).read_text())
    con = final["by_mode"]["constrained_laya"]
    payload = {
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gate_threshold": final["gate_threshold"],
        "checkpoint_revision": final["checkpoint_revision"],
        "test_n": final["n"],
        "metrics": {
            "choice_accuracy": con["accuracy"],
            "balanced_accuracy": con["balanced_accuracy"],
            "ece": con["ece"],
            "brier": con["brier"],
            "unsafe_action_rate": con["unsafe_raw_action_rate"],
            "emergency_forward_rate": con["emergency_forward_proposal_rate"],
            "emergency_executed_forward": emer.get("executed_forward"),
            "coverage_at_gate": con["coverage_at_gate"],
            "latency_p95_ms": con["latency_p95_ms"],
        },
        "tolerances": TOLERANCES,
        "note": ("Unsafe/emergency have zero tolerance by design."),
    }
    out = PROJECT_ROOT / "results" / "laya_regression_baseline.json"
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"frozen baseline (test n={payload['test_n']}): {out}")
    return payload


if __name__ == "__main__":
    main()
