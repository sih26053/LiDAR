"""Write the navigation summary roll-up (Phase 30)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> dict:
    L = lambda n: json.loads((PROJECT_ROOT / n).read_text())
    inv, sp = L("results/laya_dataset_inventory.json"), L(
        "results/laya_dataset_splits.json")
    base, fin = L("results/laya_baseline.json"), L(
        "results/laya_final_test.json")
    ood, emer = L("results/laya_ood.json"), L(
        "results/laya_emergency_validation.json")
    oo, gate = L("results/laya_option_order.json"), L(
        "models/laya/calibration/gate_selection.json")
    temp, man = L("models/laya/calibration/temperature.json"), L(
        "models/laya/manifest.json")
    summary = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {"runs": inv["n_runs"], "frames": inv["n_frames"],
                    "splits": sp["counts"],
                    "split_hash": sp["split_hash"],
                    "coverage_gate": {k: v["status"] for k, v in
                                      sp["coverage_gate"].items()}},
        "baseline_test": fin["by_mode"],
        "ood": {"n": ood["n_ood"],
                "constrained_accuracy":
                ood["by_mode"]["ood_test:constrained_laya"]["accuracy"],
                "failures": ood["constrained_failures"]},
        "emergency": {"n": emer["n_emergency"],
                      "raw_forward_rate": emer["raw_forward_rate"],
                      "executed_forward": emer["executed_forward"]},
        "option_order": {"states": oo["n_states"],
                         "flip_rate": oo["flip_rate"]},
        "gate": {"threshold": gate["current_gate_threshold"],
                 "legacy": gate["legacy_starting_threshold"],
                 "promoted": gate["promoted"],
                 "tie_break": gate.get("tie_break")},
        "temperature": {"version": temp.get("version"),
                        "runtime_group": temp.get("n_options"),
                        "deployed": {k: v.get("deployed") for k, v in
                                     temp.get("groups", {}).items()}},
        "checkpoint": {"revision": man["revision"],
                       "bytes": man["total_bytes"]},
    }
    (PROJECT_ROOT / "results" / "laya_navigation_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    md = ["# Laya navigation summary (measured)",
          f"Generated: {summary['generated_utc']}", "",
          f"Dataset: {inv['n_runs']} runs, {inv['n_frames']} frames; "
          f"splits {json.dumps(sp['counts'])}",
          f"Test constrained: acc={fin['by_mode']['constrained_laya']['accuracy']} "
          f"unsafe={fin['by_mode']['constrained_laya']['unsafe_raw_action_rate']}",
          f"OOD: n={ood['n_ood']} failures={len(ood['constrained_failures'])}",
          f"Emergency: n={emer['n_emergency']} raw_fwd={emer['raw_forward_rate']} "
          f"exec_fwd={emer['executed_forward']}",
          f"Option-order flips: {oo['n_flips']}/{oo['n_states']}",
          f"Gate: {gate['current_gate_threshold']} ({gate.get('tie_break')})",
          f"Checkpoint: {man['revision']}"]
    (PROJECT_ROOT / "results" / "laya_navigation_summary.md").write_text(
        "\n".join(md))
    print("wrote laya_navigation_summary.json/md")
    return summary


if __name__ == "__main__":
    main()
