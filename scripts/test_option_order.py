"""Option-order robustness (Phase 10): same state, permuted choice order.

For each of 12 selected states (2 per category A/B/C/D/E/F), send the
identical 4-option question with 4 criteria orders and record choice
flips. Production order is NOT changed by this script; it only
measures sensitivity. Saves results/laya_option_order.json.
"""

from __future__ import annotations

import itertools
import json
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ORDERS = [
    ["forward", "left", "right", "stop"],
    ["left", "right", "stop", "forward"],
    ["right", "stop", "forward", "left"],
    ["stop", "forward", "left", "right"],
]

# 12 states: 2 per category from validation/test/ood bench labels.
WANT = {"A_OPEN_FORWARD": 2, "B_FORWARD_BLOCKED_LEFT_OPEN": 2,
        "C_FORWARD_BLOCKED_RIGHT_OPEN": 2, "D_BOTH_SIDES_AVAILABLE": 2,
        "E_NO_SAFE_DIRECTION": 2, "F_EMERGENCY": 2}


def main() -> dict:
    from src.decision.jev_state_adapter import enrich_for_jev, to_named_state
    from src.decision.laya_client import (
        load_laya_config, resolve_endpoint, resolve_model)
    from src.simulation.stages_runner import run_stages_1_to_6
    from backend.services import laya_manager, store
    import numpy as np

    boot = laya_manager.ensure_started(wait_s=600.0)
    assert boot["health"].get("healthy")
    endpoint, _ = resolve_endpoint()
    model = resolve_model()
    cfg_q = load_laya_config().get("question", {}) or {}
    instructions = cfg_q.get("instructions", "")
    criteria_all = cfg_q.get("criteria", {})

    bench = json.loads((PROJECT_ROOT / "results" / "benchmark_labels.json"
                        ).read_text())["rows"]
    picked: dict = {k: [] for k in WANT}
    for r in bench:
        if r["category"] in picked and len(picked[r["category"]]) < 2:
            picked[r["category"]].append(r)
    assert all(len(v) == 2 for v in picked.values()), picked

    results = []
    for cat, rows in picked.items():
        for r in rows:
            entry = store.get_frame(r["frame_id"])
            pts = np.load(str(PROJECT_ROOT / entry["points_path"])).astype(
                "float64")
            out = run_stages_1_to_6(pts, r["frame_id"], entry.get("timestamp"))
            st = out["state"]
            named = enrich_for_jev(
                to_named_state(st["state_vector"], st["sector_ranges_m"]),
                st["safety"])
            choices = []
            for order in ORDERS:
                body = json.dumps({
                    "model": model, "state": named,
                    "questions": {"action": {
                        "type": "choice", "instructions": instructions,
                        "criteria": {k: criteria_all[k] for k in order}}}
                }).encode()
                req = urllib.request.Request(
                    endpoint, data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST")
                try:
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        payload = json.loads(resp.read().decode())
                    ch = payload.get("answers", {}).get("action", {}).get(
                        "choice")
                except Exception as exc:  # noqa: BLE001
                    ch = f"ERROR:{type(exc).__name__}"
                choices.append(ch)
            results.append({"frame_id": r["frame_id"], "category": cat,
                            "orders": ORDERS, "choices": choices,
                            "flipped": len(set(choices)) > 1})
    flips = sum(1 for r in results if r["flipped"])
    out = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                          time.gmtime()),
           "n_states": len(results), "n_flips": flips,
           "flip_rate": round(flips / len(results), 4) if results else None,
           "production_order": ["forward", "left", "right", "stop"],
           "production_order_changed": False,
           "note": ("Diagnostic only; production option order unchanged."),
           "results": results}
    (PROJECT_ROOT / "results" / "laya_option_order.json").write_text(
        json.dumps(out, indent=2))
    print(f"states={len(results)} flips={flips} rate={out['flip_rate']}")
    store.close()
    return out


if __name__ == "__main__":
    main()
