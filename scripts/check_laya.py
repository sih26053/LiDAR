"""Local Laya live gate probe (no key, no fabrication).

Ensures the loopback server (auto-started if needed), then sends one
REAL minimal structured-state request to POST /v1/systemone and
reports READY only when a typed choice + answer_confidence succeeds.

Run:  python scripts/check_laya.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.decision.jev_state_adapter import enrich_for_laya, to_named_state  # noqa: E402
from src.decision.laya_client import LayaDecisionSystem, service_status  # noqa: E402

# Minimal VALID structured state (13-D contract bounds [0,1]); open
# forward, blocked sides -- a real state shape, not a shortcut.
PROBE_VECTOR = [1.0, 0.05, 0.05, 0.5, 0.5, 0.05, 0.05, 1.0,
                0.2, 0.0, 0.1, 0.3, 0.2]

if __name__ == "__main__":
    from backend.services import laya_manager

    try:
        boot = laya_manager.ensure_started(wait_s=120.0)
        booted = boot.get("reused") is False
    except Exception as exc:  # noqa: BLE001 - reported, probed anyway
        print(json.dumps({"status": "BLOCKED",
                          "reason": f"startup: {exc}"}, indent=2))
        raise SystemExit(0)
    st = service_status()
    if not st["available"]:
        print(json.dumps({"status": "BLOCKED", **st,
                          "booted_this_run": booted}, indent=2))
    else:
        named = to_named_state(list(PROBE_VECTOR), [])
        rec = LayaDecisionSystem().decide(enrich_for_laya(named))
        d = rec.to_dict()
        safe = {k: d.get(k) for k in
                ("proposed_action", "confidence", "answer_confidence",
                 "probabilities", "latency_ms", "status", "error",
                 "model", "device", "endpoint")}
        print(json.dumps({"status": "READY" if d["status"] == "OK" else "BLOCKED",
                          "booted_this_run": booted,
                          "service": {k: v for k, v in st.items()},
                          "live_decision": safe}, indent=2))
