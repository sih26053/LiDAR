"""LEGACY cloud-Jev live gate probe (kept for historical records).

The active backend uses local Laya (scripts/check_laya.py) and never
calls OpenRouter. This script still works where a key is configured.
"""

Key source: config/local_secrets.py (OPENROUTER_API_KEY pasted by the
user; server-side only) unless OPENROUTER_API_KEY / TYPESAFE_API_KEY is
set in the environment. Endpoint defaults to
https://openrouter.ai/api/alpha/decisions unless JEV_ENDPOINT overrides.

Sends a REAL minimal structured-state request to the configured Decisions
endpoint and reports READY only when an actual decision (choice +
confidence) succeeds. Anything else is BLOCKED with the measured reason.

Run:  python scripts/check_jev.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.decision.jev_decision import JevDecisionSystem, service_status  # noqa: E402

# Minimal VALID structured state (13-D contract Bounds [0,1]); open forward,
# blocked sides -- a real state shape, not a privileged shortcut.
PROBE_STATE = {
    "sector_0_range": 1.0, "sector_1_range": 0.05,
    "sector_2_range": 0.05, "sector_3_range": 0.5,
    "sector_4_range": 0.5, "sector_5_range": 0.05,
    "sector_6_range": 0.05, "sector_7_range": 1.0,
    "obstacle_density": 0.2, "moving_share": 0.0,
    "static_share": 0.1, "mean_importance": 0.3,
    "mean_uncertainty": 0.2, "sector_ranges_m": [],
    "state_dim": 13,
}

if __name__ == "__main__":
    st = service_status()
    if not st["available"]:
        print(json.dumps({"status": "BLOCKED", **st}, indent=2))
    else:
        from src.decision.jev_state_adapter import enrich_for_jev

        # Probe what production sends: the enriched state.
        rec = JevDecisionSystem().decide(enrich_for_jev(dict(PROBE_STATE)))
        # Never print key material: rec carries no credentials by construction.
        safe = {k: rec.get(k) for k in
                ("proposed_action", "confidence", "probabilities",
                 "latency_ms", "status", "error", "model")}
        print(json.dumps({"status": "READY" if rec["status"] == "OK" else "BLOCKED",
                          "service": {k: v for k, v in st.items() if k != "reason"},
                          "live_decision": safe}, indent=2))
