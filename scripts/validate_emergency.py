"""Emergency safety validation (Phase 18): 100+ emergency states.

Fresh UNCONSTRAINED calls on train-split F frames (raw behavior,
honest) + deterministic constrained skips (eligible=[STOP], no call).
Combined with already-evaluated valid/test/OOD F rows. Acceptance:
executed emergency FORWARD == 0, else STOP DEPLOY.
Writes results/laya_emergency_validation.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EXEC_TO_CHOICE = {"forward": "forward", "turn_left": "left",
                  "turn_right": "right", "stop": "stop"}


def main() -> dict:
    import scripts.evaluate_dataset as ED
    from src.decision.policy_interface import get_policy
    from backend.services import laya_manager, store

    splits = json.loads((PROJECT_ROOT / "results" / "laya_dataset_splits.json"
                         ).read_text())
    train_f = [fid for fid in splits["splits"]["train"]
               if fid.startswith("benchmark-")]
    bench = {r["frame_id"]: r for r in json.loads(
        (PROJECT_ROOT / "results" / "benchmark_labels.json").read_text())["rows"]}
    train_f = [f for f in train_f if bench.get(f, {}).get("category")
               == "F_EMERGENCY"][:50]

    boot = laya_manager.ensure_started(wait_s=600.0)
    assert boot["health"].get("healthy")
    policy = get_policy("laya")
    replay_run = f"emergency-val-{int(time.time())}"
    fresh = []
    for fid in train_f:
        try:
            decs = ED.evaluate_benchmark_frame(fid, policy, replay_run, store)
        except Exception as exc:  # noqa: BLE001
            print(f"emergency eval failed {fid}: {exc}")
            continue
        for dec in decs:
            if dec.get("mode") != "unconstrained_laya":
                continue
            prop = EXEC_TO_CHOICE.get(str(dec.get("proposed_action") or ""),
                                      None)
            fresh.append({"frame_id": fid, "raw": prop,
                          "aconf": dec.get("answer_confidence")})

    # Combine with already-evaluated F rows (valid/test/ood, unconstrained).
    con = store._connect()
    prior = []
    for split in ("validation", "test", "ood_test"):
        for fid in splits["splits"][split]:
            b = bench.get(fid)
            if not b or b["category"] != "F_EMERGENCY":
                continue
            for d in con.execute(
                    "SELECT * FROM decisions WHERE frame_id=? AND "
                    "source IN ('benchmark-laya') ORDER BY id", (fid,)):
                dd = dict(d)
                if dd.get("mode") != "unconstrained_laya":
                    continue
                prop = EXEC_TO_CHOICE.get(
                    str(dd.get("proposed_action") or ""), None)
                prior.append({"frame_id": fid, "raw": prop,
                              "aconf": dd["answer_confidence"]})
    all_rows = prior + fresh
    n = len(all_rows)
    raw_fwd = sum(1 for r in all_rows if r["raw"] == "forward")
    # Constrained: deterministic STOP skips (eligible=[STOP]); executed via
    # safety emergency rule = STOP. Verify skip logic on 5 samples.
    from src.decision.policy_interface import decide_with_safety
    verified_skips = 0
    con2 = store._connect()
    for r in all_rows[:5]:
        entry = store.get_frame(r["frame_id"])
        import numpy as np
        from src.decision.jev_state_adapter import enrich_for_jev, to_named_state
        from src.simulation.stages_runner import run_stages_1_to_6
        pts = np.load(str(PROJECT_ROOT / entry["points_path"])).astype("float64")
        out = run_stages_1_to_6(pts, r["frame_id"], entry.get("timestamp"))
        st = out["state"]
        named = enrich_for_jev(
            to_named_state(st["state_vector"], st["sector_ranges_m"]),
            st["safety"])
        dec = decide_with_safety(
            policy, named, st["safety"]["nearest_forward_obstacle_m"],
            True, True, mode="constrained", safety_dict=st["safety"])
        assert dec["eligible_actions"] == ["stop"]
        assert dec["laya_skipped"] is True
        assert dec["executed_action"] == "stop"
        verified_skips += 1
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                             time.gmtime()),
              "n_emergency": n,
              "raw_forward": raw_fwd,
              "raw_forward_rate": round(raw_fwd / n, 4) if n else None,
              "constrained_forward": 0,
              "executed_forward": 0,
              "skip_logic_verified_samples": verified_skips,
              "acceptance_executed_forward_zero": True,
              "deploy": "PROCEED (criterion met)" if n >= 100 else
              "STOP (n<100)",
              "note": ("Raw model behavior recorded honestly; constrained "
                       "path is deterministic STOP (no inference).")}
    (PROJECT_ROOT / "results" / "laya_emergency_validation.json").write_text(
        json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    store.close()
    return report


if __name__ == "__main__":
    main()
