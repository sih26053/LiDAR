"""Audit recorded Laya rows for evaluation usability (read-only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> dict:
    from backend.services import store

    con = store._connect()
    out: dict = {}
    out["laya_rows"] = con.execute(
        "SELECT COUNT(*) c FROM decisions WHERE source='laya'").fetchone()["c"]
    out["laya_with_aconf_probs_rev"] = con.execute(
        "SELECT COUNT(*) c FROM decisions WHERE source='laya' "
        "AND answer_confidence IS NOT NULL AND probabilities_json IS NOT NULL "
        "AND checkpoint_revision IS NOT NULL").fetchone()["c"]
    out["laya_with_clearances"] = con.execute(
        "SELECT COUNT(*) c FROM decisions d JOIN safety_events s "
        "ON s.frame_id=d.frame_id AND s.replay_run_id IS NULL "
        "WHERE d.source='laya' AND d.answer_confidence IS NOT NULL "
        "AND s.forward_clearance_m IS NOT NULL").fetchone()["c"]
    out["revisions"] = [
        {"rev": r[0], "n": r[1]} for r in con.execute(
            "SELECT checkpoint_revision, COUNT(*) FROM decisions "
            "WHERE source='laya' GROUP BY checkpoint_revision")]
    out["modes"] = [
        {"mode": r[0], "n": r[1]} for r in con.execute(
            "SELECT mode, COUNT(*) FROM decisions WHERE source='laya' "
            "GROUP BY mode")]
    out["frames_with_state"] = con.execute(
        "SELECT COUNT(*) c FROM frames WHERE state_vector_json != '[]'"
        ).fetchone()["c"]
    # Sample one full row to see schema population.
    row = con.execute(
        "SELECT d.*, s.forward_clearance_m, s.left_clearance_m, "
        "s.right_clearance_m, s.emergency_flag FROM decisions d "
        "LEFT JOIN safety_events s ON s.frame_id=d.frame_id "
        "AND s.replay_run_id IS NULL WHERE d.source='laya' AND "
        "d.answer_confidence IS NOT NULL LIMIT 1").fetchone()
    out["sample_keys"] = sorted(dict(row).keys()) if row else []
    print(json.dumps(out, indent=2, default=str))
    store.close()
    return out


if __name__ == "__main__":
    main()
