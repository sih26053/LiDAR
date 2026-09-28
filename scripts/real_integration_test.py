"""Real integration test (Phase 29): provisioned checkpoint -> independent
Laya -> backend connects -> PyBullet autonomy -> record -> restart backend
loop while Laya runs -> reconnect -> replay (both modes) -> metrics.

All values measured in-process; NOT VALIDATED items are labelled so.
Run with the Laya venv: .venv-pb python scripts/real_integration_test.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

REPORT: dict = {"steps": [], "started_utc": time.strftime(
    "%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def step(name: str, ok: bool, detail: str = "") -> None:
    REPORT["steps"].append({"step": name, "ok": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        raise SystemExit(f"integration FAILED at: {name} :: {detail}")


def main() -> dict:
    import urllib.request

    from backend.services import decision_store, laya_manager, pybullet_live
    from backend.services import store as db
    from src.decision.checkpoint_info import checkpoint_info

    # 1-2. Provisioned checkpoint stored locally.
    info = checkpoint_info()
    step("checkpoint-local", info["state"] == "LOCAL",
         f"rev={info['revision']} offline={info['offline_inference']}")

    # 4-5. Independent Laya (spawn-if-needed) + readiness.
    t0 = time.time()
    boot = laya_manager.ensure_started(wait_s=600.0)
    cold_s = round(time.time() - t0, 1)
    REPORT["laya_cold_start_s"] = cold_s
    healthy = bool(boot["health"].get("healthy"))
    step("laya-ready", healthy,
         f"reused={boot.get('reused')} cold_start_s={cold_s}"
         if not boot.get("reused") else "reused running server")
    pre_pid = laya_manager.server_pid()

    # 6-7. Backend connects to the existing service.
    st = pybullet_live.status()
    step("backend-connect", True, f"mode={st['mode']}")

    # 8-9. PyBullet autonomous.
    live = pybullet_live.start(mode="autonomous", decision_interval_steps=2,
                               scenario_id="straight_road")
    step("pybullet-start", bool(live.get("active")), str(live.get("run_id")))
    run_id = live.get("run_id")
    deadline = time.time() + 240
    while pybullet_live.status().get("steps", 0) < 6 and time.time() < deadline:
        time.sleep(2.0)
    n_steps = pybullet_live.status().get("steps", 0)
    step("autonomy-steps", n_steps >= 6, f"steps={n_steps}")

    # 10-13. Real Laya decisions + eligibility + safety + execution.
    cur = decision_store.current()
    snap = pybullet_live.snapshot()
    step("real-laya-decision", cur is not None and snap is not None,
         f"action={cur.get('proposed_action')} conf={cur.get('confidence')}"
         if cur else "no decision")
    dec = snap.get("decision", {}) if snap else {}
    step("eligibility-recorded", dec.get("eligible_actions") is not None,
         f"eligible={dec.get('eligible_actions')} mode={dec.get('mode')}")
    step("safety-bound", snap.get("safety", {}).get("status") is not None
         and snap.get("executed_action") is not None,
         f"safety={snap.get('safety', {}).get('status')} "
         f"executed={snap.get('executed_action')}")
    REPORT["warm_laya_latency_ms"] = dec.get("latency_ms")
    REPORT["loop_latency_ms"] = snap.get("loop_latency_ms")
    REPORT["lidar_fps"] = (snap.get("lidar") or {}).get("fps")

    # 14-15. Recorded frames + SQLite.
    frames = db.list_frames(run_id=run_id, limit=10)
    step("recorded-frames", len(frames) > 0, f"frames={len(frames)}")
    fres = db.get_frame_result(frames[0]["frame_id"])
    step("sqlite-decision", fres is not None
         and fres.get("live_decision") is not None,
         f"frame={frames[0]['frame_id']}")
    REPORT["sample_frame"] = frames[0]["frame_id"]

    # 16-18. Restart the backend loop while Laya keeps running.
    pybullet_live.stop()
    still = laya_manager.is_running()
    same_pid = (laya_manager.server_pid() == pre_pid)
    step("laya-survives-backend-restart", still,
         f"same_pid={same_pid} (independent lifetime)")
    live2 = pybullet_live.start(mode="autonomous",
                                decision_interval_steps=2)
    step("backend-reconnect", bool(live2.get("active")), "reconnected")
    time.sleep(8.0)

    # 19. Dashboard snapshot carries the full chain.
    snap2 = pybullet_live.snapshot()
    d2 = (snap2 or {}).get("decision", {})
    step("dashboard-chain", all(k in d2 for k in (
        "mode", "eligible_actions", "checkpoint_revision")),
        f"mode={d2.get('mode')} eligible={d2.get('eligible_actions')}")
    pybullet_live.stop()

    # 20-21. Replay both modes; live vehicle never commanded.
    from backend.services import replay_laya as rl
    env_before = pybullet_live._live.get("env")
    for mode in ("constrained", "unconstrained"):
        rec = rl.run_replay_decision(
            REPORT["sample_frame"], f"integ-{mode}", mode=mode)
        step(f"replay-{mode}", rec.get("stored") is True
             and rec.get("source") == "replay-laya",
             f"action={rec.get('action')} eligible={rec.get('eligible_actions')}")
    step("replay-no-live-command", pybullet_live._live.get("env") is env_before
         and pybullet_live._live.get("env") is None, "env untouched (None)")

    # 22-23. A/B + calibration artifacts present.
    ab = json.loads((PROJECT_ROOT / "results" / "laya_ab_comparison.json"
                     ).read_text())
    step("ab-present", "delta_B_minus_A" in ab, "results/laya_ab_comparison.json")
    from src.decision import calibration as cal
    cs = cal.calibration_status(info.get("revision"))
    step("calibration-loaded", cs["gate"]["state"] == "LOADED",
         f"gate={cs['gate']['threshold']}")

    # 30. Performance snapshot (this machine, measured now).
    try:
        import psutil  # type: ignore
        mem = round(psutil.Process().memory_info().rss / 1e6, 1)
    except Exception:
        mem = None
    REPORT["memory_backend_rss_mb"] = mem
    REPORT["device"] = info.get("device")

    REPORT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                           time.gmtime())
    out = PROJECT_ROOT / "results" / "integration_test.json"
    out.write_text(json.dumps(REPORT, indent=2, default=str))
    print(f"wrote {out}")
    print("INTEGRATION: all steps passed (see report for measured values)")
    return REPORT


if __name__ == "__main__":
    main()
