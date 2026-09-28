"""Replay decision service (Phase 4-6): recorded frame -> Stages 1-6.

Uses the EXACT SAME functions as the live path (run_stages_1_to_6,
to_named_state, enrich_for_jev/enrich_for_laya, policy + gate, safety
layer). Engine-selectable: "laya" (ACTIVE, source=replay-laya) or
"jev" (LEGACY, source=replay-jev). No replay-only client, schema, or
LiDAR interpretation. Replay NEVER touches the live PyBullet vehicle:
no executor, no env; decisions are evaluations recorded under a
replay_run_id, leaving original live rows immutable.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from backend.services.replay_service import DataLoadError, FrameNotFoundError

logger = logging.getLogger("paradox.backend.replay_jev")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPLAY_LOG_PATH = PROJECT_ROOT / "results" / "decisions" / "replay_jev_requests.jsonl"
REPLAY_LAYA_LOG_PATH = PROJECT_ROOT / "results" / "decisions" / "replay_laya_requests.jsonl"

_ENGINE_SOURCE = {"laya": "replay-laya", "jev": "replay-jev"}


def load_recorded_points(frame_id: str) -> tuple[np.ndarray, Dict[str, Any]]:
    """Load N x 4 points for a recorded-live frame (DB) or nuScenes frame.

    Returns (points, meta). Raises FrameNotFoundError / DataLoadError
    with the same semantics as the replay service.
    """
    from backend.services import store

    fid = str(frame_id)
    entry = store.get_frame(fid)
    if entry is not None:
        path = PROJECT_ROOT / str(entry["points_path"])
        if not path.is_file():
            raise DataLoadError(f"Recorded points missing for frame {fid}: {path}")
        try:
            points = np.load(path).astype(np.float64)
        except Exception as exc:
            raise DataLoadError(f"Cannot read recorded frame {fid}: {exc}") from exc
        if points.ndim != 2 or points.shape[1] != 4:
            raise DataLoadError(
                f"Recorded frame {fid} has bad shape {points.shape}, expected N x 4")
        if len(points) == 0:
            raise DataLoadError(f"Recorded frame {fid} is empty")
        meta = {"frame_id": fid, "run_id": entry.get("run_id"),
                "sequence": entry.get("sequence"),
                "scene_id": entry.get("scenario") or entry.get("run_id"),
                "scenario": entry.get("scenario"),
                "timestamp": entry.get("timestamp"),
                "source_path": entry["points_path"],
                "replay_reference": "live-recorded",
                "origin": "live-recorded"}
        logger.info("frame=%s stage=loading status=success points=%d origin=live-recorded",
                    fid, len(points))
        return points, meta
    from backend.services import replay_service

    points, meta = replay_service.load_frame_points(fid)
    meta["origin"] = "nuscenes"
    return points, meta


def run_replay_decision(frame_id: str, replay_run_id: str,
                        semantic_mode: str = "model",
                        log_path: str | Path | None = None,
                        engine: str = "laya",
                        mode: str = "constrained") -> Dict[str, Any]:
    """One recorded frame -> decision record (no vehicle commands).

    Pipeline: load N x 4 -> Stages 1-6 -> named -> enrich ->
    eligibility -> policy (constrained subset or unconstrained) ->
    confidence gate -> safety -> persist (SQLite). Failures are
    recorded, never fabricated. Engine "laya" (active) records
    source=replay-laya; "jev" (legacy) records source=replay-jev.

    `mode`: "constrained" (production behaviour) or "unconstrained"
    (diagnostics: raw model preference recorded honestly). The mode is
    recorded on every row. Replay NEVER touches the live PyBullet
    vehicle: no executor, no env.
    """
    from src.decision.jev_state_adapter import enrich_for_jev, to_named_state
    from src.decision.policy_interface import decide_with_safety, get_policy
    from src.simulation.stages_runner import run_stages_1_to_6

    from backend.services import store

    engine = (engine or "laya").lower()
    if engine not in ("laya", "jev"):
        raise ValueError("engine must be 'laya' or 'jev'.")
    mode = str(mode or "constrained").lower()
    if mode not in ("constrained", "unconstrained"):
        raise ValueError("mode must be 'constrained' or 'unconstrained'.")
    source = _ENGINE_SOURCE[engine]
    if log_path is None:
        log_path = REPLAY_LAYA_LOG_PATH if engine == "laya" else REPLAY_LOG_PATH
    fid = str(frame_id)
    t0 = time.perf_counter()
    points, meta = load_recorded_points(fid)
    out = run_stages_1_to_6(points, fid, meta.get("timestamp"),
                            semantic_mode=semantic_mode)
    st = out["state"]
    named = to_named_state(st["state_vector"], st["sector_ranges_m"])
    named = enrich_for_jev(named, st["safety"])
    dec = decide_with_safety(
        get_policy(engine), named, st["safety"]["nearest_forward_obstacle_m"],
        True, bool(st["safety"]["emergency_stop"]), frame_id=fid,
        run_id=replay_run_id, source=source,
        log_path=str(log_path), mode=mode, safety_dict=st["safety"])
    verdict_status = dec["safety_status"]
    store.record_decision(
        fid, source, model=dec.get("model"),
        proposed_action=dec.get("proposed_action"),
        confidence=dec.get("confidence"),
        probabilities=dec.get("probabilities"),
        latency_ms=dec.get("policy_latency_ms"),
        request_id=f"{replay_run_id}:{fid}",
        error=dec.get("engine_error", dec.get("jev_error")),
        replay_run_id=replay_run_id,
        answer_confidence=dec.get("answer_confidence"),
        mode=dec.get("mode"),
        eligible=dec.get("eligible_actions"),
        raw_action=dec.get("raw_laya_action"),
        constrained_action=dec.get("constrained_action"),
        checkpoint_revision=dec.get("checkpoint_revision"),
        gate_threshold=dec.get("gate_threshold"))
    store.record_safety_event(
        fid, verdict_status,
        override=bool(dec.get("safety_override")),
        reason=dec.get("safety_reason"),
        emergency_flag=bool(st["safety"]["emergency_stop"]),
        nearest_obstacle_m=st["safety"]["nearest_forward_obstacle_m"],
        forward_clearance_m=named.get("forward_clearance_m"),
        left_clearance_m=named.get("left_clearance_m"),
        right_clearance_m=named.get("right_clearance_m"),
        replay_run_id=replay_run_id)
    total_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return {
        "frame_id": fid,
        "origin": meta.get("origin"),
        "source": source,
        "engine": engine,
        "mode": dec.get("mode"),
        "replay_run_id": replay_run_id,
        "action": dec.get("proposed_action"),
        "confidence": dec.get("confidence"),
        "answer_confidence": dec.get("answer_confidence"),
        "probabilities": dec.get("probabilities"),
        "latency_ms": dec.get("policy_latency_ms"),
        "total_ms": total_ms,
        "eligible_actions": dec.get("eligible_actions"),
        "category": dec.get("category"),
        "raw_laya_action": dec.get("raw_laya_action"),
        "raw_laya_confidence": dec.get("raw_laya_confidence"),
        "constrained_action": dec.get("constrained_action"),
        "laya_skipped": dec.get("laya_skipped"),
        "gate_metric": dec.get("gate_metric"),
        "gate_threshold": dec.get("gate_threshold"),
        "checkpoint_revision": dec.get("checkpoint_revision"),
        "calibration_version": dec.get("calibration_version"),
        "safety": {
            "status": verdict_status,
            "override": bool(dec.get("safety_override")),
            "reason": dec.get("safety_reason"),
        },
        "gate": dec.get("gate"),
        "engine_status": dec.get("engine_status", dec.get("jev_status")),
        "jev_status": dec.get("jev_status"),
        "error": dec.get("engine_error", dec.get("jev_error")),
        "stored": True,
    }


def get_replay_history(frame_id: str, source: str = "replay-laya"):
    """Replay rows for a frame (default active engine source)."""
    from backend.services import store

    return store.get_replay_history(frame_id, source=source)


def run_batch(frame_ids: List[str] | None = None,
              run_id: str | None = None,
              limit: int | None = None,
              replay_run_id: str | None = None,
              semantic_mode: str = "model",
              engine: str = "laya",
              mode: str = "constrained") -> Dict[str, Any]:
    """Sequential replay over frames (controlled: one engine call each)."""
    from backend.services import store

    engine = (engine or "laya").lower()
    prefix = "replay-laya" if engine == "laya" else "replay"
    replay_run_id = replay_run_id or f"{prefix}-{int(time.time())}"
    if frame_ids is None:
        if not run_id:
            raise ValueError("Provide frame_ids or run_id.")
        frame_ids = [f["frame_id"] for f in store.list_frames(run_id=run_id)]
    if limit is not None:
        frame_ids = list(frame_ids)[:max(1, int(limit))]
    results, errors = [], []
    for fid in frame_ids:
        t0 = time.time()
        try:
            rec = run_replay_decision(fid, replay_run_id,
                                      semantic_mode=semantic_mode,
                                      engine=engine, mode=mode)
            results.append(rec)
            logger.info("replay frame=%s action=%s conf=%s latency_ms=%s",
                        fid, rec["action"], rec["confidence"], rec["latency_ms"])
        except Exception as exc:  # noqa: BLE001 - recorded, batch continues
            err = f"{type(exc).__name__}: {str(exc)[:300]}"
            logger.warning("replay frame=%s failed: %s", fid, err)
            errors.append({"frame_id": fid, "error": err,
                           "elapsed_s": round(time.time() - t0, 2)})
    return {"replay_run_id": replay_run_id, "results": results,
            "errors": errors, "ran": len(results),
            "requested": len(frame_ids)}
