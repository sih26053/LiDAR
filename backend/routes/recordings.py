"""Recorded live frames + replay-Jev endpoints (additive).

- GET /recordings/runs, /recordings/runs/{run_id}, /recordings/runs/{run_id}/frames
- GET /recordings/frames/{frame_id} (frame + live decision + replay history)
- GET /recordings/status (recorder + DB health)
- POST /replay/jev-decide {frame_id, replay_run_id?}
- POST /replay/jev-batch {run_id?, frame_ids?, limit?}

nuScenes replay paths are untouched; recorded frames carry
origin/source = live-recorded and are never mixed silently.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("paradox.backend.routes.recordings")

router = APIRouter()


class JevDecideBody(BaseModel):
    frame_id: str = Field(..., min_length=1)
    replay_run_id: str | None = Field(default=None, max_length=64)
    semantic_mode: str = Field(default="model", pattern="^(annotation|model)$")


class JevBatchBody(BaseModel):
    run_id: str | None = None
    frame_ids: list[str] | None = None
    limit: int | None = Field(default=None, ge=1, le=500)
    replay_run_id: str | None = Field(default=None, max_length=64)
    semantic_mode: str = Field(default="model", pattern="^(annotation|model)$")


@router.get("/recordings/status")
def recordings_status():
    from backend.services import frame_recorder, store

    st = frame_recorder.recorder_status()
    try:
        runs = store.list_runs(limit=1)
        st["db"] = {"path": str(store.db_path()), "ok": True,
                    "latest_run": (runs[0]["run_id"] if runs else None)}
    except Exception as exc:  # noqa: BLE001 - report, don't fail
        st["db"] = {"path": str(store.db_path()), "ok": False,
                    "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    return st


@router.get("/recordings/runs")
def recordings_runs(limit: int = 100):
    from backend.services import store

    try:
        return {"runs": store.list_runs(limit=limit)}
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"message": str(exc)[:300]})


@router.get("/recordings/runs/{run_id}")
def recordings_run(run_id: str):
    from backend.services import store

    rec = store.get_run(run_id)
    if rec is None:
        return JSONResponse(status_code=404, content={"message": f"Run not found: {run_id}"})
    rec["frame_count"] = len(store.list_frames(run_id=run_id, limit=5000))
    return rec


@router.get("/recordings/runs/{run_id}/frames")
def recordings_run_frames(run_id: str, limit: int = 500):
    from backend.services import store

    if store.get_run(run_id) is None:
        return JSONResponse(status_code=404, content={"message": f"Run not found: {run_id}"})
    return {"run_id": run_id, "frames": store.list_frames(run_id=run_id, limit=limit)}


@router.get("/recordings/frames/{frame_id}")
def recordings_frame(frame_id: str):
    from backend.services import store

    res = store.get_frame_result(frame_id)
    if res is None:
        return JSONResponse(status_code=404, content={"message": f"Frame not found: {frame_id}"})
    return res


@router.post("/replay/jev-decide")
def replay_jev_decide(body: JevDecideBody):
    import time

    from backend.services import replay_jev as rj
    from backend.services.replay_service import DataLoadError, FrameNotFoundError

    replay_run_id = body.replay_run_id or f"replay-{int(time.time())}"
    try:
        return rj.run_replay_decision(body.frame_id, replay_run_id,
                                      semantic_mode=body.semantic_mode)
    except FrameNotFoundError as exc:
        return JSONResponse(status_code=404, content={
            "stage": "input", "error_code": "FRAME_NOT_FOUND",
            "message": str(exc)[:400], "frame_id": body.frame_id})
    except DataLoadError as exc:
        return JSONResponse(status_code=500, content={
            "stage": "loading", "error_code": "DATA_LOAD_FAILED",
            "message": str(exc)[:400], "frame_id": body.frame_id})
    except Exception as exc:  # noqa: BLE001 - recorded downstream, report honestly
        logger.exception("replay jev failed frame=%s", body.frame_id)
        return JSONResponse(status_code=502, content={
            "stage": "decision", "error_code": "JEV_REPLAY_FAILED",
            "message": f"{type(exc).__name__}: {str(exc)[:300]}",
            "frame_id": body.frame_id, "source": "replay-jev",
            "stored": False})


@router.post("/replay/jev-batch")
def replay_jev_batch(body: JevBatchBody):
    from backend.services import replay_jev as rj

    try:
        return rj.run_batch(frame_ids=body.frame_ids, run_id=body.run_id,
                            limit=body.limit,
                            replay_run_id=body.replay_run_id,
                            semantic_mode=body.semantic_mode)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"message": str(exc)[:300]})


class LayaDecideBody(BaseModel):
    frame_id: str = Field(..., min_length=1)
    replay_run_id: str | None = Field(default=None, max_length=64)
    semantic_mode: str = Field(default="model", pattern="^(annotation|model)$")
    mode: str = Field(default="constrained",
                      pattern="^(constrained|unconstrained)$")


class LayaBatchBody(BaseModel):
    run_id: str | None = None
    frame_ids: list[str] | None = None
    limit: int | None = Field(default=None, ge=1, le=500)
    replay_run_id: str | None = Field(default=None, max_length=64)
    semantic_mode: str = Field(default="model", pattern="^(annotation|model)$")
    mode: str = Field(default="constrained",
                      pattern="^(constrained|unconstrained)$")


@router.get("/laya/status")
def laya_status():
    """Local Laya server state: READY/DEGRADED/STARTING/ERROR/STOPPED.

    Includes PID, uptime, restart count, checkpoint state/revision,
    offline availability, device, and calibration load state.
    """
    from backend.services import laya_manager

    return laya_manager.status()


@router.get("/laya/checkpoint")
def laya_checkpoint():
    """Pinned checkpoint identity + local provisioning state."""
    from src.decision.checkpoint_info import checkpoint_info

    return checkpoint_info()


@router.get("/laya/calibration")
def laya_calibration():
    """Gate selection + temperature artifacts + runtime load state."""
    from pathlib import Path as _P

    from src.decision import calibration as _cal
    from src.decision.checkpoint_info import checkpoint_info
    from src.decision.policy_interface import resolve_gate_config

    rev = checkpoint_info().get("revision")
    return {"gate_config": resolve_gate_config(),
            "runtime": _cal.calibration_status(rev),
            "gate_selection": _cal.load_gate_selection(),
            "temperature": _cal.load_temperature()}


@router.get("/laya/diagnostics")
def laya_diagnostics():
    """Measured navigation diagnostics (eval + A/B summaries, or NOT AVAILABLE)."""
    import json as _json
    from pathlib import Path as _P

    root = _P(__file__).resolve().parent.parent.parent
    out: dict = {}
    for key, name in (("navigation", "results/laya_navigation_evaluation.json"),
                      ("ab_comparison", "results/laya_ab_comparison.json")):
        path = root / name
        if path.is_file():
            try:
                payload = _json.loads(path.read_text())
                out[key] = payload.get("summary", payload)
            except (OSError, ValueError) as exc:
                out[key] = {"error": f"cannot read {name}: {exc}"}
        else:
            out[key] = {"status": "NOT AVAILABLE",
                        "note": f"run scripts/evaluate_laya_navigation.py ({name} missing)"}
    return out


@router.post("/replay/laya-decide")
def replay_laya_decide(body: LayaDecideBody):
    """Replay one recorded frame through Stages 1-6 + Laya + safety.

    source=replay-laya, stored separately; never drives the live vehicle.
    mode=constrained (production) or unconstrained (raw-model diagnostics).
    """
    import time

    from backend.services import replay_laya as rl
    from backend.services.replay_service import DataLoadError, FrameNotFoundError

    replay_run_id = body.replay_run_id or f"replay-laya-{int(time.time())}"
    try:
        return rl.run_replay_decision(body.frame_id, replay_run_id,
                                      semantic_mode=body.semantic_mode,
                                      mode=body.mode)
    except FrameNotFoundError as exc:
        return JSONResponse(status_code=404, content={
            "stage": "input", "error_code": "FRAME_NOT_FOUND",
            "message": str(exc)[:400], "frame_id": body.frame_id})
    except DataLoadError as exc:
        return JSONResponse(status_code=500, content={
            "stage": "loading", "error_code": "DATA_LOAD_FAILED",
            "message": str(exc)[:400], "frame_id": body.frame_id})
    except Exception as exc:  # noqa: BLE001 - report honestly
        logger.exception("replay laya failed frame=%s", body.frame_id)
        return JSONResponse(status_code=502, content={
            "stage": "decision", "error_code": "LAYA_REPLAY_FAILED",
            "message": f"{type(exc).__name__}: {str(exc)[:300]}",
            "frame_id": body.frame_id, "source": "replay-laya",
            "stored": False})


@router.post("/laya/restart")
def laya_restart():
    """Restart the MANAGED Laya child (dev convenience).

    Refuses clearly when the server is independently managed.
    """
    from backend.services import laya_manager

    return laya_manager.restart()


@router.get("/laya/validation")
def laya_validation():
    """Curated validation status: VERIFIED / NOT VALIDATED per item.

    Every value derives from measured artifacts; nothing is assumed.
    A badge is VERIFIED only with artifact evidence on disk.
    """
    import json as _json
    from pathlib import Path as _P

    from src.decision import calibration as _cal
    from src.decision.checkpoint_info import checkpoint_info

    root = _P(__file__).resolve().parent.parent.parent

    def _load(name: str):
        try:
            return _json.loads((root / name).read_text())
        except (OSError, ValueError):
            return None

    final = _load("results/laya_final_test.json")
    gate = _cal.load_gate_selection() or {}
    temp = _cal.load_temperature() or {}
    emer = _load("results/laya_emergency_validation.json")
    ood = _load("results/laya_ood.json")
    splits = _load("results/laya_dataset_splits.json")
    info = checkpoint_info()
    rev = info.get("revision")
    cal_state = _cal.calibration_status(rev)
    temp_groups = temp.get("groups", {}) if isinstance(temp, dict) else {}
    held_out = any(g.get("validation_n") for g in temp_groups.values()
                   if isinstance(g, dict))
    split_cats = ((splits.get("split_categories", {}) or {}).get("test", {})
                  if splits else {})
    e_count = sum(v for k, v in split_cats.items()
                  if "NO_SAFE" in str(k) or k == "E_NO_SAFE_DIRECTION")
    import shutil as _shutil
    has_docker = _shutil.which("docker") is not None
    return {
        "calibration": ("VERIFIED" if cal_state["gate"]["state"] == "LOADED"
                        and gate.get("promoted") else "NOT VALIDATED"),
        "test_n": ((final or {}).get("n_frames", (final or {}).get("n"))
                     if final else None),
        "e_category": ("OBSERVED" if e_count and e_count > 0
                       else "UNOBSERVED"),
        "e_test_n": e_count,
        "temperature": ("HELD-OUT" if cal_state["temperature"]["state"]
                        == "LOADED" and held_out
                        else ("IN-SAMPLE" if temp.get("fitted")
                              else "NOT VALIDATED")),
        "gate": {"threshold": gate.get("current_gate_threshold"),
                 "legacy": gate.get("legacy_starting_threshold"),
                 "status": ("VALIDATED-CANDIDATE" if gate.get("promoted")
                            else ("STARTING VALUE" if gate else
                                  "NOT VALIDATED"))},
        "forward_behavior": "MEASURED",
        "fine_tuning": ("PREPARED" if (root / "results" /
                                       "laya_finetuning_package" /
                                       "split_manifest.json").is_file()
                        else "NOT RUN"),
        "ood": ("VERIFIED" if ood and ood.get("n_ood") else "NOT VALIDATED"),
        "ood_n": (ood or {}).get("n_ood"),
        "docker": ("AVAILABLE-NOT-RUN" if has_docker else "NOT VALIDATED"),
        "docker_reason": ("Docker runtime unavailable on this host"
                          if not has_docker else
                          "no container executed yet"),
        "offline": info.get("offline_inference"),
        "checkpoint": info.get("state"),
        "revision": rev,
        "physical_hardware": "NOT EXECUTED",
        "emergency_executed_forward": ((emer or {}).get("executed_forward")
                                       if emer else None),
        "promotion": (_load("models/laya/promotions.json") or {}).get(
            "current_known_good", {}).get("stage"),
    }


@router.post("/replay/laya-batch")
def replay_laya_batch(body: LayaBatchBody):
    from backend.services import replay_laya as rl

    try:
        return rl.run_batch(frame_ids=body.frame_ids, run_id=body.run_id,
                            limit=body.limit,
                            replay_run_id=body.replay_run_id,
                            semantic_mode=body.semantic_mode,
                            mode=body.mode)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"message": str(exc)[:300]})
