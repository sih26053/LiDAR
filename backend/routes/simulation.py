"""Closed-loop simulation + RL telemetry APIs (offline replay loop).

Simulation runs the genuine autonomy chain per frame; CARLA-gated
training endpoints do not exist because no simulator is installed.
Unmeasurable quantities (collisions, goals) are null, never invented.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.services import simulation_service
from src import rl_agent

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

router = APIRouter()


class StartBody(BaseModel):
    frame_ids: list[str] | None = None
    semantic_mode: str = Field(default="model", pattern="^(annotation|model)$")


def _ok_or_400(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"message": str(exc)[:400]})


@router.get("/rl/status")
def rl_status():
    agent = rl_agent.get_agent()
    return {
        "dqn": rl_agent.describe(),
        "replay_capacity": len(agent.buffer),
        "stored_updates": agent.updates,
        "training": "blocked (no CARLA/simulator; train_episode refuses)",
        "safety": "rule-based override live (src/safety_controller.py)",
        "executor": "simulation default; hardware disabled",
    }


@router.get("/rl/state")
def rl_state():
    last = simulation_service.last_state()
    if not last:
        return JSONResponse(status_code=404, content={"message": "No simulation step yet; POST /simulation/start then /simulation/step."})
    return {"frame_id": last["frame_id"], "state_dim": last["state_dim"],
            "q_values": last["q_values"], "stages_executed": last["stages_executed"]}


@router.get("/rl/action")
def rl_action():
    last = simulation_service.last_state()
    if not last:
        return JSONResponse(status_code=404, content={"message": "No simulation step yet."})
    return {"frame_id": last["frame_id"], "network_action": last["network_action"],
            "final_action": last["final_action"], "safety_verdict": last["safety_verdict"],
            "trained": False}


@router.post("/rl/step")
def rl_step():
    return _ok_or_400(simulation_service.step)


@router.get("/rl/episode")
def rl_episode():
    return simulation_service.status()


@router.get("/rl/metrics")
def rl_metrics():
    return simulation_service.metrics()


@router.get("/simulation/status")
def simulation_status():
    from src.simulation.carla_lidar import status as carla_status

    body = simulation_service.status()
    body["carla"] = carla_status()
    return body


@router.post("/simulation/start")
def simulation_start(body: StartBody):
    return _ok_or_400(simulation_service.start, body.frame_ids, body.semantic_mode)


@router.post("/simulation/stop")
def simulation_stop():
    return simulation_service.stop()


@router.post("/simulation/reset")
def simulation_reset():
    return _ok_or_400(simulation_service.reset)


@router.post("/simulation/step")
def simulation_step():
    return _ok_or_400(simulation_service.step)


class PyBulletRunBody(BaseModel):
    steps: int = Field(default=20, ge=1, le=500)
    decision_mode: str = Field(default="laya", pattern="^(laya|jev|dqn)$")

@router.post("/simulation/run")
def simulation_run(body: PyBulletRunBody):
    """Attempt a live PyBullet closed loop (BLOCKED here: no pybullet wheel).

    Executes for real where the dependency exists; otherwise returns the
    honest BLOCKED trace produced by scripts/run_pybullet_closed_loop.py.
    """
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.run_pybullet_closed_loop import main

    trace = main(steps=body.steps, gui=False)
    summary = {k: v for k, v in trace.items() if k != "steps"}
    summary["n_steps_recorded"] = len(trace.get("steps", []))
    summary["decision_mode_requested"] = body.decision_mode
    return summary


@router.get("/decision/current")
def decision_current():
    from backend.services import decision_store

    cur = decision_store.current()
    if cur is None:
        return JSONResponse(status_code=404, content={
            "message": "No decision recorded yet (no closed-loop step executed)."})
    return cur


@router.get("/decision/history")
def decision_history(limit: int = 50):
    from backend.services import decision_store

    return {"count": decision_store.count(), "records": decision_store.history(limit)}


@router.get("/metrics/current")
def metrics_current():
    from backend.services import decision_store, result_service

    mets = simulation_service.metrics()
    mets["decisions_recorded"] = decision_store.count()
    mets["last_frame_id"] = result_service.last_state().get("last_frame_id")
    return mets


@router.get("/simulation/pybullet")
def simulation_pybullet():
    """Live PyBullet backend state (measured now, never assumed)."""
    from src.simulation import simulator

    state = simulator.active_backend()
    try:
        from src.decision.laya_client import service_status as laya_status
        laya = laya_status()
    except Exception as exc:  # noqa: BLE001 - report
        laya = {"available": False, "reason": f"status probe failed: {exc}"}
    return {"simulator": "pybullet (active)" if state["pybullet_importable"] else "pybullet (BLOCKED: not importable here)",
            "decision_backend": "laya (LOCAL)" if laya["available"] else "laya (STARTING/ERROR) / dqn-untrained baseline",
            "decision_engine": "laya",
            "pybullet_importable": state["pybullet_importable"],
            "carla_importable": state["carla_importable"],
            "laya": laya}


class LiveStartBody(BaseModel):
    mode: str = Field(default="autonomous", pattern="^(autonomous|manual)$")
    decision_interval_steps: int = Field(default=5, ge=1, le=100)
    scenario_id: str | None = Field(default=None, max_length=64)


@router.post("/simulation/live/start")
def live_start(body: LiveStartBody):
    """Start the in-process PyBullet live loop (real env or honest BLOCKED)."""
    from backend.services import pybullet_live

    try:
        return pybullet_live.start(mode=body.mode,
                                   decision_interval_steps=body.decision_interval_steps,
                                   scenario_id=body.scenario_id)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"message": str(exc)[:400]})


@router.post("/simulation/live/stop")
def live_stop():
    from backend.services import pybullet_live

    return pybullet_live.stop()


@router.post("/simulation/live/reset")
def live_reset():
    from backend.services import pybullet_live

    return pybullet_live.reset()


@router.get("/simulation/state")
def simulation_state():
    """Current live-loop snapshot (404 when the live loop never stepped)."""
    from backend.services import pybullet_live

    snap = pybullet_live.snapshot()
    if snap is None:
        return JSONResponse(status_code=404, content={
            "message": "No live state yet; POST /simulation/live/start first."})
    return snap


@router.post("/simulation/action/{action}")
def manual_action(action: str):
    """One safety-checked manual step; recorded with source=manual (never Laya)."""
    from backend.services import pybullet_live

    try:
        return pybullet_live.manual(action)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"message": str(exc)[:400]})
