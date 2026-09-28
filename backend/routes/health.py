"""GET /health -- API liveness, never a claim about pipeline health.

GET /system/environment -- live-detected hardware/software context for
benchmark interpretation. Values are detected at request time with the
standard library; anything that cannot be detected is reported as
"Unavailable" (never invented).
"""

from __future__ import annotations

import os
import platform
import sys
from typing import Any, Dict

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "paradox-protocol-backend"}


@router.get("/system/status")
def system_status() -> Dict[str, Any]:
    """Live system roll-up (measured now; BLOCKED rendered honestly)."""
    from src.simulation import simulator

    sim = simulator.active_backend()
    try:
        from src.decision.laya_client import resolve_model, service_status
        laya = service_status()
        laya_model = resolve_model()
    except Exception as exc:  # noqa: BLE001 - report
        laya = {"available": False, "reason": f"status probe failed: {exc}"}
        laya_model = "typed-decisions"
    try:
        from backend.services import pybullet_live
        live = pybullet_live.status()
    except Exception as exc:  # noqa: BLE001 - report
        live = {"active": False, "error": f"live probe failed: {exc}"}
    return {
        "backend": "ok",
        "simulator": "pybullet",
        "pybullet_importable": sim["pybullet_importable"],
        "decision_engine": "laya",
        "decision_model": laya_model,
        "decision_backend": "LOCAL",
        "laya": laya,
        "laya_available": bool(laya.get("available")),
        "laya_reason": laya.get("reason"),
        "live": live,
        "physical_testing": "NOT EXECUTED",
    }


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


@router.get("/system/environment")
def environment() -> Dict[str, Any]:
    """Detect benchmark environment live; "Unavailable" where undetectable."""
    try:
        cpu_model = platform.processor() or platform.machine() or "Unavailable"
    except Exception:
        cpu_model = "Unavailable"
    try:
        cpu_count = os.cpu_count()
    except Exception:
        cpu_count = None
    ram_gb: float | str = "Unavailable"
    try:
        import psutil  # type: ignore

        ram_gb = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        ram_gb = "Unavailable"
    gpu = "Unavailable (not queried — CPU-only numpy/pandas pipeline, torch not required)"
    return {
        "cpu_model": cpu_model if cpu_model else "Unavailable",
        "cpu_count": cpu_count if cpu_count else "Unavailable",
        "ram_gb": ram_gb,
        "gpu": gpu,
        "os": platform.platform() or "Unavailable",
        "python_version": sys.version.split()[0],
        "packages": {
            "numpy": _package_version("numpy") or "Unavailable",
            "pandas": _package_version("pandas") or "Unavailable",
            "fastapi": _package_version("fastapi") or "Unavailable",
            "uvicorn": _package_version("uvicorn") or "Unavailable",
            "nuscenes-devkit": _package_version("nuscenes-devkit") or "Unavailable",
            "pyquaternion": _package_version("pyquaternion") or "Unavailable",
            "torch": _package_version("torch") or "Not installed (pipeline uses only numpy/pandas)",
            "open3d": _package_version("open3d") or "Not installed (not required by the pipeline)",
        },
        "note": "Detected live at request time. Values describe the machine "
        "serving the current replay, not necessarily the machine that "
        "produced the stored benchmark.",
    }
