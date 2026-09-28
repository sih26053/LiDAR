"""Local Laya server lifecycle manager (independent-service architecture).

Production architecture: the Laya server is an INDEPENDENT process
(scripts/run_laya_server.py, docker-compose `laya`, or a platform
service). The backend NEVER owns its lifetime in production: it
connects, health-checks, reports URL/PID/uptime/checkpoint/device,
and reconnects after its own restarts while Laya keeps running.

Development convenience: ensure_started() reuses a healthy server when
one exists (including an independently started one — never spawning
duplicates); only when none is reachable does it spawn a managed child,
and the child loads the pinned LOCAL checkpoint (no floating latest).

Supervision state machine: READY | DEGRADED | STARTING | ERROR | STOPPED.
Restart uses bounded backoff (never an infinite crash loop); every
transition is reported with PID, uptime, restart count, checkpoint
revision, device, and offline availability. Loopback-only binding.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("paradox.backend.laya_manager")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PYTHON = str(PROJECT_ROOT / ".venv-pb" / "Scripts" / "python.exe")

_lock = threading.Lock()
_proc: subprocess.Popen | None = None
_last_error: str | None = None
_restarts = 0
_managed_started_utc: str | None = None


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _server_cfg() -> Dict[str, Any]:
    try:
        from src.decision.laya_client import load_laya_config

        return dict(load_laya_config().get("server", {}) or {})
    except (OSError, ValueError, AttributeError):
        return {}


def pid_file() -> Path:
    return PROJECT_ROOT / str(
        _server_cfg().get("pid_file", "models/laya/laya_server.pid"))


def _env() -> Dict[str, str]:
    from src.decision.laya_client import load_laya_config

    cfg = load_laya_config()
    env = dict(os.environ)
    env.setdefault("LAYA_HOST", str(cfg.get("host", "127.0.0.1")))
    env.setdefault("LAYA_PORT", str(cfg.get("port", 18001)))
    env.setdefault("LAYA_PRELOAD", "1")
    models = cfg.get("preload_models") or ["typed-decisions"]
    env.setdefault("LAYA_MODELS", ",".join(models))
    env.setdefault("LAYA_LOG_LEVEL", "warning")
    return env


def _endpoints() -> tuple[str, str]:
    from src.decision.laya_client import resolve_endpoint, resolve_health_endpoint

    ep, _ = resolve_endpoint()
    return ep, resolve_health_endpoint()


def health(timeout_s: float = 5.0) -> Dict[str, Any]:
    """Probe the local server (never raises)."""
    _, health_ep = _endpoints()
    try:
        with urllib.request.urlopen(health_ep, timeout=timeout_s) as r:
            payload = r.read().decode()[:2000]
        try:
            import json as _json

            data = _json.loads(payload)
        except Exception:
            data = {"raw": payload[:200]}
        data["healthy"] = True
        data["endpoint"] = health_ep
        return data
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return {"healthy": False,
                "reason": f"{type(exc).__name__}: {str(exc)[:200]}",
                "endpoint": health_ep}


def is_running() -> bool:
    return bool(health(timeout_s=2.0).get("healthy"))


def read_pid_info() -> Dict[str, Any] | None:
    """PID metadata written by the standalone server (independent or not)."""
    meta = pid_file().with_suffix(".json")
    try:
        return json.loads(meta.read_text())
    except (OSError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        # OSError for dead PIDs; SystemError on Windows (WinError 87)
        # for PIDs that no longer exist. Either way: not alive.
        return False


def server_pid() -> int | None:
    """PID of the serving process: managed child first, else pid file."""
    global _proc
    with _lock:
        proc = _proc
    if proc is not None and proc.poll() is None:
        return proc.pid
    info = read_pid_info()
    if info and isinstance(info.get("pid"), int) \
            and _pid_alive(info["pid"]):
        return int(info["pid"])
    return None


def server_uptime_s() -> float | None:
    """Uptime of the serving process (independent of backend restarts)."""
    info = read_pid_info()
    if not info or not info.get("started_utc"):
        return None
    try:
        import calendar

        started = calendar.timegm(time.strptime(
            info["started_utc"], "%Y-%m-%dT%H:%M:%SZ"))
        return max(0.0, time.time() - started)
    except (ValueError, TypeError, OverflowError):
        return None


def ensure_started(wait_s: float | None = None,
                   allow_spawn: bool = True) -> Dict[str, Any]:
    """Reuse a healthy server or (dev convenience) spawn a managed child.

    Production callers pass allow_spawn=False: the independent service
    must already be up; a missing server is reported, never spawned.
    Returns {"started", "reused", "managed", "health"}.
    Raises RuntimeError with a human-readable reason on failure.
    """
    from src.decision.laya_client import load_laya_config

    global _proc, _last_error, _restarts, _managed_started_utc
    with _lock:
        h = health(timeout_s=3.0)
        if h.get("healthy"):
            managed = _proc is not None and _proc.poll() is None
            return {"started": False, "reused": True,
                    "managed": managed, "health": h}
        if not allow_spawn:
            raise RuntimeError(
                "Laya server not reachable and allow_spawn=False "
                "(production mode expects the independent service). "
                f"Start it: .venv-pb python scripts/run_laya_server.py "
                f"-- last: {h.get('reason', 'no response')}")
        if _proc is not None and _proc.poll() is None:
            pass  # previous spawn still warming; wait below
        else:
            _spawn_locked()
        timeout = float(wait_s if wait_s is not None
                        else load_laya_config().get("startup_timeout_s", 600))
        t0 = time.time()
        last: Dict[str, Any] = {}
        proc = _proc
        while time.time() - t0 < timeout:
            if proc is not None and proc.poll() is not None:
                _last_error = (f"laya server exited during startup "
                               f"(code {proc.poll()}); check CPU deps/log")
                _proc = None
                raise RuntimeError(_last_error)
            last = health(timeout_s=5.0)
            if last.get("healthy"):
                _last_error = None
                return {"started": True, "reused": False,
                        "managed": True, "health": last}
            time.sleep(5.0)
        raise RuntimeError(
            f"Laya not READY after {timeout:.0f}s: {last.get('reason', 'no response')}")


def _spawn_locked() -> None:
    """Spawn the standalone server entry as a managed child (dev mode)."""
    global _proc, _restarts, _managed_started_utc
    env = _env()
    host = env["LAYA_HOST"]
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise RuntimeError(
            f"refusing non-loopback LAYA_HOST={host!r} (loopback-only policy)")
    from src.decision.checkpoint_info import checkpoint_info

    info = checkpoint_info()
    if info["state"] != "LOCAL":
        raise RuntimeError(
            "Laya checkpoint: MISSING (run scripts/provision_laya.py first). "
            "Refusing to start a server that would silently download.")
    python = PYTHON if Path(PYTHON).is_file() else sys.executable
    try:
        _proc = subprocess.Popen(
            [python, "scripts/run_laya_server.py",
             "--port", env["LAYA_PORT"], "--offline"],
            cwd=str(PROJECT_ROOT), env=env, stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT)
        _managed_started_utc = _utc()
    except Exception as exc:
        _last_error = f"spawn failed: {exc}"
        raise RuntimeError(_last_error) from exc


def restart(wait_s: float = 60.0) -> Dict[str, Any]:
    """Restart the MANAGED child with bounded backoff (dev convenience).

    Never touches an independently started server (refuses clearly).
    Bounded: one attempt per call; repeated failures surface as ERROR
    with restart_count instead of looping forever.
    """
    global _proc, _restarts, _last_error
    with _lock:
        managed = _proc is not None
    if not managed:
        if is_running():
            return {"restarted": False,
                    "reason": "server is independently managed; "
                              "restart it via its own supervisor"}
        return {"restarted": False, "reason": "no managed process to restart"}
    backoff = [1, 2, 4, 8, 30]
    delay = backoff[min(_restarts, len(backoff) - 1)]
    time.sleep(delay)
    stop()
    try:
        out = ensure_started(wait_s=wait_s)
        _restarts += 1
        out["restart_count"] = _restarts
        out["restarted"] = True
        return out
    except RuntimeError as exc:
        _restarts += 1
        _last_error = str(exc)[:300]
        return {"restarted": False, "reason": _last_error,
                "restart_count": _restarts}


def stop() -> Dict[str, Any]:
    """Terminate the MANAGED child only (independent servers untouched)."""
    global _proc
    with _lock:
        proc, _proc = _proc, None
    if proc is None:
        return {"stopped": False,
                "reason": "no managed process (independent server untouched)"}
    try:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()
        return {"stopped": True, "managed": True}
    except Exception as exc:  # noqa: BLE001
        return {"stopped": False, "reason": str(exc)[:200]}


def status() -> Dict[str, Any]:
    """Manager + server + device + checkpoint state for the dashboard."""
    from src.decision import calibration as _cal
    from src.decision.checkpoint_info import checkpoint_info
    from src.decision.laya_client import detect_device, resolve_model

    h = health(timeout_s=3.0)
    alive = _proc is not None and _proc.poll() is None
    pid = server_pid()
    info = checkpoint_info()
    if h.get("healthy"):
        state = "READY"
    elif alive:
        state = "STARTING"
    elif pid is not None:
        state = "DEGRADED"  # pid file claims life but health fails
    else:
        state = "ERROR" if _last_error else "STOPPED"
    return {"state": state, "healthy": bool(h.get("healthy")),
            "managed_process": alive,
            "independent": (pid is not None and not alive),
            "pid": pid,
            "uptime_s": server_uptime_s(),
            "restart_count": _restarts,
            "model": resolve_model(),
            "checkpoint": info["state"],
            "checkpoint_revision": info.get("revision"),
            "offline": info.get("offline_inference"),
            "device": detect_device(),
            "calibration": _cal.calibration_status(info.get("revision")),
            "health": h, "last_error": _last_error}


atexit.register(stop)
