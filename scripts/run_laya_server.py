"""Standalone Laya server (independent process, production architecture).

Laya runs HERE as its own service; the backend only connects to it
(health/readiness), never owns its lifetime:

    LAYA SERVER (this script, independent process/service)
        |--- local checkpoint (models/laya/checkpoint, pinned revision)
        |--- health/readiness, restart supervision, offline inference
        v
    Backend (connects via LAYA_ENDPOINT, reuses if already running)
        v
    Dashboard

Run:
    python scripts/run_laya_server.py [--port 18001] [--local-path ...]
        [--offline] [--device cpu] [--no-preload]

--offline sets HF_HUB_OFFLINE=1 so inference provably needs no network
(the local checkpoint must be provisioned; missing checkpoint FAILS
CLEARLY instead of silently downloading).

Writes a PID file (models/laya/laya_server.pid) so duplicate processes
are prevented and the backend can report PID/uptime. Logs to
models/laya/laya_server.log.

Must run with the Laya venv (.venv-pb).
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_server_cfg() -> dict:
    try:
        return json.loads(
            (PROJECT_ROOT / "config" / "laya_config.json").read_text()
        ).get("server", {}) or {}
    except (OSError, ValueError):
        return {}


def main() -> None:
    srv_cfg = _load_server_cfg()
    ap = argparse.ArgumentParser(description="Standalone Laya server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18001)
    ap.add_argument("--local-path", default="models/laya/checkpoint",
                    help="project-managed checkpoint root (provisioned)")
    ap.add_argument("--offline", action="store_true",
                    help="HF_HUB_OFFLINE=1: no Hub requests at runtime")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-preload", action="store_true")
    ap.add_argument("--pid-file", default=srv_cfg.get(
        "pid_file", "models/laya/laya_server.pid"))
    ap.add_argument("--log-file", default=srv_cfg.get(
        "log_file", "models/laya/laya_server.log"))
    args = ap.parse_args()

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit(
            f"refusing non-loopback host {args.host!r} (loopback-only policy)")

    from src.decision.checkpoint_info import read_manifest

    manifest = read_manifest()
    if manifest is None:
        raise SystemExit(
            "Laya checkpoint: MISSING (no models/laya/manifest.json). "
            "Run: python scripts/provision_laya.py  (no silent download here)")
    local_root = PROJECT_ROOT / args.local_path
    sub = local_root / manifest.get("subfolder", "typed-decisions")
    if not (sub / "model.safetensors").is_file():
        raise SystemExit(
            f"Laya checkpoint: MISSING ({sub} incomplete). "
            "Run: python scripts/provision_laya.py")

    if args.offline or os.environ.get("LAYA_OFFLINE") == "1":
        os.environ["HF_HUB_OFFLINE"] = "1"

    pid_file = PROJECT_ROOT / args.pid_file
    if pid_file.is_file():
        try:
            old_pid = int(pid_file.read_text().strip())
            try:
                os.kill(old_pid, 0)
                alive = True
            except Exception:
                # OSError/SystemError (incl. Windows WinError 87 for dead
                # PIDs): treat as stale.
                alive = False
            if alive:
                raise SystemExit(
                    f"another Laya server appears live (pid {old_pid}); "
                    f"refusing duplicate (remove {pid_file} if stale)")
        except SystemExit:
            raise
        except Exception:
            pass  # unreadable pid file; overwrite below
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    pid_file.with_suffix(".json").write_text(json.dumps({
        "pid": os.getpid(), "started_utc": started_utc,
        "revision": manifest.get("revision"),
        "local_path": str(local_root),
        "offline": os.environ.get("HF_HUB_OFFLINE") == "1",
        "endpoint": f"http://{args.host}:{args.port}/v1/systemone",
    }, indent=2))
    atexit.register(lambda: pid_file.unlink(missing_ok=True))

    os.environ["LAYA_HOST"] = args.host
    os.environ["LAYA_PORT"] = str(args.port)
    if args.device:
        os.environ["LAYA_DEVICE"] = args.device

    from laya.router import Router
    from laya.serve import create_app

    log_path = PROJECT_ROOT / args.log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    router = Router(
        models={"typed-decisions": (str(local_root), manifest.get(
            "subfolder", "typed-decisions"))},
        device=args.device or None)
    if not args.no_preload:
        print(f"preloading typed-decisions from {local_root} "
              f"(offline={os.environ.get('HF_HUB_OFFLINE') == '1'})...",
              flush=True)
        router.preload(["typed-decisions"])
        print(f"loaded: {router.loaded}", flush=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"{started_utc} laya-server pid={os.getpid()} "
                 f"rev={manifest.get('revision')} "
                 f"port={args.port} loaded={router.loaded}\n")

    import uvicorn

    uvicorn.run(create_app(router), host=args.host, port=args.port,
                log_level="warning")


if __name__ == "__main__":
    main()
