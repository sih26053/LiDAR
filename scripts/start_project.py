"""First-run bootstrap: checkpoint -> Laya service -> backend -> dashboard.

Flow:
    START PROJECT
      -> local Laya checkpoint present? NO -> provision once (pinned rev)
      -> start/reuse independent Laya service (scripts/run_laya_server.py
         as a managed child in dev; production expects it already up)
      -> wait for readiness, verify model/checkpoint/revision
      -> backend connects (uvicorn backend.app, optionally --with-backend)
      -> dashboard shows LIVE (frontend dev server, --with-frontend note)

Subsequent starts skip the download (manifest + local files present).
No repeated user intervention, no silent re-downloads, no floating
"latest" checkpoint.

Run with the Laya venv (.venv-pb).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
PY = str(PROJECT_ROOT / ".venv-pb" / "Scripts" / "python.exe")


def _run(*args: str) -> int:
    exe = PY if Path(PY).is_file() else sys.executable
    return subprocess.call([exe, *args], cwd=str(PROJECT_ROOT))


def main() -> dict:
    ap = argparse.ArgumentParser(description="Bootstrap the project")
    ap.add_argument("--production", action="store_true",
                    help="expect the independent Laya service already up; "
                         "never spawn (allow_spawn=False)")
    ap.add_argument("--with-backend", action="store_true",
                    help="launch uvicorn backend in-process (blocking)")
    args = ap.parse_args()
    report: dict = {"steps": []}

    def mark(name: str, ok: bool, detail: str = "") -> None:
        report["steps"].append({"step": name, "ok": ok, "detail": detail})
        print(f"[{'OK' if ok else 'FAIL'}] {name} {detail}", flush=True)

    from src.decision.checkpoint_info import checkpoint_info
    info = checkpoint_info()
    if info["state"] != "LOCAL":
        mark("checkpoint-missing", True, "provisioning once...")
        rc = _run("scripts/provision_laya.py")
        mark("provision", rc == 0, f"exit={rc}")
        if rc != 0:
            raise SystemExit("provisioning failed; see output above")
        info = checkpoint_info()
    mark("checkpoint", info["state"] == "LOCAL",
         f"rev={info['revision']} offline={info['offline_inference']}")
    if info["state"] != "LOCAL":
        raise SystemExit("checkpoint still missing after provisioning")

    from backend.services import laya_manager
    try:
        boot = laya_manager.ensure_started(
            wait_s=600.0, allow_spawn=not args.production)
        mark("laya-service", bool(boot["health"].get("healthy")),
             f"reused={boot.get('reused')} managed={boot.get('managed')}")
    except RuntimeError as exc:
        mark("laya-service", False, str(exc)[:200])
        raise SystemExit("Laya service not ready; "
                         "start it: .venv-pb python scripts/run_laya_server.py --offline")

    st = laya_manager.status()
    mark("laya-ready", st["state"] == "READY",
         f"state={st['state']} rev={st.get('checkpoint_revision')} "
         f"calibration={st.get('calibration', {}).get('gate', {}).get('state')}")
    if st.get("checkpoint_revision") != info.get("revision"):
        mark("revision-match", False,
             f"server={st.get('checkpoint_revision')} "
             f"manifest={info.get('revision')}")
        raise SystemExit("checkpoint revision mismatch; refusing to continue")

    from src.decision import calibration as cal
    cs = cal.calibration_status(info.get("revision"))
    mark("calibration", True, f"gate={cs['gate']['state']} "
                              f"threshold={cs['gate'].get('threshold')}")
    mark("backend-next",
         True, "run: uvicorn backend.app:app --host 127.0.0.1 --port 8000 "
               "(or pass --with-backend)")
    mark("dashboard-next",
         True, "frontend: npm --prefix frontend run dev "
               "(Live Control + Laya Navigation Diagnostics panels)")
    if args.with_backend:
        import os
        os.execvp(exe if (exe := (PY if Path(PY).is_file()
                                  else sys.executable)) else sys.executable,
                  [exe, "-m", "uvicorn", "backend.app:app",
                   "--host", "127.0.0.1", "--port", "8000"])
    return report


if __name__ == "__main__":
    main()
