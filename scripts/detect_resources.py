"""Resource detection (Phase 33): CUDA/GPU/CPU/RAM/Docker/systemd/versions.

Chooses nothing by itself; records what is available so orchestration
(master command) takes the right path automatically. Writes
results/resources.json.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> dict:
    res: dict = {"detected_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                               time.gmtime()),
                 "os": platform.platform(), "python": sys.version.split()[0]}
    try:
        import torch
        res["torch"] = torch.__version__
        res["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            res["gpu_count"] = torch.cuda.device_count()
            res["gpu_vram_gb"] = round(torch.cuda.get_device_properties(
                0).total_memory / 1e9, 1)
        else:
            res["gpu_count"] = 0
    except (ImportError, OSError) as exc:
        res["torch"] = f"unavailable: {exc}"
        res["cuda_available"] = False
        res["gpu_count"] = 0
    res["cpu_count"] = os.cpu_count()
    try:
        import psutil  # type: ignore
        res["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except (ImportError, OSError):
        res["ram_gb"] = "Unavailable (psutil missing)"
    try:
        from importlib.metadata import version as _v
        res["laya_version"] = _v("laya")
    except Exception:
        res["laya_version"] = None
    res["docker"] = shutil.which("docker") is not None
    res["systemd"] = Path("/run/systemd/system").exists()
    res["paths"] = {
        "finetune": ("GPU attempt possible" if res["cuda_available"]
                     else "CPU-only: package only"),
        "docker": ("execute tests" if res["docker"]
                   else "prepare config, mark unvalidated"),
        "systemd": ("test unit" if res["systemd"]
                    else "ship unit, mark unvalidated"),
    }
    out = PROJECT_ROOT / "results" / "resources.json"
    out.write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str))
    return res


if __name__ == "__main__":
    main()
