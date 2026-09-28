"""Pre-flight check for the CARLA machine. Run from the project root:

    python scripts/check_carla.py

Reports real capability (import probe + server ping); exits 0 only when a
CARLA server answers, else exits 2 with setup instructions. Never fakes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation import carla_lidar as C


def main() -> int:
    report = {"carla_package": C.carla_available(), "server": False, "world": None}
    print(f"CARLA package installed: {report['carla_package']}")
    if not report["carla_package"]:
        print("BLOCKED: install the CARLA Python API and launch the server first.")
        print("See CARLA_SETUP.md (root) and docs/carla_setup.md.")
        print(json.dumps(report, indent=2))
        return 2
    try:
        world = C.connect_to_carla()
        report["server"] = True
        try:
            report["world"] = world.get_map().name
        except Exception:
            report["world"] = "connected (map name unreadable)"
        print(f"CARLA server reachable. World: {report['world']}")
    except Exception as exc:
        print(f"BLOCKED: package present but no server: {exc}")
        print("Launch CARLA (e.g. CarlaUE4.exe) then rerun this script.")
        print(json.dumps(report, indent=2))
        return 2
    print(json.dumps(report, indent=2))
    print("READY for run_carla_closed_loop.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
