"""Probe A-category geometry: close lateral big boxes."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.probe_geometry import probe


def main():
    from src.simulation.pybullet_env import PyBulletEnv
    from src.simulation.pybullet_lidar import PyBulletLidar

    env = PyBulletEnv(gui=False)
    env.connect()
    lidar = PyBulletLidar()

    def box(x, y, hx=1.2, hy=1.2):
        return {"xyz": [x, y, 1.0], "half": [hx, hy, 1.0]}

    SX = -50.0
    for dx, dy in ((1.0, 2.0), (1.5, 2.0), (1.0, 2.5), (2.0, 2.0),
                   (0.5, 2.0), (1.0, -2.0), (1.5, -2.5), (2.5, 1.5)):
        probe(env, lidar, [box(SX + dx, dy)], f"A-cand dx={dx} dy={dy}")
    env.close()


if __name__ == "__main__":
    main()
