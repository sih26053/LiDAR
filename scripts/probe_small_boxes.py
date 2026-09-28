"""Grid-probe small obstacles -> measured clearances/category."""

from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def probe(env, lidar, obs, tag):
    from src.decision.action_eligibility import categorize_frame
    from src.decision.jev_state_adapter import enrich_for_laya, to_named_state
    from src.simulation.pybullet_lidar import to_lidar_frame
    from src.simulation.stages_runner import run_stages_1_to_6

    env.reset(seed=1, scenario={"obstacles": obs})
    pts = lidar.to_points(lidar.scan(env))
    frame = to_lidar_frame(pts, "probe", 0.0)
    res = run_stages_1_to_6(frame.points, frame.frame_id, frame.timestamp)
    st = res["state"]
    named = enrich_for_laya(
        to_named_state(st["state_vector"], st["sector_ranges_m"]),
        st["safety"])
    cat = categorize_frame(named, st["safety"])
    print(f"{tag}: fwd={named['forward_clearance_m']} left={named['left_clearance_m']} "
          f"right={named['right_clearance_m']} emer={st['safety']['emergency_stop']} -> {cat}")


def main():
    from src.simulation.pybullet_env import PyBulletEnv
    from src.simulation.pybullet_lidar import PyBulletLidar

    env = PyBulletEnv(gui=False)
    env.connect()
    lidar = PyBulletLidar()

    def box(x, y, h=0.4):
        return {"xyz": [x, y, 0.5], "half": [h, h, 0.5]}

    SX = -50.0
    for ang in (35, 45, 55, 65):
        for dist in (2.0, 3.0, 4.0):
            dx = dist * math.cos(math.radians(ang))
            dy = dist * math.sin(math.radians(ang))
            probe(env, lidar, [box(SX + dx, dy)], f"solitary ang={ang} d={dist}")
    # E candidates: two small peripheral + nothing ahead
    for ang, dist in ((50, 2.2), (55, 2.5), (45, 2.5)):
        dx = dist * math.cos(math.radians(ang))
        dy = dist * math.sin(math.radians(ang))
        probe(env, lidar, [box(SX + dx, dy), box(SX + dx, -dy)],
              f"E-cand ang={ang} d={dist}")
    env.close()


if __name__ == "__main__":
    main()
