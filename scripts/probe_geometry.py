"""Probe obstacle geometry -> measured category (fast, no Laya)."""

from __future__ import annotations

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
          f"right={named['right_clearance_m']} emer={st['safety']['emergency_stop']} "
          f"nearest={st['safety']['nearest_forward_obstacle_m']} -> {cat}")


def main():
    from src.simulation.pybullet_env import PyBulletEnv
    from src.simulation.pybullet_lidar import PyBulletLidar

    env = PyBulletEnv(gui=False)
    env.connect()
    lidar = PyBulletLidar()

    def box(x, y, hx=1.5, hy=1.0):
        return {"xyz": [x, y, 1.0], "half": [hx, hy, 1.0]}

    SX = -50.0
    # forward blockers at varying distance, dead ahead
    for dx in (3.0, 4.0, 5.0, 6.0, 8.0):
        probe(env, lidar, [box(SX + dx, 0.0)], f"ahead dx={dx}")
    # side blockers
    for dy in (2.0, 3.0, 4.0):
        probe(env, lidar, [box(SX + 4.0, dy)], f"left dx=4 dy={dy}")
        probe(env, lidar, [box(SX + 4.0, -dy)], f"right dx=4 dy={-dy}")
    # diagonal close (trap candidates)
    for ang, dist in ((45, 2.0), (45, 3.0), (60, 2.0), (35, 2.5)):
        import math
        dx, dy = dist * math.cos(math.radians(ang)), dist * math.sin(math.radians(ang))
        probe(env, lidar, [box(SX + dx, dy), box(SX + dx, -dy)],
              f"trap ang={ang} d={dist}")
    # combined: ahead + one side
    probe(env, lidar, [box(SX + 5.0, 0.0), box(SX + 4.0, 3.0)], "ahead5+left3")
    probe(env, lidar, [box(SX + 5.0, 0.0), box(SX + 4.0, -3.0)], "ahead5+right3")
    probe(env, lidar, [box(SX + 4.0, 0.0), box(SX + 3.5, 3.0)], "ahead4+left3")
    env.close()


if __name__ == "__main__":
    main()
