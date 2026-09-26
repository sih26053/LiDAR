"""Phase 4 — PyBullet simulation environment (active simulator backend).

Vehicle + road + boundaries + obstacles + sensor + control, with explicit
stepping, configurable timestep, deterministic reset, and contact-based
collision detection. GUI for demo, DIRECT for headless evaluation.

pybullet is imported LAZILY (inside methods) so this module imports cleanly
where pybullet is uninstallable; every live method raises
PyBulletUnavailable with the exact remedy instead of failing obscurely.
Vehicle motion is kinematic (base-velocity control, documented surrogate --
not a dynamics model); LiDAR comes from real rayTest queries (no fabrication).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class PyBulletUnavailable(RuntimeError):
    """pybullet package missing (this host: no wheel for Python 3.14.4)."""


def _require_pybullet():
    try:
        import pybullet  # type: ignore
        return pybullet
    except ImportError as exc:
        raise PyBulletUnavailable(
            "pybullet is not installed here (no wheel builds for Python "
            f"3.14.4: {exc}). Install on a compatible interpreter "
            "(<=3.12 per published wheels) or see HUMAN ACTION in reports.") from exc


def load_simulation_config() -> Dict[str, Any]:
    try:
        return json.loads((PROJECT_ROOT / "config" / "simulation_config.json").read_text())
    except (OSError, ValueError):
        return {}


class PyBulletEnv:
    """Minimal deterministic driving world (Phase 4 requirements)."""

    def __init__(self, gui: bool = False, config: Dict[str, Any] | None = None):
        self.cfg = dict(config or load_simulation_config())
        self.gui = gui
        self._p = None
        self._cid = None
        
        self._vehicle = None
        self._plane = None
        self._obstacles: List[int] = []
        self._frame = 0
        self._sim_time = 0.0
        self._collisions = 0
        self._was_in_contact = False

    @property
    def timestep(self) -> float:
        return float(self.cfg.get("timestep_s", 1.0 / 60.0))

    def connect(self) -> Dict[str, Any]:
        p = _require_pybullet()
        self._p = p
        self._cid = p.connect(p.GUI if self.gui else p.DIRECT)
        if self._cid < 0:
            raise RuntimeError("pybullet connect returned negative client id")
        import pybullet_data  # type: ignore

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(),
            physicsClientId=self._cid,
        )
        return {"connected": True, "mode": "GUI" if self.gui else "DIRECT", "client": self._cid}

    def reset(self, seed: int = 42, scenario: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """(Re)build world deterministically: road, walls, obstacles, vehicle."""
        p = self._p or _require_pybullet()
        if self._cid is None:
            self.connect()
            p = self._p
        assert self._cid is not None
        p.resetSimulation(physicsClientId=self._cid)
        g = self.cfg.get("gravity", [0, 0, -9.81])
        p.setGravity(*g, physicsClientId=self._cid)
        p.setPhysicsEngineParameter(fixedTimeStep=self.timestep,
                                    physicsClientId=self._cid)
        road = self.cfg.get("road", {})
        L = float(road.get("length_m", 120.0))
        W = float(road.get("width_m", 12.0))
        plane = p.loadURDF("plane.urdf", physicsClientId=self._cid)
        assert plane >= 0
        self._plane = plane
        self._make_box([0, 0, -0.06], [L / 2, W / 2, 0.05], mass=0.0)  # road slab
        bh = float(self.cfg.get("boundaries", {}).get("height_m", 1.0))
        bt = float(self.cfg.get("boundaries", {}).get("thickness_m", 0.5))
        self._make_box([0, W / 2 + bt, bh / 2], [L / 2, bt / 2, bh / 2], mass=0.0)
        self._make_box([0, -W / 2 - bt, bh / 2], [L / 2, bt / 2, bh / 2], mass=0.0)
        self._obstacles = []
        scen_obs = (scenario or {}).get("obstacles", None)
        if scen_obs is None:
            scen_obs = self._default_obstacles(seed)
        for ob in scen_obs:
            self._obstacles.append(
                self._make_box(ob["xyz"], ob["half"], mass=0.0))
        v = self.cfg.get("vehicle", {})
        sx, sy = v.get("start_xy", [-50.0, 0.0])
        yaw = math.radians(float(v.get("start_yaw_deg", 0.0)))
        he = v.get("half_extents_m", [2.2, 1.0, 0.7])
        self._vehicle = self._make_box([sx, sy, he[2] + 0.02], he, mass=800.0)
        p.resetBasePositionAndOrientation(
            self._vehicle, [sx, sy, he[2] + 0.02],
            p.getQuaternionFromEuler([0, 0, yaw]), physicsClientId=self._cid)
        self._frame = 0
        self._sim_time = 0.0
        self._collisions = 0
        self._was_in_contact = False
        return {"reset": True, "vehicle": self._vehicle,
                "obstacles": len(self._obstacles), "seed": seed}

    def _default_obstacles(self, seed: int) -> List[Dict[str, Any]]:
        import random
        rng = random.Random(seed)
        obs = []
        for i in range(4):
            obs.append({"xyz": [-20.0 + i * 18.0 + rng.uniform(-2, 2),
                                rng.choice([-3.0, 3.0]), 1.0],
                        "half": [1.5, 1.0, 1.0]})
        return obs

    def _make_box(self, pos, half, mass: float) -> int:
        p = self._p
        assert p is not None and self._cid is not None
        shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half,
                                       physicsClientId=self._cid)
        return p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=shape,
                                 basePosition=pos, physicsClientId=self._cid)

    def step(self, action: str) -> Dict[str, Any]:
        """Apply kinematic control for one action, advance one fixed step."""
        from src.simulation.pybullet_action_executor import action_to_velocity
        p = self._p or _require_pybullet()
        if self._vehicle is None:
            raise RuntimeError("env not reset; call reset() first")
        vx, yaw_rate = action_to_velocity(action, self.cfg)
        pos, orn = p.getBasePositionAndOrientation(self._vehicle, physicsClientId=self._cid)
        yaw = p.getEulerFromQuaternion(orn)[2]
        p.resetBaseVelocity(self._vehicle,
                            linearVelocity=[vx * math.cos(yaw), vx * math.sin(yaw), 0],
                            angularVelocity=[0, 0, yaw_rate],
                            physicsClientId=self._cid)
        p.stepSimulation(physicsClientId=self._cid)
        self._frame += 1
        self._sim_time += self.timestep
        contacts = p.getContactPoints(bodyA=self._vehicle,
                                      physicsClientId=self._cid)
        # Ground-contact filter: resting contact with the ground plane (or any
        # near-vertical contact normal) is normal ground rest, NOT a vehicle
        # collision. Only non-ground contacts count.
        ground, solid = 0, []
        for c in contacts:
            try:
                normal_z = abs(float(c[7][2]))
            except (TypeError, ValueError, IndexError):
                normal_z = 0.0
            if c[2] == self._plane or normal_z > 0.95:
                ground += 1
            else:
                solid.append(int(c[2]))
        in_contact = len(solid) > 0
        if in_contact and not self._was_in_contact:
            self._collisions += 1
        self._was_in_contact = in_contact
        return {"frame": self._frame, "sim_time": round(self._sim_time, 4),
                "collision": in_contact, "collisions_total": self._collisions,
                "ground_contacts_filtered": ground,
                "collision_body": solid[0] if solid else None}

    def get_vehicle_state(self) -> Dict[str, Any]:
        p = self._p or _require_pybullet()
        if self._vehicle is None:
            raise RuntimeError("env not reset")
        pos, orn = p.getBasePositionAndOrientation(self._vehicle, physicsClientId=self._cid)
        lin, ang = p.getBaseVelocity(self._vehicle, physicsClientId=self._cid)
        return {"x": pos[0], "y": pos[1], "z": pos[2],
                "yaw_deg": math.degrees(p.getEulerFromQuaternion(orn)[2]),
                "speed_ms": math.hypot(lin[0], lin[1]), "yaw_rate": ang[2]}

    def get_obstacles(self) -> List[Dict[str, Any]]:
        p = self._p or _require_pybullet()
        out = []
        for b in self._obstacles:
            pos, _ = p.getBasePositionAndOrientation(b, physicsClientId=self._cid)
            aabb = p.getAABB(b, physicsClientId=self._cid)
            out.append({"id": b, "xyz": list(pos),
                        "aabb_min": list(aabb[0]), "aabb_max": list(aabb[1])})
        return out

    def get_simulation_time(self) -> float:
        return self._sim_time

    def get_frame_id(self) -> int:
        return self._frame

    def raycast(self, origins, endpoints):
        """Thin wrapper so the LiDAR module never touches pybullet directly."""
        p = self._p or _require_pybullet()
        return p.rayTestBatch(origins, endpoints, physicsClientId=self._cid)

    def close(self) -> Dict[str, Any]:
        if self._p is not None and self._cid is not None:
            try:
                self._p.disconnect(physicsClientId=self._cid)
            except Exception:
                pass
        self._p, self._cid, self._vehicle = None, None, None
        self._plane = None
        return {"closed": True}
