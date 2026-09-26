"""Phase 3 — PyBullet health check (executed, never assumed).

1. Import PyBullet. 2/3. DIRECT (headless) connect; GUI only with --gui.
4. Verify connection. 5. Basic physics world. 6. Step. 7. Disconnect.
8. Report real results as JSON (PASS/FAIL per check + versions).

Run:  python scripts/check_pybullet.py [--gui]
"""

from __future__ import annotations

import json
import sys
import time


def main(gui: bool = False) -> dict:
    checks: dict = {}
    try:
        import pybullet as p  # type: ignore

        checks["import"] = {"ok": True, "module": getattr(p, "__file__", "?")}
    except ImportError as exc:
        return {"status": "FAIL", "checks": {"import": {"ok": False, "error": str(exc)[:300]}}}
    mode = p.GUI if gui else p.DIRECT
    try:
        cid = p.connect(mode)
        checks["connect"] = {"ok": cid >= 0, "mode": "GUI" if gui else "DIRECT", "client": cid}
        if cid < 0:
            raise RuntimeError("connect returned negative client id")
        import pybullet_data  # type: ignore

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=cid)
        plane = p.loadURDF("plane.urdf", physicsClientId=cid)
        checks["world"] = {"ok": plane >= 0}
        box = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], physicsClientId=cid)
        body = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=box,
                                 basePosition=[0, 0, 2.0], physicsClientId=cid)
        t0 = time.perf_counter()
        for _ in range(120):
            p.stepSimulation(physicsClientId=cid)
        checks["step"] = {"ok": True, "steps": 120,
                          "step_ms": round((time.perf_counter() - t0) * 1000.0 / 120.0, 4)}
        pos, _ = p.getBasePositionAndOrientation(body, physicsClientId=cid)
        checks["physics"] = {"ok": pos[2] < 2.0, "z": round(pos[2], 3),
                             "note": "body fell under gravity => dynamics live"}
        p.disconnect(physicsClientId=cid)
        checks["disconnect"] = {"ok": True}
    except Exception as exc:  # noqa: BLE001 - report, don't crash
        checks["error"] = {"ok": False, "error": str(exc)[:300]}
    ok = all(v.get("ok", False) for k, v in checks.items() if k != "error") and "error" not in checks
    return {"status": "PASS" if ok else "FAIL", "checks": checks}


if __name__ == "__main__":
    result = main(gui="--gui" in sys.argv[1:])
    print(json.dumps(result, indent=2))
