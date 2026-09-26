"""Generic simulator interface (Phase 2).

Active backend: PyBullet (`pybullet_env`). CARLA modules
(`carla_lidar`, `carla_env`, `carla_manager`, `carla_action_executor`)
are legacy/optional and must NOT leak into Stages 1-6: all simulator
output enters the pipeline only as N x 4 [x,y,z,intensity] or a
`LiDARFrame`. Nothing here imports carla or pybullet at module load.
"""

from __future__ import annotations

from typing import Any, Dict

ACTIVE_SIMULATOR = "pybullet"
LEGACY_SIMULATOR = "carla"


class SimulatorUnavailable(RuntimeError):
    """Raised when the requested simulator backend cannot run."""


def active_backend() -> Dict[str, Any]:
    """Report which simulator backend is active/available (measured)."""
    try:
        import pybullet  # type: ignore  # noqa: F401

        pybullet_importable = True
    except ImportError:
        pybullet_importable = False
    try:
        import carla  # type: ignore  # noqa: F401

        carla_importable = True
    except ImportError:
        carla_importable = False
    return {"active": ACTIVE_SIMULATOR, "legacy": LEGACY_SIMULATOR,
            "pybullet_importable": pybullet_importable,
            "carla_importable": carla_importable}
