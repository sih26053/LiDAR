"""Laya checkpoint identity + provisioning state (no downloads here).

Every recorded Laya decision carries model_id / checkpoint_revision /
device / runtime_version / calibration_version so decisions are
reproducible and revisions are never mixed under one anonymous label.

Provisioning itself lives in scripts/provision_laya.py; this module only
reads the manifest (models/laya/manifest.json) and the pinned revision.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_ROOT = PROJECT_ROOT / "models" / "laya"
MANIFEST_PATH = MODELS_ROOT / "manifest.json"
BUNDLE_REPO = "convaiinnovations/laya"
CHECKPOINT_SUBFOLDER = "typed-decisions"


def local_checkpoint_dir() -> Path:
    """Project-managed local checkpoint (provisioned copy)."""
    return MODELS_ROOT / "checkpoint" / CHECKPOINT_SUBFOLDER


def read_manifest() -> Dict[str, Any] | None:
    try:
        return json.loads(MANIFEST_PATH.read_text())
    except (OSError, ValueError):
        return None


def checkpoint_info() -> Dict[str, Any]:
    """Measured local state: LOCAL / MISSING, revision, offline readiness."""
    from src.decision.laya_client import detect_device, resolve_model

    try:
        from importlib.metadata import version as _pkg_version
        runtime_version = _pkg_version("laya")
    except Exception:
        runtime_version = None
    manifest = read_manifest()
    local_dir = local_checkpoint_dir()
    required = ("model.safetensors", "rl_agent_config.json")
    present = {f: (local_dir / f).is_file() for f in required}
    enc_tok = ((local_dir / "encoder").is_dir()
               and (local_dir / "tokenizer").is_dir())
    provisioned = all(present.values()) and enc_tok
    revision = (manifest or {}).get("revision")
    return {
        "model_id": resolve_model(),
        "repo": (manifest or {}).get("repository", BUNDLE_REPO),
        "subfolder": (manifest or {}).get("subfolder", CHECKPOINT_SUBFOLDER),
        "revision": revision,
        "state": "LOCAL" if provisioned else "MISSING",
        "local_path": str(local_dir),
        "files_present": present,
        "encoder_tokenizer_present": enc_tok,
        "offline_inference": ("AVAILABLE" if provisioned else "UNAVAILABLE"),
        "device": detect_device(),
        "runtime_version": runtime_version,
        "manifest": manifest,
    }


def decision_version_stamp() -> Dict[str, Any]:
    """Per-decision reproducibility stamp (recorded with every decision)."""
    from src.decision import calibration as _cal

    info = checkpoint_info()
    gate = _cal.load_gate_selection()
    return {
        "model_id": info["model_id"],
        "checkpoint_revision": info["revision"],
        "device": info["device"],
        "runtime_version": info["runtime_version"],
        "calibration_version": (
            f"gate@{gate.get('dataset_hash')}" if gate else None),
        "gate_threshold": (gate or {}).get("selected_threshold"),
    }
