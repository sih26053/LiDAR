"""Provision the Laya checkpoint once into project-managed local storage.

Downloads ONLY the typed-decisions subfolder files needed at runtime
(model.safetensors, rl_agent_config.json, encoder/*, tokenizer/*) from
the pinned revision, verifies presence + sizes, and writes
models/laya/manifest.json. Subsequent starts load from the local path
with no Hub re-download.

Run:
    python scripts/provision_laya.py [--revision REV] [--force]

Must run with the Laya venv (.venv-pb) so huggingface_hub is available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

REPO = "convaiinnovations/laya"
SUBFOLDER = "typed-decisions"
DEFAULT_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
ALLOW = [f"{SUBFOLDER}/rl_agent_config.json", f"{SUBFOLDER}/model.safetensors",
         f"{SUBFOLDER}/tokenizer/*", f"{SUBFOLDER}/encoder/*"]

MANIFEST_PATH = PROJECT_ROOT / "models" / "laya" / "manifest.json"
LOCAL_ROOT = PROJECT_ROOT / "models" / "laya" / "checkpoint"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> dict:
    ap = argparse.ArgumentParser(description="Provision Laya checkpoint once")
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    if MANIFEST_PATH.is_file() and not args.force:
        manifest = json.loads(MANIFEST_PATH.read_text())
        if manifest.get("revision") == args.revision and \
                (LOCAL_ROOT / SUBFOLDER / "model.safetensors").is_file():
            print(f"already provisioned at revision {args.revision}; "
                  "--force to re-download")
            return manifest

    print(f"downloading {REPO} subfolder {SUBFOLDER} @ {args.revision} "
          f"(typed-decisions files only)...", flush=True)
    t0 = time.time()
    snap = Path(snapshot_download(
        REPO, revision=args.revision, allow_patterns=ALLOW,
        local_dir=str(LOCAL_ROOT)))
    dt = round(time.time() - t0, 1)
    sub = snap / SUBFOLDER
    required = [sub / "model.safetensors", sub / "rl_agent_config.json"]
    missing = [str(p) for p in required if not p.is_file()]
    if missing or not (sub / "encoder").is_dir() \
            or not (sub / "tokenizer").is_dir():
        raise SystemExit(f"provisioning incomplete, missing: {missing}")
    files = []
    for p in sorted(sub.rglob("*")):
        if p.is_file():
            rel = p.relative_to(LOCAL_ROOT).as_posix()
            files.append({"path": rel, "bytes": p.stat().st_size,
                          "sha256": _sha256(p)})
    manifest = {
        "repository": REPO,
        "subfolder": SUBFOLDER,
        "revision": args.revision,
        "local_dir": "models/laya/checkpoint",
        "allow_patterns": ALLOW,
        "files": files,
        "total_bytes": sum(f["bytes"] for f in files),
        "provisioned_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                         time.gmtime()),
        "download_s": dt,
        "model_version": f"{REPO}/{SUBFOLDER}@{args.revision}",
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"provisioned {len(files)} files "
          f"({manifest['total_bytes'] / 1e9:.2f} GB) in {dt}s")
    print(f"manifest: {MANIFEST_PATH}")
    return manifest


if __name__ == "__main__":
    main()
