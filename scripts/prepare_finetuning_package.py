"""Prepare the domain fine-tuning package (CPU-only: stops at training).

Exports TRAIN-split oracle-labelled navigation examples in supervised
form {state, question, label_choice} + split manifest + training config
+ reproducibility metadata into results/laya_finetuning_package/.

Labels come ONLY from the deterministic oracle (never Laya outputs).
Calibration/validation/test/OOD frames are excluded (leakage-checked).

Training itself is NOT run here: the installed laya 0.3.20 package is
inference-only (no training API) and this host is CPU-only (421M
params). Status: PREPARED, NOT RUN.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUT = PROJECT_ROOT / "results" / "laya_finetuning_package"


def main() -> dict:
    from src.decision.laya_client import load_laya_config
    from backend.services import store

    splits = json.loads((PROJECT_ROOT / "results" / "laya_dataset_splits.json"
                         ).read_text())
    train_ids = set(splits["splits"]["train"])
    recorded = {r["frame_id"]: r for r in json.loads(
        (PROJECT_ROOT / "results" / "laya_recorded_dataset.json"
         ).read_text())["rows"]}
    bench = {r["frame_id"]: r for r in json.loads(
        (PROJECT_ROOT / "results" / "benchmark_labels.json").read_text())["rows"]}
    cfg_q = load_laya_config().get("question", {}) or {}
    question = {"type": "choice",
                "instructions": cfg_q.get("instructions", ""),
                "criteria": cfg_q.get("criteria", {})}

    con = store._connect()
    examples, skipped = [], 0
    for fid in sorted(train_ids):
        if fid in bench:
            b = bench[fid]
            state = {"state_vector": b["state_vector"],
                     "clearances_m": b["clearances_m"],
                     "emergency_flag": b["emergency"],
                     "origin": "benchmark-generated"}
            oracle_pref, rule = b["oracle_preferred"], b["oracle_rule"]
        elif fid in recorded:
            r = recorded[fid]
            row = con.execute(
                "SELECT state_vector_json FROM frames WHERE frame_id=?",
                (fid,)).fetchone()
            try:
                sv = json.loads(row["state_vector_json"])
            except (TypeError, ValueError):
                skipped += 1
                continue
            state = {"state_vector": sv, "sector_ranges_m": None,
                     "origin": "live-recorded",
                     "note": "sector_ranges_m not stored; meters via "
                             "documented normalized fallback"}
            oracle_pref, rule = r["oracle_preferred"], r.get("oracle_rule", "")
        else:
            skipped += 1
            continue
        examples.append({"frame_id": fid, "state": state,
                         "question": question,
                         "label_choice": oracle_pref,
                         "oracle_rule": rule})
    dist = {}
    for e in examples:
        dist[e["label_choice"]] = dist.get(e["label_choice"], 0) + 1
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "dataset.jsonl", "w") as fh:
        for e in examples:
            fh.write(json.dumps(e) + "\n")
    train_ids_sorted = sorted(e["frame_id"] for e in examples)
    manifest = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_checkpoint": "convaiinnovations/laya/typed-decisions",
        "base_revision": "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
        "n_examples": len(examples), "n_skipped": skipped,
        "label_distribution": dist,
        "label_source": "deterministic oracle ONLY (no Laya pseudo-labels)",
        "excluded": ["calibration", "validation", "test", "ood_test"],
        "split_hash": splits["split_hash"],
        "dataset_sha256": hashlib.sha256(
            "\n".join(train_ids_sorted).encode()).hexdigest(),
    }
    (OUT / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUT / "training_config.json").write_text(json.dumps({
        "task": "supervised choice fine-tune (typed-decisions head)",
        "base": manifest["base_checkpoint"] + "@" + manifest["base_revision"],
        "seed": 20260927,
        "suggested": {"epochs": 3, "lr": 2e-5, "batch": 16,
                      "context": "single-GPU 24GB VRAM assumed; adjust",
                      "eval_splits": ["validation", "test", "ood_test"],
                      "shuffle_options_during_training": True,
                      "option_order_robustness": "flip_rate 0.0 measured; "
                      "keep deterministic production order"},
        "status": "NOT RUN (CPU-only host; installed laya is inference-only)",
    }, indent=2))
    (OUT / "README.md").write_text(
        "# Laya navigation fine-tuning package (PREPARED, NOT RUN)\n\n"
        "Contents: dataset.jsonl (oracle-labelled TRAIN states),\n"
        "split_manifest.json, training_config.json.\n\n"
        "To run on a GPU host with laya training support:\n"
        "1. Load dataset.jsonl states + question schema.\n"
        "2. Supervised-train the typed-decisions head from the pinned base.\n"
        "3. Save the checkpoint, then evaluate with scripts/evaluate_dataset.py\n"
        "   against the SAME untouched test/OOD splits.\n"
        "4. Promote only via the EXPERIMENTAL->CANDIDATE->KNOWN_GOOD gates.\n\n"
        "No trained checkpoint exists yet; no model-level claim is made.\n")
    print(f"examples={len(examples)} skipped={skipped} dist={dist}")
    print(f"package: {OUT} (PREPARED, NOT RUN)")
    store.close()
    return manifest


if __name__ == "__main__":
    main()
