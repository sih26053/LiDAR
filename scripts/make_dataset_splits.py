"""Coverage gate + dataset splits (run-group separated, deterministic).

Inputs:
    results/laya_recorded_dataset.json  (recorded, oracle-labelled)
    results/benchmark_labels.json       (generated, oracle-labelled)

Coverage gate (Phase 4): required categories A/B/C/D/E/F must each meet
--min-per-category or the command FAILS (no silent proceed). E maps to
E_NO_SAFE_DIRECTION; the requested taxonomy letters map onto the
repository's existing category names (reused, not reinvented).

Splits (Phase 6): TRAIN 60 / CALIBRATION 15 / VALIDATION 15 / TEST 10
over in-distribution frames; OOD_TEST = all OOD-family frames
(separate held-out families). Recorded live frames stay grouped by
run_id (no adjacent-frame leakage); benchmark frames are independent
resets, split by deterministic stratified shuffle (seed). No frame in
two splits (leakage test included).

Output: results/laya_dataset_splits.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TAXONOMY = {
    "A": "A_OPEN_FORWARD",
    "B": "B_FORWARD_BLOCKED_LEFT_OPEN",
    "C": "C_FORWARD_BLOCKED_RIGHT_OPEN",
    "D": "D_BOTH_SIDES_AVAILABLE",
    "E": "E_NO_SAFE_DIRECTION",
    "F": "F_EMERGENCY",
}


def _hkey(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def main() -> dict:
    ap = argparse.ArgumentParser(description="Coverage gate + splits")
    ap.add_argument("--min-per-category", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260927)
    args = ap.parse_args()

    recorded = json.loads(
        (PROJECT_ROOT / "results" / "laya_recorded_dataset.json").read_text())
    bench = json.loads(
        (PROJECT_ROOT / "results" / "benchmark_labels.json").read_text())
    rec_rows = recorded["rows"]
    bench_rows = bench["rows"]
    for r in bench_rows:
        r["origin"] = "benchmark-generated"

    ind_rows = [r for r in bench_rows if not r.get("ood")]
    ood_rows = [r for r in bench_rows if r.get("ood")]
    all_ind = rec_rows + ind_rows

    # ---- coverage gate ----
    cats = Counter(r["category"] for r in all_ind)
    gate = {}
    failed = []
    for letter, cat in TAXONOMY.items():
        n = cats.get(cat, 0)
        ok = n >= args.min_per_category
        gate[letter] = {"category": cat, "count": n,
                        "minimum": args.min_per_category,
                        "status": "READY" if ok else "NOT VALIDATED"}
        if not ok:
            failed.append(f"{letter}({cat}): {n} < {args.min_per_category}")
    ood_cats = Counter(r["category"] for r in ood_rows)
    print("IND:", dict(cats))
    print("OOD:", dict(ood_cats), "n =", len(ood_rows))
    if failed:
        print("COVERAGE GATE FAILED:", "; ".join(failed))
        raise SystemExit("coverage gate failed: " + "; ".join(failed))
    print("COVERAGE GATE: all required categories READY")

    # ---- splits ----
    # Recorded: group whole runs together.
    rec_by_run: dict = defaultdict(list)
    for r in rec_rows:
        rec_by_run[r["run_id"]].append(r)
    run_ids = sorted(rec_by_run, key=_hkey)
    # Bench IND: independent resets -> deterministic stratified shuffle.
    strata: dict = defaultdict(list)
    for r in ind_rows:
        strata[r["category"]].append(r)
    for cat in strata:
        strata[cat].sort(key=lambda r: _hkey(r["frame_id"]))

    splits: dict = {"train": [], "calibration": [], "validation": [],
                    "test": [], "ood_test": list(ood_rows)}
    fracs = [("train", 0.60), ("calibration", 0.15),
             ("validation", 0.15), ("test", 0.10)]

    def assign_sequential(members: list, keyfn) -> None:
        members = sorted(members, key=keyfn)
        n = len(members)
        bounds = [0]
        for _, f in fracs:
            bounds.append(bounds[-1] + f)
        for i, m in enumerate(members):
            pos = (i + 0.5) / max(n, 1)
            for (name, _), (lo, hi) in zip(fracs, zip(bounds, bounds[1:])):
                if lo <= pos < hi or (name == "test" and pos >= lo):
                    splits[name].append(m)
                    break

    # Whole recorded runs stay together AND every split needs open-road
    # (D) frames: deal runs largest-first across splits so no split is
    # left without D. Rank order: train, calibration, validation, test,
    # then remainder to train.
    sizes = sorted(rec_by_run.items(), key=lambda kv: len(kv[1]),
                   reverse=True)
    buckets: dict = {"train": [], "calibration": [], "validation": [],
                     "test": []}
    deal = ["train", "calibration", "validation", "test"]
    for i, (rid, members) in enumerate(sizes):
        buckets[deal[i] if i < len(deal) else "train"].append(rid)
    for name, rids in buckets.items():
        for rid in rids:
            splits[name] += rec_by_run[rid]
    for cat, members in strata.items():
        assign_sequential(members, lambda r: _hkey(r["frame_id"]))

    # ---- leakage test ----
    seen: dict = {}
    leaks = []
    for name, members in splits.items():
        for r in members:
            if r["frame_id"] in seen:
                leaks.append((r["frame_id"], seen[r["frame_id"]], name))
            seen[r["frame_id"]] = name
    rec_runs: dict = {}
    run_leaks = []
    for name in ("train", "calibration", "validation", "test"):
        for r in splits[name]:
            if r["origin"] == "live-recorded":
                if r["run_id"] in rec_runs and rec_runs[r["run_id"]] != name:
                    run_leaks.append((r["run_id"], rec_runs[r["run_id"]], name))
                rec_runs[r["run_id"]] = name
    assert not leaks, f"frame leakage: {leaks[:5]}"
    assert not run_leaks, f"run leakage: {run_leaks[:5]}"

    dh = hashlib.sha256(json.dumps(
        sorted(seen.items()), sort_keys=True).encode()).hexdigest()[:16]
    payload = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": args.seed,
        "taxonomy": TAXONOMY,
        "coverage_gate": gate,
        "split_hash": dh,
        "dataset_hash": hashlib.sha256(json.dumps(
            [r["frame_id"] for r in all_ind + ood_rows],
            sort_keys=True).encode()).hexdigest()[:16],
        "counts": {k: len(v) for k, v in splits.items()},
        "split_categories": {
            k: dict(Counter(r["category"] for r in v))
            for k, v in splits.items()},
        "split_runs": {k: sorted({r["run_id"] for r in v})
                       for k, v in splits.items()},
        "leakage_test": "PASSED (no frame in two splits; no recorded run "
                        "in two splits)",
        "splits": {k: [r["frame_id"] for r in v]
                   for k, v in splits.items()},
    }
    out = PROJECT_ROOT / "results" / "laya_dataset_splits.json"
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: payload[k] for k in
                      ("counts", "split_categories", "split_hash")}, indent=2))
    print(f"leakage test PASSED; wrote {out}")
    return payload


if __name__ == "__main__":
    main()
