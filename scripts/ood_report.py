"""OOD report (Phase 17): held-out scenario families, frozen config.

Uses the fresh OOD decisions from the baseline run (no new calls, no
changes after seeing results — the OOD set was fixed at generation).
Writes results/laya_ood.{json,csv,md}. Poor generalization, if any,
is reported with exact failure categories.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> dict:
    import scripts.evaluate_dataset as ED

    baseline = json.loads((PROJECT_ROOT / "results" / "laya_baseline.json"
                           ).read_text())
    bench = {r["frame_id"]: r for r in json.loads(
        (PROJECT_ROOT / "results" / "benchmark_labels.json").read_text())["rows"]}
    ood_families = sorted({r["family"] for r in bench.values()
                           if r.get("ood")})
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                             time.gmtime()),
              "ood_families": ood_families,
              "n_ood": baseline["by_split_mode"]["ood_test:constrained_laya"]["n"],
              "by_mode": {k: v for k, v in baseline["by_split_mode"].items()
                          if k.startswith("ood_test")},
              "by_family": {},
              "note": "OOD set fixed at generation; evaluated once."}
    # Per-family breakdown from baseline rows is unavailable (baseline
    # stores aggregates); recompute from SQLite decisions + labels.
    from backend.services import store
    con = store._connect()
    fam_rows: dict = {}
    for fid, b in bench.items():
        if not b.get("ood"):
            continue
        for d in con.execute(
                "SELECT * FROM decisions WHERE frame_id=? AND "
                "source='benchmark-laya' ORDER BY id", (fid,)):
            dd = dict(d)
            if dd.get("mode") not in ("constrained_laya",
                                      "unconstrained_laya"):
                continue
            from scripts.evaluate_dataset import EXEC_TO_CHOICE
            prop = EXEC_TO_CHOICE.get(str(dd.get("proposed_action") or ""),
                                      None)
            fam_rows.setdefault((b["family"], dd["mode"]), []).append({
                "oracle_preferred": b["oracle_preferred"],
                "proposed_action": prop,
                "answer_confidence": dd["answer_confidence"],
                "latency_ms": dd["latency_ms"],
                "category": b["category"],
                "unsafe": (prop not in (b["oracle_admissible"] or []))
                if prop else None})
    for (fam, mode), rows in sorted(fam_rows.items()):
        report["by_family"][f"{fam}:{mode}"] = ED.metrics_for(rows)
    # Failure categories: where constrained != preferred on OOD.
    fails = Counter()
    for (fam, mode), rows in fam_rows.items():
        if mode != "constrained_laya":
            continue
        for r in rows:
            if r["oracle_preferred"] != r["proposed_action"]:
                fails[(fam, r["category"], r["oracle_preferred"],
                       r["proposed_action"])] += 1
    report["constrained_failures"] = [
        {"family": k[0], "category": k[1], "oracle": k[2],
         "proposed": k[3], "n": v} for k, v in fails.items()]
    (PROJECT_ROOT / "results" / "laya_ood.json").write_text(
        json.dumps(report, indent=2, default=str))
    with open(PROJECT_ROOT / "results" / "laya_ood.csv", "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["slice", "n", "accuracy", "balanced_accuracy",
                    "macro_f1", "unsafe_rate", "emer_fwd_rate", "ece"])
        for k, m in {**report["by_mode"], **report["by_family"]}.items():
            w.writerow([k, m["n"], m["accuracy"], m["balanced_accuracy"],
                        m["macro_f1"], m["unsafe_raw_action_rate"],
                        m["emergency_forward_proposal_rate"], m["ece"]])
    md = ["# Laya OOD test (held-out families, frozen config)",
          f"Date: {report['generated_utc']}",
          f"Families: {', '.join(ood_families)}", ""]
    for k, m in report["by_mode"].items():
        md.append(f"## {k}: acc={m['accuracy']} bal={m['balanced_accuracy']} "
                  f"F1={m['macro_f1']} unsafe={m['unsafe_raw_action_rate']} "
                  f"ece={m['ece']}")
    md.append("")
    md.append(f"Constrained failures: {len(report['constrained_failures'])}")
    for f in report["constrained_failures"][:20]:
        md.append(f"- {f['family']} {f['category']}: oracle={f['oracle']} "
                  f"proposed={f['proposed']} (n={f['n']})")
    (PROJECT_ROOT / "results" / "laya_ood.md").write_text("\n".join(md))
    print(f"ood n={report['n_ood']} failures={len(report['constrained_failures'])}")
    store.close()
    return report


if __name__ == "__main__":
    main()
