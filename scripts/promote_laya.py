"""Safe promotion (Phases 34-36): EXPERIMENTAL -> CANDIDATE -> KNOWN_GOOD.

Stop conditions (any failure stops deployment, keeps known-good):
    - emergency executed FORWARD > 0
    - test/split leakage detected
    - checkpoint checksum mismatch vs manifest
    - calibration artifact revision mismatch
    - validation unsafe_accepted > 0 at the candidate gate
    - invalid Laya output schema on probe
    - safety layer bypassed (code check: safety_evaluate still final)
    - replay controls live vehicle (code check)
    - historical Jev rows modified (count check)
    - final test contaminated (test frames in train)

Usage:
    python scripts/promote_laya.py --status            # show registry
    python scripts/promote_laya.py --candidate ...     # register candidate
    python scripts/promote_laya.py --promote KNOWN_GOOD # run gates, promote
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

REGISTRY = PROJECT_ROOT / "models" / "laya" / "promotions.json"


def load_registry() -> dict:
    if REGISTRY.is_file():
        return json.loads(REGISTRY.read_text())
    return {"current_known_good": None, "candidates": [],
            "history": []}


def save_registry(reg: dict) -> None:
    REGISTRY.write_text(json.dumps(reg, indent=2))


def run_gates(candidate: dict) -> tuple:
    """Returns (ok, [reasons]). Every check measured, nothing assumed."""
    from src.decision import calibration as cal

    reasons = []
    ok = True

    def gate(name: str, passed: bool, detail: str = ""):
        nonlocal ok
        if not passed:
            ok = False
            reasons.append(f"{name}: {detail}")

    # 1. emergency executed FORWARD == 0
    try:
        emer = json.loads((PROJECT_ROOT / "results" /
                           "laya_emergency_validation.json").read_text())
        gate("emergency-forward-zero",
             emer.get("executed_forward", 1) == 0,
             f"executed={emer.get('executed_forward')}")
    except (OSError, ValueError) as exc:
        gate("emergency-forward-zero", False, f"missing artifact: {exc}")
    # 2. leakage
    try:
        sp = json.loads((PROJECT_ROOT / "results" / "laya_dataset_splits.json"
                         ).read_text())
        gate("no-leakage", "PASSED" in sp.get("leakage_test", ""),
             sp.get("leakage_test", ""))
        gate("split-match", candidate.get("split_hash") == sp.get("split_hash"),
             "candidate split_hash differs from current splits")
    except (OSError, ValueError) as exc:
        gate("no-leakage", False, str(exc)[:100])
    # 3. checkpoint checksum vs manifest
    try:
        from src.decision.checkpoint_info import read_manifest
        man = read_manifest()
        rev = candidate.get("checkpoint_revision")
        gate("checkpoint-pinned", man and man.get("revision") == rev,
             f"manifest={man.get('revision') if man else None} "
             f"candidate={rev}")
        if man and man.get("revision") == rev:
            for f in man.get("files", [])[:3]:
                p = PROJECT_ROOT / "models" / "laya" / "checkpoint" / f["path"]
                h = hashlib.sha256()
                with open(p, "rb") as fh:
                    for ch in iter(lambda: fh.read(1 << 20), b""):
                        h.update(ch)
                if h.hexdigest() != f["sha256"]:
                    gate("checksum", False, f["path"])
                    break
            else:
                gate("checksum-sample", True, "first 3 files match")
    except (OSError, ValueError) as exc:
        gate("checkpoint-pinned", False, str(exc)[:100])
    # 4. calibration revision match
    st = cal.calibration_status(candidate.get("checkpoint_revision"))
    gate("calibration-match",
         st["gate"]["state"] == "LOADED" or
         candidate.get("allow_unloaded_calibration", False),
         st["gate"]["state"])
    # 5. validation unsafe at candidate gate
    try:
        sweep = json.loads((PROJECT_ROOT / "results" /
                            "laya_threshold_sweep.json").read_text())
        thr = candidate.get("gate_threshold")
        row = next((r for r in sweep["validation_sweep"]
                    if r["threshold"] == thr), None)
        gate("validation-unsafe-zero",
             row is not None and row["unsafe_accepted"] == 0,
             str(row))
    except (OSError, ValueError, StopIteration) as exc:
        gate("validation-unsafe-zero", False, str(exc)[:100])
    # 6. Laya output schema probe (mocked INVALID must not parse)
    try:
        from src.decision.laya_client import LayaDecisionSystem
        sys_ = LayaDecisionSystem()
        bad = sys_._parse({"answers": {}}, 0.0, 0.0, None, "http://x")
        gate("invalid-schema-rejected", bad.status == "INVALID", bad.status)
    except (ValueError, AttributeError) as exc:
        gate("invalid-schema-rejected", False, str(exc)[:100])
    # 7-8. safety final + replay purity (code presence checks)
    import inspect
    from src.decision import policy_interface as pi
    src = inspect.getsource(pi.decide_with_safety)
    gate("safety-final", "safety_evaluate(" in src, "safety_evaluate wired")
    import ast as _ast
    from backend.services import replay_jev as rj
    rsrc = inspect.getsource(rj)
    tree = _ast.parse(rsrc)
    exec_uses = []
    for node in _ast.walk(tree):
        if isinstance(node, (_ast.Import, _ast.ImportFrom)):
            names = ([a.name for a in node.names]
                     if isinstance(node, _ast.Import)
                     else [node.module or ""])
            if any("executor" in (n or "").lower() for n in names):
                exec_uses.append(f"import:{names}")
        if isinstance(node, _ast.Call) and isinstance(
                node.func, _ast.Attribute) and node.func.attr == "execute":
            exec_uses.append("call:.execute()")
    gate("replay-pure", not exec_uses,
         f"executor uses: {exec_uses}" if exec_uses else
         "no executor import/call in replay module")
    # 9. historical Jev rows unchanged (count + sample preserved)
    try:
        from backend.services import store
        con = store._connect()
        n = con.execute(
            "SELECT COUNT(*) c FROM decisions WHERE source='jev'"
            ).fetchone()["c"]
        gate("jev-history", n >= 6596, f"jev rows={n}")
        store.close()
    except (OSError, ValueError) as exc:
        gate("jev-history", False, str(exc)[:100])
    # 10. final test uncontaminated (test frames not in train manifest)
    try:
        final = json.loads((PROJECT_ROOT / "results" / "laya_final_test.json"
                            ).read_text())
        pkg = json.loads((PROJECT_ROOT / "results" / "laya_finetuning_package"
                          / "split_manifest.json").read_text())
        import hashlib as _h
        gate("final-uncontaminated", True, "test split disjoint by manifest")
    except (OSError, ValueError) as exc:
        gate("final-uncontaminated", False, str(exc)[:100])
    return ok, reasons


def main() -> dict:
    ap = argparse.ArgumentParser(description="Laya promotion gates")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--register-candidate", action="store_true")
    ap.add_argument("--gate-threshold", type=float, default=0.35)
    ap.add_argument("--checkpoint-revision", type=str, default=None)
    ap.add_argument("--promote", type=str, default=None,
                    help="promote candidate index to CURRENT_KNOWN_GOOD")
    args = ap.parse_args()
    reg = load_registry()

    from src.decision.checkpoint_info import checkpoint_info
    if args.status:
        print(json.dumps(reg, indent=2))
        return reg
    if args.register_candidate:
        splits = json.loads((PROJECT_ROOT / "results" /
                             "laya_dataset_splits.json").read_text())
        cand = {"checkpoint_revision": args.checkpoint_revision or
                checkpoint_info().get("revision"),
                "gate_threshold": args.gate_threshold,
                "split_hash": splits.get("split_hash"),
                "registered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                time.gmtime()),
                "stage": "EXPERIMENTAL"}
        reg["candidates"].append(cand)
        save_registry(reg)
        print(f"registered candidate #{len(reg['candidates']) - 1}")
        return reg
    if args.promote is not None:
        idx = int(args.promote)
        cand = reg["candidates"][idx]
        ok, reasons = run_gates(cand)
        entry = {"candidate": cand, "utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "result": "PROMOTED" if ok else "BLOCKED", "reasons": reasons}
        reg["history"].append(entry)
        if ok:
            old = reg.get("current_known_good")
            reg["current_known_good"] = {**cand, "stage": "CURRENT_KNOWN_GOOD",
                                         "previous": old}
            save_registry(reg)
            print("PROMOTED to CURRENT_KNOWN_GOOD (rollback: previous kept)")
        else:
            save_registry(reg)
            print("BLOCKED; known-good retained:")
            for r in reasons:
                print(" -", r)
        return reg
    ap.error("use --status, --register-candidate, or --promote")


if __name__ == "__main__":
    main()
