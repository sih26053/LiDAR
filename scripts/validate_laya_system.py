"""Master validation command (Phase 39): inventory -> splits -> eval baseline
-> option order -> finetune package -> calibration -> final test -> OOD ->
emergency -> live integration -> promotion gates -> regression freeze.

Skips only steps needing unavailable resources (GPU training, Docker,
detached-server survival) with explicit NOT VALIDATED marks. Reuses
existing artifacts when fresh (no redundant Laya calls unless --fresh).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PY = str(PROJECT_ROOT / ".venv-pb" / "Scripts" / "python.exe")


def exe() -> str:
    return PY if Path(PY).is_file() else sys.executable


def run_step(name: str, *args: str, allow_fail: bool = False) -> bool:
    print(f"\n===== {name}: {' '.join(args)} =====", flush=True)
    rc = subprocess.call([exe(), *args], cwd=str(PROJECT_ROOT))
    ok = rc == 0
    print(f"===== {name}: {'OK' if ok else f'FAILED({rc})'} =====",
          flush=True)
    if not ok and not allow_fail:
        raise SystemExit(f"master stopped at failing step: {name}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate the Laya system")
    ap.add_argument("--fresh", action="store_true",
                    help="re-run Laya-call-heavy steps (baseline, calib); "
                         "default reuses measured artifacts")
    ap.add_argument("--skip-live", action="store_true",
                    help="skip the live PyBullet integration run")
    ap.add_argument("--from-step", type=str, default=None)
    args = ap.parse_args()

    steps = [
        ("resources", ["scripts/detect_resources.py"], False),
        ("inventory", ["scripts/inventory_laya_dataset.py"], False),
        ("audit-recorded", ["scripts/audit_recorded_laya.py"], False),
        ("build-dataset", ["scripts/build_navigation_dataset.py"], False),
        ("splits", ["scripts/make_dataset_splits.py",
                    "--min-per-category", "50"], False),
    ]
    if args.fresh:
        steps += [
            ("baseline", ["scripts/evaluate_dataset.py"], False),
            ("option-order", ["scripts/test_option_order.py"], False),
            ("calibration", ["scripts/calibrate_from_splits.py",
                             "--fresh-calib"], False),
        ]
    steps += [
        ("finetune-package", ["scripts/prepare_finetuning_package.py"], False),
        ("final-test", ["scripts/final_test.py"], False),
        ("ood", ["scripts/ood_report.py"], False),
        ("ab", ["scripts/evaluate_laya_ab.py"], True),
    ]
    if args.fresh:
        steps.append(("emergency", ["scripts/validate_emergency.py"], False))
    if not args.skip_live:
        steps.append(("live", ["scripts/real_integration_test.py"], False))
    steps += [
        ("regression-freeze", ["scripts/freeze_regression_baseline.py"], False),
        ("promotion-status", ["scripts/promote_laya.py", "--status"], False),
    ]
    started = args.from_step is None
    for name, cmd, allow_fail in steps:
        if not started:
            if name == args.from_step:
                started = True
            else:
                print(f"--- skip {name} (before --from-step) ---")
                continue
        run_step(name, *cmd, allow_fail=allow_fail)
    print(f"\nDone {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}. "
          "See results/LAYA_VALIDATION_REPORT.md for the master report.")


if __name__ == "__main__":
    main()
