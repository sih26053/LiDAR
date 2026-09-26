"""Safety-controller rule evaluation (honest scope).

Sweeps the genuine rule over synthetic inputs: obstacle distances x
DQN actions x map-validity x emergency flag. Output is a measurement
of RULE LOGIC, not of driving episodes -- labelled as such in every row
and in the dashboard. No simulator, vehicle, or collision data involved.

Output: results/rl/safety_evaluation.csv

Run:  python scripts/evaluate_safety.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.safety_controller import evaluate  # noqa: E402

DISTANCES = [0.5, 1.0, 2.5, 4.9, 5.0, 5.1, 10.0, 40.0, None]
ACTIONS = ["forward", "turn_left", "turn_right", "stop", "teleport"]


def main() -> None:
    rows = []
    for d in DISTANCES:
        for a in ACTIONS:
            for valid in (True, False):
                for emerg in (False, True):
                    v = evaluate(a, d, map_valid=valid, emergency=emerg)
                    rows.append({
                        "nearest_obstacle_m": d,
                        "dqn_action": a,
                        "map_valid": valid,
                        "emergency": emerg,
                        "final_action": v["final_action"],
                        "verdict": v["verdict"],
                        "scope": "rule-logic sweep (synthetic inputs; NOT a driving episode)",
                    })
    df = pd.DataFrame(rows)
    out = PROJECT_ROOT / "results" / "rl"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "safety_evaluation.csv", index=False)
    over = int((df["verdict"] == "OVERRIDE_TO_STOP").sum())
    print(f"cases: {len(df)} | overrides: {over} | "
          f"forward-clear safe: {int(((df.dqn_action == 'forward') & (df.nearest_obstacle_m == 40.0) & df.map_valid & ~df.emergency & (df.verdict == 'SAFE_TO_EXECUTE')).sum())}",
          flush=True)


if __name__ == "__main__":
    main()
