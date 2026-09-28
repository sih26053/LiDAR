"""Analyze probe-log category yields per family."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

rows = [json.loads(l) for l in
        open(PROJECT_ROOT / "results" / "probe_log.jsonl")]
cats = Counter((r["family"], r["category"]) for r in rows)
print("attempts:", len(rows))
for k in sorted(cats):
    print(k, cats[k])
