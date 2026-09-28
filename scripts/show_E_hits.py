"""Show G-category (non-emergency) probe rows with geometry."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

for line in open(PROJECT_ROOT / "results" / "probe_log.jsonl"):
    r = json.loads(line)
    if r["category"] == "G_NO_SAFE_ACTION" and not r["emergency"]:
        print(json.dumps(r))
