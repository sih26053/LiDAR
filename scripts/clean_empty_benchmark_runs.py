"""Remove empty benchmark probe runs (zero frames)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.services import store

con = store._connect()
n = con.execute(
    "SELECT COUNT(*) c FROM runs WHERE run_id LIKE 'benchmark-%'").fetchone()["c"]
print("benchmark runs:", n)
con.execute(
    "DELETE FROM runs WHERE run_id LIKE 'benchmark-%' AND run_id NOT IN "
    "(SELECT DISTINCT run_id FROM frames WHERE run_id LIKE 'benchmark-%')")
con.commit()
print("cleaned empty probe runs")
store.close()
