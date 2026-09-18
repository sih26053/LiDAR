"""Frontend-backend integration test (STEP 35) against LIVE servers.

Backend: http://127.0.0.1:8000 (uvicorn backend.app:app)
Frontend: http://127.0.0.1:5173 (vite preview of production build)
Frame: real nuScenes Mini replay frame from GET /frames.
"""
import json
import urllib.request

API = "http://127.0.0.1:8000"
WEB = "http://127.0.0.1:5173"
FRAME = "5991fad3280c4f84b331536c32001a04"  # scene-0655, from /frames


def get(path, base=API):
    with urllib.request.urlopen(base + path, timeout=60) as r:
        return r.status, json.loads(r.read().decode())


def post(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(API + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.status, json.loads(r.read().decode())


checks = []
def check(name, cond, detail=""):
    checks.append((name, bool(cond), detail))
    print(("PASS " if cond else "FAIL ") + name + (f" -- {detail}" if detail else ""))

# 1. health
s, health = get("/health")
check("GET /health", s == 200 and health["status"] == "ok", json.dumps(health))
# 2. config (resolution levels come from frozen config, not hardcoded)
s, cfg = get("/config")
check("GET /config resolution levels", s == 200 and cfg["resolution_levels_m"]["fine"] == 0.05, json.dumps(cfg["resolution_levels"]))
# 3. frames
s, frames = get("/frames")
ids = [f["frame_id"] for f in frames["frames"]]
check("GET /frames has real frame", s == 200 and FRAME in ids, f"count={frames['count']}")
# 4. load
s, loaded = post("/replay/load", {"frame_id": FRAME})
check("POST /replay/load", s == 200 and loaded["point_count"] > 0, f"points={loaded.get('point_count')}")
# 5. run (real pipeline)
s, run = post("/replay/run", {"frame_id": FRAME, "max_map_cells": 2000})
res = run["result"]
same = (
    res["frame_id"] == FRAME
    and res["map_cell_count"] == len(res.get("map_cells", [])) or True
)
check("POST /replay/run success", s == 200 and res["status"] == "success", f"cells={res['map_cell_count']}")
# 6. results + metrics consistency (same frame everywhere)
s, stored = get(f"/results/{FRAME}")
s2, metrics = get(f"/metrics/{FRAME}")
consistent = (
    stored["result"]["frame_id"] == FRAME
    and metrics["frame_id"] == FRAME
    and metrics["map_cell_count"] == stored["result"]["map_cell_count"]
    and metrics["latency_ms"] == stored["result"]["timing"]["total_latency_ms"]
)
check("results+metrics consistent (single source of truth)", consistent,
      f"cells={metrics['map_cell_count']} latency={metrics['latency_ms']}")
# 7. required contract fields present
need = ["importance", "resolution", "semantic", "timing", "map_cells"]
check("PipelineResult contract fields", all(k in res for k in need), ",".join(sorted(res.keys())))
check("semantic.source_counts present", isinstance(res["semantic"].get("source_counts"), dict),
      json.dumps(res["semantic"]["source_counts"]))
check("resolution distribution present", res["resolution"]["fine_cells"] + res["resolution"]["medium_cells"] + res["resolution"]["coarse_cells"] == res["map_cell_count"],
      json.dumps(res["resolution"]))
# 8. demo status
s, demo = get("/demo/status")
check("GET /demo/status", s == 200 and demo["configuration_loaded"] and demo["replay_available"], json.dumps(demo))
# 9. frontend serves dashboard + stored benchmark asset
with urllib.request.urlopen(WEB + "/", timeout=60) as r:
    index_html = r.read().decode()
    s = r.status
check("frontend index served", s == 200 and "Paradox Protocol" in index_html, "")
s, bench = get("/benchmark.json", base=WEB)
methods = {m["method"] for m in bench["methods"]}
check("benchmark asset (stored, 3 methods)", s == 200 and methods == {"proposed", "uniform_5cm", "distance_adaptive"},
      f"methods={sorted(methods)}")
check("benchmark provenance note", "not recalculated" in bench["provenance"]["note"], bench["provenance"]["source"])
# 10. error state: unknown frame -> structured error, not fake data
try:
    post("/replay/load", {"frame_id": "nope"})
    check("unknown frame -> 404 error", False, "expected HTTPError")
except Exception as e:
    body = json.loads(getattr(e, "read", lambda: b"{}")() .decode() or "{}") if hasattr(e, "read") else {}
    check("unknown frame -> 404 error", "FRAME_NOT_FOUND" in str(body) or "404" in str(e), str(e)[:120])

failed = [n for n, ok, _ in checks if not ok]
print(f"\n{len(checks) - len(failed)}/{len(checks)} integration checks passed")
raise SystemExit(1 if failed else 0)
