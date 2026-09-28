"""SQLite persistence for recorded live frames (Phase 2/9).

Single local file (default ``data/paradox_protocol.db``), stdlib
``sqlite3`` only — no server. Tables: runs, frames, map_cells,
decisions, safety_events, executions (+ schema_version). Additive:
in-memory decision_store/result_service keep working; this store is
the durable layer underneath.

Conventions: parameterized SQL everywhere, WAL mode, foreign keys,
indexes on all query paths, portable RELATIVE paths in the DB.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("paradox.backend.store")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "paradox_protocol.db"

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    mode TEXT NOT NULL,
    simulator TEXT NOT NULL DEFAULT 'pybullet',
    scenario TEXT,
    jev_model TEXT,
    decision_model TEXT,
    total_steps INTEGER NOT NULL DEFAULT 0,
    collisions INTEGER NOT NULL DEFAULT 0,
    distance REAL NOT NULL DEFAULT 0.0,
    status TEXT NOT NULL DEFAULT 'RUNNING',
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS frames (
    frame_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    timestamp REAL,
    sequence INTEGER NOT NULL,
    scenario TEXT,
    points_path TEXT NOT NULL,
    point_count INTEGER NOT NULL,
    state_vector_json TEXT NOT NULL DEFAULT '[]',
    vehicle_pose_json TEXT NOT NULL DEFAULT '{}',
    pipeline_timing_json TEXT NOT NULL DEFAULT '{}',
    metadata_path TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS map_cells (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id TEXT NOT NULL REFERENCES frames(frame_id),
    x REAL NOT NULL, y REAL NOT NULL,
    elevation REAL, occupancy REAL, resolution REAL,
    importance REAL, semantic_class TEXT, confidence REAL,
    dynamic_relevance REAL, uncertainty REAL,
    extra_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id TEXT NOT NULL REFERENCES frames(frame_id),
    replay_run_id TEXT,
    source TEXT NOT NULL,
    model TEXT,
    proposed_action TEXT,
    confidence REAL,
    probabilities_json TEXT,
    latency_ms REAL,
    request_id TEXT,
    error TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS safety_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id TEXT NOT NULL REFERENCES frames(frame_id),
    replay_run_id TEXT,
    status TEXT,
    override INTEGER NOT NULL DEFAULT 0,
    reason TEXT,
    emergency_flag INTEGER NOT NULL DEFAULT 0,
    nearest_obstacle_m REAL,
    forward_clearance_m REAL,
    left_clearance_m REAL,
    right_clearance_m REAL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS executions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id TEXT NOT NULL REFERENCES frames(frame_id),
    replay_run_id TEXT,
    source TEXT NOT NULL,
    executed_action TEXT,
    dx REAL, dy REAL, dyaw REAL,
    collision INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_frames_run ON frames(run_id);
CREATE INDEX IF NOT EXISTS idx_frames_ts ON frames(timestamp);
CREATE INDEX IF NOT EXISTS idx_frames_scenario ON frames(scenario);
CREATE INDEX IF NOT EXISTS idx_cells_frame ON map_cells(frame_id);
CREATE INDEX IF NOT EXISTS idx_decisions_frame ON decisions(frame_id);
CREATE INDEX IF NOT EXISTS idx_decisions_source ON decisions(source);
CREATE INDEX IF NOT EXISTS idx_safety_frame ON safety_events(frame_id);
CREATE INDEX IF NOT EXISTS idx_exec_frame ON executions(frame_id);
"""

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None
_db_path: Path = DEFAULT_DB_PATH


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def configure(path: str | Path | None = None) -> Path:
    """Set the DB file path (tests use a temp file). Reopens lazily."""
    global _conn, _db_path
    with _lock:
        if _conn is not None:
            try:
                _conn.close()
            except Exception:
                pass
            _conn = None
        _db_path = Path(path) if path else DEFAULT_DB_PATH
        return _db_path


def _connect() -> sqlite3.Connection:
    global _conn
    with _lock:
        if _conn is None:
            _db_path.parent.mkdir(parents=True, exist_ok=True)
            _conn = sqlite3.connect(str(_db_path), check_same_thread=False,
                                    timeout=30.0)
            _conn.row_factory = sqlite3.Row
            _conn.execute("PRAGMA journal_mode=WAL;")
            _conn.execute("PRAGMA foreign_keys=ON;")
            _conn.executescript(_SCHEMA)
            cols = [r["name"] for r in
                    _conn.execute("PRAGMA table_info(runs)").fetchall()]
            if "decision_model" not in cols:
                _conn.execute("ALTER TABLE runs ADD COLUMN decision_model TEXT")
                _conn.execute("UPDATE runs SET decision_model = jev_model"
                              " WHERE decision_model IS NULL")
                _conn.commit()
            # Additive Laya-mitigation columns (historical rows stay NULL;
            # historical Jev rows are never rewritten).
            dcols = [r["name"] for r in
                     _conn.execute("PRAGMA table_info(decisions)").fetchall()]
            for _col, _typ in (
                    ("answer_confidence", "REAL"),
                    ("mode", "TEXT"),
                    ("eligible_json", "TEXT"),
                    ("raw_action", "TEXT"),
                    ("constrained_action", "TEXT"),
                    ("checkpoint_revision", "TEXT"),
                    ("gate_threshold", "REAL")):
                if _col not in dcols:
                    _conn.execute(
                        f"ALTER TABLE decisions ADD COLUMN {_col} {_typ}")
            _conn.commit()
            row = _conn.execute(
                "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1").fetchone()
            if row is None:
                _conn.execute("INSERT INTO schema_version (version, applied_utc) VALUES (?, ?)",
                              (SCHEMA_VERSION, _utc()))
                _conn.commit()
        return _conn


def close() -> None:
    global _conn
    with _lock:
        if _conn is not None:
            try:
                _conn.close()
            except Exception:
                pass
            _conn = None


def db_path() -> Path:
    return _db_path


def _dumps(obj: Any) -> str:
    return json.dumps(obj, default=str)


# ---- runs ---------------------------------------------------------------

def create_run(run_id: str, mode: str, scenario: str | None = None,
               jev_model: str | None = None,
               decision_model: str | None = None,
               metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
    con = _connect()
    rec = {"run_id": run_id, "started_at": _utc(), "ended_at": None,
           "mode": mode, "simulator": "pybullet", "scenario": scenario,
           "jev_model": jev_model,
           "decision_model": decision_model or jev_model,
           "total_steps": 0, "collisions": 0,
           "distance": 0.0, "status": "RUNNING",
           "metadata_json": _dumps(metadata or {})}
    with _lock:
        con.execute(
            "INSERT OR REPLACE INTO runs (run_id, started_at, ended_at, mode, simulator,"
            " scenario, jev_model, decision_model, total_steps, collisions, distance,"
            " status, metadata_json)"
            " VALUES (:run_id, :started_at, :ended_at, :mode, :simulator, :scenario,"
            " :jev_model, :decision_model, :total_steps, :collisions, :distance,"
            " :status, :metadata_json)", rec)
        con.commit()
    return rec


def finish_run(run_id: str, total_steps: int = 0, collisions: int = 0,
               distance: float = 0.0, status: str = "COMPLETED") -> None:
    con = _connect()
    with _lock:
        con.execute("UPDATE runs SET ended_at=?, total_steps=?, collisions=?,"
                    " distance=?, status=? WHERE run_id=?",
                    (_utc(), int(total_steps), int(collisions),
                     float(distance), status, run_id))
        con.commit()


def get_run(run_id: str) -> Dict[str, Any] | None:
    con = _connect()
    row = con.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
    return dict(row) if row else None


def list_runs(limit: int = 100) -> List[Dict[str, Any]]:
    con = _connect()
    rows = con.execute(
        "SELECT r.*, (SELECT COUNT(*) FROM frames f WHERE f.run_id=r.run_id) AS frame_count"
        " FROM runs r ORDER BY r.started_at DESC LIMIT ?", (max(1, min(int(limit), 1000)),))
    return [dict(r) for r in rows]


# ---- frames + maps ------------------------------------------------------

def create_frame(frame_id: str, run_id: str, timestamp: float | None,
                 sequence: int, scenario: str | None, points_path: str,
                 point_count: int, state_vector: List[float],
                 vehicle_pose: Dict[str, Any], pipeline_timing: Dict[str, Any],
                 metadata_path: str) -> Dict[str, Any]:
    con = _connect()
    rec = {"frame_id": frame_id, "run_id": run_id, "timestamp": timestamp,
           "sequence": int(sequence), "scenario": scenario,
           "points_path": points_path, "point_count": int(point_count),
           "state_vector_json": _dumps(list(state_vector or [])),
           "vehicle_pose_json": _dumps(vehicle_pose or {}),
           "pipeline_timing_json": _dumps(pipeline_timing or {}),
           "metadata_path": metadata_path, "created_at": _utc()}
    with _lock:
        con.execute(
            "INSERT OR REPLACE INTO frames (frame_id, run_id, timestamp, sequence, scenario,"
            " points_path, point_count, state_vector_json, vehicle_pose_json,"
            " pipeline_timing_json, metadata_path, created_at)"
            " VALUES (:frame_id, :run_id, :timestamp, :sequence, :scenario, :points_path,"
            " :point_count, :state_vector_json, :vehicle_pose_json, :pipeline_timing_json,"
            " :metadata_path, :created_at)", rec)
        con.commit()
    return rec


def create_map_cells(frame_id: str, cells: List[Dict[str, Any]]) -> int:
    con = _connect()
    rows = []
    for c in cells or []:
        known = {k: c.get(k) for k in
                 ("x", "y", "elevation", "occupancy", "resolution", "importance",
                  "semantic_class", "confidence", "dynamic_relevance", "uncertainty")}
        extra = {k: v for k, v in c.items() if k not in known}
        rows.append((frame_id, known["x"], known["y"], known["elevation"],
                     known["occupancy"], known["resolution"], known["importance"],
                     str(known["semantic_class"]) if known["semantic_class"] is not None else None,
                     known["confidence"], known["dynamic_relevance"], known["uncertainty"],
                     _dumps(extra)))
    with _lock:
        con.executemany(
            "INSERT INTO map_cells (frame_id, x, y, elevation, occupancy, resolution,"
            " importance, semantic_class, confidence, dynamic_relevance, uncertainty, extra_json)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        con.commit()
    return len(rows)


def get_frame(frame_id: str) -> Dict[str, Any] | None:
    con = _connect()
    row = con.execute("SELECT * FROM frames WHERE frame_id=?", (frame_id,)).fetchone()
    return dict(row) if row else None


def list_frames(run_id: str | None = None, scenario: str | None = None,
                limit: int = 500) -> List[Dict[str, Any]]:
    con = _connect()
    q = ("SELECT f.*, (SELECT COUNT(*) FROM map_cells m WHERE m.frame_id=f.frame_id)"
         " AS map_cell_count FROM frames f")
    clauses, args = [], []
    if run_id:
        clauses.append("f.run_id=?")
        args.append(run_id)
    if scenario:
        clauses.append("f.scenario=?")
        args.append(scenario)
    if clauses:
        q += " WHERE " + " AND ".join(clauses)
    q += " ORDER BY f.run_id, f.sequence LIMIT ?"
    args.append(max(1, min(int(limit), 5000)))
    return [dict(r) for r in con.execute(q, args)]


def get_map_cells(frame_id: str) -> List[Dict[str, Any]]:
    con = _connect()
    rows = con.execute("SELECT * FROM map_cells WHERE frame_id=? ORDER BY id", (frame_id,))
    out = []
    for r in rows:
        d = dict(r)
        try:
            d.update(json.loads(d.pop("extra_json") or "{}"))
        except (TypeError, ValueError):
            d.pop("extra_json", None)
        out.append(d)
    return out


# ---- decisions / safety / executions ------------------------------------

def record_decision(frame_id: str, source: str, model: str | None = None,
                    proposed_action: str | None = None,
                    confidence: float | None = None,
                    probabilities: Dict[str, Any] | None = None,
                    latency_ms: float | None = None,
                    request_id: str | None = None,
                    error: str | None = None,
                    replay_run_id: str | None = None,
                    answer_confidence: float | None = None,
                    mode: str | None = None,
                    eligible: List[str] | None = None,
                    raw_action: str | None = None,
                    constrained_action: str | None = None,
                    checkpoint_revision: str | None = None,
                    gate_threshold: float | None = None) -> int:
    con = _connect()
    with _lock:
        cur = con.execute(
            "INSERT INTO decisions (frame_id, replay_run_id, source, model, proposed_action,"
            " confidence, probabilities_json, latency_ms, request_id, error, created_at,"
            " answer_confidence, mode, eligible_json, raw_action,"
            " constrained_action, checkpoint_revision, gate_threshold)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?, ?,?,?, ?,?,?,?)",
            (frame_id, replay_run_id, source, model, proposed_action, confidence,
             _dumps(probabilities) if probabilities is not None else None,
             latency_ms, request_id, error, _utc(),
             answer_confidence, mode,
             _dumps(eligible) if eligible is not None else None,
             raw_action, constrained_action, checkpoint_revision,
             gate_threshold))
        con.commit()
        return int(cur.lastrowid)


def record_safety_event(frame_id: str, status: str | None,
                        override: bool = False, reason: str | None = None,
                        emergency_flag: bool = False,
                        nearest_obstacle_m: float | None = None,
                        forward_clearance_m: float | None = None,
                        left_clearance_m: float | None = None,
                        right_clearance_m: float | None = None,
                        replay_run_id: str | None = None) -> int:
    con = _connect()
    with _lock:
        cur = con.execute(
            "INSERT INTO safety_events (frame_id, replay_run_id, status, override, reason,"
            " emergency_flag, nearest_obstacle_m, forward_clearance_m, left_clearance_m,"
            " right_clearance_m, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (frame_id, replay_run_id, status, 1 if override else 0, reason,
             1 if emergency_flag else 0, nearest_obstacle_m, forward_clearance_m,
             left_clearance_m, right_clearance_m, _utc()))
        con.commit()
        return int(cur.lastrowid)


def record_execution(frame_id: str, source: str, executed_action: str | None,
                     dx: float | None = None, dy: float | None = None,
                     dyaw: float | None = None, collision: bool = False,
                     replay_run_id: str | None = None) -> int:
    con = _connect()
    with _lock:
        cur = con.execute(
            "INSERT INTO executions (frame_id, replay_run_id, source, executed_action,"
            " dx, dy, dyaw, collision, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (frame_id, replay_run_id, source, executed_action, dx, dy, dyaw,
             1 if collision else 0, _utc()))
        con.commit()
        return int(cur.lastrowid)


def get_decision(frame_id: str, source: str | None = None,
                 replay_run_id: str | None = None) -> Dict[str, Any] | None:
    """Latest matching decision (original live row immutable; replays keyed)."""
    con = _connect()
    q = "SELECT * FROM decisions WHERE frame_id=?"
    args: list = [frame_id]
    if source:
        q += " AND source=?"
        args.append(source)
    if replay_run_id:
        q += " AND replay_run_id=?"
        args.append(replay_run_id)
    q += " ORDER BY id DESC LIMIT 1"
    row = con.execute(q, args).fetchone()
    return dict(row) if row else None


def get_replay_history(frame_id: str, source: str = "replay-jev") -> List[Dict[str, Any]]:
    con = _connect()
    rows = con.execute(
        "SELECT * FROM decisions WHERE frame_id=? AND source=?"
        " ORDER BY id", (frame_id, source))
    return [dict(r) for r in rows]


def get_frame_result(frame_id: str) -> Dict[str, Any] | None:
    """Frame + live decision + safety + execution + map count (dashboard)."""
    fr = get_frame(frame_id)
    if fr is None:
        return None
    con = _connect()
    live = get_decision(frame_id, source="jev") or get_decision(frame_id, source="fallback")
    if live is None:  # any non-replay decision (e.g. manual)
        row = con.execute(
            "SELECT * FROM decisions WHERE frame_id=? AND source!='replay-jev'"
            " ORDER BY id DESC LIMIT 1", (frame_id,)).fetchone()
        live = dict(row) if row else None
    safety = con.execute(
        "SELECT * FROM safety_events WHERE frame_id=? AND replay_run_id IS NULL"
        " ORDER BY id DESC LIMIT 1", (frame_id,)).fetchone()
    exe = con.execute(
        "SELECT * FROM executions WHERE frame_id=? AND replay_run_id IS NULL"
        " ORDER BY id DESC LIMIT 1", (frame_id,)).fetchone()
    n = con.execute("SELECT COUNT(*) AS c FROM map_cells WHERE frame_id=?",
                    (frame_id,)).fetchone()
    return {"frame": fr, "live_decision": live,
            "safety": dict(safety) if safety else None,
            "execution": dict(exe) if exe else None,
            "map_cell_count": n["c"] if n else 0,
            "replay_history": get_replay_history(frame_id),
            "replay_laya_history": get_replay_history(frame_id, "replay-laya")}
