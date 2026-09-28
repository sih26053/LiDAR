"""Tests for model / tracking / stream / environment endpoints (all real-data)."""

from fastapi.testclient import TestClient

from backend.app import app

client = TestClient(app)
REAL_FRAME = "5991fad3280c4f84b331536c32001a04"


def test_model_info_reports_measured_metrics():
    r = client.get("/model/info")
    assert r.status_code == 200, r.text[:300]
    body = r.json()
    assert body["classes"] == ["vehicle", "pedestrian_vru", "static_manmade",
                               "road_driveable", "unknown"]
    assert body["trained"] is True
    m = body["metrics_held_out"]
    assert m is not None, "held-out evaluation must be generated (scripts/evaluate_segmentation.py)"
    assert 0.0 <= m["accuracy"] <= 1.0
    assert 0.0 <= m["f1_macro"] <= 1.0
    assert set(m["f1_per_class"]) == set(body["classes"])
    assert body["model_file"].endswith(".pkl")


def test_model_predict_live_values():
    r = client.post(f"/model/predict/{REAL_FRAME}")
    assert r.status_code == 200, r.text[:300]
    body = r.json()
    assert body["input_point_count"] > 30_000
    dist = body["predicted_distribution"]
    assert sum(dist.values()) == body["input_point_count"]
    assert 0.0 <= body["mean_confidence"] <= 1.0
    assert 0.0 <= (body["agreement_with_annotation_foreground"] or 0.0) <= 1.0
    assert body["timing"]["inference_latency_ms"] > 0


def test_replay_run_model_mode_has_model_cells():
    r = client.post("/replay/run", json={"frame_id": REAL_FRAME, "semantic_mode": "model"})
    assert r.status_code == 200, r.text[:300]
    res = r.json()["result"]
    assert "trained" in res["semantic"]["mode"].lower()
    assert res["semantic"]["source_counts"].get("model", 0) == res["map_cell_count"] > 0


def test_replay_run_default_mode_unchanged():
    r = client.post("/replay/run", json={"frame_id": REAL_FRAME})
    assert r.status_code == 200, r.text[:300]
    res = r.json()["result"]
    assert "model" not in res["semantic"]["mode"].lower()


def test_tracking_scene_sequence():
    r = client.post("/tracking/run", json={"scene_id": "scene-0061", "scene_samples": 4})
    assert r.status_code == 200, r.text[:300]
    body = r.json()
    assert len(body["frame_ids"]) == 4
    assert body["n_tracks"] > 0
    assert body["n_scene_breaks"] == 0
    multi = [t for t in body["tracks"] if t["age_frames"] >= 2]
    assert len(multi) > 0
    speeds = [s["speed_m_s"] for t in multi for s in t["states"] if s["speed_m_s"] is not None]
    assert all(v is not None and v >= 0 for v in speeds)


def test_tracking_refuses_cross_scene_links():
    r = client.post("/tracking/run", json={})
    assert r.status_code == 200, r.text[:300]
    body = r.json()
    # Manifest frames span scenes -> every boundary must be a recorded break,
    # never a cross-scene association.
    assert body["n_scene_breaks"] == len(body["frame_ids"]) - 1


def test_stream_simulate_matches_live_path():
    r = client.post("/stream/simulate", json={"frame_id": REAL_FRAME, "semantic_mode": "annotation"})
    assert r.status_code == 200, r.text[:300]
    res = r.json()["result"]
    assert res["semantic"]["source_counts"].get("fallback", 0) == res["map_cell_count"] > 0
    assert "simulated" in (res["input_source"] or "")


def test_stream_push_external_points():
    import numpy as np

    pts = (np.random.RandomState(7).rand(2000, 4) * 40 - 20).tolist()
    r = client.post("/stream/push", json={"points": pts, "semantic_mode": "model"})
    assert r.status_code == 200, r.text[:300]
    res = r.json()["result"]
    assert res["map_cell_count"] > 0
    assert res["input_source"] == "external-stream"


def test_stream_push_rejects_bad_input():
    r = client.post("/stream/push", json={"points": [[1.0, 2.0, 3.0]] * 200})
    assert r.status_code == 400
    # NaN cannot survive client-side JSON encoding, so send a raw body.
    raw = ('{"points": [[1.0, 2.0, 3.0, NaN], ' +
           ', '.join(['[1.0, 2.0, 3.0, 0.0]'] * 199) + ']}')
    r = client.post("/stream/push", content=raw,
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 400


def test_replay_load_reports_io_latency():
    r = client.post("/replay/load", json={"frame_id": REAL_FRAME})
    assert r.status_code == 200, r.text[:300]
    assert r.json()["load_latency_ms"] is not None
    assert r.json()["load_latency_ms"] >= 0


def test_environment_detects_real_values():
    r = client.get("/system/environment")
    assert r.status_code == 200, r.text[:300]
    body = r.json()
    assert body["cpu_model"] not in ("", "Unavailable")
    assert body["python_version"]
    assert body["packages"]["numpy"] not in ("", "Unavailable")
