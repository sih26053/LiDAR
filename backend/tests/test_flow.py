"""Tests for the 9-stage methodology flow (all real-data, nothing faked)."""

import numpy as np
from fastapi.testclient import TestClient

from backend.app import app

client = TestClient(app)
REAL_FRAME = "5991fad3280c4f84b331536c32001a04"


def test_flow_status_reports_nine_stages():
    r = client.get("/flow/status")
    assert r.status_code == 200, r.text[:300]
    body = r.json()
    assert len(body["stages"]) == 9
    by_box = {s["box"]: s for s in body["stages"]}
    assert by_box[2]["status"] == "LIVE"
    assert "untrained" in by_box[7]["status"].lower()
    assert "BLOCKED" in by_box[9]["status"]
    assert all(s["evidence"] for s in body["stages"])
    assert set(body["perception_taxonomy"].values()) == {
        "Road", "Vehicle", "Pedestrian", "Static Obstacle", "Others"}


def test_rl_decide_uses_stored_result():
    c0 = client.post("/replay/run", json={"frame_id": REAL_FRAME, "semantic_mode": "model"})
    assert c0.status_code == 200, c0.text[:300]
    r = client.get(f"/rl/decide/{REAL_FRAME}")
    assert r.status_code == 200, r.text[:300]
    body = r.json()
    assert body["state"]["state_dim"] == 13
    assert len(body["state"]["state_vector"]) == 13
    assert len(body["decision"]["q_values"]) == 4
    assert body["decision"]["trained"] is False
    assert body["decision"]["final_action"] in ("forward", "turn_left", "turn_right", "stop")
    assert body["decide_latency_ms"] > 0


def test_rl_decide_missing_frame_is_404():
    r = client.get("/rl/decide/00000000000000000000000000000000")
    assert r.status_code == 404


def test_quadtree_derived_from_stored_cells():
    client.post("/replay/run", json={"frame_id": REAL_FRAME, "semantic_mode": "model"})
    r = client.get(f"/map/quadtree/{REAL_FRAME}")
    assert r.status_code == 200, r.text[:300]
    body = r.json()
    assert body["n_leaves"] > 0
    assert sum(body["leaf_resolution_mix"].values()) == body["n_leaves"]
    assert set(body["leaf_resolution_mix"]) <= {"0.05", "0.1", "0.2", "0.5"}


def test_scene_analysis_six_factors():
    from backend.config import load_failsafe_config, load_final_config
    from backend.services import replay_service
    from backend.services.pipeline_service import _nusc_dataset
    from src.scene_analysis import analyze_frame, scene_summary
    from src.semantic_adapter import aggregate_to_regions, build_perception_from_semantics
    from src.semantic_model import predict_points

    nusc = _nusc_dataset()
    raw, _ = replay_service.load_frame_points(REAL_FRAME)
    pts = np.asarray(raw)
    labels, sources, conf, _ = predict_points(pts)
    pr, src_arr, _ = build_perception_from_semantics(
        REAL_FRAME, pts, labels.tolist(), sources.tolist(),
        confidence=np.asarray(conf))
    _, detail_rows = aggregate_to_regions(pr, src_arr, cell_size=2.0)
    sdf = analyze_frame(detail_rows)
    assert {"distance_from_lidar_m", "point_density", "object_density",
            "object_importance", "elevation_variation",
            "prediction_uncertainty"} <= set(sdf.columns)
    s = scene_summary(sdf)
    assert s["n_regions"] == len(sdf) > 0


def test_map_cells_carry_diagram_fields():
    r = client.post("/replay/run", json={"frame_id": REAL_FRAME, "semantic_mode": "model"})
    assert r.status_code == 200, r.text[:300]
    cell = r.json()["result"]["map_cells"][0]
    # Flow box 5: Elevation (Z), Semantic Class, Occupancy, Confidence, Resolution.
    for field in ("elevation", "semantic_class", "occupancy", "confidence", "resolution"):
        assert field in cell, field
    assert cell["confidence"] is not None  # measured softmax output in model channel
