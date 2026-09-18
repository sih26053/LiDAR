"""Schema contract tests: stable request/response shapes for the frontend."""

from backend.schemas.input import InputFrame
from backend.schemas.output import PipelineResult


def test_input_frame_contract():
    f = InputFrame(frame_id="5991fad3280c4f84b331536c32001a04", scene_id="scene-0655",
                   timestamp=1535385092150099.0)
    assert f.frame_id
    assert f.timestamp is not None


def test_pipeline_result_contract_keys():
    r = PipelineResult(frame_id="x", timestamp=1.0)
    d = r.model_dump()
    for key in ("frame_id", "timestamp", "map_cells", "importance",
                "resolution", "semantic", "timing"):
        assert key in d


def test_error_contract_keys():
    from backend.schemas.errors import ErrorResponse

    e = ErrorResponse(stage="mapping", error_code="MAPPING_FAILED",
                      message="m", frame_id="x")
    assert set(e.model_dump().keys()) == {"stage", "error_code", "message", "frame_id"}
