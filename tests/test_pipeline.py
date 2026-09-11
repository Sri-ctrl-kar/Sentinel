"""Tests for the wiring of video -> detection -> tracking -> events.

These run against a ``MockDetector``, so the tracking and event paths are the
real ones but no model weights, GPU or video file are needed.
"""

import pytest

from app.config import PipelineConfig
from app.perception.backends.mock import MockDetector
from app.pipeline import PerceptionPipeline, run_pipeline
from app.perception.types import Detection

from helpers import make_detection, make_frames


def build(detector=None, **config_kwargs):
    config = PipelineConfig(detector="mock", classes=None, confidence=0.0, **config_kwargs)
    return PerceptionPipeline(config=config, detector=detector or MockDetector())


def test_pipeline_produces_events_from_a_frame_stream():
    pipeline = build()
    result = pipeline.run_frames(make_frames(30))

    assert result.frames_processed == 30
    assert len(result.events) > 0
    assert result.fps > 0


def test_pipeline_assigns_persistent_ids():
    pipeline = build()
    result = pipeline.run_frames(make_frames(30))

    entities = result.memory.entities()
    # The synthetic mock emits exactly one person and one truck.
    assert sorted(entities) == ["person_1", "truck_2"]


def test_every_entity_appears_once_and_disappears_once():
    pipeline = build()
    result = pipeline.run_frames(make_frames(30))

    for entity_id in result.memory.entities():
        actions = [e.action for e in result.memory.by_entity(entity_id)]
        assert actions.count("appeared") == 1
        assert actions.count("disappeared") == 1
        assert actions[0] == "appeared"
        assert actions[-1] == "disappeared"


def test_events_are_time_ordered():
    pipeline = build()
    result = pipeline.run_frames(make_frames(30))
    timestamps = [e.timestamp for e in result.events]
    assert timestamps == sorted(timestamps)


def test_all_events_carry_the_required_fields():
    pipeline = build()
    result = pipeline.run_frames(make_frames(30))

    for event in result.events:
        assert isinstance(event.timestamp, float)
        assert event.entity_id
        assert event.action
        assert event.position and len(event.position) == 2
        assert event.bbox and len(event.bbox) == 4
        assert event.attributes["class_name"]
        assert 0.0 <= event.attributes["confidence"] <= 1.0


def test_scripted_detections_drive_expected_entities():
    person = make_detection(10, 100, class_id=0, class_name="person")
    car = make_detection(300, 100, class_id=2, class_name="car")
    script = [[person, car]] * 12
    pipeline = build(detector=MockDetector(script=script))

    result = pipeline.run_frames(make_frames(12))
    assert sorted(result.memory.entities()) == ["car_2", "person_1"]


def test_pipeline_metadata_describes_the_backend():
    pipeline = build()
    result = pipeline.run_frames(make_frames(10))

    assert result.metadata["detector"]["backend"] == "mock"
    assert result.metadata["tracker"] == "byte_iou"
    assert result.metadata["frames_processed"] == 10


def test_pipeline_run_is_repeatable():
    """State must not leak between runs of the same pipeline object."""
    pipeline = build(detector=MockDetector(script=[[make_detection(10, 100)]] * 10))
    first = pipeline.run_frames(make_frames(10))
    pipeline._detector.calls = 0  # rewind the scripted detector
    second = pipeline.run_frames(make_frames(10))

    assert [e.to_dict() for e in first.events] == [e.to_dict() for e in second.events]


def test_empty_stream_yields_no_events():
    pipeline = build()
    result = pipeline.run_frames([])
    assert result.frames_processed == 0
    assert result.events == []


def test_run_without_a_video_path_raises():
    with pytest.raises(ValueError, match="No video path"):
        build().run()


def test_run_with_a_missing_file_raises():
    pytest.importorskip("cv2")
    with pytest.raises(FileNotFoundError):
        build().run("/nonexistent/clip.mp4")


def test_config_only_passes_model_args_to_model_backends():
    assert PipelineConfig(detector="mock").detector_kwargs() == {}
    yolo_kwargs = PipelineConfig(detector="yolo").detector_kwargs()
    assert yolo_kwargs["weights"] == "yolov8n.pt"
    assert yolo_kwargs["device"] == "auto"


# ---------------------------------------------------------------------------
# M0.2: CLI helpers
# ---------------------------------------------------------------------------
def test_parse_inline_zone():
    from app.main import parse_inline_zone

    zone = parse_inline_zone("loading_bay=10,20,110,120")
    assert zone.name == "loading_bay"
    assert zone.bounds == (10.0, 20.0, 110.0, 120.0)


@pytest.mark.parametrize(
    "spec", ["nocoords", "bay=1,2,3", "bay=1,2,3,4,5", "bay=a,b,c,d"]
)
def test_parse_inline_zone_rejects_bad_input(spec):
    from app.main import parse_inline_zone

    with pytest.raises(ValueError):
        parse_inline_zone(spec)


def test_config_reports_zones_and_pixel_units():
    from app.spatial import Zone, ZoneSet

    config = PipelineConfig(
        detector="mock", zones=ZoneSet([Zone.from_rect("bay", (0, 0, 10, 10))])
    )
    payload = config.to_dict()
    assert payload["zones"] == ["bay"]
    assert payload["coordinate_space"] == "image_pixels"
    # Thresholds are named in pixels so a consumer cannot mistake them for metres.
    assert "movement_threshold_px" in payload
    assert "stationary_threshold_px" in payload
