"""Full-stack test: a real video file in, a real JSON event log out.

Uses the dependency-free ``blob`` detector so this runs anywhere OpenCV does —
no model weights, no network, no GPU. The synthetic clip has known ground
truth: three entities, one crossing left-to-right, one right-to-left, one
static and only visible for the middle of the clip.
"""

import json

import pytest

pytest.importorskip("cv2")

from app.config import PipelineConfig  # noqa: E402
from app.main import main  # noqa: E402
from app.pipeline import run_pipeline  # noqa: E402
from app.storage.memory import EventMemory  # noqa: E402


@pytest.fixture
def run_result(demo_video, tmp_path):
    config = PipelineConfig(
        video_path=demo_video,
        detector="blob",
        classes=["person", "truck", "car"],
        confidence=0.0,
        output_path=str(tmp_path / "events.json"),
    )
    return run_pipeline(config), config


def test_pipeline_finds_the_three_expected_entities(run_result):
    result, _ = run_result
    entities = sorted(result.memory.entities())
    assert entities == ["car_3", "person_1", "truck_2"]


def test_every_entity_is_opened_and_closed(run_result):
    result, _ = run_result
    summary = result.memory.summary()
    assert summary["events_by_action"]["appeared"] == 3
    assert summary["events_by_action"]["disappeared"] == 3


def test_moving_entities_emit_movement_events(run_result):
    result, _ = run_result
    moved = {e.entity_id for e in result.memory.by_action("moved")}
    assert {"person_1", "truck_2"} <= moved
    # The green box never moves, so it must not report motion.
    assert "car_3" not in moved


def test_person_travels_left_to_right(run_result):
    result, _ = run_result
    positions = [e.position[0] for e in result.memory.by_entity("person_1")]
    assert positions == sorted(positions)
    assert positions[-1] - positions[0] > 400


def test_truck_travels_right_to_left(run_result):
    result, _ = run_result
    positions = [e.position[0] for e in result.memory.by_entity("truck_2")]
    assert positions == sorted(positions, reverse=True)


def test_output_json_is_written_and_reloadable(run_result):
    result, config = run_result
    reloaded = EventMemory.load_json(config.output_path)
    assert len(reloaded) == len(result.events)

    payload = json.loads(open(config.output_path, encoding="utf-8").read())
    assert payload["metadata"]["video"]["fps"] == pytest.approx(20.0, abs=0.1)
    assert payload["metadata"]["detector"]["backend"] == "blob"
    assert payload["metadata"]["frames_processed"] == 80


def test_timestamps_span_the_clip(run_result):
    result, _ = run_result
    timestamps = [e.timestamp for e in result.events]
    assert min(timestamps) < 0.5
    assert max(timestamps) > 3.0


def test_cli_end_to_end(demo_video, tmp_path, capsys):
    output = str(tmp_path / "cli-events.json")
    exit_code = main(
        [
            demo_video,
            "--detector",
            "blob",
            "--classes",
            "person",
            "truck",
            "car",
            "--confidence",
            "0.0",
            "-o",
            output,
        ]
    )
    assert exit_code == 0

    payload = json.loads(open(output, encoding="utf-8").read())
    assert payload["event_count"] > 0
    assert payload["metadata"]["summary"]["unique_entities"] == 3

    printed = json.loads(capsys.readouterr().out)
    assert printed["frames_processed"] == 80


def test_cli_reports_a_missing_video_file(tmp_path, capsys):
    assert main([str(tmp_path / "nope.mp4"), "--detector", "blob"]) == 1
    assert "error" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# M0.2: temporal memory over a real video run
# ---------------------------------------------------------------------------
@pytest.fixture
def zoned_run(demo_video, tmp_path):
    from app.spatial import Zone, ZoneSet

    config = PipelineConfig(
        video_path=demo_video,
        detector="blob",
        classes=["person", "truck", "car"],
        confidence=0.0,
        stationary_duration=1.0,
        zones=ZoneSet(
            [
                Zone.from_rect("loading_bay", (380, 180, 640, 320)),
                Zone.from_rect("walkway", (0, 240, 380, 384)),
            ]
        ),
        output_path=str(tmp_path / "events.json"),
        memory_path=str(tmp_path / "memory.json"),
    )
    return run_pipeline(config), config


def test_pipeline_populates_a_temporal_memory(zoned_run):
    result, _ = zoned_run
    assert result.temporal is not None
    assert len(result.temporal) == len(result.events)
    assert sorted(result.temporal.entities()) == ["car_3", "person_1", "truck_2"]


def test_temporal_memory_and_event_log_agree(zoned_run):
    result, _ = zoned_run
    assert [e.event_id for e in result.temporal] == [e.event_id for e in result.events]


def test_zone_events_are_produced_on_real_video(zoned_run):
    result, _ = zoned_run
    entered = result.temporal.stream(actions=["entered_zone"])
    exited = result.temporal.stream(actions=["exited_zone"])
    assert entered and exited
    # Every entry is eventually closed by an exit.
    assert len(entered) == len(exited)


def test_the_static_car_is_reported_stationary(zoned_run):
    result, _ = zoned_run
    stationary = result.temporal.stream(actions=["stationary"])
    assert {e.entity_id for e in stationary} == {"car_3"}


def test_entities_present_changes_over_the_clip(zoned_run):
    result, _ = zoned_run
    early = result.temporal.present_entities(at=0.2)
    middle = result.temporal.present_entities(at=2.0)
    assert "person_1" in early
    assert len(middle) >= len(early)
    assert result.temporal.present_entities() == []


def test_temporal_memory_is_written_and_reloadable(zoned_run):
    from app.memory import TemporalEventMemory

    result, config = zoned_run
    reloaded = TemporalEventMemory.load_json(config.memory_path)
    assert len(reloaded) == len(result.temporal)
    assert reloaded.entities() == result.temporal.entities()


def test_cli_prints_a_timeline(demo_video, tmp_path, capsys):
    output = str(tmp_path / "cli-events.json")
    exit_code = main(
        [
            demo_video, "--detector", "blob",
            "--classes", "person", "truck", "car",
            "--confidence", "0.0",
            "--zone", "loading_bay=380,180,640,320",
            "--timeline",
            "-o", output,
        ]
    )
    assert exit_code == 0
    printed = capsys.readouterr().out
    assert "Timeline — all entities" in printed
    assert "entered_zone" in printed
    assert "Entity states" in printed


def test_cli_rejects_a_malformed_zone(demo_video, tmp_path, capsys):
    assert main([demo_video, "--detector", "blob", "--zone", "bay=1,2,3"]) == 2
    assert "error" in capsys.readouterr().err
