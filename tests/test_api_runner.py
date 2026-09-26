"""The API's frame loop must not change what Sentinel produces (M0.9).

The runner drives detection, tracking and event generation itself so it can keep
per-frame tracks for the frontend. That is a real risk: a second loop can drift
from the pipeline's and nobody would notice until a demo disagreed with the CLI.

So the central test here runs the same clip twice — once through
``PerceptionPipeline.run()``, once through ``app.api.runner`` — and asserts the
events, entities, states and counts are identical. Everything else in this file
checks the API's own addition: the per-frame record, the timeline, and the
semantic device payload.

No GPU, no weights, no network: the detector is ``MockDetector`` and the clip is
the synthetic one the suite already renders.
"""

from __future__ import annotations

import pytest

from app.api import runner
from app.api.store import (
    STATUS_COMPLETE,
    STATUS_FAILED,
    STATUS_QUEUED,
    AnalysisRecord,
    AnalysisStore,
    Progress,
)
from app.config import PipelineConfig
from app.pipeline import PerceptionPipeline
from app.spatial import Zone, ZoneSet


def make_config(video: str, **overrides) -> PipelineConfig:
    """A mock-detector config: identical inputs for both code paths."""
    settings = dict(
        video_path=video,
        detector="mock",
        device="cpu",
        sample_interval=0.1,
        max_frames=40,
        classes=None,
        output_path="",
    )
    settings.update(overrides)
    return PipelineConfig(**settings)


def make_record(config: PipelineConfig, **overrides) -> AnalysisRecord:
    settings = dict(
        analysis_id="test-analysis",
        video_path=config.video_path,
        original_filename="clip.mp4",
        config=config,
    )
    settings.update(overrides)
    return AnalysisRecord(**settings)


@pytest.fixture
def completed(demo_video):
    record = runner.run_analysis(make_record(make_config(demo_video)))
    assert record.status == STATUS_COMPLETE, record.error
    return record


# ---------------------------------------------------------------------------
# The guarantee: the API runner agrees with the pipeline
# ---------------------------------------------------------------------------
def test_the_runner_produces_the_same_events_as_the_pipeline(demo_video):
    config = make_config(demo_video)

    pipeline = PerceptionPipeline(config=make_config(demo_video))
    try:
        expected = pipeline.run(demo_video)
    finally:
        pipeline.close()

    actual = runner.run_analysis(make_record(config))

    assert actual.status == STATUS_COMPLETE, actual.error
    assert [e.to_dict() for e in actual.result.events] == [
        e.to_dict() for e in expected.events
    ]


def test_the_runner_produces_the_same_counts_as_the_pipeline(demo_video):
    pipeline = PerceptionPipeline(config=make_config(demo_video))
    try:
        expected = pipeline.run(demo_video)
    finally:
        pipeline.close()

    actual = runner.run_analysis(make_record(make_config(demo_video))).result

    assert actual.frames_processed == expected.frames_processed
    assert actual.detections_total == expected.detections_total
    assert actual.track_reports_total == expected.track_reports_total
    assert actual.track_ids_created == expected.track_ids_created
    assert actual.track_frames == expected.track_frames


def test_the_runner_produces_the_same_temporal_memory_as_the_pipeline(demo_video):
    pipeline = PerceptionPipeline(config=make_config(demo_video))
    try:
        expected = pipeline.run(demo_video)
    finally:
        pipeline.close()

    actual = runner.run_analysis(make_record(make_config(demo_video))).result

    assert actual.temporal.entities() == expected.temporal.entities()
    assert {k: v.to_dict() for k, v in actual.temporal.states().items()} == {
        k: v.to_dict() for k, v in expected.temporal.states().items()
    }
    assert actual.temporal.summary() == expected.temporal.summary()


def test_the_runner_produces_the_same_risk_as_the_pipeline(demo_video):
    """The point of all of this: acceleration of access, not of semantics."""
    from app.reasoning import RiskEngine

    pipeline = PerceptionPipeline(config=make_config(demo_video))
    try:
        expected = pipeline.run(demo_video)
    finally:
        pipeline.close()
    actual = runner.run_analysis(make_record(make_config(demo_video))).result

    left = RiskEngine().assess_timeline(expected.temporal, step=0.2)
    right = RiskEngine().assess_timeline(actual.temporal, step=0.2)
    assert [r.to_dict() for r in right] == [r.to_dict() for r in left]


def test_the_runner_is_deterministic(demo_video):
    first = runner.run_analysis(make_record(make_config(demo_video)))
    second = runner.run_analysis(make_record(make_config(demo_video)))
    assert [f.to_dict() for f in first.frames] == [f.to_dict() for f in second.frames]


# ---------------------------------------------------------------------------
# The API's addition: per-frame tracks
# ---------------------------------------------------------------------------
def test_every_processed_frame_gets_a_record(completed):
    assert len(completed.frames) == completed.result.frames_processed
    assert [f.frame_index for f in completed.frames] == sorted(
        f.frame_index for f in completed.frames
    )


def test_frame_records_carry_what_an_overlay_needs(completed):
    populated = [f for f in completed.frames if f.tracks]
    assert populated, "no frame carried a track"
    payload = populated[0].to_dict()
    assert set(payload) == {"frame_index", "timestamp", "tracks", "events"}
    track = payload["tracks"][0]
    for field in (
        "track_id",
        "entity_id",
        "bbox",
        "position",
        "confidence",
        "class_id",
        "class_name",
        "age",
        "hits",
    ):
        assert field in track
    assert len(track["bbox"]) == 4


def test_frame_records_transcribe_the_tracker_rather_than_smoothing_it(completed):
    """Boxes must be the tracker's own, not interpolated or filtered."""
    for frame in completed.frames:
        for track in frame.tracks:
            x1, y1, x2, y2 = track.bbox
            assert x2 >= x1 and y2 >= y1
            assert track.track_id > 0
            assert track.hits >= 1


def test_per_frame_data_is_denser_than_the_event_stream(completed):
    """Requirement 9: the overlay channel, not the event channel, carries density.

    With the validated 1.0 s default sampling this gap is much larger still;
    the fixture uses 0.1 s and the frame records are still denser.
    """
    frames_with_tracks = sum(1 for f in completed.frames if f.tracks)
    assert frames_with_tracks > len(completed.result.events) / 2
    assert frames_with_tracks > 1


def test_event_timestamps_are_untouched_by_the_api(completed):
    """Events keep their own timestamps and meanings; nothing is resampled."""
    from_frames = [
        (e.timestamp, e.entity_id, e.action)
        for frame in completed.frames
        for e in frame.events
    ]
    from_memory = [
        (e.timestamp, e.entity_id, e.action) for e in completed.result.events
    ]
    assert sorted(from_frames) == sorted(from_memory)


def test_the_default_sampling_interval_is_not_changed_by_the_api():
    """The API must not quietly redefine the validated event default."""
    assert PipelineConfig().sample_interval == 1.0
    record = make_record(PipelineConfig(video_path="x.mp4", detector="mock"))
    assert record.config.sample_interval == 1.0


# ---------------------------------------------------------------------------
# Progress and state machine
# ---------------------------------------------------------------------------
def test_a_record_starts_queued_and_ends_complete(demo_video):
    record = make_record(make_config(demo_video))
    assert record.status == STATUS_QUEUED
    assert record.started_at is None
    runner.run_analysis(record)
    assert record.status == STATUS_COMPLETE
    assert record.started_at is not None and record.finished_at is not None
    assert record.elapsed_seconds >= 0.0


def test_progress_reaches_the_frame_count(completed):
    assert completed.progress.frames_processed == completed.result.frames_processed
    assert completed.progress.percent is not None


def test_progress_percentages_are_bounded():
    progress = Progress(frames_processed=200, total_frames=100)
    assert progress.percent == 100.0
    assert Progress().percent is None
    assert Progress(frames_processed=5, total_frames=20).percent == 25.0


def test_a_missing_video_fails_the_record_without_raising():
    record = make_record(make_config("/no/such/clip.mp4"))
    runner.run_analysis(record)
    assert record.status == STATUS_FAILED
    assert record.error["type"] == "FileNotFoundError"
    assert record.result is None


def test_an_unreadable_video_fails_the_record(tmp_path):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"this is not a video")
    record = make_record(make_config(str(broken)))
    runner.run_analysis(record)
    assert record.status == STATUS_FAILED
    assert record.error["message"]


def test_a_failure_records_the_error_as_data_not_a_traceback():
    record = make_record(make_config("/no/such/clip.mp4"))
    runner.run_analysis(record)
    assert set(record.error) == {"type", "message"}
    assert "Traceback" not in record.error["message"]


# ---------------------------------------------------------------------------
# Timeline assembly
# ---------------------------------------------------------------------------
def test_the_timeline_has_a_row_per_frame(completed):
    rows = runner.build_timeline(completed)
    assert len(rows) == len(completed.frames)
    assert all(set(r) == {"frame_index", "timestamp", "tracks", "events", "risk"} for r in rows)


def test_the_timeline_paginates(completed):
    rows = runner.build_timeline(completed, limit=3, offset=2)
    assert len(rows) == 3
    assert rows[0]["frame_index"] == completed.frames[2].frame_index


def test_risk_is_attached_only_where_it_was_assessed(completed):
    from app.reasoning import RiskEngine

    completed.risk_step = 0.2
    reports = RiskEngine().assess_timeline(completed.temporal, step=0.2)
    rows = runner.build_timeline(completed, reports)

    attached = [r for r in rows if r["risk"] is not None]
    assert attached, "no frame received a risk assessment"
    assert len(attached) < len(rows), "risk is coarser than the frame rate"
    for row in attached:
        assert row["risk"]["severity"] in (
            "normal",
            "low",
            "medium",
            "high",
            "critical",
        )
        assert row["risk"]["coordinate_space"] == "image_pixels"


def test_risk_scores_in_the_timeline_are_the_engines_own(completed):
    from app.reasoning import RiskEngine

    completed.risk_step = 0.2
    reports = RiskEngine().assess_timeline(completed.temporal, step=0.2)
    published = {
        round(r.timestamp, 4): round(r.max_score, 2) for r in reports if r.assessments
    }
    for row in runner.build_timeline(completed, reports):
        if row["risk"] is not None:
            stamp = row["risk"]["timestamp"]
            assert row["risk"]["risk_score"] == published[stamp]


def test_no_risk_is_invented_for_an_unassessed_moment(completed):
    """A nearest-match association, never an interpolation."""
    rows = runner.build_timeline(completed, [])
    assert all(row["risk"] is None for row in rows)


def test_risk_at_respects_the_window(completed):
    from app.reasoning import RiskEngine

    reports = RiskEngine().assess_timeline(completed.temporal, step=0.5)
    assert runner.risk_at(reports, timestamp=-99.0, window=0.25) is None
    nearest = runner.risk_at(reports, timestamp=reports[0].timestamp, window=0.25)
    assert nearest is reports[0]


# ---------------------------------------------------------------------------
# Zones flow through unchanged
# ---------------------------------------------------------------------------
def test_zones_reach_the_event_generator(demo_video):
    zones = ZoneSet([Zone.from_rect("bay", [0.0, 0.0, 640.0, 384.0])])
    record = runner.run_analysis(
        make_record(make_config(demo_video, zones=zones))
    )
    assert record.status == STATUS_COMPLETE, record.error
    actions = {e.action for e in record.result.events}
    assert "entered_zone" in actions


# ---------------------------------------------------------------------------
# Device reporting (M0.9 requirement 12)
# ---------------------------------------------------------------------------
def test_the_device_payload_reports_the_semantic_kind():
    payload = runner.device_payload(PipelineConfig(device="cpu", detector="mock"))
    assert payload["kind"] == "cpu"
    assert payload["semantic_device"] == "cpu"
    assert payload["requested"] == "cpu"
    assert payload["label"] == "CPU"


def test_a_rocm_device_is_reported_as_rocm_not_cuda(monkeypatch):
    """The M0.8 lesson, held at the API boundary too."""
    import sys

    sys.path.insert(0, "tests")
    from test_accel_device import fake_torch

    from app.accel import device as accel

    monkeypatch.setattr(
        accel,
        "_torch",
        lambda: fake_torch(
            cuda_available=True, hip="6.2", names=("AMD Instinct MI300X",)
        ),
    )
    payload = runner.device_payload(PipelineConfig(device="rocm", detector="mock"))
    assert payload["kind"] == "rocm"
    assert payload["semantic_device"] == "rocm"
    assert payload["label"] == "AMD ROCm / HIP"
    # torch_device is carried under its own name, and is not the semantic device.
    assert payload["torch_device"] == "cuda"
    assert payload["semantic_device"] != payload["torch_device"]


def test_an_unavailable_device_fails_the_record(demo_video):
    from app.accel import gpu_kind

    if gpu_kind() == "rocm":  # pragma: no cover - only on an AMD host
        pytest.skip("this machine has an AMD GPU")
    record = make_record(make_config(demo_video, device="rocm"))
    runner.run_analysis(record)
    assert record.status == STATUS_FAILED
    assert "AMD ROCm" in record.error["message"]


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------
def test_the_store_round_trips_a_record(tmp_path):
    store = AnalysisStore(upload_dir=str(tmp_path))
    record = store.add(make_record(make_config("clip.mp4"), analysis_id="abc"))
    assert store.get("abc") is record
    assert len(store) == 1
    assert store.list() == [record]
    assert store.get("missing") is None


def test_the_store_names_uploads_by_id_not_by_user_input(tmp_path):
    store = AnalysisStore(upload_dir=str(tmp_path))
    path = store.video_path_for("abc123", "../../etc/passwd.mp4")
    assert path == str(tmp_path / "abc123.mp4")
    assert ".." not in path


def test_deleting_an_analysis_removes_its_video(tmp_path):
    store = AnalysisStore(upload_dir=str(tmp_path))
    video = tmp_path / "abc.mp4"
    video.write_bytes(b"bytes")
    store.add(make_record(make_config(str(video)), analysis_id="abc", video_path=str(video)))
    assert store.delete("abc") is True
    assert not video.exists()
    assert store.delete("abc") is False


def test_ids_are_unique(tmp_path):
    store = AnalysisStore(upload_dir=str(tmp_path))
    assert len({store.new_id() for _ in range(100)}) == 100
