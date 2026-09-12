"""Real-footage smoke-test CLI and report.

These use the generated demo clip, not real footage: no video may be committed
to this repository. They cover the new input path (``--diagnostics``,
``--calibration``) and the report's honesty guarantees.
"""

import json

import pytest

from app.evaluation.smoke import (
    SHORT_TRACK_FRAMES,
    SMOKE_TEST_BANNER,
    SmokeTestReport,
    TrackSummary,
    build_report,
)
from app.main import load_calibration, main
from app.spatial import GROUND_PLANE_METERS, IMAGE_PIXELS

CALIBRATION = {
    "name": "test_calibration",
    "world_units": "meters",
    "image_points": [[100, 100], [900, 100], [900, 500], [100, 500]],
    "world_points": [[0, 0], [20, 0], [20, 10], [0, 10]],
}


@pytest.fixture
def calibration_file(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(CALIBRATION), encoding="utf-8")
    return str(path)


# ===========================================================================
# Calibration input path (requirements 5 and 6)
# ===========================================================================
def test_no_calibration_argument_means_image_space():
    assert load_calibration(None) is None
    assert load_calibration("") is None


def test_calibration_file_is_loaded_through_the_existing_loader(calibration_file):
    calibration = load_calibration(calibration_file)
    assert calibration is not None
    assert calibration.name == "test_calibration"
    assert calibration.coordinate_space == GROUND_PLANE_METERS


def test_a_missing_calibration_file_is_an_error_not_a_silent_fallback(
    demo_video, capsys
):
    """Falling back silently would hide that metres were expected."""
    exit_code = main([demo_video, "--detector", "blob", "--calibration", "nope.json"])
    assert exit_code == 2
    assert "error" in capsys.readouterr().err


def test_a_degenerate_calibration_file_is_rejected(demo_video, tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "image_points": [[0, 0], [1, 0], [2, 0], [3, 0]],
                "world_points": [[0, 0], [1, 0], [2, 0], [3, 0]],
            }
        ),
        encoding="utf-8",
    )
    assert main([demo_video, "--detector", "blob", "--calibration", str(bad)]) == 2
    assert "error" in capsys.readouterr().err


# ===========================================================================
# The --diagnostics CLI path
# ===========================================================================
def test_diagnostics_reports_on_the_demo_clip(demo_video, capsys):
    exit_code = main(
        [demo_video, "--detector", "blob", "--classes", "person", "truck", "car",
         "--confidence", "0.0", "--diagnostics", "-o", ""]
    )
    assert exit_code == 0
    output = capsys.readouterr().out

    for banner in SMOKE_TEST_BANNER:
        assert banner in output
    for section in ("VIDEO", "PROCESSING", "DETECTION", "TRACKING", "EVENTS",
                    "RISK", "SPATIAL", "INTERPRETATION"):
        assert section in output


def test_diagnostics_reports_the_video_properties(demo_video, capsys):
    main([demo_video, "--detector", "blob", "--confidence", "0.0",
          "--diagnostics", "-o", ""])
    output = capsys.readouterr().out
    assert "640x384" in output
    assert "source fps" in output
    assert "frame count" in output


def test_diagnostics_labels_image_space_fallback(demo_video, capsys):
    main([demo_video, "--detector", "blob", "--confidence", "0.0",
          "--diagnostics", "-o", ""])
    output = capsys.readouterr().out
    assert IMAGE_PIXELS in output
    assert "none supplied - image-space fallback" in output


def test_diagnostics_labels_active_calibration(demo_video, calibration_file, capsys):
    main([demo_video, "--detector", "blob", "--confidence", "0.0",
          "--calibration", calibration_file, "--diagnostics", "-o", ""])
    output = capsys.readouterr().out
    assert GROUND_PLANE_METERS in output
    assert "ACTIVE (test_calibration)" in output


def test_diagnostics_is_deterministic(demo_video, capsys):
    main([demo_video, "--detector", "blob", "--confidence", "0.0",
          "--diagnostics", "-o", ""])
    first = capsys.readouterr().out
    main([demo_video, "--detector", "blob", "--confidence", "0.0",
          "--diagnostics", "-o", ""])
    second = capsys.readouterr().out
    # Throughput varies between runs; everything else must not.
    def strip_timing(text):
        noisy = ("processing_fps", "processing fps", "realtime factor",
                 "processing_seconds")
        return "\n".join(
            line for line in text.splitlines()
            if not any(token in line for token in noisy)
        )

    assert strip_timing(first) == strip_timing(second)


# ===========================================================================
# Report honesty (requirement 7)
# ===========================================================================
def empty_report(**overrides):
    defaults = dict(
        video_path="clip.mp4", width=640, height=384, source_fps=20.0,
        frame_count=80, frames_processed=80, processing_fps=100.0,
        detector_backend="blob", detector_model=None, accelerator="cpu",
        detections_total=200, track_ids_created=3,
    )
    defaults.update(overrides)
    return SmokeTestReport(**defaults)


def test_report_declares_it_is_not_an_accuracy_benchmark():
    payload = empty_report().to_dict()
    assert payload["is_accuracy_benchmark"] is False
    assert "No ground truth" in payload["disclaimer"]


def test_report_banner_states_both_lines():
    assert SMOKE_TEST_BANNER == (
        "REAL-FOOTAGE SMOKE TEST",
        "NOT A GROUND-TRUTH ACCURACY BENCHMARK",
    )
    assert "NOT A GROUND-TRUTH ACCURACY BENCHMARK" in empty_report().to_text()


def test_report_refuses_to_report_id_switches():
    """Without ground truth an ID-switch count cannot be computed at all."""
    payload = empty_report().to_dict()
    assert payload["tracking"]["id_switches"] is None
    assert "not measurable without ground truth" in payload["tracking"][
        "id_switches_note"
    ]
    assert "not measurable without ground truth" in empty_report().to_text()


def test_report_never_calls_the_risk_score_a_probability():
    payload = empty_report().to_dict()
    assert "NOT a calibrated probability" in payload["risk"]["score_interpretation"]


def test_report_distinguishes_frames_present_from_events_recorded():
    """Conflating them would overstate or understate persistence."""
    report = empty_report(
        tracks=[TrackSummary("person_1", "person", 180, 24, 0.0, 18.0)]
    )
    payload = report.to_dict()["tracking"]["tracks"][0]
    assert payload["frames_present"] == 180
    assert payload["events_recorded"] == 24
    assert payload["duration_seconds"] == 18.0


def test_fragmentation_indicator_counts_short_lived_tracks():
    report = empty_report(
        tracks=[
            TrackSummary("a", "person", 100, 10, 0.0, 10.0),
            TrackSummary("b", "person", SHORT_TRACK_FRAMES, 1, 0.0, 0.3),
        ]
    )
    assert len(report.short_lived_tracks) == 1
    assert report.fragmentation_indicator == pytest.approx(0.5)


def test_report_flags_a_run_that_detected_nothing():
    class Result:
        metadata = {"video": {}, "detector": {}}
        frames_processed = 50
        detections_total = 0
        track_ids_created = 0
        track_frames = {}
        temporal = None
        fps = 10.0

    report = build_report(Result(), video_path="clip.mp4")
    assert any("No detections at all" in note for note in report.notes)


def test_report_flags_a_video_that_could_not_be_read():
    class Result:
        metadata = {"video": {}, "detector": {}}
        frames_processed = 0
        detections_total = 0
        track_ids_created = 0
        track_frames = {}
        temporal = None
        fps = 0.0

    report = build_report(Result(), video_path="clip.mp4")
    assert any("could not be read" in note for note in report.notes)


def test_report_points_at_the_real_benchmark_for_measured_quality():
    assert "python -m app.evaluation --clip" in empty_report().to_text()
