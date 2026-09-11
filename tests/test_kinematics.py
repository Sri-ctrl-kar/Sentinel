"""Tests for image-space motion estimation.

Nothing here may be described in metres or m/s — that is the point of the
module and several of these tests exist to enforce it.
"""

import pytest

from app.events.schema import Event
from app.reasoning.kinematics import (
    ImageMotion,
    MotionEstimator,
    closest_approach,
)


def positioned(timestamp, x, y, entity_id="person_1", class_name="person"):
    return Event(
        timestamp=timestamp,
        entity_id=entity_id,
        action="moved",
        attributes={"class_name": class_name, "confidence": 0.9},
        position=[x, y],
        bbox=[x - 20, y - 45, x + 20, y + 45],
        event_id=f"evt_{entity_id}_{timestamp:.2f}",
    )


def track(start_x, vx_per_frame, count, y=100.0, fps=10.0, entity_id="person_1", **kw):
    """A straight-line history, one event per frame."""
    return [
        positioned(i / fps, start_x + vx_per_frame * i, y, entity_id, **kw)
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# Velocity estimation
# ---------------------------------------------------------------------------
def test_velocity_is_estimated_in_pixels_per_second():
    # 10px per frame at 10fps = 100px/s.
    motion = MotionEstimator().estimate(track(0, 10, 10), at=0.9)
    assert motion.velocity_px_per_s[0] == pytest.approx(100.0)
    assert motion.velocity_px_per_s[1] == pytest.approx(0.0, abs=1e-6)
    assert motion.speed_px_per_s == pytest.approx(100.0)


def test_velocity_is_estimated_on_both_axes():
    events = [positioned(i / 10.0, i * 10, i * 20) for i in range(10)]
    motion = MotionEstimator().estimate(events, at=0.9)
    assert motion.velocity_px_per_s[0] == pytest.approx(100.0)
    assert motion.velocity_px_per_s[1] == pytest.approx(200.0)


def test_least_squares_fit_resists_a_single_outlier():
    events = track(0, 10, 10)
    events[5].position = [500.0, 100.0]  # one wildly wrong detection
    motion = MotionEstimator().estimate(events, at=0.9)
    # Endpoint differencing would be unaffected, but a mean-based estimator
    # would be destroyed; least squares degrades gracefully.
    assert 100.0 < motion.velocity_px_per_s[0] < 250.0


def test_coordinate_space_is_always_image_pixels():
    motion = MotionEstimator().estimate(track(0, 10, 10), at=0.9)
    assert motion.coordinate_space == "image_pixels"
    assert motion.to_dict()["coordinate_space"] == "image_pixels"


def test_velocity_field_is_named_in_pixels_per_second():
    """Guard against anyone renaming this to something metric-sounding."""
    payload = MotionEstimator().estimate(track(0, 10, 10), at=0.9).to_dict()
    assert "velocity_px_per_s" in payload
    assert "speed_px_per_s" in payload
    assert not any("m_per_s" in key or "meters" in key for key in payload)


# ---------------------------------------------------------------------------
# Insufficient history
# ---------------------------------------------------------------------------
def test_too_few_samples_is_not_estimable():
    motion = MotionEstimator(min_samples=3).estimate(track(0, 10, 2), at=0.1)
    assert motion.is_estimable is False
    assert motion.velocity_px_per_s == (0.0, 0.0)
    assert motion.confidence == 0.0


def test_too_short_a_time_span_is_not_estimable():
    events = [positioned(i * 0.01, i * 10, 100) for i in range(5)]
    motion = MotionEstimator(min_time_span=0.2).estimate(events, at=0.04)
    assert motion.is_estimable is False


def test_non_estimable_motion_refuses_to_extrapolate():
    motion = MotionEstimator().estimate(track(0, 10, 2), at=0.1)
    assert motion.predict_position_px(5.0) == motion.position_px


def test_estimable_motion_extrapolates_linearly():
    motion = MotionEstimator().estimate(track(0, 10, 10), at=0.9)
    predicted = motion.predict_position_px(1.0)
    assert predicted[0] == pytest.approx(motion.position_px[0] + 100.0)


def test_no_positioned_events_returns_none():
    assert MotionEstimator().estimate([], at=1.0) is None


def test_stale_observations_return_none():
    """An entity not seen recently is unobservable, not 'still going'."""
    events = track(0, 10, 10)
    assert MotionEstimator(max_staleness_seconds=0.5).estimate(events, at=5.0) is None


def test_events_after_the_assessment_time_are_ignored():
    """The engine must never see the future when assessing a past moment."""
    motion = MotionEstimator().estimate(track(0, 10, 20), at=0.5)
    assert motion.position_px[0] == pytest.approx(50.0)
    assert motion.last_seen <= 0.5


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------
def test_confidence_grows_with_evidence():
    estimator = MotionEstimator()
    few = estimator.estimate(track(0, 10, 4), at=0.3)
    many = estimator.estimate(track(0, 10, 12), at=1.1)
    assert 0.0 < few.confidence < many.confidence
    assert many.confidence == pytest.approx(1.0)


def test_confidence_is_bounded():
    motion = MotionEstimator().estimate(track(0, 10, 100), at=9.9)
    assert motion.confidence <= 1.0


# ---------------------------------------------------------------------------
# Stationarity
# ---------------------------------------------------------------------------
def test_a_still_entity_is_flagged_stationary():
    motion = MotionEstimator().estimate(track(100, 0, 10), at=0.9)
    assert motion.is_estimable is True
    assert motion.is_stationary is True
    assert motion.speed_px_per_s == pytest.approx(0.0, abs=1e-6)


def test_sub_threshold_jitter_counts_as_stationary():
    events = [positioned(i / 10.0, 100 + (i % 2), 100) for i in range(10)]
    motion = MotionEstimator(stationary_speed_px_per_s=8.0).estimate(events, at=0.9)
    assert motion.is_stationary is True


def test_a_walking_entity_is_not_stationary():
    motion = MotionEstimator().estimate(track(0, 10, 10), at=0.9)
    assert motion.is_stationary is False


# ---------------------------------------------------------------------------
# Closest approach
# ---------------------------------------------------------------------------
def build_pair(a_events, b_events, at):
    estimator = MotionEstimator()
    return (
        estimator.estimate(a_events, at, "person_1"),
        estimator.estimate(b_events, at, "forklift_2"),
    )


def test_head_on_convergence_predicts_a_zero_miss():
    a, b = build_pair(
        track(0, 10, 10, y=100),
        track(500, -10, 10, y=100, entity_id="forklift_2"),
        at=0.9,
    )
    result = closest_approach(a, b)
    assert result.is_converging is True
    assert result.distance_px == pytest.approx(0.0, abs=1.0)
    assert result.seconds_to_closest_approach == pytest.approx(1.6, abs=0.05)
    assert result.closing_speed_px_per_s == pytest.approx(200.0, abs=1.0)


def test_parallel_offset_paths_predict_a_near_miss():
    a, b = build_pair(
        track(0, 10, 10, y=100),
        track(500, -10, 10, y=300, entity_id="forklift_2"),
        at=0.9,
    )
    result = closest_approach(a, b)
    assert result.is_converging is True
    assert result.distance_px == pytest.approx(200.0, abs=1.0)


def test_diverging_entities_are_not_converging():
    a, b = build_pair(
        track(300, -10, 10, y=100),
        track(400, 10, 10, y=100, entity_id="forklift_2"),
        at=0.9,
    )
    result = closest_approach(a, b)
    assert result.is_converging is False
    assert result.closing_speed_px_per_s < 0
    assert result.seconds_to_closest_approach == 0.0


def test_both_stationary_never_converges():
    a, b = build_pair(
        track(300, 0, 10, y=100),
        track(400, 0, 10, y=100, entity_id="forklift_2"),
        at=0.9,
    )
    result = closest_approach(a, b)
    assert result.is_converging is False
    assert result.closing_speed_px_per_s == pytest.approx(0.0, abs=1e-6)
    assert result.current_separation_px == pytest.approx(100.0)


def test_identical_velocity_holds_separation():
    a, b = build_pair(
        track(0, 10, 10, y=100),
        track(200, 10, 10, y=100, entity_id="forklift_2"),
        at=0.9,
    )
    result = closest_approach(a, b)
    assert result.is_converging is False
    assert result.current_separation_px == pytest.approx(200.0)


def test_closest_approach_is_not_estimable_without_history():
    a, b = build_pair(track(0, 10, 2), track(500, -10, 2, entity_id="forklift_2"), at=0.1)
    result = closest_approach(a, b)
    assert result.is_estimable is False
    assert result.is_converging is False
    # The current separation is still a fact, even with no trajectory.
    assert result.current_separation_px > 0


def test_closest_approach_reports_pixel_space():
    a, b = build_pair(track(0, 10, 10), track(500, -10, 10, entity_id="forklift_2"), at=0.9)
    payload = closest_approach(a, b).to_dict()
    assert payload["coordinate_space"] == "image_pixels"
    assert "closest_approach_distance_px" in payload
    assert "closing_speed_px_per_s" in payload
