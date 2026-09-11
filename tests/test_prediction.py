"""M0.5 trajectory prediction tests.

The predictor is closed-form, so every expectation here is checkable by hand.
"""

import pytest

from app.calibration.examples import warehouse_calibration
from app.events.schema import Event
from app.reasoning.kinematics import MotionEstimator
from app.reasoning.prediction import (
    CURRENTLY_UNSAFE_PROXIMITY,
    NO_PREDICTED_CONFLICT,
    PREDICTED_TRAJECTORY_CONFLICT,
    PREDICTED_UNSAFE_PROXIMITY,
    PREDICTION_UNAVAILABLE,
    REASON_BOTH_STATIONARY,
    REASON_INSUFFICIENT_HISTORY,
    TrajectoryPredictor,
    _time_to_separation,
)
from app.spatial import GROUND_PLANE_METERS, IMAGE_PIXELS

CALIBRATION = warehouse_calibration()  # 40 px == 1 m


def walking(entity_id, start_x_m, speed_m_per_s, count=10, fps=10.0, y_m=5.0,
            class_name="person"):
    """Events for an entity walking along x on the calibrated floor."""
    events = []
    for i in range(count):
        seconds = i / fps
        x = start_x_m + speed_m_per_s * seconds
        foot = CALIBRATION.world_to_image((x, y_m))
        events.append(
            Event(
                timestamp=round(seconds, 4),
                entity_id=entity_id,
                action="appeared" if i == 0 else "moved",
                attributes={"class_name": class_name, "confidence": 0.9},
                position=[foot[0], foot[1] - 45.0],
                bbox=[foot[0] - 20.0, foot[1] - 90.0, foot[0] + 20.0, foot[1]],
                event_id=f"evt_{entity_id}_{i:02d}",
            )
        )
    return events


def motion(events, at=0.9, calibration=CALIBRATION):
    return MotionEstimator(calibration=calibration).estimate(
        events, at, events[0].entity_id
    )


def predict(a_events, b_events, at=0.9, unsafe=2.0, conflict=2.0, horizon=6.0,
            space=GROUND_PLANE_METERS, calibration=CALIBRATION):
    predictor = TrajectoryPredictor(horizon_seconds=horizon)
    return predictor.predict_pair(
        motion(a_events, at, calibration),
        motion(b_events, at, calibration),
        coordinate_space=space,
        unsafe_separation=unsafe,
        conflict_radius=conflict,
    )


# ===========================================================================
# Closed-form threshold crossing
# ===========================================================================
def test_time_to_separation_head_on():
    # 8 m apart, closing at 8 m/s, threshold 2 m -> (8-2)/8 = 0.75 s.
    assert _time_to_separation(8, 0, -8, 0, 2.0, 6.0) == pytest.approx(0.75)


def test_time_to_separation_returns_zero_when_already_inside():
    assert _time_to_separation(1.5, 0, -1, 0, 2.0, 6.0) == 0.0


def test_time_to_separation_returns_none_when_never_reached():
    assert _time_to_separation(8, 0, 0, -1, 2.0, 6.0) is None


def test_time_to_separation_respects_the_horizon():
    # Closing at 0.5 m/s from 8 m needs 12 s to reach 2 m.
    assert _time_to_separation(8, 0, -0.5, 0, 2.0, 6.0) is None
    assert _time_to_separation(8, 0, -0.5, 0, 2.0, 20.0) == pytest.approx(12.0)


def test_time_to_separation_with_no_relative_motion():
    assert _time_to_separation(8, 0, 0, 0, 2.0, 6.0) is None


# ===========================================================================
# Predictor configuration
# ===========================================================================
def test_horizon_must_be_positive():
    with pytest.raises(ValueError, match="horizon_seconds"):
        TrajectoryPredictor(horizon_seconds=0)


def test_timestep_must_be_positive():
    with pytest.raises(ValueError, match="timestep_seconds"):
        TrajectoryPredictor(timestep_seconds=0)


def test_horizon_is_configurable():
    short = predict(walking("a", 2.0, 2.0), walking("b", 18.0, -2.0, class_name="forklift"), horizon=1.0)
    long = predict(walking("a", 2.0, 2.0), walking("b", 18.0, -2.0, class_name="forklift"), horizon=10.0)
    assert short.horizon_seconds == 1.0
    assert short.seconds_to_unsafe_separation is None  # beyond the short horizon
    assert long.seconds_to_unsafe_separation is not None


def test_sampled_curve_covers_the_horizon():
    prediction = predict(
        walking("a", 2.0, 2.0), walking("b", 18.0, -2.0, class_name="forklift")
    )
    assert prediction.samples
    assert prediction.samples[0].seconds_ahead == 0.0
    assert prediction.samples[-1].seconds_ahead == pytest.approx(6.0, abs=0.3)


def test_sampled_curve_agrees_with_the_closed_form_minimum():
    prediction = predict(
        walking("a", 4.0, 2.0), walking("b", 16.0, -3.0, class_name="forklift")
    )
    sampled_minimum = min(s.separation for s in prediction.samples)
    assert prediction.minimum_separation <= sampled_minimum + 1e-6


# ===========================================================================
# Outcomes
# ===========================================================================
def test_converging_paths_predict_a_trajectory_conflict():
    prediction = predict(
        walking("a", 4.0, 2.0), walking("b", 16.0, -3.0, class_name="forklift")
    )
    assert prediction.outcome == PREDICTED_TRAJECTORY_CONFLICT
    assert prediction.predicts_conflict is True
    assert prediction.minimum_separation == pytest.approx(0.0, abs=0.01)
    assert prediction.seconds_to_unsafe_separation > 0


def test_a_wide_miss_is_unsafe_proximity_not_a_conflict():
    """Comes within the unsafe threshold but not within the conflict radius."""
    prediction = predict(
        walking("a", 4.0, 2.0, y_m=3.0),
        walking("b", 16.0, -3.0, y_m=4.5, class_name="forklift"),
        unsafe=2.5,
        conflict=1.0,
    )
    assert prediction.outcome == PREDICTED_UNSAFE_PROXIMITY
    assert prediction.minimum_separation > 1.0


def test_parallel_lanes_predict_no_conflict():
    prediction = predict(
        walking("a", 4.0, 2.0, y_m=2.0),
        walking("b", 16.0, -3.0, y_m=6.0, class_name="forklift"),
        unsafe=2.0,
    )
    assert prediction.outcome == NO_PREDICTED_CONFLICT
    assert prediction.enters_unsafe_separation is False
    assert prediction.minimum_separation == pytest.approx(4.0, abs=0.1)


def test_diverging_entities_predict_no_conflict():
    prediction = predict(
        walking("a", 8.0, -2.0), walking("b", 12.0, 2.0, class_name="forklift")
    )
    assert prediction.outcome == NO_PREDICTED_CONFLICT
    assert prediction.is_converging is False


def test_currently_unsafe_is_not_reported_as_a_prediction():
    """Taking credit for predicting something already happening would be wrong."""
    prediction = predict(
        walking("a", 9.5, 1.0), walking("b", 10.5, -1.0, class_name="forklift")
    )
    assert prediction.outcome == CURRENTLY_UNSAFE_PROXIMITY
    assert prediction.is_currently_unsafe is True
    assert prediction.predicts_conflict is False
    assert prediction.seconds_to_unsafe_separation == 0.0


def test_stationary_pair_predicts_no_conflict():
    prediction = predict(
        walking("a", 6.0, 0.0), walking("b", 12.0, 0.0, class_name="forklift")
    )
    assert prediction.outcome == NO_PREDICTED_CONFLICT
    assert prediction.unavailable_reason == REASON_BOTH_STATIONARY
    assert prediction.minimum_separation == pytest.approx(6.0, abs=0.05)


def test_insufficient_history_yields_no_prediction():
    prediction = predict(
        walking("a", 4.0, 2.0, count=2),
        walking("b", 16.0, -3.0, count=2, class_name="forklift"),
        at=0.1,
    )
    assert prediction.outcome == PREDICTION_UNAVAILABLE
    assert prediction.unavailable_reason == REASON_INSUFFICIENT_HISTORY
    assert prediction.minimum_separation is None
    assert prediction.seconds_to_unsafe_separation is None
    assert prediction.is_available is False


def test_insufficient_history_still_reports_the_observed_separation():
    """Refusing to predict is not refusing to observe."""
    prediction = predict(
        walking("a", 4.0, 2.0, count=2),
        walking("b", 16.0, -3.0, count=2, class_name="forklift"),
        at=0.1,
    )
    # At t=0.1 they have already closed from 12.0 m to 11.5 m.
    assert prediction.current_separation == pytest.approx(11.5, abs=0.05)


# ===========================================================================
# Units and coordinate space
# ===========================================================================
def test_world_prediction_labels_metres():
    prediction = predict(
        walking("a", 4.0, 2.0), walking("b", 16.0, -3.0, class_name="forklift")
    )
    assert prediction.coordinate_space == GROUND_PLANE_METERS
    assert prediction.units == "meters"
    payload = prediction.to_dict()
    assert "minimum_separation_m" in payload
    assert "closing_speed_m_per_s" in payload
    assert not any(key.endswith("_px") for key in payload)


def test_image_space_prediction_labels_pixels():
    predictor = TrajectoryPredictor()
    a = MotionEstimator().estimate(walking("a", 4.0, 2.0), 0.9, "a")
    b = MotionEstimator().estimate(
        walking("b", 16.0, -3.0, class_name="forklift"), 0.9, "b"
    )
    prediction = predictor.predict_pair(
        a, b, coordinate_space=IMAGE_PIXELS, unsafe_separation=90.0
    )
    assert prediction.coordinate_space == IMAGE_PIXELS
    assert prediction.units == "pixels"
    payload = prediction.to_dict()
    assert "minimum_separation_px" in payload
    assert not any(key.endswith("_m") for key in payload)


def test_requesting_world_space_without_calibration_is_unavailable():
    predictor = TrajectoryPredictor()
    a = MotionEstimator().estimate(walking("a", 4.0, 2.0), 0.9, "a")
    b = MotionEstimator().estimate(
        walking("b", 16.0, -3.0, class_name="forklift"), 0.9, "b"
    )
    prediction = predictor.predict_pair(
        a, b, coordinate_space=GROUND_PLANE_METERS, unsafe_separation=2.0
    )
    assert prediction.outcome == PREDICTION_UNAVAILABLE
    assert prediction.unavailable_reason == "no_motion_data"


def test_unknown_coordinate_space_is_rejected():
    predictor = TrajectoryPredictor()
    a = MotionEstimator().estimate(walking("a", 4.0, 2.0), 0.9, "a")
    with pytest.raises(ValueError, match="Unknown coordinate space"):
        predictor.predict_pair(a, a, coordinate_space="furlongs", unsafe_separation=1.0)


# ===========================================================================
# Single-entity prediction
# ===========================================================================
def test_entity_trajectory_extrapolates_linearly():
    predictor = TrajectoryPredictor()
    trajectory = predictor.predict_entity(
        motion(walking("a", 4.0, 2.0)), GROUND_PLANE_METERS
    )
    assert trajectory.is_estimable is True
    start = trajectory.position_at(0.0)
    later = trajectory.position_at(2.0)
    assert later[0] - start[0] == pytest.approx(4.0, abs=0.05)


def test_entity_trajectory_refuses_without_history():
    predictor = TrajectoryPredictor()
    trajectory = predictor.predict_entity(
        motion(walking("a", 4.0, 2.0, count=2), at=0.1), GROUND_PLANE_METERS
    )
    assert trajectory.is_estimable is False
    assert trajectory.velocity == (0.0, 0.0)
    assert trajectory.position_at(5.0) == trajectory.origin


def test_prediction_is_deterministic():
    first = predict(
        walking("a", 4.0, 2.0), walking("b", 16.0, -3.0, class_name="forklift")
    ).to_dict()
    second = predict(
        walking("a", 4.0, 2.0), walking("b", 16.0, -3.0, class_name="forklift")
    ).to_dict()
    assert first == second
