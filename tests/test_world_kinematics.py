"""M0.4: ground-plane motion, and the pixel-space fallback.

The two halves of this file are equally important. One checks that metric
measurements are correct when a calibration exists; the other checks that
*nothing* becomes metric when one does not.
"""

import pytest

from app.calibration.examples import perspective_calibration, warehouse_calibration
from app.events.schema import Event
from app.memory import TemporalEventMemory
from app.reasoning import RiskConfig, RiskEngine
from app.reasoning.kinematics import (
    MotionEstimator,
    closest_approach,
    ground_closest_approach,
    ground_displacement,
    ground_separation,
)
from app.spatial import GROUND_PLANE_METERS, IMAGE_PIXELS

CALIBRATION = warehouse_calibration()  # 40 px == 1 m on both axes


def walking(entity_id, start_x, step_px, count=10, fps=10.0, bottom_y=300.0, class_name="person"):
    """Events for an entity walking horizontally, with ground-contact at bottom_y."""
    events = []
    for i in range(count):
        x = start_x + step_px * i
        events.append(
            Event(
                timestamp=round(i / fps, 4),
                entity_id=entity_id,
                action="appeared" if i == 0 else "moved",
                attributes={"class_name": class_name, "confidence": 0.9},
                position=[x, bottom_y - 45.0],
                bbox=[x - 20.0, bottom_y - 90.0, x + 20.0, bottom_y],
                event_id=f"evt_{entity_id}_{i:02d}",
            )
        )
    return events


def estimate(events, at=0.9, calibration=CALIBRATION, entity_id=None):
    return MotionEstimator(calibration=calibration).estimate(
        events, at, entity_id or events[0].entity_id
    )


# ===========================================================================
# World-space position
# ===========================================================================
def test_world_position_is_produced_when_calibrated():
    motion = estimate(walking("person_1", 500.0, 0.0))
    assert motion.world is not None
    assert motion.world.coordinate_space == GROUND_PLANE_METERS
    assert motion.world.units == "meters"


def test_world_position_matches_the_hand_calculation():
    # Ground contact at image (500, 300) -> world (10, 5).
    motion = estimate(walking("person_1", 500.0, 0.0))
    assert motion.world.position_m == pytest.approx((10.0, 5.0), abs=1e-9)


def test_world_position_uses_ground_contact_not_the_box_centre():
    """Projecting an object's centre would report where it is not standing."""
    motion = estimate(walking("person_1", 500.0, 0.0, bottom_y=300.0))
    centre_projection = CALIBRATION.image_to_world([500.0, 255.0])
    assert motion.world.position_m[1] != pytest.approx(centre_projection.y)
    assert motion.world.position_m == pytest.approx((10.0, 5.0), abs=1e-9)


def test_image_position_is_still_reported_alongside():
    motion = estimate(walking("person_1", 500.0, 0.0))
    assert motion.coordinate_space == IMAGE_PIXELS
    assert motion.position_px is not None
    assert motion.to_dict()["coordinate_space"] == IMAGE_PIXELS
    assert motion.to_dict()["world"]["coordinate_space"] == GROUND_PLANE_METERS


# ===========================================================================
# World-space velocity
# ===========================================================================
def test_world_velocity_is_in_metres_per_second():
    # 40 px/frame at 10 fps = 400 px/s = 10 m/s in this fixture.
    motion = estimate(walking("person_1", 200.0, 40.0))
    assert motion.world.velocity_m_per_s[0] == pytest.approx(10.0, abs=1e-6)
    assert motion.world.speed_m_per_s == pytest.approx(10.0, abs=1e-6)


def test_world_velocity_is_fitted_from_ground_positions_not_rescaled_pixels():
    """Under perspective, scaling image velocity would be wrong by a varying factor."""
    calibration = perspective_calibration()
    # Walk from the far part of the floor toward the camera.
    events = []
    for i in range(10):
        y = 200.0 + 30.0 * i
        events.append(
            Event(
                timestamp=round(i / 10.0, 4),
                entity_id="person_1",
                action="moved",
                attributes={"class_name": "person", "confidence": 0.9},
                position=[500.0, y - 45.0],
                bbox=[480.0, y - 90.0, 520.0, y],
                event_id=f"evt_{i}",
            )
        )
    motion = estimate(events, calibration=calibration)

    naive_scale = motion.speed_px_per_s / 40.0  # pretending a fixed px-per-m
    assert motion.world.speed_m_per_s > 0
    # The honest fit and the naive rescale genuinely disagree.
    assert motion.world.speed_m_per_s != pytest.approx(naive_scale, rel=0.05)


def test_world_displacement_over_time():
    motion = estimate(walking("person_1", 200.0, 40.0))
    assert ground_displacement(motion.world, 2.0) == pytest.approx(20.0, abs=1e-6)


def test_a_stationary_entity_has_no_world_velocity():
    motion = estimate(walking("person_1", 500.0, 0.0))
    assert motion.world.speed_m_per_s == pytest.approx(0.0, abs=1e-9)
    assert motion.world.is_stationary is True


def test_world_stationarity_uses_a_metric_threshold():
    estimator = MotionEstimator(calibration=CALIBRATION, stationary_speed_m_per_s=0.2)
    slow = estimator.estimate(walking("person_1", 500.0, 0.4), 0.9, "person_1")
    assert slow.world.speed_m_per_s < 0.2
    assert slow.world.is_stationary is True


def test_insufficient_history_leaves_world_motion_unestimable():
    motion = estimate(walking("person_1", 200.0, 40.0, count=2), at=0.1)
    assert motion.is_estimable is False
    assert motion.world is None or motion.world.is_estimable is False


def test_world_motion_refuses_to_extrapolate_when_not_estimable():
    motion = estimate(walking("person_1", 200.0, 40.0, count=2), at=0.1)
    if motion.world is not None:
        assert motion.world.predict_position_m(5.0) == motion.world.position_m


# ===========================================================================
# World-space separation and closing speed
# ===========================================================================
def test_world_separation_between_two_entities():
    a = estimate(walking("person_1", 300.0, 0.0))
    b = estimate(walking("forklift_2", 700.0, 0.0, class_name="forklift"))
    # 400 px apart == 10 m in this fixture.
    assert ground_separation(a, b) == pytest.approx(10.0, abs=1e-9)


def test_world_closing_speed_and_closest_approach():
    a = estimate(walking("person_1", 200.0, 20.0))           # +5 m/s
    b = estimate(walking("forklift_2", 800.0, -20.0, class_name="forklift"))  # -5 m/s
    approach = ground_closest_approach(a, b)

    assert approach is not None
    assert approach.units == "meters"
    assert approach.closing_speed_m_per_s == pytest.approx(10.0, abs=1e-6)
    assert approach.is_converging is True
    assert approach.distance_m == pytest.approx(0.0, abs=1e-6)


def test_world_and_image_approaches_agree_on_direction():
    a = estimate(walking("person_1", 200.0, 20.0))
    b = estimate(walking("forklift_2", 800.0, -20.0, class_name="forklift"))
    assert closest_approach(a, b).is_converging == ground_closest_approach(a, b).is_converging


def test_diverging_entities_are_not_converging_in_world_space():
    a = estimate(walking("person_1", 400.0, -20.0))
    b = estimate(walking("forklift_2", 600.0, 20.0, class_name="forklift"))
    approach = ground_closest_approach(a, b)
    assert approach.is_converging is False
    assert approach.closing_speed_m_per_s < 0


def test_world_approach_propagates_the_extrapolation_flag():
    a = estimate(walking("person_1", 500.0, 0.0))
    b = estimate(walking("forklift_2", 40.0, 0.0, class_name="forklift"))  # outside
    approach = ground_closest_approach(a, b)
    assert approach.in_calibrated_region is False


# ===========================================================================
# 10 & 11. No calibration available / pixel-space fallback
# ===========================================================================
def test_without_calibration_no_world_motion_is_produced():
    motion = MotionEstimator().estimate(walking("person_1", 200.0, 40.0), 0.9, "person_1")
    assert motion.world is None
    assert motion.has_world_motion is False


def test_without_calibration_everything_stays_in_pixels():
    motion = MotionEstimator().estimate(walking("person_1", 200.0, 40.0), 0.9, "person_1")
    payload = motion.to_dict()
    assert payload["coordinate_space"] == IMAGE_PIXELS
    assert "world" not in payload
    assert not any("_m_per_s" in key or key.endswith("_m") for key in payload)


def test_estimator_reports_which_space_it_can_produce():
    assert MotionEstimator().coordinate_space == IMAGE_PIXELS
    assert MotionEstimator(calibration=CALIBRATION).coordinate_space == GROUND_PLANE_METERS


def test_world_helpers_return_none_rather_than_falling_back_to_pixels():
    """Asking for metres must never silently yield pixels."""
    a = MotionEstimator().estimate(walking("person_1", 200.0, 20.0), 0.9, "person_1")
    b = MotionEstimator().estimate(
        walking("forklift_2", 800.0, -20.0, class_name="forklift"), 0.9, "forklift_2"
    )
    assert ground_separation(a, b) is None
    assert ground_closest_approach(a, b) is None


def test_image_space_measurements_are_unchanged_by_calibration():
    """Adding a calibration must not perturb any pixel-space number."""
    events = walking("person_1", 200.0, 40.0)
    plain = MotionEstimator().estimate(events, 0.9, "person_1")
    calibrated = MotionEstimator(calibration=CALIBRATION).estimate(events, 0.9, "person_1")

    assert plain.position_px == calibrated.position_px
    assert plain.velocity_px_per_s == calibrated.velocity_px_per_s
    assert plain.confidence == calibrated.confidence
    assert plain.is_estimable == calibrated.is_estimable


# ===========================================================================
# 12. Risk engine integration — scores must not move
# ===========================================================================
def build_memory():
    events = walking("person_1", 200.0, 20.0, count=12) + walking(
        "forklift_2", 800.0, -20.0, count=12, class_name="forklift"
    )
    return TemporalEventMemory.from_events(events)


def config():
    return RiskConfig(vehicle_classes=["forklift"], report_threshold=0.0)


def test_calibration_does_not_change_any_risk_score():
    memory = build_memory()
    plain = RiskEngine(config()).assess(memory, at=1.1)
    calibrated = RiskEngine(config(), calibration=CALIBRATION).assess(memory, at=1.1)

    assert [a.risk_score for a in plain] == [a.risk_score for a in calibrated]
    assert [a.severity for a in plain] == [a.severity for a in calibrated]
    assert [a.incident_type for a in plain] == [a.incident_type for a in calibrated]


def test_scoring_remains_image_space_and_says_so():
    report = RiskEngine(config(), calibration=CALIBRATION).assess(build_memory(), at=1.1)
    assert report.metadata["scoring_coordinate_space"] == IMAGE_PIXELS
    assert report.coordinate_space == IMAGE_PIXELS
    for assessment in report:
        assert assessment.coordinate_space == IMAGE_PIXELS


def test_calibrated_assessments_report_metric_measurements():
    report = RiskEngine(config(), calibration=CALIBRATION).assess(build_memory(), at=1.1)
    pair = next(a for a in report if len(a.involved_entity_ids) == 2)
    ground = pair.details["ground_plane"]

    assert ground["coordinate_space"] == GROUND_PLANE_METERS
    assert ground["units"] == "meters"
    assert ground["separation_m"] > 0
    assert "image-space" in ground["note"]
    assert set(ground["positions_m"]) == set(pair.involved_entity_ids)


def test_uncalibrated_assessments_carry_no_metric_block():
    report = RiskEngine(config()).assess(build_memory(), at=1.1)
    for assessment in report:
        assert "ground_plane" not in assessment.details


def test_report_metadata_records_whether_a_calibration_was_used():
    memory = build_memory()
    assert RiskEngine(config()).assess(memory, at=1.1).metadata["calibration"] is None
    used = RiskEngine(config(), calibration=CALIBRATION).assess(memory, at=1.1)
    assert used.metadata["calibration"]["coordinate_space"] == GROUND_PLANE_METERS


def test_metric_details_are_measurements_not_probabilities():
    """Guard the vocabulary: a risk score is not a probability."""
    report = RiskEngine(config(), calibration=CALIBRATION).assess(build_memory(), at=1.1)
    payload = report.to_dict()
    serialised = str(payload).lower()
    assert "probability" not in serialised
