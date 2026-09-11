"""M0.4 ground-plane calibration tests.

The synthetic rectangle used throughout maps 800x400 px onto 20x10 m, so
40 px = 1 m on both axes and every expectation below is checkable by hand.
It is a mathematical fixture, not evidence of real-world accuracy.
"""

import json

import pytest

from app.calibration import (
    MIN_CORRESPONDENCES,
    DegenerateConfigurationError,
    GroundPlaneCalibration,
    GroundPoint,
    Homography,
)
from app.calibration.examples import perspective_calibration, warehouse_calibration
from app.calibration.homography import (
    mat3_determinant,
    mat3_inverse,
    mat3_multiply,
    solve_linear_system,
)
from app.spatial import GROUND_PLANE_METERS, IMAGE_PIXELS, UNIT_METERS

IMAGE_RECT = [(100, 100), (900, 100), (900, 500), (100, 500)]
WORLD_RECT = [(0, 0), (20, 0), (20, 10), (0, 10)]


def rect_calibration(**kwargs):
    return GroundPlaneCalibration(IMAGE_RECT, WORLD_RECT, **kwargs)


# ===========================================================================
# Linear algebra foundations
# ===========================================================================
def test_linear_solver_solves_a_known_system():
    solution = solve_linear_system([[2, 1], [1, 3]], [5, 10])
    assert solution[0] == pytest.approx(1.0)
    assert solution[1] == pytest.approx(3.0)


def test_linear_solver_rejects_a_singular_system():
    with pytest.raises(DegenerateConfigurationError, match="Singular"):
        solve_linear_system([[1, 2], [2, 4]], [3, 6])


def test_matrix_inverse_round_trips():
    matrix = ((2.0, 0.0, 1.0), (0.0, 3.0, 2.0), (0.0, 0.0, 1.0))
    identity = mat3_multiply(matrix, mat3_inverse(matrix))
    for i in range(3):
        for j in range(3):
            assert identity[i][j] == pytest.approx(1.0 if i == j else 0.0, abs=1e-12)


def test_singular_matrix_cannot_be_inverted():
    with pytest.raises(DegenerateConfigurationError, match="not invertible"):
        mat3_inverse(((1.0, 2.0, 3.0), (2.0, 4.0, 6.0), (1.0, 1.0, 1.0)))


def test_homography_determinant_is_non_zero_for_a_valid_fit():
    assert mat3_determinant(Homography.from_correspondences(IMAGE_RECT, WORLD_RECT).matrix) != 0


# ===========================================================================
# 1. Known four-point rectangle calibration
# ===========================================================================
def test_four_point_rectangle_reproduces_its_own_corners():
    calibration = rect_calibration()
    for image_point, world_point in zip(IMAGE_RECT, WORLD_RECT):
        mapped = calibration.image_to_world(image_point)
        assert mapped.x == pytest.approx(world_point[0], abs=1e-9)
        assert mapped.y == pytest.approx(world_point[1], abs=1e-9)


def test_four_point_calibration_reports_zero_residual_and_says_why():
    calibration = rect_calibration()
    quality = calibration.quality
    assert quality.point_count == 4
    assert quality.rms_world_error == pytest.approx(0.0, abs=1e-9)
    # ...and is explicit that this validates nothing.
    assert quality.is_exactly_determined is True
    assert any("validates nothing" in note for note in calibration.limitations())


def test_from_rectangle_helper_matches_the_explicit_form():
    helper = GroundPlaneCalibration.from_rectangle(IMAGE_RECT, 20.0, 10.0)
    explicit = rect_calibration()
    assert helper.image_to_world((500, 300)).as_tuple == pytest.approx(
        explicit.image_to_world((500, 300)).as_tuple
    )


def test_from_rectangle_rejects_the_wrong_corner_count():
    with pytest.raises(ValueError, match="exactly 4 image corners"):
        GroundPlaneCalibration.from_rectangle(IMAGE_RECT[:3], 20.0, 10.0)


def test_from_rectangle_rejects_non_positive_dimensions():
    with pytest.raises(ValueError, match="must both be positive"):
        GroundPlaneCalibration.from_rectangle(IMAGE_RECT, 0.0, 10.0)


def test_more_than_four_points_are_accepted_and_fitted():
    calibration = GroundPlaneCalibration(
        IMAGE_RECT + [(500, 300)], WORLD_RECT + [(10, 5)]
    )
    assert calibration.quality.point_count == 5
    assert calibration.quality.is_exactly_determined is False
    assert calibration.quality.rms_world_error == pytest.approx(0.0, abs=1e-6)


def test_a_bad_correspondence_shows_up_as_residual_when_over_determined():
    """The point of supplying more than four: errors become visible."""
    good = GroundPlaneCalibration(IMAGE_RECT + [(500, 300)], WORLD_RECT + [(10, 5)])
    bad = GroundPlaneCalibration(IMAGE_RECT + [(500, 300)], WORLD_RECT + [(3, 9)])
    assert bad.quality.rms_world_error > good.quality.rms_world_error
    assert bad.quality.rms_world_error > 0.1


# ===========================================================================
# 2. Image -> world mapping
# ===========================================================================
@pytest.mark.parametrize(
    "image_point,expected",
    [
        ((500, 300), (10.0, 5.0)),    # centre
        ((100, 100), (0.0, 0.0)),     # corner
        ((300, 200), (5.0, 2.5)),     # quarter
        ((900, 500), (20.0, 10.0)),   # far corner
    ],
)
def test_image_to_world_mapping(image_point, expected):
    mapped = rect_calibration().image_to_world(image_point)
    assert mapped.as_tuple == pytest.approx(expected, abs=1e-9)


def test_image_to_world_labels_its_coordinate_space_and_units():
    mapped = rect_calibration().image_to_world((500, 300))
    assert mapped.coordinate_space == GROUND_PLANE_METERS
    assert mapped.units == UNIT_METERS
    assert mapped.to_dict()["coordinate_space"] == GROUND_PLANE_METERS


def test_forty_pixels_is_one_metre_in_this_fixture():
    calibration = rect_calibration()
    a = calibration.image_to_world((500, 300))
    b = calibration.image_to_world((540, 300))
    assert a.distance_to(b) == pytest.approx(1.0, abs=1e-9)


def test_image_box_maps_via_its_ground_contact_anchor():
    """A box is projected by its feet, not its centre."""
    calibration = rect_calibration()
    bbox = (480, 200, 520, 300)  # bottom-centre is (500, 300)
    assert calibration.image_box_to_world(bbox).as_tuple == pytest.approx(
        calibration.image_to_world((500, 300)).as_tuple
    )


def test_image_box_anchor_is_selectable():
    calibration = rect_calibration()
    bbox = (480, 200, 520, 300)
    centre = calibration.image_box_to_world(bbox, anchor="center")
    feet = calibration.image_box_to_world(bbox, anchor="bottom_center")
    assert centre.y < feet.y


def test_mapping_many_points_at_once():
    mapped = rect_calibration().image_points_to_world(IMAGE_RECT)
    assert len(mapped) == 4
    assert all(isinstance(p, GroundPoint) for p in mapped)


# ===========================================================================
# 3. World -> image mapping
# ===========================================================================
@pytest.mark.parametrize(
    "world_point,expected",
    [((10, 5), (500, 300)), ((0, 0), (100, 100)), ((20, 10), (900, 500))],
)
def test_world_to_image_mapping(world_point, expected):
    assert rect_calibration().world_to_image(world_point) == pytest.approx(
        expected, abs=1e-6
    )


def test_world_to_image_is_useful_for_drawing_metric_zones():
    calibration = rect_calibration()
    metric_zone = [(5, 2), (15, 2), (15, 8), (5, 8)]
    pixels = [calibration.world_to_image(p) for p in metric_zone]
    assert pixels[0] == pytest.approx((300, 180), abs=1e-6)
    assert pixels[2] == pytest.approx((700, 420), abs=1e-6)


# ===========================================================================
# 4. Round-trip transformation
# ===========================================================================
@pytest.mark.parametrize(
    "point", [(500, 300), (150, 120), (880, 480), (321, 456), (100, 500)]
)
def test_image_world_image_round_trip(point):
    calibration = rect_calibration()
    world = calibration.image_to_world(point)
    back = calibration.world_to_image(world.as_tuple)
    assert back == pytest.approx(point, abs=1e-6)


@pytest.mark.parametrize("point", [(0, 0), (10, 5), (19.5, 9.5), (3.3, 7.7)])
def test_world_image_world_round_trip(point):
    calibration = rect_calibration()
    pixels = calibration.world_to_image(point)
    back = calibration.image_to_world(pixels)
    assert back.as_tuple == pytest.approx(point, abs=1e-6)


def test_round_trip_survives_a_genuinely_projective_calibration():
    """The affine fixture is the easy case; perspective is the real one."""
    calibration = perspective_calibration()
    for point in [(400, 300), (500, 200), (700, 450), (200, 480)]:
        world = calibration.image_to_world(point)
        assert calibration.world_to_image(world.as_tuple) == pytest.approx(
            point, abs=1e-6
        )


def test_homography_inverse_is_its_own_inverse():
    homography = Homography.from_correspondences(IMAGE_RECT, WORLD_RECT)
    twice = homography.inverse().inverse()
    for i in range(3):
        for j in range(3):
            assert twice.matrix[i][j] == pytest.approx(homography.matrix[i][j], abs=1e-9)


# ===========================================================================
# 5. Non-degenerate validation
# ===========================================================================
def test_a_valid_trapezoid_is_accepted():
    calibration = perspective_calibration()
    assert calibration.quality.point_count == 4
    assert calibration.coordinate_space == GROUND_PLANE_METERS


def test_a_convex_quadrilateral_in_any_order_is_accepted():
    calibration = GroundPlaneCalibration(
        [(100, 500), (900, 500), (900, 100), (100, 100)],
        [(0, 10), (20, 10), (20, 0), (0, 0)],
    )
    assert calibration.image_to_world((500, 300)).as_tuple == pytest.approx(
        (10.0, 5.0), abs=1e-9
    )


def test_calibration_records_the_region_it_was_fitted_over():
    calibration = rect_calibration()
    assert len(calibration.image_region) == 4
    assert len(calibration.world_region) == 4


# ===========================================================================
# 6. Degenerate / collinear rejection
# ===========================================================================
def test_collinear_image_points_are_rejected():
    with pytest.raises(DegenerateConfigurationError, match="Degenerate image"):
        GroundPlaneCalibration(
            [(100, 100), (200, 100), (300, 100), (400, 100)], WORLD_RECT
        )


def test_collinear_world_points_are_rejected():
    with pytest.raises(DegenerateConfigurationError, match="Degenerate world"):
        GroundPlaneCalibration(IMAGE_RECT, [(0, 0), (5, 0), (10, 0), (15, 0)])


def test_three_collinear_of_four_is_rejected():
    """Three points on a line leave only 3 independent constraints."""
    with pytest.raises(DegenerateConfigurationError, match="Degenerate image"):
        GroundPlaneCalibration(
            [(100, 100), (200, 100), (300, 100), (400, 500)], WORLD_RECT
        )


def test_duplicate_points_are_rejected():
    with pytest.raises(DegenerateConfigurationError, match="Duplicate image"):
        GroundPlaneCalibration(
            [(100, 100), (100, 100), (900, 500), (100, 500)], WORLD_RECT
        )


def test_duplicate_world_points_are_rejected():
    with pytest.raises(DegenerateConfigurationError, match="Duplicate world"):
        GroundPlaneCalibration(IMAGE_RECT, [(0, 0), (0, 0), (20, 10), (0, 10)])


def test_all_coincident_points_are_rejected():
    with pytest.raises(DegenerateConfigurationError):
        GroundPlaneCalibration([(5, 5)] * 4, WORLD_RECT)


def test_five_points_with_three_collinear_still_works():
    """Degeneracy is about whether ANY four points are in general position."""
    calibration = GroundPlaneCalibration(
        IMAGE_RECT + [(500, 100)], WORLD_RECT + [(10, 0)]
    )
    assert calibration.quality.point_count == 5


def test_near_collinear_points_are_rejected_by_the_tolerance():
    with pytest.raises(DegenerateConfigurationError):
        GroundPlaneCalibration(
            [(100, 100), (500, 100.0001), (900, 100), (400, 100.00005)], WORLD_RECT
        )


# ===========================================================================
# 7. Incorrect point counts
# ===========================================================================
def test_too_few_points_are_rejected():
    with pytest.raises(ValueError, match="at least 4 correspondences"):
        GroundPlaneCalibration(IMAGE_RECT[:3], WORLD_RECT[:3])


def test_mismatched_counts_are_rejected():
    with pytest.raises(ValueError, match="Point count mismatch"):
        GroundPlaneCalibration(IMAGE_RECT, WORLD_RECT[:3])


def test_empty_point_lists_are_rejected():
    with pytest.raises(ValueError):
        GroundPlaneCalibration([], [])


def test_non_2d_points_are_rejected():
    with pytest.raises(ValueError, match="must be a 2D point"):
        GroundPlaneCalibration(
            [(100, 100, 5), (900, 100), (900, 500), (100, 500)], WORLD_RECT
        )


def test_minimum_correspondence_constant_is_four():
    assert MIN_CORRESPONDENCES == 4


# ===========================================================================
# 8. Coordinates outside the calibration polygon
# ===========================================================================
def test_points_inside_the_region_are_flagged_as_calibrated():
    assert rect_calibration().image_to_world((500, 300)).in_calibrated_region is True


@pytest.mark.parametrize("point", [(50, 50), (950, 300), (500, 50), (500, 550)])
def test_points_outside_the_region_are_flagged_as_extrapolated(point):
    mapped = rect_calibration().image_to_world(point)
    assert mapped.in_calibrated_region is False


def test_extrapolated_points_still_return_a_value():
    """Flagged, not refused: the caller decides whether to trust it."""
    mapped = rect_calibration().image_to_world((50, 50))
    assert mapped.in_calibrated_region is False
    assert mapped.x < 0 and mapped.y < 0


def test_points_on_the_region_boundary_count_as_inside():
    assert rect_calibration().contains_image_point((100, 300)) is True


def test_world_region_containment():
    calibration = rect_calibration()
    assert calibration.contains_world_point((10, 5)) is True
    assert calibration.contains_world_point((100, 100)) is False


# ===========================================================================
# 9. Multiple entities sharing one calibration
# ===========================================================================
def test_one_calibration_serves_many_entities_consistently():
    calibration = rect_calibration()
    entities = {
        "worker": (300, 300),
        "forklift": (700, 300),
        "pallet": (500, 400),
    }
    mapped = {name: calibration.image_to_world(p) for name, p in entities.items()}

    assert mapped["worker"].as_tuple == pytest.approx((5.0, 5.0), abs=1e-9)
    assert mapped["forklift"].as_tuple == pytest.approx((15.0, 5.0), abs=1e-9)
    assert mapped["worker"].distance_to(mapped["forklift"]) == pytest.approx(10.0)


def test_calibration_is_stateless_across_entities():
    """Mapping one entity must not affect the next."""
    calibration = rect_calibration()
    first = calibration.image_to_world((500, 300)).as_tuple
    for _ in range(50):
        calibration.image_to_world((123, 456))
    assert calibration.image_to_world((500, 300)).as_tuple == pytest.approx(first)


def test_separation_is_symmetric_and_depth_independent():
    """The property pixel space lacks: 5m is 5m anywhere on the plane."""
    calibration = perspective_calibration()
    near_a = calibration.image_to_world((300, 480))
    near_b = calibration.world_to_image((near_a.x + 5.0, near_a.y))
    far_a = calibration.image_to_world((480, 200))
    far_b = calibration.world_to_image((far_a.x + 5.0, far_a.y))

    near_gap = calibration.image_to_world(near_b).distance_to(near_a)
    far_gap = calibration.image_to_world(far_b).distance_to(far_a)
    assert near_gap == pytest.approx(5.0, abs=1e-6)
    assert far_gap == pytest.approx(5.0, abs=1e-6)

    # ...while the same two separations differ wildly in pixels.
    near_px = abs(near_b[0] - 300)
    far_px = abs(far_b[0] - 480)
    assert near_px > far_px * 1.5


# ===========================================================================
# Metadata, serialisation and honesty
# ===========================================================================
def test_metadata_describes_the_transform_and_its_limits():
    metadata = rect_calibration(name="bay_cam").metadata()
    assert metadata["name"] == "bay_cam"
    assert metadata["source_space"] == IMAGE_PIXELS
    assert metadata["coordinate_space"] == GROUND_PLANE_METERS
    assert metadata["world_units"] == UNIT_METERS
    assert metadata["point_count"] == 4
    assert len(metadata["limitations"]) >= 5


def test_quality_states_that_residuals_are_not_accuracy():
    payload = rect_calibration().quality.to_dict()
    assert "not accuracy" in payload["residuals_measure"]
    assert "rms_world_error_meters" in payload


def test_limitations_mention_the_plane_assumption():
    notes = " ".join(rect_calibration().limitations()).lower()
    assert "plane" in notes
    assert "flat floor" in notes or "ramps" in notes
    assert "camera pose" in notes or "re-mount" in notes


def test_unsupported_units_are_refused_not_relabelled():
    with pytest.raises(ValueError, match="Unsupported world_units"):
        GroundPlaneCalibration(IMAGE_RECT, WORLD_RECT, world_units="feet")


def test_json_round_trip(tmp_path):
    original = rect_calibration(name="bay_cam", description="synthetic")
    path = original.save_json(str(tmp_path / "calibration.json"))
    reloaded = GroundPlaneCalibration.load_json(path)

    assert reloaded.name == "bay_cam"
    assert reloaded.image_to_world((500, 300)).as_tuple == pytest.approx(
        original.image_to_world((500, 300)).as_tuple
    )
    payload = json.loads(open(path, encoding="utf-8").read())
    assert payload["metadata"]["coordinate_space"] == GROUND_PLANE_METERS


def test_calibration_is_deterministic():
    a = rect_calibration().image_to_world((321, 456)).to_dict()
    b = rect_calibration().image_to_world((321, 456)).to_dict()
    assert a == b


def test_ground_points_of_different_spaces_cannot_be_compared():
    a = rect_calibration().image_to_world((500, 300))
    b = GroundPoint(1.0, 2.0, coordinate_space=IMAGE_PIXELS)
    with pytest.raises(ValueError, match="Cannot measure between"):
        a.distance_to(b)


def test_the_synthetic_example_declares_itself_synthetic():
    for calibration in (warehouse_calibration(), perspective_calibration()):
        assert "no accuracy claim" in calibration.description
