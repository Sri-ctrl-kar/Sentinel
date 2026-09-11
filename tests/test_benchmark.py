"""M0.6: the benchmark end to end — annotations, clips, lead time, honesty."""

import json

import pytest

from app.calibration.examples import perspective_calibration, warehouse_calibration
from app.evaluation import (
    OUTCOME_NO_TRANSITION,
    OUTCOME_TRUE_POSITIVE,
    ScenarioEvaluator,
    evaluate_calibration_conditioning,
    load_annotation,
    perspective_distinction_check,
    run_benchmark,
)
from app.evaluation.protocol import EntityTrack, parse_annotation
from app.evaluation.report import evaluate_clip, main as benchmark_main
from app.reasoning.config import RiskConfig
from app.scenarios import (
    SCENARIO_G_FAR_DEPTH_M,
    SCENARIO_G_NEAR_DEPTH_M,
    SCENARIO_G_PIXEL_GAP,
)

CLIP = "data/clips/rendered_bay.json"


# ===========================================================================
# Annotation format
# ===========================================================================
def test_minimal_annotation_needs_only_entities():
    annotation = parse_annotation(
        {"entities": [{"id": "w", "class_name": "person", "boxes": {"0": [0, 0, 10, 10]}}]}
    )
    assert annotation.entities[0].entity_id == "w"
    assert annotation.has_calibration is False
    assert annotation.events == []


def test_annotation_without_entities_is_rejected():
    with pytest.raises(ValueError, match="entities"):
        parse_annotation({"name": "x"})


def test_entity_without_boxes_is_rejected():
    with pytest.raises(ValueError, match="no annotated boxes"):
        parse_annotation({"entities": [{"id": "w", "class_name": "person", "boxes": {}}]})


def test_malformed_box_is_rejected():
    with pytest.raises(ValueError, match="4 values"):
        parse_annotation(
            {"entities": [{"id": "w", "class_name": "person", "boxes": {"0": [1, 2]}}]}
        )


def test_boxes_are_interpolated_between_annotated_frames():
    """So a human marks a box every 10 frames, not every frame."""
    track = EntityTrack("w", "person", {0: (0, 0, 10, 10), 10: (100, 0, 110, 10)})
    assert track.box_at(5) == pytest.approx((50, 0, 60, 10))
    assert track.box_at(0) == (0, 0, 10, 10)


def test_interpolation_does_not_invent_an_entity_outside_its_span():
    track = EntityTrack("w", "person", {5: (0, 0, 10, 10), 10: (50, 0, 60, 10)})
    assert track.box_at(4) is None
    assert track.box_at(11) is None


def test_annotated_actions_default_to_the_actions_present():
    annotation = parse_annotation(
        {
            "entities": [{"id": "w", "class_name": "person", "boxes": {"0": [0, 0, 1, 1]}}],
            "events": [{"entity": "w", "action": "appeared", "time": 0.0}],
        }
    )
    assert annotation.annotated_actions == ["appeared"]


def test_annotation_can_carry_calibration_and_zones():
    annotation = parse_annotation(
        {
            "entities": [{"id": "w", "class_name": "person", "boxes": {"0": [0, 0, 1, 1]}}],
            "calibration": {
                "image_points": [[100, 100], [900, 100], [900, 500], [100, 500]],
                "world_points": [[0, 0], [20, 0], [20, 10], [0, 10]],
            },
            "zones": [{"name": "bay", "rect": [400, 200, 800, 500]}],
            "operating_zones": ["bay"],
        }
    )
    assert annotation.has_calibration
    assert annotation.zones.names == ["bay"]
    assert annotation.operating_zones == ["bay"]


# ===========================================================================
# The rendered clip
# ===========================================================================
@pytest.fixture(scope="module")
def clip_result():
    pytest.importorskip("cv2")
    import os

    if not os.path.exists(CLIP):
        pytest.skip("rendered clip not generated")
    return evaluate_clip(load_annotation(CLIP))


def test_clip_ground_truth_is_independent_of_the_pipeline(clip_result):
    """Boxes come from the renderer's drawing commands, not from detection."""
    annotation = load_annotation(CLIP)
    assert "RENDERED" in annotation.notes
    assert "independent of the pipeline" in annotation.notes
    assert len(annotation.entities) == 3


def test_clip_tracking_finds_every_entity(clip_result):
    assert clip_result.ground_truth_entities == 3
    assert clip_result.predicted_tracks == 3
    assert clip_result.tracking.id_switches == 0


def test_clip_tracking_scores_are_high_but_not_perfect(clip_result):
    """A perfect score on a rendered clip would suggest a broken metric."""
    metrics = clip_result.tracking
    assert 0.9 < metrics.mota <= 1.0
    assert 0.9 < metrics.idf1 <= 1.0
    assert metrics.false_negatives + metrics.false_positives > 0


def test_clip_event_recognition_is_scored_only_where_annotated(clip_result):
    scored = {s.action for s in clip_result.events.scored_actions}
    assert scored == {"appeared", "disappeared"}
    assert "moved" in clip_result.events.unscored_actions


def test_clip_appearance_detection_is_exact(clip_result):
    appeared = clip_result.events.scores["appeared"]
    assert appeared.true_positives == 3
    assert appeared.false_positives == 0
    assert appeared.false_negatives == 0


def test_clip_surfaces_the_disappearance_latency(clip_result):
    """A real finding: max_age delays disappearance, so one entity is late.

    The car leaves at 3.15s but is reported at 3.95s, because a track must
    coast for max_age frames before retirement. Documented rather than hidden
    by widening the tolerance.
    """
    disappeared = clip_result.events.scores["disappeared"]
    assert disappeared.false_negatives == 1
    assert disappeared.false_positives == 1


def test_clip_evaluation_is_deterministic(clip_result):
    second = evaluate_clip(load_annotation(CLIP))
    assert clip_result.tracking.to_dict() == second.tracking.to_dict()


# ===========================================================================
# Lead time (requirement D/E)
# ===========================================================================
@pytest.fixture(scope="module")
def risk_report():
    return ScenarioEvaluator().evaluate_all()


def test_lead_time_is_measured_from_the_first_valid_prediction(risk_report):
    result = risk_report.by_name("C")
    assert result.first_valid_prediction_time is not None
    assert result.prediction_lead_time_seconds == pytest.approx(
        result.first_unsafe_time - result.first_valid_prediction_time, abs=1e-6
    )


def test_a_prediction_must_precede_the_transition_to_count(risk_report):
    for result in risk_report:
        if result.prediction_lead_time_seconds is not None:
            assert result.first_valid_prediction_time < result.first_unsafe_time
            assert result.prediction_lead_time_seconds > 0


def test_no_lead_time_without_an_unsafe_transition(risk_report):
    for result in risk_report:
        if result.first_unsafe_time is None:
            assert result.prediction_lead_time_seconds is None


def test_lead_time_statistics_cover_only_valid_predictions(risk_report):
    stats = risk_report.lead_time
    valid = [r for r in risk_report if r.prediction_lead_time_seconds is not None]
    assert stats.count == len(valid)
    assert stats.minimum <= stats.median <= stats.maximum
    assert stats.to_dict()["valid_predictions"] == len(valid)


def test_lead_time_statistics_are_empty_when_nothing_qualifies():
    from app.evaluation import LeadTimeStats

    empty = LeadTimeStats()
    assert empty.count == 0
    assert empty.mean is None and empty.median is None
    assert empty.minimum is None and empty.maximum is None


def test_present_tense_unsafe_proximity_does_not_count_as_prediction(risk_report):
    """G is unsafely close throughout, so it must earn no predictive credit."""
    result = risk_report.by_name("G")
    assert result.outcome_status == OUTCOME_NO_TRANSITION
    assert result.prediction_lead_time_seconds is None


def test_correct_predictions_are_true_positives(risk_report):
    for name in ("C", "D", "H"):
        assert risk_report.by_name(name).outcome_status == OUTCOME_TRUE_POSITIVE


def test_no_missed_or_late_predictions(risk_report):
    assert risk_report.missed == []
    assert risk_report.late_predictions == []


# ===========================================================================
# Calibration validation (requirement G)
# ===========================================================================
def test_a_well_conditioned_calibration_is_accepted_and_stable():
    report = evaluate_calibration_conditioning(
        [(100, 100), (900, 100), (900, 500), (100, 500)],
        [(0, 0), (20, 0), (20, 10), (0, 10)],
        "rectangle",
    )
    assert report.accepted and report.is_well_conditioned
    assert report.worst_case_error_m is not None


def test_noise_in_the_correspondences_costs_world_accuracy():
    """Calibration quality bounds world-space accuracy — measured, not asserted."""
    small = evaluate_calibration_conditioning(
        [(100, 100), (900, 100), (900, 500), (100, 500)],
        [(0, 0), (20, 0), (20, 10), (0, 10)],
        "rect",
        perturbation_px=1.0,
    )
    large = evaluate_calibration_conditioning(
        [(100, 100), (900, 100), (900, 500), (100, 500)],
        [(0, 0), (20, 0), (20, 10), (0, 10)],
        "rect",
        perturbation_px=8.0,
    )
    assert large.worst_case_error_m > small.worst_case_error_m


#: Points that pass validation but are nearly in a line in the image — a
#: camera almost at floor level. Accepted, yet badly conditioned.
ILL_CONDITIONED_IMAGE = [(100, 300), (400, 338), (700, 360), (900, 320)]
ILL_CONDITIONED_WORLD = [(0, 0), (7, 1.5), (14, 0), (18, 4)]


def test_a_poorly_conditioned_calibration_is_flagged_even_though_it_fits():
    report = evaluate_calibration_conditioning(
        ILL_CONDITIONED_IMAGE, ILL_CONDITIONED_WORLD, "flat_view"
    )
    assert report.accepted is True, "this configuration is legal, just bad"
    assert report.is_well_conditioned is False
    assert report.minimum_triangle_sine < 0.1


def test_a_poorly_conditioned_calibration_amplifies_the_same_noise():
    """The point of conditioning: identical survey error, wildly worse metres.

    This is the concrete form of "calibration quality bounds world-space
    accuracy" — a 2px error costs centimetres on a well-spread rectangle and
    tens of metres on a nearly-collinear one.
    """
    good = evaluate_calibration_conditioning(
        [(100, 100), (900, 100), (900, 500), (100, 500)],
        [(0, 0), (20, 0), (20, 10), (0, 10)],
        "rect",
        perturbation_px=2.0,
    )
    poor = evaluate_calibration_conditioning(
        ILL_CONDITIONED_IMAGE, ILL_CONDITIONED_WORLD, "flat_view", perturbation_px=2.0
    )
    assert poor.worst_case_error_m > good.worst_case_error_m * 50


def test_a_collinear_calibration_is_rejected_outright():
    report = evaluate_calibration_conditioning(
        [(100, 100), (300, 100), (500, 100), (700, 100)],
        [(0, 0), (5, 0), (10, 0), (15, 0)],
        "line",
    )
    assert report.accepted is False
    assert report.rejection_reason
    assert report.is_well_conditioned is False


def test_conditioning_report_states_it_is_not_an_accuracy_claim():
    report = evaluate_calibration_conditioning(
        [(100, 100), (900, 100), (900, 500), (100, 500)],
        [(0, 0), (20, 0), (20, 10), (0, 10)],
    )
    assert "not a measurement of" in report.to_dict()["interpretation"]


# ===========================================================================
# Perspective distinction (requirement H) — permanent regression
# ===========================================================================
def test_equal_pixel_gaps_map_to_substantially_different_world_distances():
    check = perspective_distinction_check(
        perspective_calibration(),
        near_depth_m=SCENARIO_G_NEAR_DEPTH_M,
        far_depth_m=SCENARIO_G_FAR_DEPTH_M,
        pixel_gap=SCENARIO_G_PIXEL_GAP,
    )
    assert check.pixel_gaps_match
    assert check.world_ratio >= 1.5
    assert check.passed


def test_perspective_check_fails_loudly_if_the_distinction_disappears():
    from app.evaluation.spatial import PerspectiveCheck

    identical = PerspectiveCheck(110.0, 110.0, 2.0, 2.05)
    assert identical.pixel_gaps_match
    assert identical.passed is False


def test_an_affine_calibration_shows_no_perspective_distinction():
    """The control: without perspective there is nothing for world space to add."""
    check = perspective_distinction_check(
        warehouse_calibration(), near_depth_m=2.0, far_depth_m=8.0, pixel_gap=110.0
    )
    assert check.world_ratio == pytest.approx(1.0, abs=0.01)
    assert check.passed is False


# ===========================================================================
# Threshold transparency (requirement I)
# ===========================================================================
def test_every_threshold_is_audited_with_units_and_provenance():
    rows = RiskConfig().threshold_audit()
    assert len(rows) >= 20
    for row in rows:
        assert row["parameter"] and row["units"] and row["purpose"]
        assert row["provenance"] in ("engineering-selected", "empirically-validated")


def test_the_headline_thresholds_are_labelled_engineering_assumptions():
    rows = {r["parameter"]: r for r in RiskConfig().threshold_audit()}
    for name, value in (
        ("critical_radius_m", 1.5),
        ("unsafe_separation_m", 2.0),
        ("closing_speed_reference_m_per_s", 4.0),
    ):
        assert rows[name]["value"] == value
        assert rows[name]["provenance"] == "engineering-selected"


def test_no_threshold_claims_empirical_validation_yet():
    """M0.6 provides no data that could calibrate a threshold."""
    rows = RiskConfig().threshold_audit()
    assert all(r["provenance"] == "engineering-selected" for r in rows)


# ===========================================================================
# The assembled benchmark (requirement J)
# ===========================================================================
@pytest.fixture(scope="module")
def benchmark():
    return run_benchmark()


def test_benchmark_reports_all_five_metric_families(benchmark):
    payload = benchmark.to_dict()
    for section in ("tracking", "risk", "spatial", "thresholds", "validity"):
        assert section in payload


def test_benchmark_never_collapses_to_a_single_accuracy_number(benchmark):
    payload = json.dumps(benchmark.to_dict()).lower()
    assert '"accuracy"' not in payload
    assert "overall_score" not in payload


def test_benchmark_text_has_the_required_sections(benchmark):
    text = benchmark.to_text()
    for heading in ("TRACKING", "EVENTS", "RISK", "SPATIAL", "THRESHOLDS", "VALIDITY"):
        assert heading in text


def test_benchmark_states_that_clips_are_rendered_not_real(benchmark):
    text = benchmark.to_text()
    assert "RENDERED, not real footage" in text
    assert "establish NOTHING about accuracy on real cameras" in text


def test_benchmark_states_scores_are_not_probabilities(benchmark):
    assert "NOT calibrated probabilities" in benchmark.to_text()


def test_benchmark_reports_the_perspective_verdict(benchmark):
    assert benchmark.perspective.passed
    assert "perspective distinction: PASS" in benchmark.to_text()


def test_benchmark_counts_calibrated_and_uncalibrated_cases(benchmark):
    assert benchmark.calibrated_scenarios == 7
    assert benchmark.uncalibrated_scenarios == 1


def test_benchmark_is_deterministic():
    assert run_benchmark().to_dict() == run_benchmark().to_dict()


def test_benchmark_cli_text(capsys):
    assert benchmark_main([]) == 0
    assert "SENTINEL BENCHMARK" in capsys.readouterr().out


def test_benchmark_cli_json(capsys):
    assert benchmark_main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["risk"]["scenario_count"] == 8
    assert payload["thresholds"]


def test_benchmark_survives_a_missing_clip(capsys):
    assert benchmark_main(["--clip", "does/not/exist.json", "--no-default-clip"]) == 0
    assert "skipped" in capsys.readouterr().out
