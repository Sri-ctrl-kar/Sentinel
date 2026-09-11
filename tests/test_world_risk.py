"""M0.5 world-space risk: scenarios A-H, space selection and time-to-risk."""

import pytest

from app.evaluation import (
    OUTCOME_NO_TRANSITION,
    OUTCOME_POSSIBLE_FALSE_POSITIVE,
    OUTCOME_TRUE_NEGATIVE,
    OUTCOME_TRUE_POSITIVE,
    ScenarioEvaluator,
)
from app.reasoning import Explainer, RiskEngine
from app.reasoning.models.risk import SEVERITIES, TimeToRisk
from app.reasoning.prediction import (
    CURRENTLY_UNSAFE_PROXIMITY,
    NO_PREDICTED_CONFLICT,
    PREDICTED_TRAJECTORY_CONFLICT,
    PREDICTION_UNAVAILABLE,
)
from app.scenarios import ALL_WORLD_SCENARIOS, SCENARIO_G_PIXEL_GAP, load_world
from app.spatial import GROUND_PLANE_METERS, IMAGE_PIXELS, pixel_distance


def assess(name):
    """Return (report, best pair assessment) for a world scenario."""
    scenario = load_world(name)
    report = RiskEngine(scenario.config, calibration=scenario.calibration).assess(
        scenario.memory, at=scenario.at
    )
    pairs = [a for a in report.assessments if len(a.involved_entity_ids) == 2]
    return scenario, report, (max(pairs, key=lambda a: a.risk_score) if pairs else None)


def rank(severity):
    return SEVERITIES.index(severity)


# ===========================================================================
# A — moving apart
# ===========================================================================
def test_scenario_a_is_low_risk():
    _s, _r, assessment = assess("A")
    assert rank(assessment.severity) <= rank("low")
    assert assessment.risk_score < 20.0


def test_scenario_a_predicts_no_conflict():
    _s, _r, assessment = assess("A")
    assert assessment.prediction_outcome == NO_PREDICTED_CONFLICT
    prediction = assessment.details["prediction"]
    assert prediction["enters_unsafe_separation"] is False
    assert prediction["closing_speed_m_per_s"] < 0


def test_scenario_a_has_no_time_to_risk():
    _s, _r, assessment = assess("A")
    assert assessment.time_to_risk.status == TimeToRisk.STATUS_NOT_PREDICTED
    assert assessment.time_to_risk.seconds is None


# ===========================================================================
# B — approaching but safely separated
# ===========================================================================
def test_scenario_b_is_elevated_but_not_critical():
    _s, _r, assessment = assess("B")
    assert rank("low") <= rank(assessment.severity) < rank("critical")


def test_scenario_b_predicts_no_unsafe_approach():
    _s, _r, assessment = assess("B")
    prediction = assessment.details["prediction"]
    assert assessment.prediction_outcome == NO_PREDICTED_CONFLICT
    assert prediction["minimum_separation_m"] > prediction["unsafe_separation_threshold_m"]
    assert prediction["closing_speed_m_per_s"] > 0  # genuinely approaching


def test_scenario_b_gives_no_time_to_risk():
    _s, _r, assessment = assess("B")
    assert assessment.time_to_risk.seconds is None


# ===========================================================================
# C — predicted unsafe closest approach
# ===========================================================================
def test_scenario_c_is_high_risk():
    _s, _r, assessment = assess("C")
    assert rank(assessment.severity) >= rank("high")


def test_scenario_c_predicts_a_trajectory_conflict():
    _s, _r, assessment = assess("C")
    assert assessment.prediction_outcome == PREDICTED_TRAJECTORY_CONFLICT
    prediction = assessment.details["prediction"]
    assert prediction["minimum_separation_m"] < prediction["conflict_radius_m"]


def test_scenario_c_estimates_time_to_the_threshold():
    _s, _r, assessment = assess("C")
    time_to_risk = assessment.time_to_risk
    assert time_to_risk.status == TimeToRisk.STATUS_PREDICTED
    assert 0.0 < time_to_risk.seconds < 6.0
    assert time_to_risk.units == "meters"


def test_scenario_c_scores_above_scenario_b():
    _b, _br, b = assess("B")
    _c, _cr, c = assess("C")
    assert c.risk_score > b.risk_score


# ===========================================================================
# D — zone entry while another entity approaches
# ===========================================================================
def test_scenario_d_is_the_highest_risk_scenario():
    _s, _r, assessment = assess("D")
    assert assessment.severity == "critical"
    others = [assess(n)[2].risk_score for n in "ABCEFG"]
    assert assessment.risk_score >= max(others)


def test_scenario_d_combines_spatial_and_contextual_evidence():
    _s, _r, assessment = assess("D")
    assert assessment.factor("zone").score > 0
    assert assessment.factor("trajectory").score > 0
    assert assessment.factor("closing_speed").score > 0
    assert assessment.escalation_multiplier > 1.0


def test_scenario_d_scores_above_the_same_approach_without_a_zone():
    """Context must add something: D is C plus an operating-zone breach."""
    _c, _cr, c = assess("C")
    _d, _dr, d = assess("D")
    assert d.risk_score > c.risk_score


# ===========================================================================
# E — stationary entities
# ===========================================================================
def test_scenario_e_has_no_trajectory_risk():
    _s, _r, assessment = assess("E")
    assert assessment.factor("trajectory").score == 0.0
    assert assessment.factor("trajectory").details["both_stationary"] is True


def test_scenario_e_predicts_no_conflict():
    _s, _r, assessment = assess("E")
    assert assessment.prediction_outcome == NO_PREDICTED_CONFLICT
    assert assessment.details["prediction"]["closing_speed_m_per_s"] == 0.0


def test_scenario_e_stays_below_medium():
    _s, _r, assessment = assess("E")
    assert rank(assessment.severity) < rank("medium")


# ===========================================================================
# F — insufficient history
# ===========================================================================
def test_scenario_f_fabricates_no_prediction():
    _s, _r, assessment = assess("F")
    assert assessment.prediction_outcome == PREDICTION_UNAVAILABLE
    assert assessment.details["prediction"]["unavailable_reason"] == "insufficient_history"


def test_scenario_f_reports_no_time_to_risk_with_a_reason():
    _s, _r, assessment = assess("F")
    time_to_risk = assessment.time_to_risk
    assert time_to_risk.status == TimeToRisk.STATUS_NOT_PREDICTED
    assert time_to_risk.seconds is None
    assert time_to_risk.reason == "insufficient_history"


def test_scenario_f_still_measures_in_world_space():
    """Refusing to predict is not refusing to observe."""
    _s, _r, assessment = assess("F")
    assert assessment.coordinate_space == GROUND_PLANE_METERS
    assert "separation_m" in assessment.factor("proximity").details


# ===========================================================================
# G — equal pixel gaps, different world distances
# ===========================================================================
def test_scenario_g_pairs_have_equal_pixel_separation():
    scenario, report, _ = assess("G")
    near = [a for a in report if set(a.involved_entity_ids) == {"person_1", "forklift_2"}][0]
    far = [a for a in report if set(a.involved_entity_ids) == {"person_3", "forklift_4"}][0]

    def pixel_gap(assessment):
        a, b = assessment.involved_entity_ids
        return pixel_distance(
            scenario.memory.state_at(a, scenario.at).position,
            scenario.memory.state_at(b, scenario.at).position,
        )

    assert pixel_gap(near) == pytest.approx(pixel_gap(far), abs=1.0)
    assert pixel_gap(near) == pytest.approx(SCENARIO_G_PIXEL_GAP, abs=2.0)


def test_scenario_g_world_separations_differ_substantially():
    _s, report, _ = assess("G")
    near = [a for a in report if set(a.involved_entity_ids) == {"person_1", "forklift_2"}][0]
    far = [a for a in report if set(a.involved_entity_ids) == {"person_3", "forklift_4"}][0]

    near_m = near.factor("proximity").details["separation_m"]
    far_m = far.factor("proximity").details["separation_m"]
    assert far_m > near_m * 1.8, (near_m, far_m)


def test_scenario_g_risk_follows_world_geometry_not_pixel_geometry():
    """Identical in pixels, different on the ground — and scored differently."""
    _s, report, _ = assess("G")
    near = [a for a in report if set(a.involved_entity_ids) == {"person_1", "forklift_2"}][0]
    far = [a for a in report if set(a.involved_entity_ids) == {"person_3", "forklift_4"}][0]
    assert near.risk_score > far.risk_score
    assert near.coordinate_space == GROUND_PLANE_METERS


# ===========================================================================
# H — uncalibrated fallback
# ===========================================================================
def test_scenario_h_falls_back_to_image_space():
    _s, report, assessment = assess("H")
    assert assessment.coordinate_space == IMAGE_PIXELS
    assert report.metadata["scoring_coordinate_space"] == IMAGE_PIXELS
    assert report.metadata["calibration"] is None


def test_scenario_h_remains_functional():
    _s, _r, assessment = assess("H")
    assert assessment.risk_score > 0
    assert assessment.prediction_outcome == PREDICTED_TRAJECTORY_CONFLICT
    assert assessment.time_to_risk.seconds is not None


def test_scenario_h_uses_pixel_units_throughout():
    _s, _r, assessment = assess("H")
    assert assessment.time_to_risk.units == "pixels"
    for factor in assessment.contributing_factors:
        for key in factor.details:
            assert not key.endswith("_m"), f"{factor.name}.{key}"
            assert not key.endswith("_m_per_s"), f"{factor.name}.{key}"


# ===========================================================================
# Space selection
# ===========================================================================
@pytest.mark.parametrize("name", ["A", "B", "C", "D", "E", "F", "G"])
def test_calibrated_scenarios_score_in_world_space(name):
    _s, _r, assessment = assess(name)
    assert assessment.coordinate_space == GROUND_PLANE_METERS
    assert assessment.space_fallback_reason is None


@pytest.mark.parametrize("name", sorted(ALL_WORLD_SCENARIOS))
def test_no_scenario_mixes_the_two_spaces(name):
    _s, _r, assessment = assess(name)
    suffix = "_m" if assessment.coordinate_space == GROUND_PLANE_METERS else "_px"
    wrong = "_px" if suffix == "_m" else "_m"
    for factor in assessment.contributing_factors:
        for key in factor.details:
            assert not key.endswith(wrong), f"{name}: {factor.name}.{key}"


@pytest.mark.parametrize("name", sorted(ALL_WORLD_SCENARIOS))
def test_every_scenario_is_deterministic(name):
    assert assess(name)[2].to_dict() == assess(name)[2].to_dict()


@pytest.mark.parametrize("name", sorted(ALL_WORLD_SCENARIOS))
def test_every_scenario_states_its_space_and_score_bounds(name):
    _s, report, assessment = assess(name)
    assert report.coordinate_space in (GROUND_PLANE_METERS, IMAGE_PIXELS)
    assert 0.0 <= assessment.risk_score <= 100.0
    assert "NOT a calibrated probability" in report.to_dict()["score_interpretation"]


# ===========================================================================
# Explainability
# ===========================================================================
def test_explanation_exposes_world_space_evidence():
    _s, _r, assessment = assess("D")
    text = Explainer().explain(assessment)

    assert "Coordinate space: ground_plane_meters" in text
    assert " m" in text          # metric measurements present
    assert "Prediction:" in text
    assert "Time-to-risk:" in text
    assert "predicted minimum separation" in text
    assert "NOT a probability" in text


def test_explanation_of_an_uncalibrated_scenario_says_pixels():
    _s, _r, assessment = assess("H")
    text = Explainer().explain(assessment)
    assert "Coordinate space: image_pixels" in text
    assert "IMAGE PIXELS" in text


def test_explanation_reports_refusal_rather_than_a_number():
    _s, _r, assessment = assess("F")
    text = Explainer().explain(assessment)
    assert "PREDICTION_UNAVAILABLE" in text
    assert "insufficient_history" in text


# ===========================================================================
# Evaluation harness
# ===========================================================================
def test_harness_evaluates_every_scenario():
    report = ScenarioEvaluator().evaluate_all()
    assert len(report) == len(ALL_WORLD_SCENARIOS)
    assert {r.scenario for r in report} == set(ALL_WORLD_SCENARIOS)


def test_harness_reports_the_required_columns():
    result = ScenarioEvaluator().evaluate_all().by_name("C")
    payload = result.to_dict()
    for key in (
        "scenario", "risk_score", "severity", "coordinate_space",
        "predicted_conflict", "time_to_risk_seconds",
        "minimum_separation_m", "closing_speed_m_per_s",
        "outcome_status", "prediction_lead_time_seconds",
    ):
        assert key in payload, key


def test_harness_measures_lead_time_for_a_genuine_approach():
    result = ScenarioEvaluator().evaluate_all().by_name("C")
    assert result.outcome_status == OUTCOME_TRUE_POSITIVE
    assert result.prediction_lead_time_seconds is not None
    assert result.prediction_lead_time_seconds > 0


def test_harness_reports_no_false_positives_on_the_safe_scenarios():
    report = ScenarioEvaluator().evaluate_all()
    for name in ("A", "B", "E"):
        assert report.by_name(name).outcome_status == OUTCOME_TRUE_NEGATIVE


def test_harness_distinguishes_static_closeness_from_a_missed_prediction():
    """G's pair is close but never transitions, so there is nothing to predict."""
    report = ScenarioEvaluator().evaluate_all()
    assert report.by_name("G").outcome_status == OUTCOME_NO_TRANSITION
    assert report.by_name("G").unsafe_at_start is True


def test_harness_finds_no_missed_detections():
    assert ScenarioEvaluator().evaluate_all().missed == []


def test_lead_time_is_measured_against_observed_data_not_the_predictor():
    """The onset time must come from recorded separation, or the metric is circular.

    M0.6 changed the other half of this subtraction. It used to measure from
    ``first_alert_time`` (a severity crossing), which would have credited the
    system for "predicting" an unsafe state that had already begun. It now
    measures from ``first_valid_prediction_time``, which is required to be a
    forward-looking prediction made before the transition. The non-circularity
    this test guards — that the onset comes from recorded positions, never from
    the predictor — is unchanged.
    """
    report = ScenarioEvaluator().evaluate_all()
    result = report.by_name("C")
    assert result.first_unsafe_time is not None
    assert result.first_valid_prediction_time is not None
    assert result.first_valid_prediction_time < result.first_unsafe_time
    assert result.prediction_lead_time_seconds == pytest.approx(
        result.first_unsafe_time - result.first_valid_prediction_time, abs=1e-6
    )


def test_harness_table_carries_the_metric_caveat():
    text = ScenarioEvaluator().evaluate_all().to_table()
    assert "NOT accuracy" in text
    assert "NOT probabilities" in text


def test_harness_is_deterministic():
    assert (
        ScenarioEvaluator().evaluate_all().to_dict()
        == ScenarioEvaluator().evaluate_all().to_dict()
    )


def test_harness_rejects_an_unknown_alert_severity():
    with pytest.raises(ValueError, match="Unknown severity"):
        ScenarioEvaluator(alert_severity="catastrophic")
