"""Scenarios A-G from the M0.3 specification.

Each scenario scripts detections through the real tracker, event generator,
temporal memory and risk engine. Nothing is mocked and nothing is random, so
these scores are exact and reproducible.

Scores are asserted as *bands* rather than exact values wherever the exact
number is a consequence of tuning rather than of the requirement. Where the
requirement IS the number (a refusal to predict, a zero score), it is asserted
exactly.
"""

import pytest

from app.reasoning import RiskEngine
from app.reasoning.models.risk import SEVERITIES

from scenarios import ALL_SCENARIOS, load


def assess(scenario, report_threshold=0.0):
    """Assess a scenario, keeping sub-threshold results visible for inspection."""
    config = scenario.config
    config.report_threshold = report_threshold
    report = RiskEngine(config).assess(scenario.memory, at=scenario.at)
    pairs = [a for a in report.assessments if len(a.involved_entity_ids) == 2]
    return report, (pairs[0] if pairs else None)


def severity_rank(severity):
    return SEVERITIES.index(severity)


# ===========================================================================
# SCENARIO A — moving apart
# ===========================================================================
def test_scenario_a_diverging_entities_are_low_risk():
    _report, assessment = assess(load("A"))
    assert assessment.risk_score < 20.0
    assert severity_rank(assessment.severity) <= severity_rank("low")


def test_scenario_a_reports_nothing_at_the_default_threshold():
    scenario = load("A")
    report = RiskEngine(scenario.config).assess(scenario.memory, at=scenario.at)
    assert report.assessments == []
    assert report.max_score == 0.0
    assert report.severity == "normal"


def test_scenario_a_has_no_motion_based_risk():
    _report, assessment = assess(load("A"))
    assert assessment.factor("closing_speed").score == 0.0
    assert assessment.factor("trajectory").score == 0.0
    assert assessment.predicted_time_to_incident_seconds is None


# ===========================================================================
# SCENARIO B — converging but not intersecting
# ===========================================================================
def test_scenario_b_is_elevated_but_not_critical():
    _report, assessment = assess(load("B"))
    assert severity_rank("low") <= severity_rank(assessment.severity)
    assert severity_rank(assessment.severity) < severity_rank("critical")


def test_scenario_b_detects_convergence_without_claiming_collision():
    _report, assessment = assess(load("B"))
    assert assessment.factor("closing_speed").score > 0.5
    assert assessment.factor("trajectory").score > 0.0
    assert assessment.incident_type == "CONVERGING_TRAJECTORIES"


def test_scenario_b_gives_no_time_to_incident_for_a_near_miss():
    """They converge, but they miss — that is not a time to incident."""
    _report, assessment = assess(load("B"))
    trajectory = assessment.factor("trajectory")
    assert trajectory.details["closest_approach_distance_px"] > 90.0
    assert assessment.predicted_time_to_incident_seconds is None


# ===========================================================================
# SCENARIO C — intersecting trajectories
# ===========================================================================
def test_scenario_c_intersecting_trajectories_are_high_risk():
    _report, assessment = assess(load("C"))
    assert severity_rank(assessment.severity) >= severity_rank("high")
    assert assessment.risk_score >= 65.0


def test_scenario_c_predicts_a_collision_and_when():
    _report, assessment = assess(load("C"))
    assert assessment.incident_type == "PERSON_VEHICLE_COLLISION_RISK"
    tti = assessment.predicted_time_to_incident_seconds
    assert tti is not None
    assert 0.0 < tti < 2.0


def test_scenario_c_closest_approach_is_essentially_zero():
    _report, assessment = assess(load("C"))
    assert assessment.factor("trajectory").details["closest_approach_distance_px"] < 90.0


def test_scenario_c_scores_higher_than_the_near_miss():
    """The only difference between B and C is whether the paths actually meet."""
    _b, b_assessment = assess(load("B"))
    _c, c_assessment = assess(load("C"))
    assert c_assessment.risk_score > b_assessment.risk_score


# ===========================================================================
# SCENARIO D — person in operating zone, vehicle approaching
# ===========================================================================
def test_scenario_d_is_critical():
    _report, assessment = assess(load("D"))
    assert assessment.severity == "critical"
    assert assessment.risk_score >= 90.0


def test_scenario_d_identifies_the_zone_breach():
    _report, assessment = assess(load("D"))
    zone = assessment.factor("zone")
    assert zone.score > 0.0
    assert "forklift_bay" in zone.details["breached_zones"]


def test_scenario_d_names_the_collision_and_recommends_stopping_the_vehicle():
    _report, assessment = assess(load("D"))
    assert assessment.incident_type == "PERSON_VEHICLE_COLLISION_RISK"
    assert "stop" in assessment.recommended_intervention.lower()


def test_scenario_d_involves_both_entities():
    _report, assessment = assess(load("D"))
    assert sorted(assessment.involved_entity_ids) == ["forklift_2", "person_1"]


def test_scenario_d_cites_zone_entry_as_evidence():
    _report, assessment = assess(load("D"))
    entries = [
        e.event_id
        for e in load("D").memory.stream(actions=["entered_zone"], entity_id="person_1")
    ]
    assert entries
    assert any(eid in assessment.evidence_event_ids for eid in entries)


# ===========================================================================
# SCENARIO E — many weak signals
# ===========================================================================
def test_scenario_e_reaches_high_risk():
    _report, assessment = assess(load("E"))
    assert severity_rank(assessment.severity) >= severity_rank("high")


def test_scenario_e_signals_are_individually_weak():
    """The whole point: no single factor is alarming on its own."""
    _report, assessment = assess(load("E"))
    for factor in assessment.contributing_factors:
        if factor.name == "escalation":
            continue
        assert factor.contribution < 20.0, (
            f"{factor.name} contributed {factor.contribution} alone — "
            "scenario E is supposed to be built from weak signals"
        )


def test_scenario_e_is_driven_by_corroboration():
    _report, assessment = assess(load("E"))
    escalation = assessment.factor("escalation")
    assert escalation.details["signal_count"] >= 4
    assert assessment.escalation_multiplier > 1.0
    # Without corroboration it would not have reached high.
    assert assessment.details["subtotal_points"] < 65.0


def test_scenario_e_persistence_recognises_the_repetition():
    _report, assessment = assess(load("E"))
    persistence = assessment.factor("persistence")
    assert persistence.score == 1.0
    assert persistence.details["violation_count"] >= 3


def test_scenario_e_repetition_belongs_to_one_entity():
    """Fragmented identities would hide the pattern entirely."""
    scenario = load("E")
    assert scenario.memory.entities() == ["person_1", "forklift_2"]


# ===========================================================================
# SCENARIO F — everything stationary
# ===========================================================================
def test_scenario_f_produces_no_trajectory_risk():
    _report, assessment = assess(load("F"))
    trajectory = assessment.factor("trajectory")
    assert trajectory.score == 0.0
    assert trajectory.details["both_stationary"] is True


def test_scenario_f_produces_no_closing_speed_risk():
    _report, assessment = assess(load("F"))
    assert assessment.factor("closing_speed").score == 0.0


def test_scenario_f_makes_no_prediction():
    _report, assessment = assess(load("F"))
    assert assessment.predicted_time_to_incident_seconds is None


def test_scenario_f_is_not_escalated():
    _report, assessment = assess(load("F"))
    assert assessment.escalation_multiplier == 1.0


def test_scenario_f_stays_below_medium():
    """Proximity alone must never manufacture a developing situation."""
    _report, assessment = assess(load("F"))
    assert severity_rank(assessment.severity) < severity_rank("medium")
    assert assessment.incident_type == "CLOSE_PROXIMITY"


# ===========================================================================
# SCENARIO G — insufficient history
# ===========================================================================
def test_scenario_g_refuses_to_predict_a_trajectory():
    _report, assessment = assess(load("G"))
    trajectory = assessment.factor("trajectory")
    assert trajectory.score == 0.0
    assert trajectory.details["insufficient_history"] is True


def test_scenario_g_says_which_entities_it_could_not_model():
    _report, assessment = assess(load("G"))
    missing = assessment.factor("trajectory").details["entities_without_trajectory"]
    assert sorted(missing) == ["forklift_2", "person_1"]


def test_scenario_g_makes_no_time_to_incident_claim():
    _report, assessment = assess(load("G"))
    assert assessment.predicted_time_to_incident_seconds is None


def test_scenario_g_still_reports_what_it_does_know():
    """Refusing to predict is not refusing to observe: proximity is a fact."""
    _report, assessment = assess(load("G"))
    assert assessment.factor("proximity").score > 0.0
    assert assessment.factor("proximity").details["separation_px"] > 0


def test_scenario_g_stays_low_despite_a_real_collision_course():
    """The entities ARE converging; the engine simply cannot know that yet."""
    _report, assessment = assess(load("G"))
    assert severity_rank(assessment.severity) < severity_rank("medium")


# ===========================================================================
# Cross-scenario properties
# ===========================================================================
@pytest.mark.parametrize("name", sorted(ALL_SCENARIOS))
def test_every_scenario_labels_its_coordinate_space(name):
    report, assessment = assess(load(name))
    assert report.coordinate_space == "image_pixels"
    assert assessment.coordinate_space == "image_pixels"
    for factor in assessment.contributing_factors:
        assert factor.coordinate_space == "image_pixels"


@pytest.mark.parametrize("name", sorted(ALL_SCENARIOS))
def test_every_scenario_is_deterministic(name):
    first_report, first = assess(load(name))
    second_report, second = assess(load(name))
    assert first.to_dict() == second.to_dict()
    assert first_report.to_dict() == second_report.to_dict()


@pytest.mark.parametrize("name", sorted(ALL_SCENARIOS))
def test_every_scenario_scores_within_bounds(name):
    _report, assessment = assess(load(name))
    assert 0.0 <= assessment.risk_score <= 100.0
    assert 0.0 <= assessment.confidence <= 1.0
    assert assessment.severity in SEVERITIES


def test_scenario_ordering_matches_intuition():
    """A < F/G < B < C < E < D."""
    scores = {name: assess(load(name))[1].risk_score for name in ALL_SCENARIOS}
    assert scores["A"] < scores["F"]
    assert scores["F"] < scores["B"]
    assert scores["B"] < scores["C"]
    assert scores["C"] < scores["E"]
    assert scores["E"] < scores["D"]
