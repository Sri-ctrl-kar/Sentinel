"""Hallucination and grounding regression tests (M0.7 section G).

Eight named failure modes, each one a specific lie a language model could
tell about a Sentinel incident. Every one is caught by code, not by asking the
model nicely — the prompt reduces these; this file is what proves they cannot
reach a human unnoticed.

These tests are permanent. If a future provider, prompt or model regresses,
one of them fails.
"""

from __future__ import annotations

import pytest

from app.intelligence import (
    GroundedReasoner,
    IncidentReasoner,
    build_reasoner,
    check_grounding,
)
from app.intelligence.evidence import PredictionEvidence, TimeToRiskEvidence
from app.intelligence.grounding import (
    CLAIMED_INTERVENTION,
    COORDINATE_SPACE_MISMATCH,
    ENTITY_COUNT_INFLATION,
    INVENTED_LOCATION,
    INVENTED_NUMBER,
    MISSING_PREDICTION_FRAMING,
    PAST_TENSE_CLAIM,
    PROBABILITY_LANGUAGE,
    UNACKNOWLEDGED_LIMITATION,
    UNIT_MISMATCH,
    UNKNOWN_ENTITY,
    GroundingError,
    assert_grounded,
)
from app.intelligence.schema import ExplanationSchemaError, IncidentExplanation
from app.spatial import GROUND_PLANE_METERS, IMAGE_PIXELS
from incident_helpers import make_evidence, make_explanation, scenario_evidence


class ScriptedReasoner(IncidentReasoner):
    """A provider that returns exactly the output a test tells it to.

    Stands in for a model that has hallucinated, so the gate can be tested end
    to end rather than only through the checker.
    """

    provider = "scripted"
    model = "adversarial-fixture"
    is_language_model = True

    def __init__(self, payload):
        self.payload = payload

    def reason(self, evidence):
        return IncidentExplanation.from_model_output(
            self.payload,
            incident_id=evidence.incident_id,
            provider=self.provider,
            model=self.model,
            is_language_model=True,
        )


# ---------------------------------------------------------------------------
# The control: the deterministic mock must always pass
# ---------------------------------------------------------------------------
def test_the_mock_reasoner_is_grounded_on_every_scenario():
    """If this fails, the checker is broken — the mock only restates evidence."""
    from app.scenarios import ALL_WORLD_SCENARIOS

    reasoner = build_reasoner("mock")
    for name in ALL_WORLD_SCENARIOS:
        for calibrated in (True, False):
            evidence = scenario_evidence(name, calibrated=calibrated)
            report = check_grounding(reasoner.reason(evidence), evidence)
            assert report.ok, f"{name} (calibrated={calibrated}): {report.summary()}"


# ---------------------------------------------------------------------------
# 1. A location that does not exist
# ---------------------------------------------------------------------------
def test_case_1_an_invented_location_is_caught():
    evidence = make_evidence()
    explanation = make_explanation(
        evidence,
        summary=(
            "A worker was detected in the north aisle of the warehouse near "
            "the loading dock."
        ),
    )
    report = check_grounding(explanation, evidence)
    assert report.has(INVENTED_LOCATION)
    assert any("aisle" in v.detail or "warehouse" in v.detail for v in report.violations)


def test_case_1_a_zone_name_from_the_evidence_is_not_a_hallucination():
    """Zones are evidence. Only places the evidence lacks are inventions."""
    evidence = make_evidence(zones=["forklift_bay"])
    explanation = make_explanation(
        evidence, summary="person_1 is inside the forklift_bay zone."
    )
    assert not check_grounding(explanation, evidence).has(INVENTED_LOCATION)


# ---------------------------------------------------------------------------
# 2. Pixels are not metres
# ---------------------------------------------------------------------------
def test_case_2_pixel_evidence_reported_in_metres_is_caught():
    evidence = make_evidence(coordinate_space=IMAGE_PIXELS)
    explanation = make_explanation(
        evidence,
        coordinate_space=IMAGE_PIXELS,
        evidence_points=["the two are 3.90 metres apart"],
    )
    report = check_grounding(explanation, evidence)
    assert report.has(UNIT_MISMATCH)


def test_case_2_pixel_evidence_reported_in_any_physical_unit_is_caught():
    evidence = make_evidence(coordinate_space=IMAGE_PIXELS)
    for phrase in (
        "3.90 m apart",
        "closing at 5.50 m/s",
        "about 3.9 feet away",
        "travelling at 5.5 mph",
    ):
        explanation = make_explanation(
            evidence, coordinate_space=IMAGE_PIXELS, evidence_points=[phrase]
        )
        report = check_grounding(explanation, evidence)
        assert report.has(UNIT_MISMATCH), phrase


def test_case_2_pixel_evidence_reported_in_pixels_passes():
    evidence = make_evidence(coordinate_space=IMAGE_PIXELS)
    explanation = make_explanation(
        evidence,
        coordinate_space=IMAGE_PIXELS,
        evidence_points=["current separation: 3.90 px"],
    )
    assert not check_grounding(explanation, evidence).has(UNIT_MISMATCH)


def test_case_2_calibrated_evidence_called_pixels_is_also_caught():
    """The mistake is symmetric, and so is the check."""
    evidence = make_evidence(coordinate_space=GROUND_PLANE_METERS)
    explanation = make_explanation(
        evidence, evidence_points=["the pair are 3.90 pixels apart"]
    )
    assert check_grounding(explanation, evidence).has(UNIT_MISMATCH)


def test_case_2_a_mislabelled_coordinate_space_is_caught():
    evidence = make_evidence(coordinate_space=IMAGE_PIXELS)
    explanation = make_explanation(evidence, coordinate_space=GROUND_PLANE_METERS)
    assert check_grounding(explanation, evidence).has(COORDINATE_SPACE_MISMATCH)


# ---------------------------------------------------------------------------
# 3. A prediction is not an event
# ---------------------------------------------------------------------------
def test_case_3_a_predicted_conflict_described_as_having_happened_is_caught():
    evidence = make_evidence()
    explanation = make_explanation(
        evidence,
        summary="forklift_2 collided with person_1.",
        predicted_outcome="The impact occurred at the edge of the zone.",
    )
    report = check_grounding(explanation, evidence)
    assert report.has(PAST_TENSE_CLAIM)


def test_case_3_predictive_wording_passes():
    evidence = make_evidence()
    explanation = make_explanation(
        evidence,
        predicted_outcome=(
            "On the current paths the pair is predicted to reach the unsafe "
            "separation threshold; no contact has occurred."
        ),
    )
    report = check_grounding(explanation, evidence)
    assert not report.has(PAST_TENSE_CLAIM)
    assert not report.has(MISSING_PREDICTION_FRAMING)


def test_case_3_a_forward_looking_incident_must_be_framed_as_a_prediction():
    evidence = make_evidence()
    explanation = make_explanation(
        evidence,
        summary="person_1 and forklift_2 are close together.",
        predicted_outcome="The separation is small.",
        evidence_points=["current separation: 3.90 m"],
        uncertainty="None stated.",
    )
    assert check_grounding(explanation, evidence).has(MISSING_PREDICTION_FRAMING)


def test_case_3_a_present_unsafe_condition_may_be_described_in_the_present():
    """When the engine says it is unsafe now, saying so is not a hallucination."""
    evidence = make_evidence(
        incident_state="current",
        time_to_risk=TimeToRiskEvidence(status="already_unsafe", seconds=0.0),
        prediction=PredictionEvidence(
            outcome="CURRENTLY_UNSAFE_PROXIMITY",
            horizon_seconds=6.0,
            is_forward_looking=False,
        ),
    )
    explanation = make_explanation(
        evidence,
        summary="The pair is inside the unsafe separation threshold now.",
        predicted_outcome="The unsafe condition exists at this moment.",
    )
    report = check_grounding(explanation, evidence)
    assert not report.has(PAST_TENSE_CLAIM)
    assert not report.has(MISSING_PREDICTION_FRAMING)


# ---------------------------------------------------------------------------
# 4. A risk score is not a probability
# ---------------------------------------------------------------------------
def test_case_4_a_risk_score_restated_as_a_percentage_is_caught():
    evidence = make_evidence(risk_score=72.0)
    explanation = make_explanation(
        evidence,
        severity_explanation="There is a 72% probability of a collision.",
    )
    report = check_grounding(explanation, evidence)
    assert report.has(PROBABILITY_LANGUAGE)


@pytest.mark.parametrize(
    "phrase",
    [
        "a 72 percent chance of impact",
        "the likelihood of a collision is high",
        "the odds of contact are 72 in 100",
    ],
)
def test_case_4_probability_vocabulary_is_caught(phrase):
    evidence = make_evidence(risk_score=72.0)
    explanation = make_explanation(evidence, severity_explanation=phrase)
    assert check_grounding(explanation, evidence).has(PROBABILITY_LANGUAGE)


def test_case_4_the_disclaimer_itself_is_allowed():
    """Saying it is *not* a probability must not trip the probability check."""
    evidence = make_evidence(risk_score=72.0)
    explanation = make_explanation(
        evidence,
        severity_explanation=(
            "Risk score 72.0 is an ordinal engineering signal and not a "
            "probability."
        ),
    )
    assert not check_grounding(explanation, evidence).has(PROBABILITY_LANGUAGE)


# ---------------------------------------------------------------------------
# 5. Insufficient history must be admitted
# ---------------------------------------------------------------------------
def test_case_5_an_unacknowledged_prediction_failure_is_caught():
    evidence = make_evidence(
        prediction=PredictionEvidence(
            outcome="PREDICTION_UNAVAILABLE",
            unavailable_reason="insufficient_history",
        ),
        time_to_risk=TimeToRiskEvidence(
            status="not_predicted", reason="insufficient_history"
        ),
        incident_state="observed",
    )
    explanation = make_explanation(
        evidence,
        predicted_outcome="The pair will converge within two seconds.",
        uncertainty="Everything is well established.",
    )
    report = check_grounding(explanation, evidence)
    assert report.has(UNACKNOWLEDGED_LIMITATION)


def test_case_5_acknowledging_the_limitation_passes():
    evidence = make_evidence(
        prediction=PredictionEvidence(
            outcome="PREDICTION_UNAVAILABLE",
            unavailable_reason="insufficient_history",
        ),
        time_to_risk=TimeToRiskEvidence(
            status="not_predicted", reason="insufficient_history"
        ),
        incident_state="observed",
    )
    explanation = make_explanation(
        evidence,
        predicted_outcome=(
            "No prediction is available: the deterministic layer had "
            "insufficient history for this pair."
        ),
        uncertainty="Nothing can be said about what happens next.",
    )
    assert not check_grounding(explanation, evidence).has(UNACKNOWLEDGED_LIMITATION)


def test_case_5_the_mock_reasoner_admits_it_by_itself():
    evidence = make_evidence(
        prediction=PredictionEvidence(
            outcome="PREDICTION_UNAVAILABLE",
            unavailable_reason="insufficient_history",
        ),
        time_to_risk=TimeToRiskEvidence(
            status="not_predicted", reason="insufficient_history"
        ),
        incident_state="observed",
    )
    explanation = build_reasoner("mock").reason(evidence)
    assert "insufficient_history" in explanation.uncertainty
    assert check_grounding(explanation, evidence).ok


# ---------------------------------------------------------------------------
# 6. Two entities do not become three
# ---------------------------------------------------------------------------
def test_case_6_an_inflated_entity_count_is_caught():
    evidence = make_evidence()
    assert len(evidence.entities) == 2
    explanation = make_explanation(
        evidence, summary="Three entities are converging in the same area."
    )
    report = check_grounding(explanation, evidence)
    assert report.has(ENTITY_COUNT_INFLATION)


def test_case_6_an_extra_person_is_caught():
    evidence = make_evidence()
    explanation = make_explanation(
        evidence, summary="Two workers are standing in front of the forklift."
    )
    report = check_grounding(explanation, evidence)
    assert report.has(ENTITY_COUNT_INFLATION)


def test_case_6_an_invented_entity_id_is_caught():
    evidence = make_evidence()
    explanation = make_explanation(
        evidence, summary="person_1, forklift_2 and person_3 are converging."
    )
    report = check_grounding(explanation, evidence)
    assert report.has(UNKNOWN_ENTITY)
    assert any("person_3" in v.detail for v in report.violations)


def test_case_6_the_correct_count_passes():
    evidence = make_evidence()
    explanation = make_explanation(
        evidence, summary="Two entities are involved: person_1 and forklift_2."
    )
    report = check_grounding(explanation, evidence)
    assert not report.has(ENTITY_COUNT_INFLATION)
    assert not report.has(UNKNOWN_ENTITY)


# ---------------------------------------------------------------------------
# 7. Sentinel never acts
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "claim",
    [
        "The forklift was stopped automatically.",
        "The operator has been notified.",
        "Sentinel halted the vehicle.",
        "The alarm was sounded in the zone.",
    ],
)
def test_case_7_a_claimed_intervention_is_caught(claim):
    evidence = make_evidence()
    explanation = make_explanation(evidence, recommended_action=claim)
    assert check_grounding(explanation, evidence).has(CLAIMED_INTERVENTION)


def test_case_7_a_recommendation_is_not_an_intervention():
    evidence = make_evidence()
    explanation = make_explanation(
        evidence,
        recommended_action=(
            "Recommend that the operator slow the vehicle and that the person "
            "clear the zone. Sentinel takes no action itself."
        ),
    )
    assert not check_grounding(explanation, evidence).has(CLAIMED_INTERVENTION)


def test_case_7_the_mock_frames_its_action_as_a_recommendation():
    evidence = scenario_evidence("D")
    explanation = build_reasoner("mock").reason(evidence)
    assert explanation.recommended_action.lower().startswith("recommendation")
    assert check_grounding(explanation, evidence).ok


# ---------------------------------------------------------------------------
# 8. Malformed output never becomes an explanation
# ---------------------------------------------------------------------------
def test_case_8_malformed_model_output_is_rejected_by_schema_validation():
    evidence = make_evidence()
    reasoner = ScriptedReasoner({"summary": "something happened"})
    with pytest.raises(ExplanationSchemaError):
        reasoner.reason(evidence)


def test_case_8_non_json_output_is_rejected():
    evidence = make_evidence()
    reasoner = ScriptedReasoner("I'm sorry, I can't help with that.")
    with pytest.raises(ExplanationSchemaError):
        reasoner.reason(evidence)


def test_case_8_a_fabricated_extra_field_is_rejected():
    evidence = make_evidence()
    payload = dict(
        summary="A risk was detected.",
        severity_explanation="High severity.",
        evidence_points=["current separation: 3.90 m"],
        predicted_outcome="Predicted to close further.",
        recommended_action="Recommend slowing the vehicle.",
        urgency="elevated",
        uncertainty="Constant velocity assumed.",
        coordinate_space=GROUND_PLANE_METERS,
        collision_probability=0.87,
    )
    with pytest.raises(ExplanationSchemaError, match="unexpected field"):
        ScriptedReasoner(payload).reason(evidence)


# ---------------------------------------------------------------------------
# Invented numbers
# ---------------------------------------------------------------------------
def test_a_number_that_is_not_in_the_evidence_is_caught():
    evidence = make_evidence()
    explanation = make_explanation(
        evidence, evidence_points=["the pair are 17.25 apart"]
    )
    report = check_grounding(explanation, evidence)
    assert report.has(INVENTED_NUMBER)


def test_numbers_that_are_in_the_evidence_pass():
    evidence = make_evidence()
    explanation = make_explanation(
        evidence,
        evidence_points=[
            "current separation: 3.90 m",
            "closing speed: 5.50 m/s",
            "predicted crossing in 0.35 s",
        ],
    )
    assert not check_grounding(explanation, evidence).has(INVENTED_NUMBER)


def test_rounding_an_evidence_value_is_not_an_invention():
    evidence = make_evidence()
    explanation = make_explanation(evidence, evidence_points=["about 3.9 m apart"])
    assert not check_grounding(explanation, evidence).has(INVENTED_NUMBER)


def test_a_sign_flip_is_an_invention():
    """Closing at 5.5 and separating at -5.5 are different facts."""
    evidence = make_evidence(closing_speed=5.5)
    explanation = make_explanation(
        evidence, evidence_points=["closing speed: -5.50 m/s"]
    )
    assert check_grounding(explanation, evidence).has(INVENTED_NUMBER)


def test_entity_ids_are_not_read_as_measurements():
    evidence = make_evidence()
    explanation = make_explanation(
        evidence, evidence_points=["person_1 and forklift_2 are involved"]
    )
    assert not check_grounding(explanation, evidence).has(INVENTED_NUMBER)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------
def test_a_grounded_reasoner_refuses_to_pass_a_hallucination_through():
    evidence = make_evidence()
    hallucinating = ScriptedReasoner(
        dict(
            summary="Three people were struck in the warehouse aisle.",
            severity_explanation="There is an 87% probability of injury.",
            evidence_points=["they were 17.25 metres apart"],
            predicted_outcome="The collision occurred.",
            recommended_action="The forklift was stopped automatically.",
            urgency="immediate",
            uncertainty="None.",
            coordinate_space=GROUND_PLANE_METERS,
        )
    )
    gate = GroundedReasoner(hallucinating, strict=True)
    with pytest.raises(GroundingError) as excinfo:
        gate.reason(evidence)

    codes = set(excinfo.value.report.codes)
    assert {
        INVENTED_LOCATION,
        PROBABILITY_LANGUAGE,
        PAST_TENSE_CLAIM,
        CLAIMED_INTERVENTION,
        ENTITY_COUNT_INFLATION,
        INVENTED_NUMBER,
    } <= codes


def test_a_non_strict_gate_returns_the_output_with_its_report():
    evidence = make_evidence()
    gate = GroundedReasoner(
        ScriptedReasoner(
            dict(
                summary="The collision occurred in the warehouse.",
                severity_explanation="Severity is high.",
                evidence_points=["current separation: 3.90 m"],
                predicted_outcome="It already happened.",
                recommended_action="Recommend review.",
                urgency="immediate",
                uncertainty="None.",
                coordinate_space=GROUND_PLANE_METERS,
            )
        ),
        strict=False,
    )
    explanation = gate.reason(evidence)
    assert explanation.summary.startswith("The collision")
    assert not gate.last_report.ok


def test_the_gate_passes_grounded_output_untouched():
    evidence = scenario_evidence("D")
    gate = GroundedReasoner(build_reasoner("mock"), strict=True)
    explanation = gate.reason(evidence)
    assert explanation == build_reasoner("mock").reason(evidence)
    assert gate.last_report.ok


def test_assert_grounded_raises_with_a_readable_report():
    evidence = make_evidence()
    explanation = make_explanation(evidence, summary="Seen in the warehouse.")
    with pytest.raises(GroundingError, match="invented_location"):
        assert_grounded(explanation, evidence)


def test_a_grounding_report_serialises():
    evidence = make_evidence()
    report = check_grounding(make_explanation(evidence, summary="In the warehouse."), evidence)
    payload = report.to_dict()
    assert payload["ok"] is False
    assert payload["violations"][0]["code"] == INVENTED_LOCATION
