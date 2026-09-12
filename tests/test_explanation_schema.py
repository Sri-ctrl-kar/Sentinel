"""The structured explanation schema (M0.7 section C).

The schema is the second line of defence after the prompt: whatever a model
returns, only output matching this shape ever reaches a human. Most of these
tests are about rejection, because rejection is the feature.
"""

from __future__ import annotations

import json

import pytest

from app.intelligence.schema import (
    EXPLANATION_SCHEMA_VERSION,
    MAX_EVIDENCE_POINTS,
    MAX_FIELD_CHARS,
    MODEL_AUTHORED_FIELDS,
    URGENCIES,
    ExplanationSchemaError,
    IncidentExplanation,
    explanation_json_schema,
)
from app.spatial import GROUND_PLANE_METERS, IMAGE_PIXELS

VALID = {
    "summary": "person_1 and forklift_2 are closing on the ground plane.",
    "severity_explanation": "Proximity and zone factors dominate the score.",
    "evidence_points": ["current separation: 3.90 m", "closing speed: 5.50 m/s"],
    "predicted_outcome": "Predicted to cross the unsafe threshold in 0.35 s.",
    "recommended_action": "Recommend slowing the vehicle.",
    "urgency": "immediate",
    "uncertainty": "Constant-velocity assumption.",
    "coordinate_space": GROUND_PLANE_METERS,
}


def test_a_valid_payload_is_accepted():
    explanation = IncidentExplanation.from_model_output(
        VALID, incident_id="INC-1", provider="mock", model="template"
    )
    assert explanation.urgency == "immediate"
    assert explanation.incident_id == "INC-1"
    assert explanation.provider == "mock"
    assert explanation.schema_version == EXPLANATION_SCHEMA_VERSION


def test_a_json_string_is_accepted():
    explanation = IncidentExplanation.from_model_output(json.dumps(VALID))
    assert explanation.summary == VALID["summary"]


def test_the_explanation_round_trips():
    explanation = IncidentExplanation.from_model_output(VALID, incident_id="INC-1")
    assert IncidentExplanation.from_json(explanation.to_json()) == explanation


# ---------------------------------------------------------------------------
# Rejection
# ---------------------------------------------------------------------------
def test_invalid_json_is_rejected():
    with pytest.raises(ExplanationSchemaError, match="not valid JSON"):
        IncidentExplanation.from_model_output("{definitely not json")


def test_a_non_object_is_rejected():
    with pytest.raises(ExplanationSchemaError, match="JSON object"):
        IncidentExplanation.from_model_output(["a list of things"])


@pytest.mark.parametrize("field", MODEL_AUTHORED_FIELDS)
def test_every_missing_field_is_rejected(field):
    payload = {k: v for k, v in VALID.items() if k != field}
    with pytest.raises(ExplanationSchemaError, match="missing field"):
        IncidentExplanation.from_model_output(payload)


def test_an_extra_field_is_rejected():
    """A model must not smuggle in a field nobody validates."""
    payload = dict(VALID, collision_probability=0.87)
    with pytest.raises(ExplanationSchemaError, match="unexpected field"):
        IncidentExplanation.from_model_output(payload)


def test_a_wrongly_typed_field_is_rejected():
    with pytest.raises(ExplanationSchemaError, match="must be a string"):
        IncidentExplanation.from_model_output(dict(VALID, summary={"text": "hi"}))


def test_an_empty_field_is_rejected():
    with pytest.raises(ExplanationSchemaError, match="must not be empty"):
        IncidentExplanation.from_model_output(dict(VALID, summary="   "))


def test_an_unknown_urgency_is_rejected():
    with pytest.raises(ExplanationSchemaError, match="urgency"):
        IncidentExplanation.from_model_output(dict(VALID, urgency="catastrophic"))


def test_an_unknown_coordinate_space_is_rejected():
    with pytest.raises(ExplanationSchemaError, match="coordinate_space"):
        IncidentExplanation.from_model_output(
            dict(VALID, coordinate_space="metres_probably")
        )


def test_evidence_points_must_be_a_non_empty_list_of_strings():
    with pytest.raises(ExplanationSchemaError, match="must be a list"):
        IncidentExplanation.from_model_output(dict(VALID, evidence_points="one point"))
    with pytest.raises(ExplanationSchemaError, match="must not be empty"):
        IncidentExplanation.from_model_output(dict(VALID, evidence_points=[]))
    with pytest.raises(ExplanationSchemaError, match="must be a string"):
        IncidentExplanation.from_model_output(dict(VALID, evidence_points=[3.9]))


def test_an_overlong_evidence_list_is_rejected():
    payload = dict(VALID, evidence_points=["point"] * (MAX_EVIDENCE_POINTS + 1))
    with pytest.raises(ExplanationSchemaError, match="at most"):
        IncidentExplanation.from_model_output(payload)


def test_an_overlong_field_is_rejected():
    payload = dict(VALID, summary="x" * (MAX_FIELD_CHARS + 1))
    with pytest.raises(ExplanationSchemaError, match="exceeds"):
        IncidentExplanation.from_model_output(payload)


# ---------------------------------------------------------------------------
# Shape of the contract
# ---------------------------------------------------------------------------
def test_the_schema_has_no_field_for_a_probability_or_a_measurement():
    """There is nowhere for a model to put a number it made up."""
    properties = explanation_json_schema()["properties"]
    assert set(properties) == set(MODEL_AUTHORED_FIELDS)
    for name in properties:
        assert "probab" not in name
        assert "confidence" not in name
        assert "distance" not in name
        assert "speed" not in name


def test_the_json_schema_is_closed_and_fully_required():
    schema = explanation_json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(MODEL_AUTHORED_FIELDS)


def test_the_json_schema_constrains_the_enumerated_fields():
    properties = explanation_json_schema()["properties"]
    assert properties["urgency"]["enum"] == list(URGENCIES)
    assert set(properties["coordinate_space"]["enum"]) == {
        IMAGE_PIXELS,
        GROUND_PLANE_METERS,
    }


def test_metadata_is_attached_by_the_host_not_the_model():
    """A model cannot declare what it is; the provider does."""
    assert "provider" not in MODEL_AUTHORED_FIELDS
    assert "model" not in MODEL_AUTHORED_FIELDS
    assert "is_language_model" not in MODEL_AUTHORED_FIELDS
    explanation = IncidentExplanation.from_model_output(
        VALID, provider="anthropic", model="claude-opus-5", is_language_model=True
    )
    assert explanation.is_language_model is True


def test_text_fields_cover_every_piece_of_prose():
    explanation = IncidentExplanation.from_model_output(VALID)
    text = explanation.full_text()
    for value in VALID.values():
        if isinstance(value, str) and value not in URGENCIES and "_" not in value:
            assert value in text
    for point in VALID["evidence_points"]:
        assert point in text
