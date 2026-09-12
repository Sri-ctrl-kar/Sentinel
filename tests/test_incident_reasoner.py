"""The reasoner interface, the registry and the two providers (sections B, E, F).

No test here touches the network. The Anthropic provider is exercised with a
stub client, which is the point of building it around an injectable client:
its prompt construction, its response parsing and its error handling are all
testable with no credentials.
"""

from __future__ import annotations

import json

import pytest

from app.intelligence import (
    IncidentEvidence,
    IncidentReasoner,
    available_reasoners,
    build_reasoner,
    check_grounding,
    register_reasoner,
)
from app.intelligence.providers.anthropic_claude import (
    AnthropicIncidentReasoner,
    extract_json,
    response_text,
)
from app.intelligence.providers.mock import MOCK_LABEL, MOCK_MODEL, MockIncidentReasoner
from app.intelligence.reasoner import ReasonerError, ReasonerUnavailable
from app.intelligence.schema import ExplanationSchemaError, IncidentExplanation
from app.intelligence.settings import ReasonerSettings
from incident_helpers import make_evidence, scenario_evidence

VALID_PAYLOAD = {
    "summary": "person_1 and forklift_2 are closing.",
    "severity_explanation": "Proximity and zone factors dominate.",
    "evidence_points": ["current separation: 3.90 m"],
    "predicted_outcome": "Predicted to close further.",
    "recommended_action": "Recommend slowing the vehicle.",
    "urgency": "immediate",
    "uncertainty": "Constant velocity assumed.",
    "coordinate_space": "ground_plane_meters",
}


class Block:
    def __init__(self, type_, text=None):
        self.type = type_
        self.text = text


class Response:
    def __init__(self, content):
        self.content = content


class StubClient:
    """Records the request and replays a canned response."""

    def __init__(self, response):
        self.response = response
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def stub_reasoner(payload=VALID_PAYLOAD, **kwargs):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    client = StubClient(Response([Block("text", text)]))
    return AnthropicIncidentReasoner(client=client, **kwargs), client


# ---------------------------------------------------------------------------
# The interface
# ---------------------------------------------------------------------------
def test_a_reasoner_is_anything_with_a_reason_method():
    class Minimal(IncidentReasoner):
        provider = "minimal"
        model = "none"

        def reason(self, evidence):
            return IncidentExplanation.from_model_output(
                dict(VALID_PAYLOAD, coordinate_space=evidence.coordinate_space)
            )

    assert isinstance(Minimal(), IncidentReasoner)
    assert Minimal().reason(make_evidence()).urgency == "immediate"


def test_the_interface_cannot_be_instantiated_without_reason():
    class Broken(IncidentReasoner):
        pass

    with pytest.raises(TypeError):
        Broken()


def test_a_reasoner_describes_its_provider_and_kind():
    assert MockIncidentReasoner().describe() == (
        f"mock / {MOCK_MODEL} (deterministic template)"
    )
    reasoner, _ = stub_reasoner(model="claude-opus-5")
    assert reasoner.describe() == "anthropic / claude-opus-5 (language model)"


def test_reasoning_never_mutates_the_evidence():
    evidence = scenario_evidence("D")
    before = evidence.to_json()
    build_reasoner("mock").reason(evidence)
    assert evidence.to_json() == before


def test_the_explanation_does_not_change_the_deterministic_risk_score():
    evidence = scenario_evidence("D")
    explanation = build_reasoner("mock").reason(evidence)
    assert IncidentEvidence.from_json(evidence.to_json()).risk_score == 100.0
    assert not hasattr(explanation, "risk_score")


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------
def test_both_providers_are_registered():
    assert "mock" in available_reasoners()
    assert "anthropic" in available_reasoners()


def test_the_default_provider_is_the_mock():
    assert build_reasoner().provider == "mock"


def test_an_unknown_provider_is_a_clear_error():
    with pytest.raises(ReasonerError, match="unknown reasoner"):
        build_reasoner("gpt-9000")


def test_a_new_provider_can_be_registered_without_touching_the_core():
    class Local(MockIncidentReasoner):
        provider = "local-llama"

    register_reasoner("local-llama", Local)
    try:
        assert "local-llama" in available_reasoners()
        assert build_reasoner("local-llama").provider == "local-llama"
    finally:
        from app.intelligence import reasoner as module

        module._FACTORIES.pop("local-llama", None)


def test_a_missing_sdk_or_key_is_reported_as_unavailable():
    """Neither a missing package nor a missing key may surface as a crash.

    Passes whether or not the SDK happens to be installed: without it the
    import fails, with it the absent credential does, and both must arrive as
    ReasonerUnavailable with an actionable message.
    """
    reasoner = AnthropicIncidentReasoner.__new__(AnthropicIncidentReasoner)
    reasoner.settings = ReasonerSettings(has_credentials=False)
    with pytest.raises(ReasonerUnavailable) as excinfo:
        reasoner._build_client()
    assert "mock" in str(excinfo.value)


def test_building_the_anthropic_provider_without_credentials_is_not_a_crash():
    from app.intelligence import reasoner as module

    try:
        build_reasoner(
            "anthropic", settings=ReasonerSettings(has_credentials=False)
        )
    except ReasonerUnavailable as exc:
        assert "mock" in str(exc)
    else:  # pragma: no cover - only if a key is present in the environment
        pytest.skip("credentials are present in this environment")


# ---------------------------------------------------------------------------
# The mock provider (section E)
# ---------------------------------------------------------------------------
def test_the_mock_never_claims_to_be_a_language_model():
    reasoner = MockIncidentReasoner()
    assert reasoner.is_language_model is False
    assert "template" in reasoner.model
    explanation = reasoner.reason(make_evidence())
    assert explanation.is_language_model is False
    assert MOCK_LABEL in explanation.summary


def test_the_mock_is_deterministic():
    evidence = scenario_evidence("D")
    first = MockIncidentReasoner().reason(evidence)
    second = MockIncidentReasoner().reason(evidence)
    assert first.to_dict() == second.to_dict()


def test_the_mock_output_passes_the_same_schema_validation_as_a_model():
    explanation = MockIncidentReasoner().reason(scenario_evidence("D"))
    assert IncidentExplanation.from_dict(explanation.to_dict()) == explanation


def test_the_mock_states_the_units_of_the_evidence_it_was_given():
    metric = MockIncidentReasoner().reason(scenario_evidence("D", calibrated=True))
    pixels = MockIncidentReasoner().reason(scenario_evidence("D", calibrated=False))
    assert " m" in metric.full_text()
    assert "pixel" in pixels.full_text().lower()
    assert "metre" not in pixels.full_text().lower()


def test_the_mock_urgency_follows_the_deterministic_lifecycle_state():
    assert MockIncidentReasoner().reason(make_evidence(incident_state="current")).urgency == "immediate"
    assert MockIncidentReasoner().reason(make_evidence(incident_state="developing")).urgency == "elevated"
    assert MockIncidentReasoner().reason(make_evidence(incident_state="resolved")).urgency == "informational"


# ---------------------------------------------------------------------------
# The Anthropic provider (section F)
# ---------------------------------------------------------------------------
def test_the_provider_sends_the_grounded_prompt_and_the_json_schema():
    reasoner, client = stub_reasoner()
    evidence = make_evidence()
    reasoner.reason(evidence)

    request = client.calls[0]
    assert request["model"] == reasoner.model
    assert "INVENT NOTHING" in request["system"]
    assert evidence.incident_id in request["messages"][0]["content"]
    schema = request["output_config"]["format"]["schema"]
    assert schema["additionalProperties"] is False
    assert request["thinking"] == {"type": "adaptive"}


def test_the_provider_labels_its_output_as_model_generated():
    reasoner, _ = stub_reasoner(model="claude-opus-5")
    explanation = reasoner.reason(make_evidence())
    assert explanation.provider == "anthropic"
    assert explanation.model == "claude-opus-5"
    assert explanation.is_language_model is True


def test_the_provider_rejects_malformed_output():
    reasoner, _ = stub_reasoner("not json at all")
    with pytest.raises(ExplanationSchemaError):
        reasoner.reason(make_evidence())


def test_the_provider_rejects_output_missing_a_required_field():
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "urgency"}
    reasoner, _ = stub_reasoner(payload)
    with pytest.raises(ExplanationSchemaError, match="missing field"):
        reasoner.reason(make_evidence())


def test_the_provider_cannot_change_the_evidence():
    """Even a model that returns a different risk score changes nothing."""
    evidence = make_evidence(risk_score=72.0)
    reasoner, _ = stub_reasoner(
        dict(VALID_PAYLOAD, summary="The risk score is actually 5.")
    )
    reasoner.reason(evidence)
    assert evidence.risk_score == 72.0


def test_provider_output_goes_through_the_same_grounding_check():
    evidence = make_evidence()
    reasoner, _ = stub_reasoner(
        dict(VALID_PAYLOAD, summary="Two workers were struck in the warehouse.")
    )
    report = check_grounding(reasoner.reason(evidence), evidence)
    assert not report.ok


def test_thinking_blocks_are_never_treated_as_the_answer():
    client = StubClient(
        Response(
            [
                Block("thinking", "let me consider inventing a location"),
                Block("text", json.dumps(VALID_PAYLOAD)),
            ]
        )
    )
    explanation = AnthropicIncidentReasoner(client=client).reason(make_evidence())
    assert explanation.summary == VALID_PAYLOAD["summary"]


def test_a_parsed_structured_output_is_used_when_the_sdk_supplies_one():
    class Parsed:
        content = []
        parsed_output = VALID_PAYLOAD

    assert extract_json(Parsed()) == VALID_PAYLOAD


def test_an_empty_response_is_an_error_not_an_empty_explanation():
    with pytest.raises(ReasonerError):
        response_text(Response([]))
    with pytest.raises(ReasonerError):
        response_text(Response([Block("thinking", "hmm")]))


def test_dict_shaped_responses_are_understood():
    assert response_text({"content": [{"type": "text", "text": "hello"}]}) == "hello"
