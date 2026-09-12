"""The Anthropic provider against the real SDK, with no network (M0.7 pass 2).

Every other test in the suite drives the provider with a hand-written stub. A
stub agrees with whatever you assert about it, which is exactly the weakness
this file addresses: here the request is checked against the SDK's own
parameter types, and the response path is driven with recorded payloads parsed
by the SDK's own response models.

Recorded, not invented: ``tests/fixtures/anthropic_*.json`` were captured from
the real ``anthropic`` client talking to a loopback HTTP server — see
``scripts/verify_anthropic_provider.py``, which can replay that capture or, with
credentials, make one live call.

The whole module skips when the SDK is not installed, so the suite still runs
with no vendor package present. Nothing here opens a socket.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

anthropic = pytest.importorskip(
    "anthropic", reason="the Anthropic SDK is optional; install it to verify the provider"
)

from anthropic.types import Message  # noqa: E402
from anthropic.types.message_create_params import (  # noqa: E402
    MessageCreateParamsNonStreaming,
)

import typing_extensions as te  # noqa: E402

from app.intelligence import check_grounding  # noqa: E402
from app.intelligence.providers.anthropic_claude import (  # noqa: E402
    THINKING,
    AnthropicIncidentReasoner,
    check_not_refused,
    extract_json,
    served_model,
)
from app.intelligence.reasoner import ReasonerError  # noqa: E402
from app.intelligence.schema import (  # noqa: E402
    ExplanationSchemaError,
    explanation_json_schema,
)
from app.intelligence.settings import DEFAULT_ANTHROPIC_MODEL  # noqa: E402
from incident_helpers import scenario_evidence  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def recorded(name: str) -> Message:
    """One recorded API response, parsed by the SDK's own response model."""
    with open(os.path.join(FIXTURES, f"anthropic_{name}_response.json")) as handle:
        return Message.model_validate(json.load(handle))


class RecordingClient:
    """Replays a recorded response and remembers the request it was sent."""

    def __init__(self, response: Message):
        self.response = response
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def reasoner_for(name: str = "message"):
    client = RecordingClient(recorded(name))
    return AnthropicIncidentReasoner(client=client), client


# ---------------------------------------------------------------------------
# The request, checked against the SDK rather than against our own stub
# ---------------------------------------------------------------------------
def test_the_model_id_is_one_the_sdk_knows():
    """A typo in a model string is otherwise only discoverable at call time."""
    import typing

    literal = getattr(
        __import__("anthropic.types.model", fromlist=["Model"]), "Model"
    )
    names = []
    for arg in typing.get_args(literal):
        names.extend(typing.get_args(arg) or [arg])
    assert DEFAULT_ANTHROPIC_MODEL in [n for n in names if isinstance(n, str)]


def test_the_model_id_carries_no_date_suffix():
    assert DEFAULT_ANTHROPIC_MODEL == "claude-opus-5"


def test_every_request_key_is_a_real_messages_api_parameter():
    reasoner, _ = reasoner_for()
    payload = reasoner.request_payload(scenario_evidence("D"))
    allowed = set(te.get_type_hints(MessageCreateParamsNonStreaming))
    assert set(payload) <= allowed, f"not API parameters: {set(payload) - allowed}"


def test_the_structured_output_request_matches_the_sdk_type():
    from anthropic.types.json_output_format_param import JSONOutputFormatParam
    from anthropic.types.output_config_param import OutputConfigParam

    reasoner, _ = reasoner_for()
    output_config = reasoner.request_payload(scenario_evidence("D"))["output_config"]

    assert set(output_config) <= set(te.get_type_hints(OutputConfigParam))
    fmt = output_config["format"]
    assert set(fmt) == set(te.get_type_hints(JSONOutputFormatParam))
    assert fmt["type"] == "json_schema"
    assert fmt["schema"] == explanation_json_schema()
    assert fmt["schema"]["additionalProperties"] is False


def test_the_thinking_config_matches_the_sdk_type():
    from anthropic.types.thinking_config_adaptive_param import (
        ThinkingConfigAdaptiveParam,
    )

    assert set(THINKING) <= set(te.get_type_hints(ThinkingConfigAdaptiveParam))
    assert THINKING["type"] == "adaptive"
    assert "budget_tokens" not in THINKING


def test_the_request_carries_only_evidence_and_instructions():
    """Nothing from the pipeline reaches the model except the evidence."""
    reasoner, client = reasoner_for()
    evidence = scenario_evidence("D")
    reasoner.reason(evidence)

    sent = client.calls[0]
    user_text = sent["messages"][0]["content"]
    assert evidence.to_json() in user_text
    assert "INVENT NOTHING" in sent["system"]
    assert sent["model"] == DEFAULT_ANTHROPIC_MODEL
    assert sent["max_tokens"] >= 2000


def test_no_credential_is_ever_placed_in_the_request():
    reasoner, client = reasoner_for()
    reasoner.reason(scenario_evidence("D"))
    serialised = json.dumps(client.calls[0])
    for marker in ("api_key", "ANTHROPIC_API_KEY", "authorization", "sk-"):
        assert marker not in serialised


# ---------------------------------------------------------------------------
# The response path, driven by recorded payloads through real SDK objects
# ---------------------------------------------------------------------------
def test_a_recorded_response_produces_a_valid_explanation():
    reasoner, _ = reasoner_for()
    explanation = reasoner.reason(scenario_evidence("D"))
    assert explanation.provider == "anthropic"
    assert explanation.urgency in ("immediate", "elevated", "monitor", "informational")
    assert explanation.coordinate_space == "ground_plane_meters"


def test_a_recorded_response_passes_the_grounding_gate():
    evidence = scenario_evidence("D")
    reasoner, _ = reasoner_for()
    report = check_grounding(reasoner.reason(evidence), evidence)
    assert report.ok, report.summary()


def test_the_reported_model_comes_from_the_response_not_the_request():
    """A request states an intention; only the response says what answered."""
    reasoner, _ = reasoner_for()
    reasoner.model = "claude-opus-5-but-we-asked-wrong"
    explanation = reasoner.reason(scenario_evidence("D"))
    assert explanation.model == recorded("message").model == "claude-opus-5"


def test_thinking_blocks_in_a_real_response_are_not_read_as_the_answer():
    message = recorded("message")
    assert any(block.type == "thinking" for block in message.content)
    payload = extract_json(message)
    assert "summary" in payload


def test_a_refusal_is_reported_as_a_refusal():
    with pytest.raises(ReasonerError, match="declined"):
        check_not_refused(recorded("refusal"))


def test_a_refusal_names_its_category_and_spares_the_assessment():
    with pytest.raises(ReasonerError) as excinfo:
        reasoner_for("refusal")[0].reason(scenario_evidence("D"))
    message = str(excinfo.value)
    assert "cyber" in message
    assert "deterministic assessment is unaffected" in message


def test_malformed_real_output_is_rejected_by_schema_validation():
    with pytest.raises(ExplanationSchemaError):
        reasoner_for("malformed")[0].reason(scenario_evidence("D"))


def test_served_model_reads_the_response_field():
    assert served_model(recorded("message")) == "claude-opus-5"
    assert served_model(object()) is None


# ---------------------------------------------------------------------------
# The provider has no authority over the deterministic result
# ---------------------------------------------------------------------------
def test_the_deterministic_assessment_is_identical_across_a_provider_call():
    from app.reasoning import RiskEngine
    from app.scenarios import load_world

    scenario = load_world("D")
    engine = RiskEngine(scenario.config, calibration=scenario.calibration)
    before = engine.assess(scenario.memory, at=scenario.at).to_dict()

    evidence = scenario_evidence("D")
    evidence_before = evidence.to_json()
    reasoner_for()[0].reason(evidence)

    after = engine.assess(scenario.memory, at=scenario.at).to_dict()
    assert after == before
    assert evidence.to_json() == evidence_before
    assert before["max_risk_score"] == 100.0


# ---------------------------------------------------------------------------
# SDK error translation
# ---------------------------------------------------------------------------
class FailingClient:
    def __init__(self, exc):
        self.exc = exc
        self.messages = self

    def create(self, **kwargs):
        raise self.exc


def api_error(cls, status_code):
    """Build a real SDK error object without making a request."""
    import httpx2

    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(status_code, request=request, json={"error": {}})
    return cls("boom", response=response, body=None)


def test_an_authentication_failure_is_reported_as_unavailable():
    from app.intelligence.reasoner import ReasonerUnavailable

    reasoner = AnthropicIncidentReasoner(
        client=FailingClient(api_error(anthropic.AuthenticationError, 401))
    )
    with pytest.raises(ReasonerUnavailable, match="credential"):
        reasoner.reason(scenario_evidence("D"))


def test_a_rate_limit_is_reported_as_a_reasoner_error():
    reasoner = AnthropicIncidentReasoner(
        client=FailingClient(api_error(anthropic.RateLimitError, 429))
    )
    with pytest.raises(ReasonerError, match="rate limited"):
        reasoner.reason(scenario_evidence("D"))


def test_an_unknown_model_is_reported_with_the_setting_to_change():
    reasoner = AnthropicIncidentReasoner(
        client=FailingClient(api_error(anthropic.NotFoundError, 404))
    )
    with pytest.raises(ReasonerError, match="SENTINEL_REASONER_MODEL"):
        reasoner.reason(scenario_evidence("D"))


def test_a_connection_failure_is_reported_as_unavailable():
    from app.intelligence.reasoner import ReasonerUnavailable
    import httpx2

    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    reasoner = AnthropicIncidentReasoner(
        client=FailingClient(anthropic.APIConnectionError(request=request))
    )
    with pytest.raises(ReasonerUnavailable, match="could not reach"):
        reasoner.reason(scenario_evidence("D"))


def test_an_sdk_failure_never_escapes_as_a_vendor_exception():
    """Callers handle ReasonerError; they must not need the vendor hierarchy."""
    reasoner = AnthropicIncidentReasoner(
        client=FailingClient(api_error(anthropic.APIStatusError, 503))
    )
    with pytest.raises(ReasonerError):
        reasoner.reason(scenario_evidence("D"))


# ---------------------------------------------------------------------------
# The verification script
# ---------------------------------------------------------------------------
def test_the_verification_script_refuses_to_fake_a_live_run(capsys):
    """With no credential, --live reports NOT_RUN rather than inventing one.

    Guarded so it can never fire a real request: if this machine *does* have a
    credential, there is nothing here to test and the test skips.
    """
    from app.intelligence.settings import detect_credential_source

    if detect_credential_source(dict(os.environ)) is not None:
        pytest.skip("a credential is present; --live would make a real call")

    scripts = os.path.join(os.path.dirname(FIXTURES), os.pardir, "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from verify_anthropic_provider import main

    code = main(["--live"])
    out = capsys.readouterr().out
    assert code == 2
    assert "LIVE_PROVIDER_TEST = NOT_RUN" in out
    assert "REASON = missing credentials" in out
