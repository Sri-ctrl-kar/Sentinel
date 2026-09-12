"""Anthropic Claude incident reasoner.

The only module in Sentinel that talks to a hosted model. Everything it needs
arrives as an :class:`~app.intelligence.evidence.IncidentEvidence`; everything
it returns is a validated :class:`~app.intelligence.schema.IncidentExplanation`.
It never touches the pipeline, the memory or the risk engine, and it cannot
change a risk score — the evidence it receives is frozen.

Three deliberate choices:

* **The SDK is imported lazily.** Importing this module costs nothing and
  needs no package installed, so the prompt construction and response parsing
  below are unit-testable with a stub client and no network.
* **Structured output is requested at the API boundary**, using the same JSON
  Schema the local validator enforces. Malformed output is then rejected
  twice — by the API and by :meth:`IncidentExplanation.from_model_output`.
* **Credentials come from the environment only**, through
  :mod:`app.intelligence.settings`. No key is ever passed around in code,
  logged, or written into an explanation.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..evidence import IncidentEvidence
from ..prompt import build_prompt
from ..reasoner import IncidentReasoner, ReasonerError, ReasonerUnavailable
from ..schema import (
    ExplanationSchemaError,
    IncidentExplanation,
    explanation_json_schema,
)
from ..settings import ReasonerSettings

#: Adaptive thinking: the model decides how much reasoning a given incident
#: needs. ``budget_tokens`` is not used — current models reject it. Verified
#: against ``anthropic.types.ThinkingConfigAdaptiveParam`` in SDK 1.5.0.
THINKING = {"type": "adaptive"}

#: The structured-output request, verified against
#: ``anthropic.types.JSONOutputFormatParam`` (``{"type": "json_schema",
#: "schema": {...}}``) in SDK 1.5.0.
OUTPUT_FORMAT_TYPE = "json_schema"


class AnthropicIncidentReasoner(IncidentReasoner):
    """Explains incidents with a Claude model behind the standard interface."""

    provider = "anthropic"
    is_language_model = True

    def __init__(
        self,
        client: Any = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        settings: Optional[ReasonerSettings] = None,
        thinking: Optional[dict] = None,
    ) -> None:
        self.settings = settings or ReasonerSettings.from_env()
        self.model = model or self.settings.model
        self.max_tokens = max_tokens or self.settings.max_tokens
        self.thinking = THINKING if thinking is None else thinking
        self._client = client if client is not None else self._build_client()

    # ------------------------------------------------------------------
    def _build_client(self) -> Any:
        """Create a real SDK client, or explain precisely why we cannot."""
        try:
            import anthropic  # noqa: WPS433 - deliberately lazy
        except ImportError as exc:
            raise ReasonerUnavailable(
                "the 'anthropic' package is not installed; install it or use "
                "--reasoner mock"
            ) from exc

        if not self.settings.has_credentials:
            raise ReasonerUnavailable(
                "no Anthropic credential was found: set ANTHROPIC_API_KEY or "
                "ANTHROPIC_AUTH_TOKEN (see .env.example), sign in with "
                "'ant auth login', or use --reasoner mock"
            )
        # The SDK resolves the credential itself, in its own documented order
        # (API key, auth token, profile, workload identity). Passing one here
        # would override a user's working setup with our guess at it.
        return anthropic.Anthropic(timeout=self.settings.timeout_seconds)

    # ------------------------------------------------------------------
    def request_payload(self, evidence: IncidentEvidence) -> Dict[str, Any]:
        """Exactly what goes on the wire, as a dict.

        Separated from :meth:`reason` so the request can be inspected and
        asserted on without sending it — every key here is checked against the
        SDK's own parameter types in ``tests/test_anthropic_provider.py``.
        """
        prompt = build_prompt(evidence)
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": prompt.system,
            "messages": prompt.to_messages(),
            "thinking": self.thinking,
            "output_config": {
                "format": {
                    "type": OUTPUT_FORMAT_TYPE,
                    "schema": explanation_json_schema(),
                }
            },
        }

    def reason(self, evidence: IncidentEvidence) -> IncidentExplanation:
        response = self._call(self.request_payload(evidence))
        check_not_refused(response)
        payload = extract_json(response)
        return IncidentExplanation.from_model_output(
            payload,
            incident_id=evidence.incident_id,
            provider=self.provider,
            # What actually served the request, which is not always what was
            # asked for. Reporting the request would be reporting an intention.
            model=served_model(response) or self.model,
            is_language_model=True,
        )

    # ------------------------------------------------------------------
    def _call(self, payload: Dict[str, Any]) -> Any:
        """Send one request, turning SDK failures into reasoner failures.

        A caller of :meth:`reason` handles ``ReasonerError``; it should not
        also have to know the vendor's exception hierarchy. Authentication and
        connection problems are reported as *unavailable* — those are fixable
        by the operator — while everything else is a reasoner error.
        """
        try:
            return self._client.messages.create(**payload)
        except Exception as exc:  # noqa: BLE001 - re-raised, never swallowed
            raise _translate(exc) from exc


def _translate(exc: Exception) -> Exception:
    """Map an SDK exception onto the reasoner's own error types."""
    try:
        import anthropic
    except ImportError:  # pragma: no cover - only without the SDK installed
        return ReasonerError(f"model request failed: {exc}")

    if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
        return ReasonerUnavailable(
            f"the Anthropic credential was rejected ({exc}); check it, or use "
            "--reasoner mock"
        )
    if isinstance(exc, anthropic.APIConnectionError):
        return ReasonerUnavailable(f"could not reach the Anthropic API: {exc}")
    if isinstance(exc, anthropic.NotFoundError):
        return ReasonerError(
            f"model or endpoint not found ({exc}); check SENTINEL_REASONER_MODEL"
        )
    if isinstance(exc, anthropic.RateLimitError):
        return ReasonerError(f"rate limited by the Anthropic API: {exc}")
    if isinstance(exc, anthropic.APIStatusError):
        return ReasonerError(f"Anthropic API error {exc.status_code}: {exc}")
    return ReasonerError(f"model request failed: {exc}")


def served_model(response: Any) -> Optional[str]:
    """The model the API says answered, which may differ from the request."""
    model = getattr(response, "model", None)
    if model is None and isinstance(response, dict):
        model = response.get("model")
    return model or None


def check_not_refused(response: Any) -> None:
    """Raise if the model declined, rather than reporting 'no text block'.

    A refusal is a valid HTTP 200 with ``stop_reason == "refusal"``; treating
    it as a parse failure would hide why nothing came back.
    """
    stop_reason = getattr(response, "stop_reason", None)
    if stop_reason is None and isinstance(response, dict):
        stop_reason = response.get("stop_reason")
    if stop_reason != "refusal":
        return

    details = getattr(response, "stop_details", None)
    if details is None and isinstance(response, dict):
        details = response.get("stop_details")
    category = getattr(details, "category", None)
    if category is None and isinstance(details, dict):
        category = details.get("category")
    raise ReasonerError(
        "the model declined to answer"
        + (f" (category: {category})" if category else "")
        + "; the deterministic assessment is unaffected"
    )


def response_text(response: Any) -> str:
    """Concatenate the text blocks of a Messages API response.

    Thinking blocks are skipped: they are the model's scratch work, not its
    answer, and must never reach an explanation.
    """
    blocks = getattr(response, "content", None)
    if blocks is None and isinstance(response, dict):
        blocks = response.get("content")
    if not blocks:
        raise ReasonerError("model returned no content")

    parts: List[str] = []
    for block in blocks:
        kind = getattr(block, "type", None)
        text = getattr(block, "text", None)
        if kind is None and isinstance(block, dict):
            kind, text = block.get("type"), block.get("text")
        if kind == "text" and text:
            parts.append(text)
    if not parts:
        raise ReasonerError("model returned no text block")
    return "".join(parts)


def extract_json(response: Any) -> Any:
    """The JSON object the model was asked to produce.

    Raises :class:`ExplanationSchemaError` when the response is not JSON at
    all, so a garbled generation is rejected with the same error type as a
    structurally wrong one.
    """
    parsed = getattr(response, "parsed_output", None)
    if parsed is not None:
        return parsed

    text = response_text(response).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ExplanationSchemaError(
            f"model output was not valid JSON: {exc}"
        ) from exc
