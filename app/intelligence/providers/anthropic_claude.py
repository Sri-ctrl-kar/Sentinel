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
from typing import Any, List, Optional

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
#: needs. ``budget_tokens`` is not used — current models reject it.
THINKING = {"type": "adaptive"}


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
                "ANTHROPIC_API_KEY is not set; export it (see .env.example) or "
                "use --reasoner mock"
            )
        return anthropic.Anthropic(timeout=self.settings.timeout_seconds)

    # ------------------------------------------------------------------
    def reason(self, evidence: IncidentEvidence) -> IncidentExplanation:
        prompt = build_prompt(evidence)
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=prompt.system,
            messages=prompt.to_messages(),
            thinking=self.thinking,
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": explanation_json_schema(),
                }
            },
        )
        payload = extract_json(response)
        return IncidentExplanation.from_model_output(
            payload,
            incident_id=evidence.incident_id,
            provider=self.provider,
            model=self.model,
            is_language_model=True,
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
