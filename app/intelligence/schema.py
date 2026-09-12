"""The structured explanation an incident reasoner must produce.

Free text is not an interface. If the AI layer returned a paragraph, nothing
downstream could check it, diff it, or reject it — so it returns this instead:
a small, closed set of fields, each with a job, validated on the way in.

The schema is deliberately small. Every field is a *statement about evidence
that already exists*; there is no field in which a model could put a new
measurement, a new entity, or a probability, because no such field exists.

Metadata (``provider``, ``model``, ``is_language_model``) is attached by the
provider after generation, never by the model itself — a model cannot be
trusted to report what it is.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..spatial import GROUND_PLANE_METERS, IMAGE_PIXELS

EXPLANATION_SCHEMA_VERSION = "0.7"

URGENCY_INFORMATIONAL = "informational"
URGENCY_MONITOR = "monitor"
URGENCY_ELEVATED = "elevated"
URGENCY_IMMEDIATE = "immediate"

#: Ordered quietest to loudest.
URGENCIES = (
    URGENCY_INFORMATIONAL,
    URGENCY_MONITOR,
    URGENCY_ELEVATED,
    URGENCY_IMMEDIATE,
)

COORDINATE_SPACES = (IMAGE_PIXELS, GROUND_PLANE_METERS)

#: Fields the model itself authors. Everything else is attached by the host.
MODEL_AUTHORED_FIELDS = (
    "summary",
    "severity_explanation",
    "evidence_points",
    "predicted_outcome",
    "recommended_action",
    "urgency",
    "uncertainty",
    "coordinate_space",
)

MAX_EVIDENCE_POINTS = 8
MAX_FIELD_CHARS = 1200


class ExplanationSchemaError(ValueError):
    """Raised when generated output does not satisfy the schema.

    Carrying its own type matters: a provider can catch exactly this and
    report "the model returned something unusable" without swallowing bugs.
    """


@dataclass(frozen=True)
class IncidentExplanation:
    """A human-readable interpretation of one :class:`IncidentEvidence`.

    Interpretation only. Nothing here overrides, re-scores or contradicts the
    deterministic assessment the evidence came from — and
    :mod:`app.intelligence.grounding` checks that claim mechanically rather
    than taking the model's word for it.
    """

    summary: str
    severity_explanation: str
    evidence_points: List[str]
    predicted_outcome: str
    recommended_action: str
    urgency: str
    uncertainty: str
    coordinate_space: str
    incident_id: str = ""
    provider: str = ""
    model: str = ""
    #: False for the deterministic mock provider. Consumers that need to label
    #: output as machine-written vs template-written read this, not the name.
    is_language_model: bool = False
    schema_version: str = EXPLANATION_SCHEMA_VERSION

    # ------------------------------------------------------------------
    @property
    def is_metric(self) -> bool:
        return self.coordinate_space == GROUND_PLANE_METERS

    def text_fields(self) -> List[str]:
        """Every piece of prose in this explanation, for text-level checks."""
        return [
            self.summary,
            self.severity_explanation,
            self.predicted_outcome,
            self.recommended_action,
            self.uncertainty,
            *self.evidence_points,
        ]

    def full_text(self) -> str:
        return "\n".join(self.text_fields())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "incident_id": self.incident_id,
            "summary": self.summary,
            "severity_explanation": self.severity_explanation,
            "evidence_points": list(self.evidence_points),
            "predicted_outcome": self.predicted_outcome,
            "recommended_action": self.recommended_action,
            "urgency": self.urgency,
            "uncertainty": self.uncertainty,
            "coordinate_space": self.coordinate_space,
            "provider": self.provider,
            "model": self.model,
            "is_language_model": self.is_language_model,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    # ------------------------------------------------------------------
    @classmethod
    def from_model_output(
        cls,
        payload: Any,
        incident_id: str = "",
        provider: str = "",
        model: str = "",
        is_language_model: bool = False,
    ) -> "IncidentExplanation":
        """Validate raw model output and build an explanation from it.

        Raises :class:`ExplanationSchemaError` on anything malformed: a
        non-object, a missing field, a wrong type, an empty string, an unknown
        urgency, an unknown coordinate space, or an extra field nobody asked
        for. Strictness is the point — a provider that cannot produce the
        contract fails loudly instead of returning half an explanation.
        """
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ExplanationSchemaError(f"output is not valid JSON: {exc}") from exc

        if not isinstance(payload, Mapping):
            raise ExplanationSchemaError(
                f"output must be a JSON object, got {type(payload).__name__}"
            )

        unknown = sorted(set(payload) - set(MODEL_AUTHORED_FIELDS))
        if unknown:
            raise ExplanationSchemaError(f"unexpected field(s): {', '.join(unknown)}")

        missing = [f for f in MODEL_AUTHORED_FIELDS if f not in payload]
        if missing:
            raise ExplanationSchemaError(f"missing field(s): {', '.join(missing)}")

        values: Dict[str, Any] = {}
        for name in MODEL_AUTHORED_FIELDS:
            if name == "evidence_points":
                values[name] = _validate_points(payload[name])
            else:
                values[name] = _validate_text(name, payload[name])

        if values["urgency"] not in URGENCIES:
            raise ExplanationSchemaError(
                f"urgency must be one of {URGENCIES}, got {values['urgency']!r}"
            )
        if values["coordinate_space"] not in COORDINATE_SPACES:
            raise ExplanationSchemaError(
                "coordinate_space must be one of "
                f"{COORDINATE_SPACES}, got {values['coordinate_space']!r}"
            )

        return cls(
            incident_id=incident_id,
            provider=provider,
            model=model,
            is_language_model=is_language_model,
            **values,
        )

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "IncidentExplanation":
        """Rebuild an explanation previously produced by :meth:`to_dict`."""
        model_part = {k: payload[k] for k in MODEL_AUTHORED_FIELDS if k in payload}
        return cls.from_model_output(
            model_part,
            incident_id=payload.get("incident_id", ""),
            provider=payload.get("provider", ""),
            model=payload.get("model", ""),
            is_language_model=bool(payload.get("is_language_model", False)),
        )

    @classmethod
    def from_json(cls, text: str) -> "IncidentExplanation":
        return cls.from_dict(json.loads(text))


def _validate_text(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ExplanationSchemaError(
            f"{name} must be a string, got {type(value).__name__}"
        )
    stripped = value.strip()
    if not stripped:
        raise ExplanationSchemaError(f"{name} must not be empty")
    if len(stripped) > MAX_FIELD_CHARS:
        raise ExplanationSchemaError(
            f"{name} exceeds {MAX_FIELD_CHARS} characters ({len(stripped)})"
        )
    return stripped


def _validate_points(value: Any) -> List[str]:
    if not isinstance(value, (list, tuple)):
        raise ExplanationSchemaError(
            f"evidence_points must be a list, got {type(value).__name__}"
        )
    if not value:
        raise ExplanationSchemaError("evidence_points must not be empty")
    if len(value) > MAX_EVIDENCE_POINTS:
        raise ExplanationSchemaError(
            f"evidence_points must have at most {MAX_EVIDENCE_POINTS} entries, "
            f"got {len(value)}"
        )
    return [_validate_text("evidence_points[]", item) for item in value]


def explanation_json_schema() -> Dict[str, Any]:
    """JSON Schema for the model-authored fields.

    Handed to providers that support structured output, so malformed shapes
    are rejected at the API boundary as well as here.
    """
    return {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": (
                    "One or two sentences stating what the deterministic "
                    "system detected, using only the supplied evidence."
                ),
            },
            "severity_explanation": {
                "type": "string",
                "description": (
                    "Why this severity band, referring to the risk factors "
                    "given. The risk score is an ordinal engineering signal, "
                    "never a probability or a percentage."
                ),
            },
            "evidence_points": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": MAX_EVIDENCE_POINTS,
                "description": (
                    "Short factual bullets, each restating one supplied "
                    "measurement or factor with its units."
                ),
            },
            "predicted_outcome": {
                "type": "string",
                "description": (
                    "What the deterministic prediction says will happen and "
                    "over what time horizon. Predicted events must be worded "
                    "as predictions, not as events that have occurred."
                ),
            },
            "recommended_action": {
                "type": "string",
                "description": (
                    "A recommendation for a human operator. Never describe an "
                    "action as already taken; Sentinel does not act."
                ),
            },
            "urgency": {
                "type": "string",
                "enum": list(URGENCIES),
            },
            "uncertainty": {
                "type": "string",
                "description": (
                    "What the evidence does not establish, including any "
                    "stated reason the system could not predict."
                ),
            },
            "coordinate_space": {
                "type": "string",
                "enum": list(COORDINATE_SPACES),
                "description": (
                    "Echo the coordinate space of the supplied evidence "
                    "exactly. Pixel measurements are not metres."
                ),
            },
        },
        "required": list(MODEL_AUTHORED_FIELDS),
        "additionalProperties": False,
    }
