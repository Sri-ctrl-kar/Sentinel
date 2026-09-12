"""The deterministic mock reasoner.

**This is not a language model.** It is a template: it reads the evidence and
writes sentences with the numbers slotted in. It exists so that every layer
above the provider boundary — prompting, schema validation, grounding, the
CLI, the tests — can be exercised end to end with no API key, no network and
no variance between runs.

It never claims to be an LLM. ``is_language_model`` is ``False``, the model
identifier says ``deterministic-template``, and the summary itself carries a
label, so output cannot be mistaken for generated prose downstream.

Because it is deterministic it is also the grounding suite's control: if the
mock's output ever fails a grounding check, the checker has a bug, not the
model.
"""

from __future__ import annotations

from typing import List, Optional

from ..evidence import IncidentEvidence
from ..lifecycle import (
    STATE_CURRENT,
    STATE_DEVELOPING,
    STATE_IMMINENT,
    STATE_RESOLVED,
)
from ..reasoner import IncidentReasoner
from ..schema import (
    URGENCY_ELEVATED,
    URGENCY_IMMEDIATE,
    URGENCY_INFORMATIONAL,
    URGENCY_MONITOR,
    IncidentExplanation,
)

MOCK_MODEL = "deterministic-template-v0.7"

#: Prefix on every summary. Anything that displays an explanation shows this
#: without having to inspect the metadata.
MOCK_LABEL = "[mock reasoner: deterministic template, not a language model]"


class MockIncidentReasoner(IncidentReasoner):
    """Explains an incident by restating its evidence in sentences."""

    provider = "mock"
    model = MOCK_MODEL
    is_language_model = False

    def __init__(self, label: bool = True) -> None:
        self.label = label

    # ------------------------------------------------------------------
    def reason(self, evidence: IncidentEvidence) -> IncidentExplanation:
        payload = {
            "summary": self._summary(evidence),
            "severity_explanation": self._severity(evidence),
            "evidence_points": self._points(evidence),
            "predicted_outcome": self._prediction(evidence),
            "recommended_action": self._action(evidence),
            "urgency": self._urgency(evidence),
            "uncertainty": self._uncertainty(evidence),
            "coordinate_space": evidence.coordinate_space,
        }
        # Goes through exactly the same validation as a real provider's
        # output: the mock gets no shortcut past the schema.
        return IncidentExplanation.from_model_output(
            payload,
            incident_id=evidence.incident_id,
            provider=self.provider,
            model=self.model,
            is_language_model=self.is_language_model,
        )

    # ------------------------------------------------------------------
    def _summary(self, evidence: IncidentEvidence) -> str:
        who = " and ".join(
            f"{e.entity_id} ({e.class_name or 'unclassified'})"
            for e in evidence.entities
        )
        head = f"{MOCK_LABEL} " if self.label else ""
        return (
            f"{head}Sentinel's deterministic risk engine raised "
            f"{evidence.incident_type} involving {who} at "
            f"{evidence.timestamp:.2f} s, in lifecycle state "
            f"{evidence.incident_state}."
        )

    def _severity(self, evidence: IncidentEvidence) -> str:
        active = [f for f in evidence.factors if f.contribution > 0]
        if active:
            drivers = "; ".join(
                f"{f.name} contributed {f.contribution:.1f} of {f.weight:.0f} points"
                for f in active
            )
        else:
            drivers = "no factor scored above zero"
        return (
            f"Severity band {evidence.severity} follows from risk score "
            f"{evidence.risk_score:.1f} of 100, an ordinal engineering signal "
            f"and not a probability. Contributions: {drivers}."
        )

    def _points(self, evidence: IncidentEvidence) -> List[str]:
        points = [q.describe() for q in evidence.quantities()]
        for entity in evidence.entities:
            if entity.speed is not None:
                points.append(
                    f"{entity.entity_id} speed {entity.speed:.2f} "
                    f"{evidence.speed_unit}"
                )
            if entity.zones:
                points.append(
                    f"{entity.entity_id} recorded inside zone "
                    f"{', '.join(entity.zones)}"
                )
        if not points:
            points.append(
                "the evidence records no spatial measurement for this incident"
            )
        return points[:8]

    def _prediction(self, evidence: IncidentEvidence) -> str:
        prediction = evidence.prediction
        time_to_risk = evidence.time_to_risk
        if prediction is None:
            return (
                "No trajectory prediction accompanies this incident, so nothing "
                "is stated about what happens next."
            )

        parts = [f"Deterministic prediction outcome: {prediction.outcome}."]
        if time_to_risk is not None and time_to_risk.is_already_unsafe:
            parts.append(
                "The unsafe separation threshold is crossed now; this is a "
                "present condition rather than a prediction."
            )
        elif time_to_risk is not None and time_to_risk.is_predicted:
            parts.append(
                "On the current constant-velocity paths the pair is predicted "
                f"to cross the unsafe separation threshold in "
                f"{time_to_risk.seconds:.2f} s. It has not happened."
            )
        elif prediction.is_forward_looking:
            parts.append(
                "A future unsafe approach is predicted; no crossing time was "
                "solved for."
            )
        else:
            parts.append("No future unsafe approach is predicted.")

        if prediction.minimum_separation is not None:
            closest = (
                f"Predicted closest approach {prediction.minimum_separation:.2f} "
                f"{evidence.distance_unit}"
            )
            if prediction.seconds_to_minimum_separation is not None:
                closest += (
                    f", expected {prediction.seconds_to_minimum_separation:.2f} s "
                    "from now"
                )
            parts.append(closest + ".")
        if prediction.horizon_seconds is not None:
            parts.append(
                f"Prediction horizon {prediction.horizon_seconds:.1f} s."
            )
        return " ".join(parts)

    def _action(self, evidence: IncidentEvidence) -> str:
        intervention = evidence.recommended_intervention or (
            "review the situation"
        )
        return (
            "Recommendation for a human operator (Sentinel takes no action "
            f"itself): {intervention}."
        )

    def _urgency(self, evidence: IncidentEvidence) -> str:
        if evidence.incident_state == STATE_CURRENT:
            return URGENCY_IMMEDIATE
        if evidence.incident_state == STATE_IMMINENT:
            return URGENCY_IMMEDIATE
        if evidence.incident_state == STATE_DEVELOPING:
            return URGENCY_ELEVATED
        if evidence.incident_state == STATE_RESOLVED:
            return URGENCY_INFORMATIONAL
        return URGENCY_MONITOR if evidence.risk_score > 0 else URGENCY_INFORMATIONAL

    def _uncertainty(self, evidence: IncidentEvidence) -> str:
        parts: List[str] = []
        if evidence.is_metric:
            parts.append(
                "Distances are on the calibrated ground plane; their accuracy "
                "depends on the calibration survey, which this evidence does "
                "not quantify."
            )
        else:
            parts.append(
                "All distances here are image pixels. No camera calibration "
                "was applied, so no value in this report is a physical "
                "distance and none can be converted into one."
            )
        prediction = evidence.prediction
        if prediction is not None and prediction.unavailable_reason:
            parts.append(
                "The deterministic layer could not produce a prediction: "
                f"{prediction.unavailable_reason}."
            )
        if evidence.space_fallback_reason:
            parts.append(
                "The assessment fell back from the configured coordinate "
                f"space: {evidence.space_fallback_reason}."
            )
        parts.append(
            "Prediction assumes constant velocity and does not model "
            "operator reaction or obstacles."
        )
        return " ".join(parts)
