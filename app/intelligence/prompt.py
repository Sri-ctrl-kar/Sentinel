"""Evidence-grounded prompting.

The prompt is where the model is told what it is and, more importantly, what
it is not. It is not the risk engine, it is not an observer of the scene, and
it is not a source of numbers. It receives one incident's evidence and writes
about that evidence.

Prompting alone does not make output trustworthy — :mod:`app.intelligence.
grounding` checks the result mechanically, and that check is what the tests
assert on. The prompt's job is to make violations rare; the checker's job is
to make them visible.

This module builds strings. It imports no model SDK and can be exercised in
full without a network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from ..spatial import GROUND_PLANE_METERS
from .evidence import IncidentEvidence
from .lifecycle import describe_state
from .schema import URGENCIES

SYSTEM_PROMPT = """\
You are the incident-reporting layer of Sentinel, an industrial safety \
monitoring system.

Sentinel's deterministic pipeline has ALREADY made every judgement that \
matters. It detected the objects, tracked them, measured their positions and \
velocities, predicted their paths, scored the risk, and decided that this \
incident exists. That pipeline is authoritative. Your only job is to explain \
its structured evidence in language a human operator can act on.

RULES — these are not style preferences, they are correctness requirements:

1. USE ONLY THE SUPPLIED EVIDENCE. Every fact you state must appear in the \
evidence block. If it is not there, you do not know it.
2. INVENT NOTHING. No locations, sites, buildings, aisles or streets. No \
people, vehicles or objects beyond the entities listed. No measurements, \
times, speeds or distances beyond those given. No names. No weather, no \
lighting, no intent, no emotion.
3. NEVER CONVERT UNITS. Report each number in exactly the unit given. \
Measurements in image pixels are pixels; they are NOT metres, feet, miles \
per hour, or any physical distance. Say "pixels" when the evidence says \
pixels.
4. THE RISK SCORE IS NOT A PROBABILITY. It is an ordinal 0-100 engineering \
signal. Never write it as a percentage, a likelihood, a chance, or odds. \
"Risk score 72" is correct; "72% chance of a collision" is forbidden.
5. DISTINGUISH PRESENT FROM PREDICTED. If the evidence says a conflict is \
predicted, write about it in the future or conditional tense — it has not \
happened. Only describe something as occurring now when the evidence says \
the unsafe condition exists now.
6. SENTINEL DOES NOT ACT. It observes and reports. Never state or imply that \
anything was stopped, halted, alerted, braked, shut down, evacuated or \
notified. Recommendations are recommendations for a human.
7. SAY SO WHEN THE EVIDENCE IS THIN. If the evidence records that a \
prediction was unavailable, or gives a reason such as insufficient history, \
state that limitation plainly in the uncertainty field instead of filling \
the gap.
8. EXPLAIN WHY THE SYSTEM RAISED THIS. Name the deterministic factors that \
contributed, and state the prediction together with its time horizon.

Answer with the required structured fields and nothing else."""


@dataclass(frozen=True)
class Prompt:
    """A system prompt and a user message, ready for any provider."""

    system: str
    user: str

    def to_messages(self) -> List[Dict[str, Any]]:
        return [{"role": "user", "content": self.user}]


def build_prompt(evidence: IncidentEvidence) -> Prompt:
    """The full grounded prompt for one incident."""
    return Prompt(system=SYSTEM_PROMPT, user=user_message(evidence))


def user_message(evidence: IncidentEvidence) -> str:
    sections = [
        "Explain the following incident. It was raised by Sentinel's "
        "deterministic risk engine; you are interpreting it, not deciding it.",
        "",
        evidence_block(evidence),
        "",
        _units_reminder(evidence),
        "",
        _task_block(evidence),
    ]
    return "\n".join(sections)


def evidence_block(evidence: IncidentEvidence) -> str:
    """Evidence rendered twice: readable prose, then the exact JSON.

    The prose makes the units impossible to miss; the JSON makes the values
    exact. Neither adds anything the evidence does not contain.
    """
    lines: List[str] = ["=== INCIDENT EVIDENCE (the only facts you have) ==="]
    lines.append(f"incident id: {evidence.incident_id}")
    lines.append(f"timestamp: {evidence.timestamp:.2f} s into the clip")
    lines.append(
        f"coordinate space: {evidence.coordinate_space} "
        f"(distances in {evidence.distance_unit}, speeds in {evidence.speed_unit})"
    )
    lines.append(
        f"incident type: {evidence.incident_type}; "
        f"severity band: {evidence.severity}; "
        f"risk score: {evidence.risk_score:.1f}/100 "
        "(ordinal engineering signal, NOT a probability)"
    )
    lines.append(
        f"lifecycle state: {evidence.incident_state} "
        f"({describe_state(evidence.incident_state)})"
    )

    lines.append("")
    lines.append("entities involved (this is the complete list):")
    for entity in evidence.entities:
        label = entity.class_name or "unclassified"
        parts = [f"  - {entity.entity_id} ({label})"]
        if entity.position is not None:
            coords = ", ".join(f"{v:.2f}" for v in entity.position)
            parts.append(f"position ({coords}) {evidence.distance_unit}")
        if entity.speed is not None:
            parts.append(f"speed {entity.speed:.2f} {evidence.speed_unit}")
        if entity.zones:
            parts.append(f"in zone(s): {', '.join(entity.zones)}")
        lines.append("; ".join(parts))

    lines.append("")
    lines.append("measurements:")
    quantities = evidence.quantities()
    if quantities:
        for quantity in quantities:
            lines.append(f"  - {quantity.describe()}")
    else:
        lines.append("  - none recorded")

    lines.append("")
    lines.append("deterministic risk factors:")
    for factor in evidence.factors:
        lines.append(
            f"  - {factor.name}: score {factor.score:.2f} × weight "
            f"{factor.weight:.0f} = {factor.contribution:.1f} points"
            f" — {factor.rationale}"
        )

    lines.append("")
    lines.append(_prediction_lines(evidence))
    lines.append("")
    lines.append("=== EVIDENCE AS JSON (authoritative values) ===")
    lines.append(evidence.to_json())
    return "\n".join(lines)


def _prediction_lines(evidence: IncidentEvidence) -> str:
    lines = ["prediction:"]
    prediction = evidence.prediction
    if prediction is None:
        lines.append("  - no prediction was produced for this incident")
    else:
        lines.append(f"  - outcome: {prediction.outcome}")
        if prediction.horizon_seconds is not None:
            lines.append(
                f"  - prediction horizon: {prediction.horizon_seconds:.1f} s"
            )
        if prediction.minimum_separation is not None:
            lines.append(
                "  - predicted minimum separation: "
                f"{prediction.minimum_separation:.2f} {evidence.distance_unit}"
            )
        if prediction.seconds_to_minimum_separation is not None:
            lines.append(
                "  - seconds to that minimum: "
                f"{prediction.seconds_to_minimum_separation:.2f} s"
            )
        if prediction.unavailable_reason:
            lines.append(
                f"  - prediction unavailable because: {prediction.unavailable_reason}"
            )
        lines.append(
            "  - this describes the FUTURE"
            if prediction.is_forward_looking
            else "  - this does NOT describe a predicted future conflict"
        )

    time_to_risk = evidence.time_to_risk
    if time_to_risk is not None:
        if time_to_risk.is_already_unsafe:
            lines.append(
                "  - time to risk: the unsafe separation threshold is crossed "
                "NOW (this is a present condition, not a prediction)"
            )
        elif time_to_risk.is_predicted and time_to_risk.seconds is not None:
            lines.append(
                "  - time to risk: predicted to cross the unsafe threshold in "
                f"{time_to_risk.seconds:.2f} s (a prediction, not an event)"
            )
        else:
            reason = time_to_risk.reason or "not predicted within the horizon"
            lines.append(f"  - time to risk: not predicted ({reason})")
    return "\n".join(lines)


def _units_reminder(evidence: IncidentEvidence) -> str:
    if evidence.coordinate_space == GROUND_PLANE_METERS:
        return (
            "UNITS: this incident is calibrated to the ground plane. Distances "
            "are in metres and speeds in metres per second. Do not describe "
            "them as pixels."
        )
    return (
        "UNITS: this incident is in IMAGE PIXEL space. No camera calibration "
        "was applied, so no measurement here is a physical distance. A "
        "separation of 40 means 40 pixels in the image — it is not 40 metres, "
        "not 40 feet, and cannot be converted to either. Say 'pixels'."
    )


def _task_block(evidence: IncidentEvidence) -> str:
    lines = [
        "Write the explanation now:",
        "  - summary: what the system detected, in one or two sentences.",
        "  - severity_explanation: why this severity band, from the factors above.",
        "  - evidence_points: the supplied measurements, each with its unit.",
        "  - predicted_outcome: the prediction and its time horizon, worded as "
        "a prediction.",
        "  - recommended_action: what a human operator should consider doing. "
        "A recommendation only.",
        f"  - urgency: exactly one of {', '.join(URGENCIES)}.",
        "  - uncertainty: what this evidence does not establish.",
        f"  - coordinate_space: exactly {evidence.coordinate_space!r}.",
    ]
    if evidence.insufficient_history:
        lines.append(
            "  NOTE: the deterministic layer declined to predict because it had "
            "insufficient history. Say so; do not supply a prediction of your own."
        )
    return "\n".join(lines)
