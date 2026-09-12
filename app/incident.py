"""Incident intelligence demo (M0.7).

Runs the deterministic pipeline on a synthetic scenario, prints exactly what
the risk engine decided, and then — clearly fenced off — prints what the AI
layer says about it::

    python -m app.incident --scenario D --reasoner mock
    python -m app.incident --scenario D --reasoner anthropic

The fence is the point. Everything above the ``DETERMINISTIC SENTINEL SIGNAL``
banner is measured and computed; everything below the ``AI INCIDENT
INTERPRETATION`` banner is interpretation of those measurements and has no
authority over them. The grounding report at the end says whether the
interpretation stayed inside the evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Sequence

from .intelligence import (
    GroundingReport,
    IncidentEvidence,
    IncidentExplanation,
    build_reasoner,
    check_grounding,
    describe_state,
    evidence_from_assessment,
)
from .intelligence.reasoner import ReasonerError
from .intelligence.settings import ReasonerSettings
from .reasoning import RiskEngine
from .reasoning.models.risk import RiskAssessment
from .scenarios import ALL_WORLD_SCENARIOS, load_world

WIDTH = 72

DETERMINISTIC_BANNER = "DETERMINISTIC SENTINEL SIGNAL"
AI_BANNER = "AI INCIDENT INTERPRETATION"


# ---------------------------------------------------------------------------
def build_evidence(
    scenario_name: str,
    at: Optional[float] = None,
    calibrated: bool = True,
    limit: int = 1,
) -> List[IncidentEvidence]:
    """Run the deterministic pipeline and package its output as evidence."""
    scenario = load_world(scenario_name)
    calibration = scenario.calibration if calibrated else None
    engine = RiskEngine(scenario.config, calibration=calibration)
    report = engine.assess(scenario.memory, at=scenario.at if at is None else at)
    assessments: Sequence[RiskAssessment] = report.assessments[:limit]
    return [
        evidence_from_assessment(a, events=scenario.memory.events)
        for a in assessments
    ]


# ---------------------------------------------------------------------------
def banner(title: str, subtitle: str = "") -> List[str]:
    lines = ["=" * WIDTH, f"  {title}"]
    if subtitle:
        lines.append(f"  {subtitle}")
    lines.append("=" * WIDTH)
    return lines


def format_evidence(evidence: IncidentEvidence) -> List[str]:
    """The deterministic half: measured, computed, authoritative."""
    lines = banner(
        DETERMINISTIC_BANNER,
        "measured and computed by the pipeline — this is the source of truth",
    )
    lines.append(f"incident      : {evidence.incident_id}")
    lines.append(f"type          : {evidence.incident_type}")
    lines.append(
        f"lifecycle     : {evidence.incident_state} "
        f"({describe_state(evidence.incident_state)})"
    )
    lines.append(
        f"risk score    : {evidence.risk_score:.1f}/100  "
        f"severity {evidence.severity}   "
        "(ordinal engineering signal, NOT a probability)"
    )
    lines.append(
        f"space         : {evidence.coordinate_space} "
        f"(distances in {evidence.distance_unit}, speeds in {evidence.speed_unit})"
    )
    if evidence.space_fallback_reason:
        lines.append(f"space fallback: {evidence.space_fallback_reason}")
    lines.append(f"timestamp     : {evidence.timestamp:.2f} s")

    lines.append("")
    lines.append("entities:")
    for entity in evidence.entities:
        detail = f"  {entity.entity_id:<12} {entity.class_name or 'unclassified':<12}"
        if entity.position is not None:
            detail += (
                " at ("
                + ", ".join(f"{v:.2f}" for v in entity.position)
                + f") {evidence.distance_unit}"
            )
        if entity.speed is not None:
            detail += f"  speed {entity.speed:.2f} {evidence.speed_unit}"
        if entity.zones:
            detail += f"  zones: {', '.join(entity.zones)}"
        lines.append(detail)

    lines.append("")
    lines.append("measurements:")
    quantities = evidence.quantities()
    if quantities:
        for quantity in quantities:
            lines.append(f"  {quantity.describe()}")
    else:
        lines.append("  none recorded")

    lines.append("")
    lines.append("risk factors:")
    for factor in evidence.factors:
        lines.append(
            f"  {factor.name:<14} {factor.score:>5.2f} x {factor.weight:>4.0f}"
            f" = {factor.contribution:>6.2f} pts   {factor.rationale}"
        )

    lines.append("")
    lines.append("prediction:")
    prediction = evidence.prediction
    if prediction is None:
        lines.append("  none produced")
    else:
        lines.append(f"  outcome     : {prediction.outcome}")
        if prediction.horizon_seconds is not None:
            lines.append(f"  horizon     : {prediction.horizon_seconds:.1f} s")
        if prediction.minimum_separation is not None:
            lines.append(
                f"  min. sep.   : {prediction.minimum_separation:.2f} "
                f"{evidence.distance_unit}"
            )
        if prediction.unavailable_reason:
            lines.append(f"  unavailable : {prediction.unavailable_reason}")
    time_to_risk = evidence.time_to_risk
    if time_to_risk is not None:
        detail = time_to_risk.status
        if time_to_risk.seconds is not None:
            detail += f" ({time_to_risk.seconds:.2f} s)"
        if time_to_risk.reason:
            detail += f" — {time_to_risk.reason}"
        lines.append(f"  time to risk: {detail}")
    lines.append(f"  recommended : {evidence.recommended_intervention or 'none'}")
    return lines


def format_explanation(
    explanation: IncidentExplanation, report: GroundingReport
) -> List[str]:
    """The AI half: interpretation only, with its grounding verdict attached."""
    kind = (
        "language model" if explanation.is_language_model else "deterministic template"
    )
    lines = banner(
        AI_BANNER,
        f"explains the evidence above — it does not decide, score or override "
        f"it\n  provider: {explanation.provider} / {explanation.model} ({kind})",
    )
    lines.append(f"summary   : {explanation.summary}")
    lines.append("")
    lines.append(f"severity  : {explanation.severity_explanation}")
    lines.append("")
    lines.append("evidence  :")
    for point in explanation.evidence_points:
        lines.append(f"  - {point}")
    lines.append("")
    lines.append(f"predicted : {explanation.predicted_outcome}")
    lines.append("")
    lines.append(f"recommend : {explanation.recommended_action}")
    lines.append(f"urgency   : {explanation.urgency}")
    lines.append("")
    lines.append(f"uncertain : {explanation.uncertainty}")

    lines.append("")
    lines.append("-" * WIDTH)
    lines.append("  GROUNDING CHECK (every claim verified against the evidence)")
    lines.append("-" * WIDTH)
    if report.ok:
        lines.append("  PASS — " + report.summary())
    else:
        lines.append("  FAIL — " + report.summary())
        for violation in report.violations:
            lines.append(f"  ! {violation}")
    return lines


def explain(
    evidence: IncidentEvidence, reasoner: Any
) -> "tuple[IncidentExplanation, GroundingReport]":
    explanation = reasoner.reason(evidence)
    return explanation, check_grounding(explanation, evidence)


def as_payload(
    evidence: IncidentEvidence,
    explanation: IncidentExplanation,
    report: GroundingReport,
) -> Dict[str, Any]:
    return {
        "deterministic_evidence": evidence.to_dict(),
        "ai_explanation": explanation.to_dict(),
        "grounding": report.to_dict(),
    }


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    settings = ReasonerSettings.from_env()
    parser = argparse.ArgumentParser(
        prog="python -m app.incident",
        description=(
            "Explain a deterministic Sentinel incident with an AI reasoner. "
            "The risk engine decides; the reasoner only describes."
        ),
    )
    parser.add_argument(
        "--scenario",
        default="D",
        choices=sorted(ALL_WORLD_SCENARIOS),
        help="Which synthetic world scenario to assess (default: D)",
    )
    parser.add_argument(
        "--reasoner",
        default=settings.provider,
        help="Provider name: mock (default, no network) or anthropic",
    )
    parser.add_argument("--model", default=None, help="Override the provider's model")
    parser.add_argument(
        "--at", type=float, default=None, help="Assess at this timestamp instead"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="How many assessments to explain, highest risk first (default: 1)",
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Ignore the scenario calibration and reason in image pixels",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any explanation fails its grounding check",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    kwargs: Dict[str, Any] = {}
    if args.model:
        kwargs["model"] = args.model
    try:
        reasoner = build_reasoner(args.reasoner, **kwargs)
    except ReasonerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    evidences = build_evidence(
        args.scenario,
        at=args.at,
        calibrated=not args.no_calibration,
        limit=max(1, args.limit),
    )
    if not evidences:
        print("no assessment was produced for this scenario", file=sys.stderr)
        return 1

    payloads: List[Dict[str, Any]] = []
    blocks: List[str] = []
    failed = False
    for evidence in evidences:
        explanation, report = explain(evidence, reasoner)
        failed = failed or not report.ok
        if args.json:
            payloads.append(as_payload(evidence, explanation, report))
        else:
            blocks.append(
                "\n".join(format_evidence(evidence) + [""] + format_explanation(explanation, report))
            )

    if args.json:
        print(json.dumps(payloads, indent=2))
    else:
        print("\n\n".join(blocks))

    return 3 if (failed and args.strict) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
