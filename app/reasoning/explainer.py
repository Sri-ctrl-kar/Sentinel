"""Human-readable rendering of risk assessments.

Deliberately a plain formatter with no model behind it: at M0.3 the
explanation must be a faithful transcription of the arithmetic the engine
actually did, not a generated narrative. Every line printed here can be traced
to a field on the assessment.
"""

from __future__ import annotations

from typing import List, Optional

from ..spatial import IMAGE_PIXELS
from .models.risk import FactorScore, RiskAssessment, RiskReport


class Explainer:
    """Renders assessments as text."""

    def __init__(self, show_factor_table: bool = True, show_units_note: bool = True) -> None:
        self.show_factor_table = show_factor_table
        self.show_units_note = show_units_note

    # ------------------------------------------------------------------
    def explain(self, incident: RiskAssessment) -> str:
        """Render one assessment in the Sentinel incident-analysis format."""
        lines: List[str] = []
        lines.append("SENTINEL INCIDENT ANALYSIS")
        lines.append("=" * 26)
        lines.append("")
        lines.append(f"Risk: {incident.risk_score:.0f}/100")
        lines.append(f"Severity: {incident.severity.upper()}")
        lines.append(f"Incident: {incident.incident_type}")
        lines.append(f"Confidence: {incident.confidence:.2f}")
        lines.append(f"At: t={incident.timestamp:.2f}s")
        lines.append("")

        lines.append("Entities:")
        for entity_id in incident.involved_entity_ids:
            lines.append(f"  {entity_id}")
        lines.append("")

        lines.append("Evidence:")
        for factor in incident.active_factors:
            lines.append(f"  - {factor.rationale}")
        if not incident.active_factors:
            lines.append("  - (no factor contributed)")
        lines.append("")

        if incident.predicted_time_to_incident_seconds is not None:
            lines.append(
                f"Predicted time-to-risk: "
                f"{incident.predicted_time_to_incident_seconds:.1f} seconds"
            )
        else:
            lines.append(
                "Predicted time-to-risk: not predictable from current evidence"
            )
        lines.append("")

        lines.append("Recommended intervention:")
        lines.append(f"  {incident.recommended_intervention or '(none)'}")

        if self.show_factor_table:
            lines.append("")
            lines.extend(self._factor_table(incident))

        if self.show_units_note:
            lines.append("")
            lines.extend(self._units_note(incident))

        return "\n".join(lines)

    # ------------------------------------------------------------------
    def explain_report(self, report: RiskReport, limit: Optional[int] = None) -> str:
        """Render a whole report, highest risk first."""
        if not report.assessments:
            return (
                "SENTINEL INCIDENT ANALYSIS\n"
                "==========================\n\n"
                f"At: t={report.timestamp:.2f}s\n"
                "Risk: 0/100\n"
                "Severity: NORMAL\n\n"
                "No developing situations detected."
            )

        assessments = report.assessments[:limit] if limit else report.assessments
        blocks = [self.explain(a) for a in assessments]
        separator = "\n\n" + "-" * 60 + "\n\n"
        return separator.join(blocks)

    # ------------------------------------------------------------------
    def _factor_table(self, incident: RiskAssessment) -> List[str]:
        """Show the arithmetic, so the score can be checked by hand."""
        lines = ["Scoring breakdown (score x weight = points):"]
        for factor in incident.contributing_factors:
            if factor.name == "escalation":
                continue
            lines.append(
                f"  {factor.name:<15} {factor.score:>5.2f} x {factor.weight:>5.1f} "
                f"= {factor.contribution:>6.2f}  (confidence {factor.confidence:.2f})"
            )
        subtotal = incident.details.get("subtotal_points", 0.0)
        lines.append(f"  {'subtotal':<15} {'':>5}   {'':>5}   {subtotal:>6.2f}")
        if incident.escalation_multiplier != 1.0:
            escalation = incident.factor("escalation")
            signals = escalation.details.get("signal_count") if escalation else "?"
            lines.append(
                f"  {'escalation':<15} x{incident.escalation_multiplier:.2f} "
                f"({signals} corroborating signals)"
            )
        lines.append(f"  {'RISK SCORE':<15} {'':>5}   {'':>5}   {incident.risk_score:>6.2f}")
        return lines

    def _units_note(self, incident: RiskAssessment) -> List[str]:
        return [
            f"Coordinate space: {incident.coordinate_space}",
            "  All distances are IMAGE PIXELS and all speeds are PIXELS PER SECOND.",
            "  These are not physical distances or speeds; no camera calibration",
            "  or ground-plane homography is applied. Times are real seconds.",
        ]


def format_incident(incident: RiskAssessment) -> str:
    """Convenience wrapper around :meth:`Explainer.explain`."""
    return Explainer().explain(incident)
