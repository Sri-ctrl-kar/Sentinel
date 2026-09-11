"""Proximity and closing-speed factors.

Both answer questions about the *gap* between two entities: how big it is, and
how fast it is shrinking. Kept in one module because they share the same
measurement — the image-plane separation — and differ only in whether they look
at its value or its derivative.

Both work in whichever coordinate space the candidate is being scored in:
image pixels for an uncalibrated camera, ground-plane metres when a calibration
supplied world positions. The formula is identical; only the thresholds and the
unit labels differ, and those come from the active
:class:`~app.reasoning.config.SpatialThresholds`.
"""

from __future__ import annotations

from ..models.risk import FactorScore
from .base import RiskContext, RiskFactor, linear_falloff


class ProximityFactor(RiskFactor):
    """How close two entities are, right now.

    Scoring
    -------
    Linear from 0.0 at the interaction radius to 1.0 at the critical radius::

        score = clamp01((interaction_radius - separation) /
                        (interaction_radius - critical_radius))

    Linear rather than inverse-square on purpose: an inverse law would make the
    score explode at small separations and would be dominated by detector
    jitter exactly when the situation matters most. Linear is boring,
    predictable, and easy to reason about when reviewing an incident.

    Note the weight: proximity alone maxes out at 30 points, which is the
    "low" band. Two entities being close together is *not* an incident — it
    only becomes one in combination with motion or a zone violation.
    """

    name = "proximity"

    def evaluate(self, context: RiskContext) -> FactorScore:
        primary = context.primary_motion
        secondary = context.secondary_motion
        if primary is None or secondary is None:
            return self._inactive(
                context, "no pair to measure: proximity needs two observed entities"
            )
        geometry = context.geometry()
        if geometry is None:
            return self._inactive(
                context,
                f"separation is not measurable in {context.coordinate_space}",
            )

        thresholds = context.thresholds
        separation = geometry.separation
        score = linear_falloff(
            separation,
            full_at=thresholds.critical_radius,
            zero_at=thresholds.interaction_radius,
        )

        space_label = "on the ground plane" if context.is_world_space else "in image space"
        if score <= 0.0:
            rationale = (
                f"separation {thresholds.format_distance(separation)} is beyond the "
                f"{thresholds.format_distance(thresholds.interaction_radius)} "
                "interaction radius"
            )
        else:
            rationale = (
                f"{primary.entity_id} and {secondary.entity_id} are "
                f"{thresholds.format_distance(separation)} apart {space_label} "
                f"(critical below "
                f"{thresholds.format_distance(thresholds.critical_radius)})"
            )

        # Confidence here is about *observation*, not prediction: a current
        # separation needs both entities seen recently, nothing more.
        confidence = 1.0 if score > 0 else 0.0
        evidence = _recent_evidence(context, primary.entity_id, secondary.entity_id)

        return self._score(
            context,
            score,
            rationale,
            confidence=confidence,
            evidence_event_ids=evidence,
            **{
                thresholds.distance_key("separation"): round(separation, 3),
                thresholds.distance_key("interaction_radius"): thresholds.interaction_radius,
                thresholds.distance_key("critical_radius"): thresholds.critical_radius,
            },
        )


class ClosingSpeedFactor(RiskFactor):
    """How fast the gap between two entities is shrinking.

    Scoring
    -------
    ``closing_speed`` is the radial component of relative velocity — positive
    when the gap is closing, negative when it is opening. It is taken from the
    same closest-approach solution the trajectory factor uses, so the two can
    never disagree about which way things are moving.

    ::

        score = clamp01((closing_speed - floor) / (reference - floor))

    Diverging or steady pairs score 0. This is the factor that distinguishes
    Scenario A (moving apart, harmless) from Scenario C (rushing together)
    when the current separation happens to be identical.
    """

    name = "closing_speed"

    def evaluate(self, context: RiskContext) -> FactorScore:
        geometry = context.geometry()
        if geometry is None:
            return self._inactive(
                context, "no pair to measure: closing speed needs two observed entities"
            )

        thresholds = context.thresholds
        primary = context.primary_motion
        secondary = context.secondary_motion
        if not geometry.is_estimable:
            return self._inactive(
                context,
                "insufficient history to estimate velocity; "
                "closing speed not computed",
                insufficient_history=True,
                **{
                    thresholds.distance_key("separation"): round(
                        geometry.separation, 3
                    )
                },
            )

        closing = geometry.closing_speed
        score = linear_falloff(
            closing,
            full_at=thresholds.closing_speed_reference,
            zero_at=thresholds.closing_speed_floor,
        )

        space_label = "on the ground plane" if context.is_world_space else "in image space"
        if closing <= thresholds.closing_speed_floor:
            direction = "separating" if closing < 0 else "holding station"
            rationale = (
                f"gap is {direction} "
                f"({thresholds.format_speed(closing)} closing speed {space_label})"
            )
        else:
            rationale = (
                f"separation is shrinking at {thresholds.format_speed(closing)} "
                f"{space_label} (saturates at "
                f"{thresholds.format_speed(thresholds.closing_speed_reference)})"
            )

        confidence = min(
            primary.confidence if primary else 0.0,
            secondary.confidence if secondary else 0.0,
        )
        evidence = _recent_evidence(
            context, context.candidate.primary, context.candidate.secondary
        )

        return self._score(
            context,
            score,
            rationale,
            confidence=confidence,
            evidence_event_ids=evidence,
            is_converging=geometry.is_converging,
            **{
                thresholds.speed_key("closing_speed"): round(closing, 3),
                thresholds.distance_key("separation"): round(geometry.separation, 3),
                thresholds.speed_key("reference"): thresholds.closing_speed_reference,
            },
        )


def _recent_evidence(context: RiskContext, *entity_ids, limit: int = 3):
    """The most recent event IDs for each entity, as supporting evidence."""
    evidence = []
    for entity_id in entity_ids:
        if not entity_id:
            continue
        history = context.history(entity_id)
        evidence.extend(e.event_id for e in history[-limit:] if e.event_id)
    return evidence
