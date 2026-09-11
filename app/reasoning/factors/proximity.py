"""Proximity and closing-speed factors.

Both answer questions about the *gap* between two entities: how big it is, and
how fast it is shrinking. Kept in one module because they share the same
measurement — the image-plane separation — and differ only in whether they look
at its value or its derivative.

All distances are image pixels; all speeds are pixels per second.
"""

from __future__ import annotations

from ...spatial import pixel_distance
from ..models.risk import FactorScore
from .base import RiskContext, RiskFactor, linear_falloff


class ProximityFactor(RiskFactor):
    """How close two entities are, right now.

    Scoring
    -------
    Linear from 0.0 at ``interaction_radius_px`` to 1.0 at
    ``critical_radius_px``::

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

        config = context.config
        separation = pixel_distance(primary.position_px, secondary.position_px)
        score = linear_falloff(
            separation,
            full_at=config.critical_radius_px,
            zero_at=config.interaction_radius_px,
        )

        if score <= 0.0:
            rationale = (
                f"separation {separation:.0f}px is beyond the "
                f"{config.interaction_radius_px:.0f}px interaction radius"
            )
        else:
            rationale = (
                f"{primary.entity_id} and {secondary.entity_id} are "
                f"{separation:.0f}px apart in image space "
                f"(critical below {config.critical_radius_px:.0f}px)"
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
            separation_px=round(separation, 2),
            interaction_radius_px=config.interaction_radius_px,
            critical_radius_px=config.critical_radius_px,
        )


class ClosingSpeedFactor(RiskFactor):
    """How fast the gap between two entities is shrinking.

    Scoring
    -------
    ``closing_speed`` is the radial component of relative image velocity —
    positive when the gap is closing, negative when it is opening. It is taken
    from the same closest-approach solution the trajectory factor uses, so the
    two can never disagree about which way things are moving.

    ::

        score = clamp01((closing_speed - floor) / (reference - floor))

    Diverging or steady pairs score 0. This is the factor that distinguishes
    Scenario A (moving apart, harmless) from Scenario C (rushing together)
    when the current separation happens to be identical.
    """

    name = "closing_speed"

    def evaluate(self, context: RiskContext) -> FactorScore:
        approach = context.approach()
        if approach is None:
            return self._inactive(
                context, "no pair to measure: closing speed needs two observed entities"
            )

        primary = context.primary_motion
        secondary = context.secondary_motion
        if not approach.is_estimable:
            return self._inactive(
                context,
                "insufficient history to estimate image velocity; "
                "closing speed not computed",
                separation_px=round(approach.current_separation_px, 2),
                insufficient_history=True,
            )

        config = context.config
        closing = approach.closing_speed_px_per_s
        score = linear_falloff(
            closing,
            full_at=config.closing_speed_reference_px_per_s,
            zero_at=config.closing_speed_floor_px_per_s,
        )

        if closing <= config.closing_speed_floor_px_per_s:
            direction = "separating" if closing < 0 else "holding station"
            rationale = (
                f"gap is {direction} "
                f"({closing:+.0f}px/s image-space closing speed)"
            )
        else:
            rationale = (
                f"separation is shrinking at {closing:.0f}px/s in image space "
                f"(saturates at {config.closing_speed_reference_px_per_s:.0f}px/s)"
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
            closing_speed_px_per_s=round(closing, 2),
            separation_px=round(approach.current_separation_px, 2),
            reference_px_per_s=config.closing_speed_reference_px_per_s,
            is_converging=approach.is_converging,
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
