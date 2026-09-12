"""Deterministic incident lifecycle.

Severity says *how bad*; lifecycle says *where in its life* a situation is.
They are orthogonal on purpose — a critical incident that has already happened
and a critical incident four seconds out call for different words, and the
difference is not a matter of score.

The five states are derived here, from the risk engine's own prediction state,
and nowhere else. A language model may describe a state; it may never choose
one. That is the whole point of deriving them in a module that imports no
model runtime.

State machine
-------------
=============  ==========================================================
``observed``   Seen and scored, nothing developing.
``developing`` A future unsafe approach is predicted, beyond the imminent
               horizon.
``imminent``   Predicted to become unsafe within
               :data:`IMMINENT_HORIZON_SECONDS`.
``current``    The unsafe condition exists *now* — not a prediction.
``resolved``   Was active, and is no longer.
=============  ==========================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from ..reasoning.models.risk import RiskAssessment, TimeToRisk
from ..reasoning.prediction import (
    CURRENTLY_UNSAFE_PROXIMITY,
    PREDICTED_TRAJECTORY_CONFLICT,
    PREDICTED_UNSAFE_PROXIMITY,
)

STATE_OBSERVED = "observed"
STATE_DEVELOPING = "developing"
STATE_IMMINENT = "imminent"
STATE_CURRENT = "current"
STATE_RESOLVED = "resolved"

#: Ordered from quietest to most urgent. ``resolved`` is deliberately outside
#: the ordering: it is an exit, not a rung.
INCIDENT_STATES = (
    STATE_OBSERVED,
    STATE_DEVELOPING,
    STATE_IMMINENT,
    STATE_CURRENT,
    STATE_RESOLVED,
)

#: States in which something is actually happening or expected to.
ACTIVE_STATES = (STATE_DEVELOPING, STATE_IMMINENT, STATE_CURRENT)

#: A predicted crossing this close is called imminent rather than developing.
#: Engineering-selected: two seconds is roughly the shortest interval in which
#: a human operator can be told something and react to it. It is not tuned
#: against any benchmark.
IMMINENT_HORIZON_SECONDS = 2.0

#: Below this score a situation with no prediction and no current breach is
#: considered quiet enough to resolve. Matches the ``low`` severity floor, so
#: lifecycle and severity never disagree about whether anything is happening.
QUIET_SCORE = 20.0

#: What each state means, for display. Kept next to the derivation so the two
#: cannot drift apart.
STATE_DESCRIPTIONS = {
    STATE_OBSERVED: "seen and scored; nothing developing",
    STATE_DEVELOPING: "a future unsafe approach is predicted",
    STATE_IMMINENT: (
        f"predicted to become unsafe within {IMMINENT_HORIZON_SECONDS:.0f}s"
    ),
    STATE_CURRENT: "the unsafe condition exists now",
    STATE_RESOLVED: "was active; no longer",
}

#: Prediction outcomes that describe the future rather than the present.
FORWARD_LOOKING_OUTCOMES = (
    PREDICTED_UNSAFE_PROXIMITY,
    PREDICTED_TRAJECTORY_CONFLICT,
)


def derive_incident_state(
    assessment: RiskAssessment, previous: Optional[str] = None
) -> str:
    """The lifecycle state of ``assessment``, from deterministic signals only.

    ``previous`` is the state this same situation was last in. It is needed
    only to tell ``resolved`` (it was active, now it is not) from ``observed``
    (it was never active). Everything else is a function of the assessment.
    """
    time_to_risk = assessment.time_to_risk

    if _is_unsafe_now(assessment, time_to_risk):
        return STATE_CURRENT

    seconds = _predicted_seconds(assessment, time_to_risk)
    if seconds is not None:
        if seconds <= IMMINENT_HORIZON_SECONDS:
            return STATE_IMMINENT
        return STATE_DEVELOPING

    if assessment.prediction_outcome in FORWARD_LOOKING_OUTCOMES:
        # Predicted to become unsafe, but without a crossing time — the paths
        # conflict without the closed-form solution reporting when.
        return STATE_DEVELOPING

    if assessment.risk_score >= QUIET_SCORE:
        return STATE_OBSERVED

    if previous in ACTIVE_STATES:
        return STATE_RESOLVED
    return STATE_OBSERVED


def _is_unsafe_now(
    assessment: RiskAssessment, time_to_risk: Optional[TimeToRisk]
) -> bool:
    if time_to_risk is not None:
        return time_to_risk.status == TimeToRisk.STATUS_ALREADY_UNSAFE
    return assessment.prediction_outcome == CURRENTLY_UNSAFE_PROXIMITY


def _predicted_seconds(
    assessment: RiskAssessment, time_to_risk: Optional[TimeToRisk]
) -> Optional[float]:
    """Seconds until the unsafe threshold, only when genuinely predicted."""
    if time_to_risk is not None and time_to_risk.is_predicted:
        return time_to_risk.seconds
    return None


def describe_state(state: str) -> str:
    return STATE_DESCRIPTIONS.get(state, state)


def is_active(state: str) -> bool:
    return state in ACTIVE_STATES


def situation_key(assessment: RiskAssessment) -> Tuple[str, ...]:
    """Stable identity for "the same situation" across time steps."""
    return tuple(sorted(assessment.involved_entity_ids))


@dataclass
class LifecycleTracker:
    """Remembers each situation's last state so ``resolved`` can be detected.

    Deterministic and replayable: feeding the same sequence of assessments
    always produces the same sequence of states.
    """

    states: Dict[Tuple[str, ...], str] = field(default_factory=dict)

    def update(self, assessment: RiskAssessment) -> str:
        key = situation_key(assessment)
        state = derive_incident_state(assessment, previous=self.states.get(key))
        self.states[key] = state
        return state

    def update_all(self, assessments: Iterable[RiskAssessment]) -> List[str]:
        return [self.update(a) for a in assessments]

    def state_of(self, assessment: RiskAssessment) -> Optional[str]:
        return self.states.get(situation_key(assessment))

    def reset(self) -> None:
        self.states.clear()
