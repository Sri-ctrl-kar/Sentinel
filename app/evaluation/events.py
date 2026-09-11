"""Event recognition metrics.

Answers one narrow question: of the events a human annotated, how many did the
pipeline emit, and how many did it emit that the human did not?

Kept strictly separate from risk evaluation. An event is an observation
("this entity entered this zone at t=3.2"); a risk assessment is a judgement
about a situation. Mixing their scores would hide which layer is at fault when
the number moves.

Matching rules
--------------
A predicted event matches a ground-truth event when all of:

* the entities correspond under the identity mapping from tracking,
* the actions are identical,
* the zone matches, for zone events,
* their timestamps differ by no more than ``tolerance_seconds``.

Within those constraints the pairing is an **optimal assignment** on timing
error, not greedy, so a cluster of nearby events cannot be scored differently
depending on which one happens to be considered first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..events.schema import (
    ACTION_APPEARED,
    ACTION_DISAPPEARED,
    ACTION_ENTERED_ZONE,
    ACTION_EXITED_ZONE,
    ACTION_MOVED,
    ACTION_STATIONARY,
    Event,
)
from .assignment import solve_max_score

#: Actions the benchmark scores. ``detected`` is excluded deliberately: it is a
#: heartbeat whose rate is a configuration choice, not a recognition claim.
EVALUATED_ACTIONS: Tuple[str, ...] = (
    ACTION_APPEARED,
    ACTION_DISAPPEARED,
    ACTION_MOVED,
    ACTION_STATIONARY,
    ACTION_ENTERED_ZONE,
    ACTION_EXITED_ZONE,
)

#: How far apart a predicted and annotated event may be and still count as the
#: same event. Generous because a human annotating a 20-second clip cannot
#: place a timestamp to the frame.
DEFAULT_TOLERANCE_SECONDS = 0.5


@dataclass(frozen=True)
class AnnotatedEvent:
    """One event a human (or a renderer) says happened."""

    entity_id: str
    action: str
    timestamp: float
    zone: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "entity_id": self.entity_id,
            "action": self.action,
            "timestamp": round(self.timestamp, 3),
        }
        if self.zone:
            payload["zone"] = self.zone
        return payload


@dataclass
class ActionScore:
    """Precision/recall/F1 for a single action."""

    action: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    timing_errors: List[float] = field(default_factory=list)

    @property
    def precision(self) -> Optional[float]:
        predicted = self.true_positives + self.false_positives
        return self.true_positives / predicted if predicted else None

    @property
    def recall(self) -> Optional[float]:
        actual = self.true_positives + self.false_negatives
        return self.true_positives / actual if actual else None

    @property
    def f1(self) -> Optional[float]:
        precision, recall = self.precision, self.recall
        if precision is None or recall is None or precision + recall == 0:
            return None
        return 2 * precision * recall / (precision + recall)

    @property
    def mean_timing_error(self) -> Optional[float]:
        if not self.timing_errors:
            return None
        return sum(self.timing_errors) / len(self.timing_errors)

    @property
    def is_scored(self) -> bool:
        """False when neither annotated nor predicted — nothing to report."""
        return (
            self.true_positives + self.false_positives + self.false_negatives
        ) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": _round(self.precision),
            "recall": _round(self.recall),
            "f1": _round(self.f1),
            "mean_timing_error_seconds": _round(self.mean_timing_error),
        }


@dataclass
class EventMetrics:
    """Per-action scores plus a micro-averaged total."""

    scores: Dict[str, ActionScore] = field(default_factory=dict)
    tolerance_seconds: float = DEFAULT_TOLERANCE_SECONDS
    #: Actions the pipeline emitted that the annotation does not claim to
    #: cover. Reported as unscored rather than counted as false positives.
    unscored_actions: Dict[str, int] = field(default_factory=dict)

    @property
    def scored_actions(self) -> List[ActionScore]:
        return [s for s in self.scores.values() if s.is_scored]

    @property
    def totals(self) -> ActionScore:
        """Micro-average: counts pooled across actions, then scored once.

        Micro rather than macro because a macro average would let a single
        rare action with two instances outweigh a common one with two hundred.
        """
        total = ActionScore(action="ALL")
        for score in self.scores.values():
            total.true_positives += score.true_positives
            total.false_positives += score.false_positives
            total.false_negatives += score.false_negatives
            total.timing_errors.extend(score.timing_errors)
        return total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tolerance_seconds": self.tolerance_seconds,
            "per_action": {
                action: score.to_dict()
                for action, score in self.scores.items()
                if score.is_scored
            },
            "micro_average": self.totals.to_dict(),
            "unscored_actions": dict(self.unscored_actions),
            "note": (
                "matching is optimal assignment on timing error within the "
                "tolerance; only actions the annotation claims to cover "
                "exhaustively are scored, and 'detected' heartbeats never are"
            ),
        }


def evaluate_events(
    annotated: Sequence[AnnotatedEvent],
    predicted: Sequence[Event],
    id_mapping: Optional[Dict[str, str]] = None,
    tolerance_seconds: float = DEFAULT_TOLERANCE_SECONDS,
    actions: Sequence[str] = EVALUATED_ACTIONS,
) -> EventMetrics:
    """Score predicted events against annotations.

    ``id_mapping`` maps annotated entity IDs to predicted track IDs — normally
    the mapping produced by :func:`~app.evaluation.tracking.mot_metrics`, so
    tracking and event evaluation agree on who is who. Annotated entities with
    no mapping can never match, which is correct: an entity the tracker never
    found cannot have had its events recognised.
    """
    mapping = dict(id_mapping or {})
    metrics = EventMetrics(tolerance_seconds=tolerance_seconds)
    scored = [a for a in actions if a in set(actions)]

    for action in scored:
        truth = [e for e in annotated if e.action == action]
        guesses = [e for e in predicted if e.action == action]
        metrics.scores[action] = _score_action(
            action, truth, guesses, mapping, tolerance_seconds
        )

    # Anything emitted outside the annotated action set is reported, not judged.
    for event in predicted:
        if event.action in scored or event.action == "detected":
            continue
        metrics.unscored_actions[event.action] = (
            metrics.unscored_actions.get(event.action, 0) + 1
        )
    return metrics


def _score_action(
    action: str,
    truth: Sequence[AnnotatedEvent],
    guesses: Sequence[Event],
    mapping: Dict[str, str],
    tolerance: float,
) -> ActionScore:
    score = ActionScore(action=action)
    if not truth and not guesses:
        return score

    affinity: List[List[float]] = []
    for annotation in truth:
        row = []
        expected_id = mapping.get(annotation.entity_id)
        for guess in guesses:
            row.append(
                _affinity(annotation, guess, expected_id, tolerance)
            )
        affinity.append(row)

    pairs = (
        solve_max_score(affinity, minimum_score=0.0) if affinity and guesses else []
    )
    matched_truth = {t for t, _ in pairs}
    matched_guesses = {g for _, g in pairs}

    for truth_index, guess_index in pairs:
        score.true_positives += 1
        score.timing_errors.append(
            abs(truth[truth_index].timestamp - guesses[guess_index].timestamp)
        )
    score.false_negatives = len(truth) - len(matched_truth)
    score.false_positives = len(guesses) - len(matched_guesses)
    return score


def _affinity(
    annotation: AnnotatedEvent,
    guess: Event,
    expected_id: Optional[str],
    tolerance: float,
) -> float:
    """Match quality in ``(0, 1]``, or 0 when the pair is inadmissible."""
    if expected_id is None or guess.entity_id != expected_id:
        return 0.0
    if annotation.zone is not None:
        if guess.attributes.get("zone") != annotation.zone:
            return 0.0
    error = abs(annotation.timestamp - guess.timestamp)
    if error > tolerance:
        return 0.0
    # Closer in time is a better match; strictly positive so an exact-tolerance
    # match still counts.
    return 1.0 - (error / tolerance) * 0.999


def _round(value: Optional[float], digits: int = 4) -> Optional[float]:
    return round(value, digits) if value is not None else None
