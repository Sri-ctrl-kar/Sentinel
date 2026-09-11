"""M0.6: correctness of the evaluation metrics themselves.

A benchmark whose metrics are wrong is worse than no benchmark, so these test
the measuring instruments against hand-computable cases before any of them is
pointed at the pipeline.
"""

import itertools
import random

import pytest

from app.evaluation.assignment import solve_max_score, solve_min_cost
from app.evaluation.events import (
    AnnotatedEvent,
    EventMetrics,
    evaluate_events,
)
from app.evaluation.tracking import MOTMetrics, iou, mot_metrics
from app.events.schema import Event


# ===========================================================================
# Optimal assignment
# ===========================================================================
def brute_force_min(cost):
    rows, columns = len(cost), len(cost[0])
    k = min(rows, columns)
    return min(
        sum(cost[r][c] for r, c in zip(row_set, column_set))
        for row_set in itertools.permutations(range(rows), k)
        for column_set in itertools.permutations(range(columns), k)
    )


@pytest.mark.parametrize("seed", range(25))
def test_assignment_matches_brute_force(seed):
    """The metrics are only correct if the assignment underneath is optimal."""
    rng = random.Random(seed)
    rows, columns = rng.randint(1, 5), rng.randint(1, 5)
    cost = [[rng.randint(0, 20) for _ in range(columns)] for _ in range(rows)]

    pairs = solve_min_cost(cost)
    assert len(pairs) == min(rows, columns)
    assert sum(cost[r][c] for r, c in pairs) == brute_force_min(cost)


def test_assignment_beats_greedy_where_greedy_is_wrong():
    """The case that makes optimal matching necessary rather than pedantic."""
    cost = [[1.0, 2.0], [1.1, 90.0]]
    # Greedy takes (0,0)=1.0 then is forced into (1,1)=90.0 for a total of 91.0.
    assert sum(cost[r][c] for r, c in solve_min_cost(cost)) == pytest.approx(3.1)


def test_assignment_handles_rectangular_inputs_both_ways():
    assert len(solve_min_cost([[1, 2, 3]])) == 1
    assert len(solve_min_cost([[1], [2], [3]])) == 1


def test_assignment_of_empty_input():
    assert solve_min_cost([]) == []
    assert solve_min_cost([[]]) == []


def test_max_score_drops_pairs_at_or_below_the_floor():
    assert solve_max_score([[0.9, 0.0], [0.0, 0.0]], minimum_score=0.0) == [(0, 0)]


# ===========================================================================
# Tracking metrics
# ===========================================================================
def frames(count, boxes):
    return [dict(boxes) for _ in range(count)]


def test_iou_of_identical_boxes():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_of_disjoint_boxes():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_perfect_tracking_scores_one():
    gt = frames(5, {"a": (0, 0, 10, 10), "b": (50, 0, 60, 10)})
    pred = frames(5, {"1": (0, 0, 10, 10), "2": (50, 0, 60, 10)})
    metrics = mot_metrics(gt, pred)
    assert metrics.mota == pytest.approx(1.0)
    assert metrics.idf1 == pytest.approx(1.0)
    assert metrics.id_switches == 0


def test_one_id_switch_is_counted_and_costs_exactly_one_error():
    gt = frames(5, {"a": (0, 0, 10, 10), "b": (50, 0, 60, 10)})
    pred = frames(3, {"1": (0, 0, 10, 10), "2": (50, 0, 60, 10)}) + frames(
        2, {"9": (0, 0, 10, 10), "2": (50, 0, 60, 10)}
    )
    metrics = mot_metrics(gt, pred)
    assert metrics.id_switches == 1
    # 10 GT boxes, one error.
    assert metrics.mota == pytest.approx(0.9)
    # IDTP = 3 ("1"->a) + 5 ("2"->b) = 8; 2*8/(2*8+2+2) = 0.8
    assert metrics.idf1 == pytest.approx(0.8)


def test_missed_detections_are_counted():
    gt = frames(5, {"a": (0, 0, 10, 10), "b": (50, 0, 60, 10)})
    pred = frames(5, {"1": (0, 0, 10, 10)})
    metrics = mot_metrics(gt, pred)
    assert metrics.false_negatives == 5
    assert metrics.mota == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)


def test_mota_goes_negative_with_enough_false_positives():
    """MOTA is unbounded below; reporting it as a percentage would be wrong."""
    gt = frames(5, {"a": (0, 0, 10, 10)})
    spurious = {f"x{i}": (200 + i * 20, 0, 210 + i * 20, 10) for i in range(3)}
    pred = frames(5, {"1": (0, 0, 10, 10), **spurious})
    metrics = mot_metrics(gt, pred)
    assert metrics.mota < 0
    assert metrics.false_positives == 15


def test_boxes_below_the_iou_threshold_do_not_match():
    gt = frames(3, {"a": (0, 0, 10, 10)})
    pred = frames(3, {"1": (9, 0, 19, 10)})  # IoU ~ 0.05
    metrics = mot_metrics(gt, pred)
    assert metrics.matches == 0
    assert metrics.false_negatives == 3
    assert metrics.false_positives == 3


def test_identity_mapping_is_reported():
    gt = frames(4, {"worker": (0, 0, 10, 10)})
    pred = frames(4, {"person_1": (0, 0, 10, 10)})
    assert mot_metrics(gt, pred).id_mapping == {"worker": "person_1"}


def test_empty_sequences_report_none_rather_than_a_fake_score():
    metrics = mot_metrics([], [])
    assert metrics.mota is None
    assert metrics.idf1 is None
    assert metrics.to_dict()["mota"] is None


# ===========================================================================
# Event metrics
# ===========================================================================
def predicted(entity, action, time, **attributes):
    return Event(time, entity, action, dict(attributes))


def test_exact_event_match_scores_one():
    metrics = evaluate_events(
        [AnnotatedEvent("w", "appeared", 1.0)],
        [predicted("p1", "appeared", 1.0)],
        id_mapping={"w": "p1"},
        actions=("appeared",),
    )
    score = metrics.scores["appeared"]
    assert (score.true_positives, score.false_positives, score.false_negatives) == (1, 0, 0)
    assert score.f1 == pytest.approx(1.0)


def test_event_within_tolerance_matches():
    metrics = evaluate_events(
        [AnnotatedEvent("w", "appeared", 1.0)],
        [predicted("p1", "appeared", 1.4)],
        id_mapping={"w": "p1"},
        tolerance_seconds=0.5,
        actions=("appeared",),
    )
    assert metrics.scores["appeared"].true_positives == 1
    assert metrics.scores["appeared"].mean_timing_error == pytest.approx(0.4)


def test_event_outside_tolerance_is_a_miss_and_a_false_alarm():
    metrics = evaluate_events(
        [AnnotatedEvent("w", "appeared", 1.0)],
        [predicted("p1", "appeared", 3.0)],
        id_mapping={"w": "p1"},
        tolerance_seconds=0.5,
        actions=("appeared",),
    )
    score = metrics.scores["appeared"]
    assert (score.false_negatives, score.false_positives) == (1, 1)
    assert score.true_positives == 0


def test_events_of_an_unmatched_entity_cannot_score():
    """An entity the tracker never found cannot have had its events recognised."""
    metrics = evaluate_events(
        [AnnotatedEvent("w", "appeared", 1.0)],
        [predicted("someone_else", "appeared", 1.0)],
        id_mapping={},
        actions=("appeared",),
    )
    assert metrics.scores["appeared"].true_positives == 0
    assert metrics.scores["appeared"].false_negatives == 1


def test_zone_events_must_name_the_same_zone():
    metrics = evaluate_events(
        [AnnotatedEvent("w", "entered_zone", 1.0, zone="bay")],
        [predicted("p1", "entered_zone", 1.0, zone="aisle")],
        id_mapping={"w": "p1"},
        actions=("entered_zone",),
    )
    assert metrics.scores["entered_zone"].true_positives == 0


def test_clustered_events_are_matched_optimally_not_greedily():
    """Two annotations and two predictions must pair by best total timing."""
    metrics = evaluate_events(
        [AnnotatedEvent("w", "moved", 1.0), AnnotatedEvent("w", "moved", 1.4)],
        [predicted("p1", "moved", 1.05), predicted("p1", "moved", 1.45)],
        id_mapping={"w": "p1"},
        tolerance_seconds=0.5,
        actions=("moved",),
    )
    score = metrics.scores["moved"]
    assert score.true_positives == 2
    assert score.mean_timing_error == pytest.approx(0.05, abs=1e-6)


def test_unannotated_actions_are_reported_not_penalised():
    """The credibility rule: not annotated is not the same as wrong."""
    metrics = evaluate_events(
        [AnnotatedEvent("w", "appeared", 1.0)],
        [predicted("p1", "appeared", 1.0), predicted("p1", "moved", 2.0)],
        id_mapping={"w": "p1"},
        actions=("appeared",),
    )
    assert metrics.scores["appeared"].false_positives == 0
    assert metrics.unscored_actions == {"moved": 1}
    assert metrics.totals.precision == pytest.approx(1.0)


def test_heartbeat_events_are_never_scored_or_reported():
    metrics = evaluate_events(
        [AnnotatedEvent("w", "appeared", 1.0)],
        [predicted("p1", "appeared", 1.0), predicted("p1", "detected", 2.0)],
        id_mapping={"w": "p1"},
        actions=("appeared",),
    )
    assert "detected" not in metrics.unscored_actions


def test_micro_average_pools_counts_across_actions():
    metrics = EventMetrics()
    from app.evaluation.events import ActionScore

    metrics.scores["a"] = ActionScore("a", true_positives=3, false_positives=1)
    metrics.scores["b"] = ActionScore("b", true_positives=1, false_negatives=3)
    totals = metrics.totals
    assert totals.true_positives == 4
    assert totals.precision == pytest.approx(4 / 5)
    assert totals.recall == pytest.approx(4 / 7)


def test_precision_and_recall_are_none_when_undefined():
    from app.evaluation.events import ActionScore

    assert ActionScore("x").precision is None
    assert ActionScore("x").recall is None
    assert ActionScore("x").f1 is None
