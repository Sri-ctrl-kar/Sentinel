"""Tests for the ByteTrack-style IoU tracker.

Persistent identity is the core requirement of M0.1, so these tests focus on
ID stability under motion, occlusion, and clutter.
"""

import pytest

from app.perception.tracker import available_trackers, create_tracker
from app.perception.trackers.byte_iou import ByteIoUTracker, iou
from app.perception.types import Detection

from helpers import make_detection


def run(tracker, frames, fps=20.0):
    """Feed a list of per-frame detection lists; return per-frame track lists."""
    return [tracker.update(dets, i / fps) for i, dets in enumerate(frames)]


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------
def test_iou_identical_boxes():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_disjoint_boxes():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_partial_overlap():
    # 5x10 intersection, union = 100 + 100 - 50
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_iou_touching_edges_is_zero():
    assert iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def test_tracker_registry():
    assert "byte_iou" in available_trackers()
    assert isinstance(create_tracker("byte_iou"), ByteIoUTracker)


def test_unknown_tracker_raises():
    with pytest.raises(ValueError, match="Unknown tracker"):
        create_tracker("nope")


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
def test_id_is_stable_across_a_moving_object():
    tracker = ByteIoUTracker(min_hits=2)
    frames = [[make_detection(10 + i * 5, 10)] for i in range(10)]
    results = run(tracker, frames)

    ids = {t.track_id for frame in results for t in frame}
    assert ids == {1}, "a single moving object must keep one ID"


def test_min_hits_suppresses_single_frame_false_positives():
    tracker = ByteIoUTracker(min_hits=3)
    frames = [[make_detection(10, 10)]]
    assert run(tracker, frames)[0] == [], "a one-frame blip must not be reported"


def test_two_objects_get_distinct_ids():
    tracker = ByteIoUTracker(min_hits=2)
    frames = [
        [make_detection(10 + i * 4, 10), make_detection(400 - i * 4, 10)]
        for i in range(6)
    ]
    results = run(tracker, frames)
    final_ids = {t.track_id for t in results[-1]}
    assert len(final_ids) == 2


def test_id_survives_a_brief_occlusion():
    tracker = ByteIoUTracker(min_hits=2, max_age=10)
    # Present for 4 frames, hidden for 3, then back where the motion model
    # predicts it should be.
    frames = []
    for i in range(12):
        visible = not (4 <= i < 7)
        frames.append([make_detection(10 + i * 5, 10)] if visible else [])
    results = run(tracker, frames)

    before = results[3][0].track_id
    after = results[8][0].track_id
    assert before == after, "the same object must not be reborn with a new ID"


def test_track_is_dropped_after_max_age():
    tracker = ByteIoUTracker(min_hits=2, max_age=3)
    frames = [[make_detection(10, 10)] for _ in range(4)] + [[] for _ in range(6)]
    results = run(tracker, frames)

    assert results[-1] == []
    assert any(t.track_id == 1 for t in tracker.lost_tracks) or all(
        not frame for frame in results[6:]
    )


def test_lost_tracks_are_reported_once():
    tracker = ByteIoUTracker(min_hits=2, max_age=2)
    frames = [[make_detection(10, 10)] for _ in range(3)] + [[] for _ in range(6)]
    lost_totals = []
    for i, dets in enumerate(frames):
        tracker.update(dets, i / 20.0)
        lost_totals.append(len(tracker.lost_tracks))
    assert sum(lost_totals) == 1, "a dropped track must be announced exactly once"


def test_reappearance_after_max_age_gets_a_new_id():
    # Beyond max_age the tracker has no basis for claiming it's the same object;
    # a fresh ID is the honest answer at M0.1 (no ReID model).
    tracker = ByteIoUTracker(min_hits=2, max_age=2)
    frames = (
        [[make_detection(10, 10)] for _ in range(3)]
        + [[] for _ in range(6)]
        + [[make_detection(10, 10)] for _ in range(3)]
    )
    results = run(tracker, frames)
    assert results[2][0].track_id != results[-1][0].track_id


# ---------------------------------------------------------------------------
# Two-stage (ByteTrack) association
# ---------------------------------------------------------------------------
def test_low_confidence_detection_rescues_an_existing_track():
    tracker = ByteIoUTracker(min_hits=2, high_threshold=0.5, low_threshold=0.1)
    frames = [
        [make_detection(10, 10, confidence=0.9)],
        [make_detection(15, 10, confidence=0.9)],
        # Confidence collapses (partial occlusion) but the object is still there.
        [make_detection(20, 10, confidence=0.2)],
        [make_detection(25, 10, confidence=0.9)],
    ]
    results = run(tracker, frames)
    assert results[2], "a low-confidence detection must keep the track alive"
    assert results[2][0].track_id == results[1][0].track_id


def test_low_confidence_detection_never_creates_a_track():
    tracker = ByteIoUTracker(min_hits=1, high_threshold=0.5, low_threshold=0.1)
    frames = [[make_detection(10, 10, confidence=0.2)] for _ in range(5)]
    results = run(tracker, frames)
    assert all(frame == [] for frame in results)


def test_detections_below_low_threshold_are_ignored_entirely():
    tracker = ByteIoUTracker(min_hits=1, low_threshold=0.3)
    frames = [[make_detection(10, 10, confidence=0.05)] for _ in range(5)]
    assert all(frame == [] for frame in run(tracker, frames))


# ---------------------------------------------------------------------------
# Class handling
# ---------------------------------------------------------------------------
def test_class_aware_matching_does_not_merge_different_classes():
    tracker = ByteIoUTracker(min_hits=1, class_aware=True)
    frames = [
        [make_detection(10, 10, class_id=0, class_name="person")],
        # Same position, different class: must not inherit the person's ID.
        [make_detection(10, 10, class_id=2, class_name="car")],
    ]
    results = run(tracker, frames)
    assert results[0][0].track_id != results[1][0].track_id


def test_class_agnostic_matching_can_merge_classes():
    tracker = ByteIoUTracker(min_hits=1, class_aware=False)
    frames = [
        [make_detection(10, 10, class_id=0, class_name="person")],
        [make_detection(12, 10, class_id=2, class_name="car")],
    ]
    results = run(tracker, frames)
    assert results[0][0].track_id == results[1][0].track_id


# ---------------------------------------------------------------------------
# Motion model & misc
# ---------------------------------------------------------------------------
def test_velocity_is_estimated_from_motion():
    tracker = ByteIoUTracker(min_hits=2)
    frames = [[make_detection(10 + i * 10, 10)] for i in range(8)]
    results = run(tracker, frames)
    vx, vy = results[-1][0].velocity
    assert vx == pytest.approx(10.0, abs=1.0)
    assert vy == pytest.approx(0.0, abs=0.5)


def test_history_accumulates_positions():
    tracker = ByteIoUTracker(min_hits=2)
    frames = [[make_detection(10 + i * 5, 10)] for i in range(6)]
    results = run(tracker, frames)
    assert len(results[-1].pop().history) == 6


def test_reset_clears_all_state():
    tracker = ByteIoUTracker(min_hits=1)
    run(tracker, [[make_detection(10, 10)] for _ in range(3)])
    tracker.reset()
    result = tracker.update([make_detection(10, 10)], 0.0)
    assert result[0].track_id == 1, "IDs must restart after reset"


def test_empty_stream_produces_no_tracks():
    tracker = ByteIoUTracker()
    assert tracker.update([], 0.0) == []


def test_association_is_deterministic():
    def go():
        tracker = ByteIoUTracker(min_hits=2)
        frames = [
            [make_detection(10 + i * 4, 10), make_detection(300 - i * 4, 10)]
            for i in range(8)
        ]
        return [[(t.track_id, t.bbox) for t in f] for f in run(tracker, frames)]

    assert go() == go()
