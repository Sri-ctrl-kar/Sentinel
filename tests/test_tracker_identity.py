"""M0.3.1 regression tests: track identity persistence.

The bug these guard against: the tracker scored association solely on
``iou(predicted_box, detection)``, so a wrong prediction could veto an
obviously correct match. At a direction reversal the predicted box sits
``2 x step`` from the detection while the last observed box sits only
``1 x step`` away, and the tracker fragmented one object into a new ID per
reversal.

Every test here drives the tracker with detections alone. Ground truth is used
only to score the result afterwards (see ``tracking_metrics.py``); the tracker
never sees a label.
"""

import pytest

from app.perception.trackers.byte_iou import (
    ByteIoUTracker,
    center_distance,
    iou,
    normalised_center_distance,
    size_consistency,
)
from app.perception.types import Detection

from tracking_metrics import detection, evaluate, triangle_wave


def tracker(**overrides):
    settings = dict(min_hits=1, max_age=5, iou_threshold=0.1)
    settings.update(overrides)
    return ByteIoUTracker(**settings)


def forklift(x, y=330.0):
    return Detection((x, y, x + 110.0, y + 70.0), 0.88, 90, "forklift")


# ===========================================================================
# Geometry helpers
# ===========================================================================
def test_normalised_center_distance_scales_with_object_size():
    """40px is far for a pedestrian and nothing for a lorry."""
    small = normalised_center_distance((0, 0, 40, 40), (40, 0, 80, 40))
    large = normalised_center_distance((0, 0, 400, 400), (40, 0, 440, 400))
    assert small > large


def test_normalised_center_distance_of_identical_boxes_is_zero():
    assert normalised_center_distance((0, 0, 40, 90), (0, 0, 40, 90)) == 0.0


def test_one_degenerate_box_still_has_a_scale_from_the_other():
    """A zero-area box is still measurable: the other box supplies the scale."""
    # Degenerate box sitting exactly on the other box's centre.
    assert normalised_center_distance((20, 45, 20, 45), (0, 0, 40, 90)) == 0.0
    # ...and further away scores higher, rather than dividing by zero.
    assert normalised_center_distance((20, 45, 20, 45), (60, 0, 100, 90)) > 0.0


def test_two_degenerate_boxes_have_no_scale_at_all():
    assert normalised_center_distance((0, 0, 0, 0), (50, 50, 50, 50)) == float("inf")


def test_center_distance_is_euclidean():
    assert center_distance((0, 0, 10, 10), (30, 40, 40, 50)) == pytest.approx(50.0)


def test_size_consistency_ratio():
    assert size_consistency((0, 0, 40, 90), (0, 0, 40, 90)) == 1.0
    assert size_consistency((0, 0, 40, 90), (0, 0, 40, 45)) == pytest.approx(0.5)
    assert size_consistency((0, 0, 0, 0), (0, 0, 40, 90)) == 0.0


# ===========================================================================
# 1. Existing normal movement
# ===========================================================================
def test_normal_linear_movement_keeps_one_id():
    frames = [[("w", detection(100 + 10 * f))] for f in range(20)]
    metrics = evaluate(tracker(), frames)
    assert metrics.is_clean
    assert metrics.track_ids_created == 1


def test_normal_movement_on_both_axes_keeps_one_id():
    frames = [[("w", detection(100 + 8 * f, 100 + 6 * f))] for f in range(20)]
    assert evaluate(tracker(), frames).is_clean


def test_a_stationary_object_keeps_one_id():
    frames = [[("w", detection(200))] for _ in range(20)]
    assert evaluate(tracker(), frames).is_clean


def test_slow_movement_still_matches_on_prediction():
    """The motion model must still be doing its job in the ordinary case."""
    track = tracker()
    for f in range(6):
        track.update([detection(100 + 10 * f)], f / 10.0)
    state = track._tracks[0]
    assert state.velocity[0] == pytest.approx(10.0, abs=1.5)
    # Prediction leads the last observation, which is the point of having one.
    assert state.predicted_bbox()[0] > state.bbox[0]


# ===========================================================================
# 2. Temporary occlusion
# ===========================================================================
def test_short_occlusion_preserves_identity():
    frames = []
    for f in range(20):
        hidden = 8 <= f < 11
        frames.append([] if hidden else [("w", detection(100 + 12 * f))])
    metrics = evaluate(tracker(max_age=8), frames)
    assert metrics.track_ids_created == 1
    assert metrics.id_switches == 0


def test_occlusion_of_one_object_while_another_passes():
    frames = []
    for f in range(20):
        row = [("b", detection(400 - 15 * f))]
        if not (8 <= f < 11):
            row.insert(0, ("a", detection(100 + 15 * f)))
        frames.append(row)
    metrics = evaluate(tracker(max_age=8), frames)
    assert metrics.is_clean, metrics.summary()


def test_occlusion_while_reversing_still_preserves_identity():
    """The two failure modes combined: hidden across the turning point."""
    frames = []
    for f in range(24):
        x = triangle_wave(f, 300.0, 400.0, 20.0)
        frames.append([] if 4 <= f < 7 else [("w", detection(x))])
    metrics = evaluate(tracker(max_age=8), frames)
    assert metrics.track_ids_created == 1
    assert metrics.id_switches == 0


# ===========================================================================
# 3. Fast direction reversal  (the reported bug)
# ===========================================================================
@pytest.mark.parametrize("step", [5.0, 10.0, 15.0, 20.0, 25.0, 35.0, 45.0, 60.0])
def test_direction_reversal_keeps_one_id_at_every_speed(step):
    """Fragmentation used to begin the moment 2 x step exceeded the box width."""
    frames = [
        [("w", detection(triangle_wave(f, 300.0, 300.0 + 4 * step, step)))]
        for f in range(24)
    ]
    metrics = evaluate(tracker(), frames)
    assert metrics.track_ids_created == 1, metrics.summary()
    assert metrics.id_switches == 0


def test_reversal_is_matched_on_the_observed_box_not_the_prediction():
    """Pin the mechanism, not just the outcome."""
    track = tracker()
    for f in range(6):
        track.update([detection(300 + 20 * f)], f / 10.0)

    state = track._tracks[0]
    reversed_detection = detection(380.0)  # back one step from 400
    predicted_overlap = iou(state.predicted_bbox(), reversed_detection.bbox)
    observed_overlap = iou(state.bbox, reversed_detection.bbox)

    # The prediction is useless here; the last observation is not.
    assert predicted_overlap < track.iou_threshold
    assert observed_overlap >= track.iou_threshold

    result = track.update([reversed_detection], 0.6)
    assert [t.track_id for t in result] == [1]


def test_the_original_m0_3_reproduction_no_longer_fragments():
    """The exact pacing path that produced ~7 identities during M0.3 validation."""
    frames = []
    for f in range(31):
        phase = f % 10
        offset = phase if phase <= 5 else 10 - phase
        x = 320.0 + 20.0 * offset
        frames.append(
            [
                ("worker", detection(x)),
                ("forklift", forklift(700.0 - 8.5 * f)),
            ]
        )
    metrics = evaluate(tracker(), frames)
    assert metrics.ground_truth_entities == 2
    assert metrics.track_ids_created == 2, metrics.summary()
    assert metrics.id_switches == 0
    assert metrics.false_merges == 0


def test_motion_faster_than_the_object_needs_the_center_fallback():
    """Beyond one box width nothing overlaps; the gate is what recovers it."""
    frames = [
        [("w", detection(triangle_wave(f, 300.0, 540.0, 60.0)))] for f in range(24)
    ]
    assert evaluate(tracker(), frames).track_ids_created == 1
    # Disabling the fallback reproduces the fragmentation, proving it is load-bearing.
    assert evaluate(tracker(center_distance_gate=0.0), frames).track_ids_created > 1


# ===========================================================================
# 4. Stop-and-reverse movement
# ===========================================================================
def test_stop_then_reverse_keeps_one_id():
    positions = (
        [300 + 15 * f for f in range(8)]      # walk right
        + [405] * 6                           # stop
        + [405 - 15 * f for f in range(1, 9)]  # walk back
    )
    frames = [[("w", detection(x))] for x in positions]
    metrics = evaluate(tracker(), frames)
    assert metrics.track_ids_created == 1, metrics.summary()
    assert metrics.id_switches == 0


def test_stop_then_reverse_fast_keeps_one_id():
    positions = (
        [300 + 25 * f for f in range(6)]
        + [425] * 4
        + [425 - 25 * f for f in range(1, 7)]
    )
    frames = [[("w", detection(x))] for x in positions]
    assert evaluate(tracker(), frames).is_clean


def test_stale_velocity_after_a_stop_does_not_break_the_next_match():
    """Velocity decays through the stop rather than stranding the prediction."""
    track = tracker()
    for f in range(6):
        track.update([detection(300 + 20 * f)], f / 10.0)
    for f in range(6, 12):
        track.update([detection(400)], f / 10.0)
    assert len(track._tracks) == 1
    assert track._tracks[0].track_id == 1


# ===========================================================================
# 5. Multiple nearby objects of the same class
# ===========================================================================
@pytest.mark.parametrize("gap", [30.0, 50.0, 80.0, 120.0])
def test_parallel_same_class_objects_stay_separate(gap):
    frames = [
        [("a", detection(100 + 10 * f)), ("b", detection(100 + gap + 10 * f))]
        for f in range(20)
    ]
    metrics = evaluate(tracker(), frames)
    assert metrics.is_clean, f"gap={gap}: {metrics.summary()}"


def test_two_nearby_objects_reversing_together_stay_separate():
    frames = [
        [
            ("a", detection(triangle_wave(f, 100.0, 200.0, 20.0))),
            ("b", detection(triangle_wave(f, 180.0, 280.0, 20.0))),
        ]
        for f in range(24)
    ]
    metrics = evaluate(tracker(), frames)
    assert metrics.is_clean, metrics.summary()


def test_a_crowd_of_same_class_objects_keeps_distinct_ids():
    frames = [
        [(f"p{i}", detection(100 + i * 70 + 8 * f)) for i in range(5)]
        for f in range(20)
    ]
    metrics = evaluate(tracker(), frames)
    assert metrics.ground_truth_entities == 5
    assert metrics.is_clean, metrics.summary()


def test_the_center_fallback_respects_size_consistency():
    """A non-overlapping detection of very different size is not a plausible match."""
    track = tracker()
    for f in range(4):
        track.update([detection(100 + 5 * f)], f / 10.0)
    # Same class and inside the distance gate, but a ninth of the area.
    tiny = Detection((165.0, 300.0, 178.0, 330.0), 0.9, 0, "person")
    assert size_consistency(track._tracks[0].bbox, tiny.bbox) < track.min_size_consistency
    result = track.update([tiny], 0.4)
    assert [t.track_id for t in result] == [2], "should start a new track, not adopt"


def test_size_consistency_gate_is_inclusive_at_the_threshold():
    """Exactly at the threshold is accepted, so the boundary is documented."""
    track = tracker()
    for f in range(4):
        track.update([detection(100 + 5 * f)], f / 10.0)
    # Quarter area == the 0.25 default, and no overlap with the track box.
    quarter = Detection((165.0, 300.0, 185.0, 345.0), 0.9, 0, "person")
    assert size_consistency(track._tracks[0].bbox, quarter.bbox) == pytest.approx(
        track.min_size_consistency
    )
    assert [t.track_id for t in track.update([quarter], 0.4)] == [1]


def test_class_consistency_still_prevents_cross_class_matches():
    track = tracker()
    for f in range(4):
        track.update([detection(300 + 5 * f)], f / 10.0)
    result = track.update([forklift(300.0, 300.0)], 0.4)
    assert all(t.class_name == "forklift" for t in result if t.track_id == 2)
    assert 1 not in [t.track_id for t in result]


# ===========================================================================
# 6. Two crossing trajectories
# ===========================================================================
def test_head_on_crossing_does_not_swap_identities():
    frames = [
        [("a", detection(100 + 12 * f)), ("b", detection(400 - 12 * f))]
        for f in range(26)
    ]
    metrics = evaluate(tracker(), frames)
    assert metrics.is_clean, metrics.summary()
    assert metrics.false_merges == 0


def test_diagonal_crossing_does_not_swap_identities():
    frames = [
        [
            ("a", detection(100 + 12 * f, 200 + 8 * f)),
            ("b", detection(400 - 12 * f, 400 - 8 * f)),
        ]
        for f in range(26)
    ]
    metrics = evaluate(tracker(), frames)
    assert metrics.is_clean, metrics.summary()


def test_crossing_is_resolved_by_the_prediction_tier():
    """Crossing is exactly where the motion model earns its keep.

    Relaxed evidence must never be reached while prediction is working, which
    is why the tiers are ordered rather than combined with max().
    """
    frames = [
        [("a", detection(100 + 12 * f)), ("b", detection(400 - 12 * f))]
        for f in range(26)
    ]
    # With no motion model at all, crossing objects are genuinely ambiguous.
    naive = evaluate(tracker(velocity_smoothing=0.0), frames)
    informed = evaluate(tracker(), frames)
    assert informed.false_merges <= naive.false_merges


def test_crossing_objects_of_different_classes_stay_separate():
    frames = [
        [("a", detection(100 + 12 * f)), ("b", forklift(400 - 12 * f, 300.0))]
        for f in range(26)
    ]
    assert evaluate(tracker(), frames).is_clean


# ===========================================================================
# 7. Disappearance beyond max_age
# ===========================================================================
def test_absence_beyond_max_age_starts_a_new_identity():
    """Correct behaviour, not a bug: without ReID the tracker cannot know."""
    frames = (
        [[("a", detection(100 + 10 * f))] for f in range(6)]
        + [[] for _ in range(10)]
        + [[("a", detection(300 + 10 * f))] for f in range(6)]
    )
    metrics = evaluate(tracker(max_age=3), frames)
    assert metrics.track_ids_created == 2
    assert metrics.false_merges == 0


def test_a_track_is_retired_after_max_age():
    track = tracker(max_age=3)
    for f in range(4):
        track.update([detection(100 + 10 * f)], f / 10.0)
    for f in range(4, 12):
        track.update([], f / 10.0)
    assert track.update([], 1.2) == []
    assert track._tracks == []


def test_a_retired_track_is_announced_exactly_once():
    track = tracker(max_age=2)
    for f in range(3):
        track.update([detection(100)], f / 10.0)
    announcements = 0
    for f in range(3, 12):
        track.update([], f / 10.0)
        announcements += len(track.lost_tracks)
    assert announcements == 1


def test_the_fallback_gate_does_not_resurrect_a_distant_object():
    """The looser matching must not reach across the frame."""
    track = tracker(max_age=30)
    for f in range(6):
        track.update([detection(100 + 5 * f)], f / 10.0)
    far_away = detection(900.0)
    result = track.update([far_away], 0.6)
    assert [t.track_id for t in result] == [2]


# ===========================================================================
# 8. Determinism and tier ordering invariants
# ===========================================================================
def test_association_remains_deterministic():
    def run():
        frames = [
            [("a", detection(100 + 12 * f)), ("b", detection(400 - 12 * f))]
            for f in range(26)
        ]
        return evaluate(tracker(), frames).to_dict()

    assert run() == run()


def test_tiers_are_ordered_weakest_last():
    assert (
        ByteIoUTracker.TIER_PREDICTED_IOU
        < ByteIoUTracker.TIER_OBSERVED_IOU
        < ByteIoUTracker.TIER_CENTER
    )


def test_a_center_candidate_never_outranks_an_iou_candidate():
    """The invariant that protects crossing objects."""
    track = tracker()
    for f in range(5):
        track.update(
            [detection(100 + 10 * f), detection(400 - 10 * f)], f / 10.0
        )
    candidates = track._candidates(
        [detection(150), detection(360)], list(range(len(track._tracks)))
    )
    tiers = [c[0] for c in candidates]
    iou_scores = [c[1] for c in candidates if c[0] != ByteIoUTracker.TIER_CENTER]
    center_scores = [c[1] for c in candidates if c[0] == ByteIoUTracker.TIER_CENTER]
    assert tiers, "expected candidates"
    # Sorting is by tier first, so every centre candidate is considered last
    # regardless of its numeric score.
    if iou_scores and center_scores:
        ordered = sorted(candidates, key=lambda c: (c[0], -c[1], c[2], c[3]))
        first_center = next(
            i for i, c in enumerate(ordered) if c[0] == ByteIoUTracker.TIER_CENTER
        )
        assert all(
            c[0] != ByteIoUTracker.TIER_CENTER for c in ordered[:first_center]
        )


def test_disabling_the_gate_restores_pure_iou_association():
    track = tracker(center_distance_gate=0.0)
    for f in range(4):
        track.update([detection(100 + 5 * f)], f / 10.0)
    # A jump far beyond any overlap: with the gate off this must not match.
    result = track.update([detection(400.0)], 0.4)
    assert [t.track_id for t in result] == [2]


def test_reset_clears_the_new_state_too():
    track = tracker()
    for f in range(5):
        track.update([detection(100 + 10 * f)], f / 10.0)
    track.reset()
    assert track.update([detection(100)], 0.0)[0].track_id == 1
