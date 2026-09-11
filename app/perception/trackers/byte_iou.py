"""ByteTrack-style IoU tracker (pure Python, no torch, no ReID model).

Why this design
---------------
ByteTrack's core insight is a *two-stage* association: match confident
detections first, then use the leftover low-confidence detections to rescue
tracks that would otherwise be dropped. In a warehouse or street scene that is
precisely the occlusion case Sentinel must survive — a worker who slips behind
a pallet for half a second must keep the same entity ID, not be reborn as a new
one.

Compared to the original we keep the two-stage association and a constant
velocity motion model, but replace the Kalman filter with an exponentially
smoothed linear predictor. That keeps the module dependency-free and
deterministic (hence trivially unit-testable) at a small cost in precision that
does not matter at M0.1, where the output is discrete events rather than
sub-pixel trajectories.

Association evidence (M0.3.1)
-----------------------------
The motion model is **evidence, not a constraint**. An earlier version scored
candidates solely on ``iou(predicted_box, detection)``, which meant a wrong
prediction could veto an obviously correct match. At a direction reversal the
predicted box sits ``2 x step`` from the detection while the track's last known
box sits only ``1 x step`` away, so the tracker fragmented one object into a
new ID per reversal — and would have done better with no motion model at all.

Association now considers three tiers of evidence in strict priority order:

1. **IoU against the predicted box** — the motion model agrees. This is the
   original behaviour and still decides every ordinary frame.
2. **IoU against the last observed box** — the motion model disagrees, but the
   detection still overlaps where the object was last actually seen. This is
   the reversal case.
3. **Normalised centre distance** — neither box overlaps at all, because the
   object moved further than its own size in one frame.

Each tier is consulted only for pairs the previous tier left unmatched. That
ordering is the whole design: relaxation applies exactly where prediction
failed, and never where it succeeded.

Why tiers rather than ``max(predicted_iou, observed_iou)``: at a crossing, one
track's *last observed* box can overlap the *other* object's detection better
than its own predicted box overlaps its own detection. Taking the maximum lets
that stale overlap win and swaps the two identities. Tiering consumes every
confident predicted match first, so the ambiguous evidence is never reached
while the motion model is still doing its job.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from ..tracker import Tracker
from ..types import BBox, Detection, Track


def iou(a: BBox, b: BBox) -> float:
    """Intersection-over-union of two ``(x1, y1, x2, y2)`` boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _shift(box: BBox, dx: float, dy: float) -> BBox:
    return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)


def _center(box: BBox) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _reference_size(box: BBox) -> float:
    """A single characteristic length for a box, in pixels.

    The geometric mean of width and height: stable for both the tall thin boxes
    people produce and the wide flat ones vehicles produce, unlike using width
    or height alone.
    """
    width = max(0.0, box[2] - box[0])
    height = max(0.0, box[3] - box[1])
    return (width * height) ** 0.5


def _area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def center_distance(a: BBox, b: BBox) -> float:
    """Distance between two box centres, in pixels."""
    ax, ay = _center(a)
    bx, by = _center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def normalised_center_distance(track_box: BBox, detection_box: BBox) -> float:
    """Centre distance scaled by the mean characteristic size of the two boxes.

    Scaling matters: 40px is a long way for a distant pedestrian and nothing at
    all for a lorry filling half the frame. A single pixel gate would be wrong
    for one of them.
    """
    reference = (_reference_size(track_box) + _reference_size(detection_box)) / 2.0
    if reference <= 0:
        return float("inf")
    return center_distance(track_box, detection_box) / reference


def size_consistency(a: BBox, b: BBox) -> float:
    """Ratio of the smaller box area to the larger, in ``[0, 1]``."""
    area_a, area_b = _area(a), _area(b)
    larger = max(area_a, area_b)
    if larger <= 0:
        return 0.0
    return min(area_a, area_b) / larger


class _TrackState:
    """Internal mutable bookkeeping for one tracked entity."""

    __slots__ = (
        "track_id",
        "bbox",
        "confidence",
        "class_id",
        "class_name",
        "age",
        "hits",
        "time_since_update",
        "velocity",
        "history",
        "confirmed",
        "first_timestamp",
        "last_timestamp",
    )

    def __init__(
        self,
        track_id: int,
        detection: Detection,
        timestamp: float,
    ) -> None:
        self.track_id = track_id
        self.bbox: BBox = detection.bbox
        self.confidence = detection.confidence
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        self.velocity: Tuple[float, float] = (0.0, 0.0)
        self.history: List[Tuple[float, float]] = [detection.center]
        self.confirmed = False
        self.first_timestamp = timestamp
        self.last_timestamp = timestamp

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def predicted_bbox(self) -> BBox:
        """Where we expect this object to be in the current frame."""
        vx, vy = self.velocity
        return _shift(self.bbox, vx, vy)

    def update(self, detection: Detection, timestamp: float, smoothing: float) -> None:
        prev_cx, prev_cy = self.center
        new_cx, new_cy = detection.center
        vx, vy = self.velocity
        self.velocity = (
            smoothing * (new_cx - prev_cx) + (1.0 - smoothing) * vx,
            smoothing * (new_cy - prev_cy) + (1.0 - smoothing) * vy,
        )
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        # A class flip usually means a confused frame, not a new object: only
        # adopt the new label when the detection is clearly confident.
        if detection.class_id != self.class_id and detection.confidence >= self.confidence:
            self.class_id = detection.class_id
            self.class_name = detection.class_name
        self.hits += 1
        self.time_since_update = 0
        self.last_timestamp = timestamp
        self.history.append((new_cx, new_cy))

    def mark_missed(self) -> None:
        """Coast on the motion model for a frame where nothing matched."""
        self.bbox = self.predicted_bbox()
        self.time_since_update += 1

    def to_track(self) -> Track:
        return Track(
            track_id=self.track_id,
            bbox=self.bbox,
            confidence=self.confidence,
            class_id=self.class_id,
            class_name=self.class_name,
            age=self.age,
            hits=self.hits,
            time_since_update=self.time_since_update,
            velocity=self.velocity,
            history=list(self.history[-32:]),
        )


class ByteIoUTracker(Tracker):
    """Two-stage IoU association with a constant-velocity motion model.

    Parameters
    ----------
    high_threshold:
        Detections at or above this confidence are matched first.
    low_threshold:
        Detections below ``high_threshold`` but at or above this value are used
        in the second pass to rescue tracks. Below it they are discarded.
    iou_threshold:
        Minimum IoU for a detection/track pair to be considered a match.
    max_age:
        Frames a track may coast unmatched before it is deleted.
    min_hits:
        Matched frames required before a track is reported (suppresses
        one-frame false positives).
    class_aware:
        When true, a detection can only match a track of the same class.
    center_distance_gate:
        Fallback gate, in multiples of the mean box size, used only when no
        IoU match is available for a pair. ``0`` disables the fallback and
        restores pure IoU association. The default of 1.5 accepts a detection
        displaced by about one and a half object-widths, which covers a fast
        reversal without reaching a genuinely different object.
    min_size_consistency:
        Minimum smaller/larger area ratio for a centre-distance fallback match.
        Stops a track adopting a detection of wildly different size when only
        the loose gate is in play.
    """

    name = "byte_iou"

    def __init__(
        self,
        high_threshold: float = 0.5,
        low_threshold: float = 0.1,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 2,
        class_aware: bool = True,
        velocity_smoothing: float = 0.5,
        center_distance_gate: float = 1.5,
        min_size_consistency: float = 0.25,
    ) -> None:
        self.high_threshold = float(high_threshold)
        self.low_threshold = float(low_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.class_aware = bool(class_aware)
        self.velocity_smoothing = float(velocity_smoothing)
        self.center_distance_gate = float(center_distance_gate)
        self.min_size_consistency = float(min_size_consistency)
        self._tracks: List[_TrackState] = []
        self._next_id = 1
        self._lost: List[Track] = []
        self.frame_count = 0

    # ------------------------------------------------------------------
    # Tracker interface
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._tracks = []
        self._next_id = 1
        self._lost = []
        self.frame_count = 0

    @property
    def lost_tracks(self) -> List[Track]:
        return list(self._lost)

    def update(self, detections: Sequence[Detection], timestamp: float) -> List[Track]:
        self.frame_count += 1
        self._lost = []
        for state in self._tracks:
            state.age += 1

        high = [d for d in detections if d.confidence >= self.high_threshold]
        low = [
            d
            for d in detections
            if self.low_threshold <= d.confidence < self.high_threshold
        ]

        # --- stage 1: confident detections against every live track ---------
        unmatched_tracks = list(range(len(self._tracks)))
        matches, unmatched_high, unmatched_tracks = self._associate(
            high, unmatched_tracks
        )
        for track_idx, det_idx in matches:
            self._tracks[track_idx].update(
                high[det_idx], timestamp, self.velocity_smoothing
            )

        # --- stage 2: low-confidence detections rescue coasting tracks ------
        # Only tracks that already exist may be rescued; low-confidence
        # detections never spawn new identities.
        matches, _, unmatched_tracks = self._associate(low, unmatched_tracks)
        for track_idx, det_idx in matches:
            self._tracks[track_idx].update(
                low[det_idx], timestamp, self.velocity_smoothing
            )

        for track_idx in unmatched_tracks:
            self._tracks[track_idx].mark_missed()

        # --- spawn new identities from leftover confident detections --------
        for det_idx in unmatched_high:
            self._tracks.append(
                _TrackState(self._next_id, high[det_idx], timestamp)
            )
            self._next_id += 1

        # --- retire dead tracks --------------------------------------------
        survivors: List[_TrackState] = []
        for state in self._tracks:
            if state.time_since_update > self.max_age:
                if state.confirmed:
                    self._lost.append(state.to_track())
                continue
            survivors.append(state)
        self._tracks = survivors

        # --- report ---------------------------------------------------------
        active: List[Track] = []
        for state in self._tracks:
            if state.hits >= self.min_hits:
                state.confirmed = True
            if state.confirmed and state.time_since_update == 0:
                active.append(state.to_track())
        return active

    # ------------------------------------------------------------------
    # Association
    # ------------------------------------------------------------------
    #: Candidate tiers, lowest number wins. Every candidate in a tier is
    #: considered before any candidate in the next, so weaker evidence can only
    #: ever claim pairs that stronger evidence left unmatched.
    TIER_PREDICTED_IOU = 0
    TIER_OBSERVED_IOU = 1
    TIER_CENTER = 2

    def _associate(
        self, detections: Sequence[Detection], track_indices: Sequence[int]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Greedy best-evidence-first matching.

        Returns ``(matches, unmatched_detection_indices, unmatched_track_indices)``
        where each match is a ``(track_index, detection_index)`` pair.
        """
        if not detections or not track_indices:
            return [], list(range(len(detections))), list(track_indices)

        candidates = self._candidates(detections, track_indices)
        # Tier first, then score descending; ties broken by index so the result
        # is deterministic regardless of dict or set iteration order.
        candidates.sort(key=lambda c: (c[0], -c[1], c[2], c[3]))

        matched_tracks: set = set()
        matched_dets: set = set()
        matches: List[Tuple[int, int]] = []
        for _tier, _score, track_idx, det_idx in candidates:
            if track_idx in matched_tracks or det_idx in matched_dets:
                continue
            matched_tracks.add(track_idx)
            matched_dets.add(det_idx)
            matches.append((track_idx, det_idx))

        unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
        unmatched_tracks = [i for i in track_indices if i not in matched_tracks]
        return matches, unmatched_dets, unmatched_tracks

    def _candidates(
        self, detections: Sequence[Detection], track_indices: Sequence[int]
    ) -> List[Tuple[int, float, int, int]]:
        """Score every admissible (track, detection) pair as ``(tier, score, t, d)``."""
        candidates: List[Tuple[int, float, int, int]] = []

        for track_idx in track_indices:
            state = self._tracks[track_idx]
            predicted = state.predicted_bbox()
            observed = state.bbox

            for det_idx, det in enumerate(detections):
                if self.class_aware and det.class_id != state.class_id:
                    continue

                # --- tier 0: the motion model agrees ------------------------
                predicted_overlap = iou(predicted, det.bbox)
                if predicted_overlap >= self.iou_threshold:
                    candidates.append(
                        (self.TIER_PREDICTED_IOU, predicted_overlap, track_idx, det_idx)
                    )
                    continue

                # --- tier 1: the motion model was wrong, but the object is
                # still where we last saw it. This is the reversal case: the
                # prediction points backwards, so it must not veto the match.
                observed_overlap = iou(observed, det.bbox)
                if observed_overlap >= self.iou_threshold:
                    candidates.append(
                        (self.TIER_OBSERVED_IOU, observed_overlap, track_idx, det_idx)
                    )
                    continue

                # --- tier 2: nothing overlaps, because the object moved
                # further than its own size in a single frame.
                if self.center_distance_gate <= 0:
                    continue
                proximity = self._center_score(predicted, observed, det.bbox)
                if proximity is not None:
                    candidates.append((self.TIER_CENTER, proximity, track_idx, det_idx))

        return candidates

    def _center_score(
        self, predicted: BBox, observed: BBox, detection_box: BBox
    ) -> Optional[float]:
        """Fallback affinity in ``[0, 1)``, or ``None`` if the pair is not admissible.

        Measured from whichever of the predicted or last-observed box is nearer,
        for the same reason tier 0 takes the better of the two.
        """
        if size_consistency(observed, detection_box) < self.min_size_consistency:
            return None
        distance = min(
            normalised_center_distance(predicted, detection_box),
            normalised_center_distance(observed, detection_box),
        )
        if distance > self.center_distance_gate:
            return None
        # Nearer is better. Bounded below 1.0 so a fallback score can never be
        # confused with an IoU score when read in logs.
        return max(0.0, 1.0 - distance / self.center_distance_gate) * 0.999
