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
    ) -> None:
        self.high_threshold = float(high_threshold)
        self.low_threshold = float(low_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.class_aware = bool(class_aware)
        self.velocity_smoothing = float(velocity_smoothing)
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
    def _associate(
        self, detections: Sequence[Detection], track_indices: Sequence[int]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Greedy highest-IoU-first matching.

        Returns ``(matches, unmatched_detection_indices, unmatched_track_indices)``
        where each match is a ``(track_index, detection_index)`` pair.
        """
        if not detections or not track_indices:
            return [], list(range(len(detections))), list(track_indices)

        candidates: List[Tuple[float, int, int]] = []
        for track_idx in track_indices:
            state = self._tracks[track_idx]
            predicted = state.predicted_bbox()
            for det_idx, det in enumerate(detections):
                if self.class_aware and det.class_id != state.class_id:
                    continue
                score = iou(predicted, det.bbox)
                if score >= self.iou_threshold:
                    candidates.append((score, track_idx, det_idx))

        # Sort by IoU descending; ties broken by index so results are stable.
        candidates.sort(key=lambda c: (-c[0], c[1], c[2]))

        matched_tracks: set = set()
        matched_dets: set = set()
        matches: List[Tuple[int, int]] = []
        for _score, track_idx, det_idx in candidates:
            if track_idx in matched_tracks or det_idx in matched_dets:
                continue
            matched_tracks.add(track_idx)
            matched_dets.add(det_idx)
            matches.append((track_idx, det_idx))

        unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
        unmatched_tracks = [i for i in track_indices if i not in matched_tracks]
        return matches, unmatched_dets, unmatched_tracks
