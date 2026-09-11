"""Ground-truth evaluation of tracker identity persistence.

Used by the M0.3.1 regression tests. Ground truth is used **only to score the
tracker afterwards** — it is never visible to the tracker, which sees nothing
but a list of :class:`~app.perception.types.Detection` objects per frame.

Emitted tracks are mapped back to ground-truth objects by bounding-box
identity: the tracker copies the matched detection's bbox onto the track, and
only emits tracks matched in the current frame, so the mapping is exact and
requires no heuristics of its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.perception.types import Detection

#: One frame of scripted input: ``[(ground_truth_id, Detection), ...]``
LabelledFrame = Sequence[Tuple[str, Detection]]


@dataclass
class TrackingMetrics:
    """How well the tracker preserved identity over a scripted sequence."""

    ground_truth_entities: int
    track_ids_created: int
    id_switches: int
    false_merges: int
    fragmentation: Dict[str, int] = field(default_factory=dict)
    track_owners: Dict[int, List[str]] = field(default_factory=dict)
    gt_track_ids: Dict[str, List[int]] = field(default_factory=dict)
    unmatched_detections: int = 0

    @property
    def is_clean(self) -> bool:
        """One track ID per ground-truth object, with no merges."""
        return (
            self.track_ids_created == self.ground_truth_entities
            and self.id_switches == 0
            and self.false_merges == 0
        )

    def summary(self) -> str:
        return (
            f"gt={self.ground_truth_entities} ids={self.track_ids_created} "
            f"switches={self.id_switches} merges={self.false_merges} "
            f"fragmentation={self.fragmentation}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ground_truth_entities": self.ground_truth_entities,
            "track_ids_created": self.track_ids_created,
            "id_switches": self.id_switches,
            "false_merges": self.false_merges,
            "fragmentation": dict(self.fragmentation),
            "unmatched_detections": self.unmatched_detections,
        }


def evaluate(
    tracker,
    frames: Sequence[LabelledFrame],
    fps: float = 10.0,
) -> TrackingMetrics:
    """Run ``tracker`` over labelled frames and measure identity persistence.

    * **id_switches** — a ground-truth object was reported under a different
      track ID than the last time it was reported. This is what fragmentation
      shows up as.
    * **false_merges** — one track ID was used for two or more different
      ground-truth objects at any point in its life. This is the error that
      loosening association risks, so it is measured alongside.
    * **fragmentation** — distinct track IDs per ground-truth object; 1 is
      perfect.
    """
    gt_track_ids: Dict[str, List[int]] = {}
    track_owners: Dict[int, List[str]] = {}
    last_id_for_gt: Dict[str, int] = {}
    id_switches = 0
    unmatched = 0

    for frame_index, labelled in enumerate(frames):
        detections = [detection for _gt_id, detection in labelled]
        by_bbox = {_bbox_key(d.bbox): gt_id for gt_id, d in labelled}

        tracks = tracker.update(detections, round(frame_index / fps, 4))

        for track in tracks:
            gt_id = by_bbox.get(_bbox_key(track.bbox))
            if gt_id is None:
                # A reported track whose box matches no detection this frame.
                unmatched += 1
                continue

            gt_track_ids.setdefault(gt_id, [])
            if track.track_id not in gt_track_ids[gt_id]:
                gt_track_ids[gt_id].append(track.track_id)

            track_owners.setdefault(track.track_id, [])
            if gt_id not in track_owners[track.track_id]:
                track_owners[track.track_id].append(gt_id)

            previous = last_id_for_gt.get(gt_id)
            if previous is not None and previous != track.track_id:
                id_switches += 1
            last_id_for_gt[gt_id] = track.track_id

    all_ids = set(track_owners)
    false_merges = sum(1 for owners in track_owners.values() if len(owners) > 1)

    return TrackingMetrics(
        ground_truth_entities=len({gt for frame in frames for gt, _d in frame}),
        track_ids_created=len(all_ids),
        id_switches=id_switches,
        false_merges=false_merges,
        fragmentation={gt: len(ids) for gt, ids in sorted(gt_track_ids.items())},
        track_owners=track_owners,
        gt_track_ids=gt_track_ids,
        unmatched_detections=unmatched,
    )


def _bbox_key(bbox: Sequence[float]) -> Tuple[int, ...]:
    """Round to a tenth of a pixel so float noise cannot break the mapping."""
    return tuple(round(float(v) * 10) for v in bbox)


# ---------------------------------------------------------------------------
# Scripted motion builders (shared by the regression tests)
# ---------------------------------------------------------------------------
def box(x: float, y: float = 300.0, w: float = 40.0, h: float = 90.0) -> Tuple[float, ...]:
    return (x, y, x + w, y + h)


def detection(
    x: float,
    y: float = 300.0,
    w: float = 40.0,
    h: float = 90.0,
    confidence: float = 0.9,
    class_id: int = 0,
    class_name: str = "person",
) -> Detection:
    return Detection(box(x, y, w, h), confidence, class_id, class_name)


def triangle_wave(frame: int, low: float, high: float, step: float) -> float:
    """A back-and-forth path between ``low`` and ``high`` at ``step`` per frame."""
    span = high - low
    legs = max(1, int(round(span / step)))
    phase = frame % (2 * legs)
    offset = phase if phase <= legs else 2 * legs - phase
    return low + step * offset
