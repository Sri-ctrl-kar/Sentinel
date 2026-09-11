"""Ground-truth evaluation of tracking quality.

Ground truth is used **only to score the tracker afterwards** — it is never
visible to the tracker, which sees nothing but a list of
:class:`~app.perception.types.Detection` objects per frame.

Two families of metric live here and are deliberately kept separate:

**Identity metrics** (M0.3.1): ID switches, false merges, fragmentation. These
are exactly defined, need no matching threshold, and are what the tracker
regression tests assert against.

**Standard MOT metrics** (M0.6): MOTA and IDF1, computed with *optimal*
assignment rather than greedy, because that is how they are defined. See
:func:`mot_metrics` for the precise formulation and its caveats.

Emitted tracks are mapped back to ground-truth objects by bounding-box
identity: the tracker copies the matched detection's bbox onto the track, and
only emits tracks matched in the current frame, so the mapping is exact and
requires no heuristics of its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..perception.types import Detection
from .assignment import solve_max_score

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


# ---------------------------------------------------------------------------
# Standard MOT metrics (M0.6)
# ---------------------------------------------------------------------------
#: IoU at or above which a predicted box may be matched to a ground-truth box.
#: 0.5 is the MOTChallenge convention.
DEFAULT_IOU_THRESHOLD = 0.5


@dataclass
class MOTMetrics:
    """MOTA and IDF1, computed the way they are defined.

    MOTA
    ----
    ``MOTA = 1 - (FN + FP + IDSW) / GT`` where GT is the total number of
    ground-truth boxes across all frames. Detections are matched to ground
    truth **per frame by optimal assignment** on IoU, with pairs below
    ``iou_threshold`` rejected. MOTA is not a percentage and is **unbounded
    below**: a tracker emitting many false positives scores negative. It is
    reported as a fraction, not dressed up as accuracy.

    IDF1
    ----
    The F1 of identity assignment: predicted identities are matched to
    ground-truth identities by **one global optimal assignment** maximising the
    number of correctly identified boxes, then

    ``IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)``.

    Caveats, stated because they bound what these numbers mean
    ---------------------------------------------------------
    * Matching uses IoU only. There is no distance gating and no class
      constraint beyond what the caller supplies.
    * These are computed over whatever frames the caller provides; they say
      nothing about footage outside the evaluated clip.
    * A high MOTA on a rendered clip is evidence that the pipeline works on
      that clip. It is not evidence about real footage.
    """

    ground_truth_boxes: int
    predicted_boxes: int
    matches: int
    false_negatives: int
    false_positives: int
    id_switches: int
    idtp: int
    idfp: int
    idfn: int
    iou_threshold: float = DEFAULT_IOU_THRESHOLD
    #: Best global ground-truth -> predicted identity mapping. Reused by the
    #: event evaluator so both speak about the same entities.
    id_mapping: Dict[str, str] = field(default_factory=dict)

    @property
    def mota(self) -> Optional[float]:
        if self.ground_truth_boxes == 0:
            return None
        errors = self.false_negatives + self.false_positives + self.id_switches
        return 1.0 - errors / self.ground_truth_boxes

    @property
    def idf1(self) -> Optional[float]:
        denominator = 2 * self.idtp + self.idfp + self.idfn
        if denominator == 0:
            return None
        return 2 * self.idtp / denominator

    @property
    def precision(self) -> Optional[float]:
        if self.predicted_boxes == 0:
            return None
        return self.matches / self.predicted_boxes

    @property
    def recall(self) -> Optional[float]:
        if self.ground_truth_boxes == 0:
            return None
        return self.matches / self.ground_truth_boxes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ground_truth_boxes": self.ground_truth_boxes,
            "predicted_boxes": self.predicted_boxes,
            "matches": self.matches,
            "false_negatives": self.false_negatives,
            "false_positives": self.false_positives,
            "id_switches": self.id_switches,
            "mota": _round(self.mota),
            "idf1": _round(self.idf1),
            "detection_precision": _round(self.precision),
            "detection_recall": _round(self.recall),
            "iou_threshold": self.iou_threshold,
            "id_mapping": dict(self.id_mapping),
            "definition": (
                "MOTA = 1-(FN+FP+IDSW)/GT, unbounded below; "
                "IDF1 = 2*IDTP/(2*IDTP+IDFP+IDFN); both use optimal assignment"
            ),
        }


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two ``(x1, y1, x2, y2)`` boxes."""
    inter_w = min(a[2], b[2]) - max(a[0], b[0])
    inter_h = min(a[3], b[3]) - max(a[1], b[1])
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    intersection = inter_w * inter_h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def mot_metrics(
    ground_truth: Sequence[Dict[str, Sequence[float]]],
    predictions: Sequence[Dict[str, Sequence[float]]],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> MOTMetrics:
    """Compute MOTA and IDF1 from per-frame ``{id: bbox}`` mappings.

    ``ground_truth`` and ``predictions`` are parallel sequences, one entry per
    evaluated frame, each mapping an identity to a bounding box.
    """
    total_gt = sum(len(frame) for frame in ground_truth)
    total_pred = sum(len(frame) for frame in predictions)

    matches = 0
    false_negatives = 0
    false_positives = 0
    id_switches = 0
    previous_match: Dict[str, str] = {}
    # Co-occurrence counts for the global identity assignment behind IDF1.
    overlap: Dict[Tuple[str, str], int] = {}

    for gt_frame, pred_frame in zip(ground_truth, predictions):
        gt_ids = sorted(gt_frame)
        pred_ids = sorted(pred_frame)
        scores = [
            [iou(gt_frame[g], pred_frame[p]) for p in pred_ids] for g in gt_ids
        ]
        pairs = solve_max_score(scores, minimum_score=iou_threshold - 1e-12)
        pairs = [(g, p) for g, p in pairs if scores[g][p] >= iou_threshold]

        matched_gt = set()
        matched_pred = set()
        for gt_index, pred_index in pairs:
            gt_id, pred_id = gt_ids[gt_index], pred_ids[pred_index]
            matched_gt.add(gt_id)
            matched_pred.add(pred_id)
            matches += 1
            overlap[(gt_id, pred_id)] = overlap.get((gt_id, pred_id), 0) + 1
            if gt_id in previous_match and previous_match[gt_id] != pred_id:
                id_switches += 1
            previous_match[gt_id] = pred_id

        false_negatives += len(gt_ids) - len(matched_gt)
        false_positives += len(pred_ids) - len(matched_pred)

    idtp, mapping = _identity_assignment(overlap)
    return MOTMetrics(
        ground_truth_boxes=total_gt,
        predicted_boxes=total_pred,
        matches=matches,
        false_negatives=false_negatives,
        false_positives=false_positives,
        id_switches=id_switches,
        idtp=idtp,
        idfp=total_pred - idtp,
        idfn=total_gt - idtp,
        iou_threshold=iou_threshold,
        id_mapping=mapping,
    )


def _identity_assignment(
    overlap: Dict[Tuple[str, str], int]
) -> Tuple[int, Dict[str, str]]:
    """``(IDTP, mapping)`` under the single best global identity assignment."""
    if not overlap:
        return 0, {}
    gt_ids = sorted({gt for gt, _ in overlap})
    pred_ids = sorted({pred for _, pred in overlap})
    scores = [
        [float(overlap.get((gt, pred), 0)) for pred in pred_ids] for gt in gt_ids
    ]
    pairs = solve_max_score(scores, minimum_score=0.0)
    idtp = int(sum(scores[g][p] for g, p in pairs))
    mapping = {gt_ids[g]: pred_ids[p] for g, p in pairs}
    return idtp, mapping


def _round(value: Optional[float], digits: int = 4) -> Optional[float]:
    return round(value, digits) if value is not None else None
