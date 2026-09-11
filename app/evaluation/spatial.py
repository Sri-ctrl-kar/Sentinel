"""Calibration validation and perspective checks.

M0.4 proved the homography reproduces the points it was fitted to. That is
necessary and almost meaningless: four points always fit exactly. This module
asks the questions that actually bound world-space accuracy.

1. **Conditioning** — is the point configuration well spread, or nearly
   degenerate?
2. **Noise sensitivity** — if the surveyed points are off by a few pixels, how
   wrong are the resulting metres?
3. **Perspective distinction** — does world-space reasoning actually separate
   two situations that image space cannot tell apart?

None of this establishes real-world accuracy. It establishes how much accuracy
is *lost* to a given amount of input error, which is a different and more
honest claim: calibration quality bounds world-space accuracy, and here is the
bound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..calibration.homography import DegenerateConfigurationError
from ..calibration.planar import GroundPlaneCalibration
from ..spatial import euclidean_distance, triangle_sine


@dataclass
class CalibrationConditionReport:
    """How well-posed a calibration is, and how it degrades under noise."""

    name: str
    point_count: int
    accepted: bool
    #: Smallest |sin| over any triple of image points. Near 0 means three
    #: points are nearly collinear and the fit is ill-conditioned.
    minimum_triangle_sine: float
    #: Largest world-space error induced by perturbing the image points.
    worst_case_error_m: Optional[float] = None
    mean_error_m: Optional[float] = None
    perturbation_px: float = 0.0
    rejection_reason: Optional[str] = None
    probe_count: int = 0

    @property
    def is_well_conditioned(self) -> bool:
        """A pragmatic threshold, not a theorem.

        0.1 corresponds to roughly 6 degrees between the legs of the worst
        triple. Below that the points are close enough to collinear that small
        survey errors move the resulting metres a long way.
        """
        return self.accepted and self.minimum_triangle_sine >= 0.1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "point_count": self.point_count,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
            "minimum_triangle_sine": round(self.minimum_triangle_sine, 6),
            "well_conditioned": self.is_well_conditioned,
            "perturbation_px": self.perturbation_px,
            "mean_world_error_m": _round(self.mean_error_m),
            "worst_case_world_error_m": _round(self.worst_case_error_m),
            "probe_count": self.probe_count,
            "interpretation": (
                "world error is what a perturbation of this size in the "
                "surveyed image points costs; it is not a measurement of "
                "real-world accuracy"
            ),
        }


def minimum_triangle_sine(points: Sequence[Sequence[float]]) -> float:
    """Smallest |sin| over every triple — a scale-free conditioning proxy."""
    worst = 1.0
    count = len(points)
    for i in range(count):
        for j in range(i + 1, count):
            for k in range(j + 1, count):
                worst = min(worst, triangle_sine(points[i], points[j], points[k]))
    return worst


#: Deterministic offsets applied when perturbing calibration points. A fixed
#: cycle rather than random noise so the benchmark is reproducible.
_PERTURBATION_PATTERN: Tuple[Tuple[float, float], ...] = (
    (1.0, 0.0),
    (0.0, 1.0),
    (-1.0, 0.0),
    (0.0, -1.0),
    (0.707, 0.707),
    (-0.707, 0.707),
    (0.707, -0.707),
    (-0.707, -0.707),
)


def evaluate_calibration_conditioning(
    image_points: Sequence[Sequence[float]],
    world_points: Sequence[Sequence[float]],
    name: str = "calibration",
    perturbation_px: float = 2.0,
    probes: Optional[Sequence[Sequence[float]]] = None,
) -> CalibrationConditionReport:
    """Fit a calibration, then measure what input error costs in metres.

    Each image point is displaced by ``perturbation_px`` along a fixed
    direction from :data:`_PERTURBATION_PATTERN`, the calibration is refitted,
    and the probe points are mapped through both. The difference is the
    world-space cost of that much survey error.

    Deterministic by construction: no random noise, so the reported figure is
    reproducible and comparable between runs.
    """
    condition = minimum_triangle_sine(image_points)
    try:
        baseline = GroundPlaneCalibration(image_points, world_points, name=name)
    except (ValueError, DegenerateConfigurationError) as exc:
        return CalibrationConditionReport(
            name=name,
            point_count=len(image_points),
            accepted=False,
            minimum_triangle_sine=condition,
            rejection_reason=str(exc),
            perturbation_px=perturbation_px,
        )

    if probes is None:
        probes = _default_probes(baseline)

    perturbed_points = [
        (
            float(point[0]) + _PERTURBATION_PATTERN[i % len(_PERTURBATION_PATTERN)][0]
            * perturbation_px,
            float(point[1]) + _PERTURBATION_PATTERN[i % len(_PERTURBATION_PATTERN)][1]
            * perturbation_px,
        )
        for i, point in enumerate(image_points)
    ]

    try:
        perturbed = GroundPlaneCalibration(
            perturbed_points, world_points, name=f"{name}_perturbed"
        )
    except (ValueError, DegenerateConfigurationError) as exc:
        return CalibrationConditionReport(
            name=name,
            point_count=len(image_points),
            accepted=True,
            minimum_triangle_sine=condition,
            rejection_reason=f"perturbed fit failed: {exc}",
            perturbation_px=perturbation_px,
        )

    errors: List[float] = []
    for probe in probes:
        a = baseline.image_to_world(probe)
        b = perturbed.image_to_world(probe)
        errors.append(euclidean_distance(a.as_tuple, b.as_tuple))

    return CalibrationConditionReport(
        name=name,
        point_count=len(image_points),
        accepted=True,
        minimum_triangle_sine=condition,
        worst_case_error_m=max(errors) if errors else None,
        mean_error_m=sum(errors) / len(errors) if errors else None,
        perturbation_px=perturbation_px,
        probe_count=len(errors),
    )


def _default_probes(calibration: GroundPlaneCalibration) -> List[Tuple[float, float]]:
    """A grid across the calibrated image region."""
    x1, y1, x2, y2 = _bounds(calibration.image_region)
    probes = []
    for i in range(1, 4):
        for j in range(1, 4):
            probes.append((x1 + (x2 - x1) * i / 4.0, y1 + (y2 - y1) * j / 4.0))
    return probes


def _bounds(points: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


# ---------------------------------------------------------------------------
# Perspective distinction
# ---------------------------------------------------------------------------
@dataclass
class PerspectiveCheck:
    """Does world-space reasoning separate what image space cannot?

    The permanent regression guarding this milestone's core claim: two pairs
    with near-identical pixel separation whose ground-plane separations differ
    substantially. If this ever fails, calibrated reasoning has stopped adding
    anything over pixels.
    """

    near_pixel_gap: float
    far_pixel_gap: float
    near_world_gap_m: float
    far_world_gap_m: float
    pixel_gap_tolerance: float = 2.0
    minimum_world_ratio: float = 1.5

    @property
    def pixel_gaps_match(self) -> bool:
        return abs(self.near_pixel_gap - self.far_pixel_gap) <= self.pixel_gap_tolerance

    @property
    def world_ratio(self) -> float:
        smaller = min(self.near_world_gap_m, self.far_world_gap_m)
        larger = max(self.near_world_gap_m, self.far_world_gap_m)
        return larger / smaller if smaller > 0 else math.inf

    @property
    def passed(self) -> bool:
        return self.pixel_gaps_match and self.world_ratio >= self.minimum_world_ratio

    def to_dict(self) -> Dict[str, Any]:
        return {
            "near_pixel_gap": round(self.near_pixel_gap, 2),
            "far_pixel_gap": round(self.far_pixel_gap, 2),
            "pixel_gaps_match": self.pixel_gaps_match,
            "near_world_gap_m": round(self.near_world_gap_m, 3),
            "far_world_gap_m": round(self.far_world_gap_m, 3),
            "world_ratio": round(self.world_ratio, 3),
            "minimum_world_ratio": self.minimum_world_ratio,
            "passed": self.passed,
            "claim": (
                "equal image separations correspond to substantially different "
                "ground separations, and calibrated reasoning distinguishes them"
            ),
        }


def perspective_distinction_check(
    calibration: GroundPlaneCalibration,
    near_depth_m: float,
    far_depth_m: float,
    pixel_gap: float,
    start_x_m: float = 2.0,
) -> PerspectiveCheck:
    """Build two equal-pixel-gap pairs at different depths and compare them."""
    from ..scenarios import world_offset_for_pixel_gap

    near_offset = world_offset_for_pixel_gap(
        calibration, near_depth_m, start_x_m, pixel_gap
    )
    far_offset = world_offset_for_pixel_gap(
        calibration, far_depth_m, start_x_m, pixel_gap
    )

    def gap_px(depth: float, offset: float) -> float:
        a = calibration.world_to_image((start_x_m, depth))
        b = calibration.world_to_image((start_x_m + offset, depth))
        return euclidean_distance(a, b)

    return PerspectiveCheck(
        near_pixel_gap=gap_px(near_depth_m, near_offset),
        far_pixel_gap=gap_px(far_depth_m, far_offset),
        near_world_gap_m=near_offset,
        far_world_gap_m=far_offset,
    )


def _round(value: Optional[float], digits: int = 4) -> Optional[float]:
    return round(value, digits) if value is not None else None
