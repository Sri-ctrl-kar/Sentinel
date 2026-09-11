"""Ground-plane calibration: image pixels to metres on the floor.

What this buys us
-----------------
Pixel distance is not physical distance. The same 50px gap is centimetres near
the camera and metres at the far wall, so a single pixel threshold cannot be
correct across one frame. A homography fixes that *for points on the calibrated
plane*: two people standing on the floor 3m apart measure 3m apart wherever
they are in the image.

What it does not buy us
-----------------------
Read :meth:`GroundPlaneCalibration.limitations` before trusting a number from
this module. In short: the transform is exact only for points **on the plane**,
accuracy is bounded by the accuracy of the surveyed points you supplied, and
outside the region those points span the result is extrapolation.

This module deliberately knows nothing about detectors, trackers, events or
memory. It converts coordinates and describes its own reliability.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..spatial import (
    ANCHOR_BOTTOM_CENTER,
    GROUND_PLANE_METERS,
    IMAGE_PIXELS,
    UNIT_METERS,
    Point,
    Zone,
    anchor_point,
    convex_hull,
    euclidean_distance,
    triangle_sine,
)
from .homography import DegenerateConfigurationError, Homography

#: Minimum correspondences for a homography.
MIN_CORRESPONDENCES = 4

#: ``|sin|`` below which three points count as collinear. About 0.06 degrees —
#: tight enough to accept any honestly surveyed rectangle, loose enough to
#: reject points a human meant to put on one line.
DEFAULT_COLLINEARITY_TOLERANCE = 1e-3


@dataclass(frozen=True)
class GroundPoint:
    """A position on the calibrated ground plane.

    Carries its own coordinate space and, critically, whether it came from
    inside the region the calibration was actually fitted over. A point
    outside that region is an extrapolation and is flagged as such rather than
    being quietly returned as though it were as trustworthy as the rest.
    """

    x: float
    y: float
    in_calibrated_region: bool = True
    coordinate_space: str = GROUND_PLANE_METERS
    units: str = UNIT_METERS

    @property
    def as_tuple(self) -> Point:
        return (self.x, self.y)

    def distance_to(self, other: "GroundPoint") -> float:
        """Distance to another ground point, in this calibration's units."""
        if other.coordinate_space != self.coordinate_space:
            raise ValueError(
                f"Cannot measure between '{self.coordinate_space}' and "
                f"'{other.coordinate_space}'"
            )
        return euclidean_distance(self.as_tuple, other.as_tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "coordinate_space": self.coordinate_space,
            "units": self.units,
            "in_calibrated_region": self.in_calibrated_region,
        }


@dataclass(frozen=True)
class CalibrationQuality:
    """How well the fitted transform reproduces the points it was given.

    This is a **self-consistency** measure, not an accuracy measure. A perfect
    residual means the correspondences agree with a homography; it says nothing
    about whether the surveyed world points were measured correctly in the
    first place. Four points always fit exactly, so a zero residual from four
    points carries no information at all — which is why ``point_count`` sits
    next to it.
    """

    point_count: int
    rms_image_error_px: float
    max_image_error_px: float
    rms_world_error: float
    max_world_error: float
    world_units: str = UNIT_METERS

    @property
    def is_exactly_determined(self) -> bool:
        """True when there is no redundancy to validate the fit against."""
        return self.point_count == MIN_CORRESPONDENCES

    def to_dict(self) -> Dict[str, Any]:
        return {
            "point_count": self.point_count,
            "rms_image_error_px": round(self.rms_image_error_px, 6),
            "max_image_error_px": round(self.max_image_error_px, 6),
            f"rms_world_error_{self.world_units}": round(self.rms_world_error, 6),
            f"max_world_error_{self.world_units}": round(self.max_world_error, 6),
            "exactly_determined": self.is_exactly_determined,
            "residuals_measure": "self-consistency of the supplied points, not accuracy",
        }


class GroundPlaneCalibration:
    """Maps image pixels to ground-plane metres for a fixed camera.

    Parameters
    ----------
    image_points:
        Four or more ``(x, y)`` pixel positions of known ground features.
    world_points:
        The same features' ``(X, Y)`` positions on the floor, in ``world_units``.
    world_units:
        Only ``"meters"`` is supported. Anything else is refused rather than
        relabelled, because a length whose unit is a guess is worse than no
        length at all.
    name:
        Optional label, carried into metadata for provenance.
    """

    def __init__(
        self,
        image_points: Sequence[Sequence[float]],
        world_points: Sequence[Sequence[float]],
        world_units: str = UNIT_METERS,
        name: Optional[str] = None,
        collinearity_tolerance: float = DEFAULT_COLLINEARITY_TOLERANCE,
        description: str = "",
    ) -> None:
        if world_units != UNIT_METERS:
            raise ValueError(
                f"Unsupported world_units '{world_units}'. M0.4 supports only "
                f"'{UNIT_METERS}'; units are never inferred or relabelled."
            )

        self.name = name or "unnamed"
        self.description = description
        self.world_units = world_units
        self.collinearity_tolerance = float(collinearity_tolerance)

        self.image_points: List[Point] = _as_points(image_points, "image_points")
        self.world_points: List[Point] = _as_points(world_points, "world_points")
        self._validate()

        self._to_world = Homography.from_correspondences(
            self.image_points, self.world_points
        )
        self._to_image = self._to_world.inverse()

        self.image_region = convex_hull(self.image_points)
        self.world_region = convex_hull(self.world_points)
        self._image_zone = _zone_from_hull("calibrated_image_region", self.image_region)
        self._world_hull_zone = _zone_from_hull("calibrated_world_region", self.world_region)

        self.quality = self._measure_quality()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_rectangle(
        cls,
        image_corners: Sequence[Sequence[float]],
        width_meters: float,
        depth_meters: float,
        **kwargs: Any,
    ) -> "GroundPlaneCalibration":
        """Calibrate from four image corners of a known floor rectangle.

        Corners must be given in the order top-left, top-right, bottom-right,
        bottom-left as they appear *in the image*; they map onto
        ``(0,0), (width,0), (width,depth), (0,depth)`` on the floor.
        """
        if len(image_corners) != 4:
            raise ValueError(
                f"from_rectangle needs exactly 4 image corners, got {len(image_corners)}"
            )
        if width_meters <= 0 or depth_meters <= 0:
            raise ValueError("Rectangle width and depth must both be positive")
        world = [
            (0.0, 0.0),
            (float(width_meters), 0.0),
            (float(width_meters), float(depth_meters)),
            (0.0, float(depth_meters)),
        ]
        return cls(image_corners, world, **kwargs)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "GroundPlaneCalibration":
        return cls(
            image_points=payload["image_points"],
            world_points=payload["world_points"],
            world_units=payload.get("world_units", UNIT_METERS),
            name=payload.get("name"),
            description=payload.get("description", ""),
        )

    @classmethod
    def load_json(cls, path: str) -> "GroundPlaneCalibration":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def save_json(self, path: str, indent: int = 2) -> str:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=indent)
            handle.write("\n")
        return path

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate(self) -> None:
        if len(self.image_points) != len(self.world_points):
            raise ValueError(
                f"Point count mismatch: {len(self.image_points)} image points vs "
                f"{len(self.world_points)} world points. Each image point needs "
                "exactly one world counterpart."
            )
        if len(self.image_points) < MIN_CORRESPONDENCES:
            raise ValueError(
                f"A homography needs at least {MIN_CORRESPONDENCES} correspondences, "
                f"got {len(self.image_points)}."
            )

        _reject_duplicates(self.image_points, "image")
        _reject_duplicates(self.world_points, "world")

        for label, points in (("image", self.image_points), ("world", self.world_points)):
            if not self._has_general_position_quad(points):
                raise DegenerateConfigurationError(
                    f"Degenerate {label} configuration: no four of the supplied "
                    f"{label} points are in general position (every candidate set "
                    "has three or more points on a line). A homography cannot be "
                    "determined from such points."
                )

    def _has_general_position_quad(self, points: Sequence[Point]) -> bool:
        """Is there any set of 4 points with no 3 collinear?"""
        for quad in combinations(range(len(points)), MIN_CORRESPONDENCES):
            if all(
                triangle_sine(points[a], points[b], points[c])
                > self.collinearity_tolerance
                for a, b, c in combinations(quad, 3)
            ):
                return True
        return False

    # ------------------------------------------------------------------
    # Mapping
    # ------------------------------------------------------------------
    @property
    def coordinate_space(self) -> str:
        """The space this calibration produces."""
        return GROUND_PLANE_METERS

    @property
    def source_space(self) -> str:
        return IMAGE_PIXELS

    def image_to_world(self, point: Sequence[float]) -> GroundPoint:
        """Map an image pixel onto the ground plane."""
        x, y = self._to_world.apply(point)
        return GroundPoint(
            x=x,
            y=y,
            in_calibrated_region=self.contains_image_point(point),
            coordinate_space=self.coordinate_space,
            units=self.world_units,
        )

    def world_to_image(self, point: Sequence[float]) -> Point:
        """Map a ground-plane position back to an image pixel.

        Useful for drawing zones defined in metres onto a frame, and for
        checking a calibration by eye.
        """
        return self._to_image.apply(point)

    def image_points_to_world(
        self, points: Sequence[Sequence[float]]
    ) -> List[GroundPoint]:
        return [self.image_to_world(p) for p in points]

    def image_box_to_world(
        self, bbox: Sequence[float], anchor: str = ANCHOR_BOTTOM_CENTER
    ) -> GroundPoint:
        """Map a bounding box onto the floor via one anchor point.

        Defaults to the box's bottom-centre — the same ground-contact proxy
        zones use — because that is the only part of an upright object that is
        plausibly *on* the calibrated plane. Using the centroid would project
        a point floating in mid-air onto the floor and report the result as a
        position, which is exactly the error this milestone exists to avoid.
        """
        return self.image_to_world(anchor_point(bbox, anchor))

    def contains_image_point(self, point: Sequence[float]) -> bool:
        """Is this pixel inside the region the calibration was fitted over?"""
        if self._image_zone is None:
            return False
        return self._image_zone.contains_point(point)

    def contains_world_point(self, point: Sequence[float]) -> bool:
        if self._world_hull_zone is None:
            return False
        return self._world_hull_zone.contains_point(point)

    # ------------------------------------------------------------------
    # Quality
    # ------------------------------------------------------------------
    def _measure_quality(self) -> CalibrationQuality:
        world_errors = [
            euclidean_distance(self._to_world.apply(image), world)
            for image, world in zip(self.image_points, self.world_points)
        ]
        image_errors = [
            euclidean_distance(self._to_image.apply(world), image)
            for image, world in zip(self.image_points, self.world_points)
        ]
        return CalibrationQuality(
            point_count=len(self.image_points),
            rms_image_error_px=_rms(image_errors),
            max_image_error_px=max(image_errors),
            rms_world_error=_rms(world_errors),
            max_world_error=max(world_errors),
            world_units=self.world_units,
        )

    def limitations(self) -> List[str]:
        """What this calibration cannot tell you. Reported alongside results."""
        notes = [
            "Exact only for points ON the calibrated plane. A person's head, a "
            "raised forklift tine or anything off the floor projects to the "
            "wrong ground position.",
            "Accuracy is bounded by the accuracy of the supplied world points. "
            "Residuals below measure self-consistency, not survey accuracy.",
            "Valid only for the camera pose it was measured at. Any pan, tilt, "
            "zoom or re-mount invalidates it silently.",
            "Outside the calibrated region results are extrapolated; points are "
            "flagged with in_calibrated_region=False.",
            "Assumes a flat floor. Ramps, steps and gradients are not modelled.",
            "No lens-distortion model. Wide-angle or fisheye lenses need "
            "undistortion first or the edges of the frame will be wrong.",
        ]
        if self.quality.is_exactly_determined:
            notes.append(
                "Fitted from exactly 4 points, so the residual is necessarily "
                "zero and validates nothing. Supply 5+ points to detect a bad "
                "correspondence."
            )
        return notes

    # ------------------------------------------------------------------
    def metadata(self) -> Dict[str, Any]:
        """Everything a consumer needs to judge this calibration."""
        return {
            "name": self.name,
            "description": self.description,
            "source_space": self.source_space,
            "coordinate_space": self.coordinate_space,
            "world_units": self.world_units,
            "point_count": len(self.image_points),
            "quality": self.quality.to_dict(),
            "image_region_px": [[round(x, 2), round(y, 2)] for x, y in self.image_region],
            "world_region": [[round(x, 4), round(y, 4)] for x, y in self.world_region],
            "limitations": self.limitations(),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "world_units": self.world_units,
            "image_points": [[x, y] for x, y in self.image_points],
            "world_points": [[x, y] for x, y in self.world_points],
            "homography_image_to_world": self._to_world.to_list(),
            "metadata": self.metadata(),
        }

    def __repr__(self) -> str:
        return (
            f"GroundPlaneCalibration(name={self.name!r}, "
            f"points={len(self.image_points)}, space={self.coordinate_space!r})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _as_points(raw: Sequence[Sequence[float]], label: str) -> List[Point]:
    points: List[Point] = []
    for index, value in enumerate(raw):
        if len(value) != 2:
            raise ValueError(
                f"{label}[{index}] must be a 2D point, got {len(value)} values"
            )
        points.append((float(value[0]), float(value[1])))
    return points


def _reject_duplicates(points: Sequence[Point], label: str) -> None:
    seen: Dict[Point, int] = {}
    for index, point in enumerate(points):
        if point in seen:
            raise DegenerateConfigurationError(
                f"Duplicate {label} point {point} at indices {seen[point]} and "
                f"{index}: the same location cannot constrain two correspondences."
            )
        seen[point] = index


def _zone_from_hull(name: str, hull: Sequence[Point]) -> Optional[Zone]:
    if len(hull) < 3:
        return None
    return Zone(name=name, polygon=tuple(hull))


def _rms(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return (sum(v * v for v in values) / len(values)) ** 0.5
