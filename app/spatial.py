"""Image-space geometry: coordinate conventions and zones.

Scope warning — read this before using any distance from this module
--------------------------------------------------------------------
Every coordinate here is in **image pixel space**: the origin is the top-left
of the decoded frame, ``x`` grows right and ``y`` grows down. Nothing in
Sentinel currently knows the camera's intrinsics, its height, its tilt, or the
scene's ground plane, so:

* A pixel distance is **not** a real-world distance. Two objects 50px apart
  near the camera may be metres apart, while 50px at the horizon may be tens
  of metres.
* Pixel speed is **not** physical speed, and pixel area is not physical size.
* A pixel distance is only comparable to another pixel distance at roughly the
  same depth in the same frame.

Zones inherit that limitation: a zone is a region of the *image*, not a region
of the world. That is deliberate and sufficient for M0.2 — a fixed camera's
loading bay is a fixed polygon on screen. Real-world units require a homography
or camera calibration, which is not in scope.

This module has no dependency on any model, framework or perception code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

#: Raw image space: pixels, origin top-left of the decoded frame. Everything
#: perception produces lives here, and it is the only space available without
#: a calibration.
IMAGE_PIXELS = "image_pixels"

#: Metric ground-plane space, produced by :mod:`app.calibration` from a
#: homography. Coordinates are metres on the floor plane. This name exists so
#: that a metric measurement can never be mistaken for a pixel one: a value is
#: only in metres if the thing carrying it says ``ground_plane_meters``.
GROUND_PLANE_METERS = "ground_plane_meters"

#: The world unit M0.4 supports. Calibration explicitly refuses anything else
#: rather than silently relabelling unknown units as metres.
UNIT_METERS = "meters"

#: Every coordinate space Sentinel understands.
COORDINATE_SPACES = (IMAGE_PIXELS, GROUND_PLANE_METERS)

Point = Tuple[float, float]
Box = Tuple[float, float, float, float]  # (x1, y1, x2, y2)

#: How a zone derives the single test point it uses for a bounding box.
ANCHOR_CENTER = "center"
ANCHOR_BOTTOM_CENTER = "bottom_center"
ANCHORS = (ANCHOR_CENTER, ANCHOR_BOTTOM_CENTER)


def box_center(box: Sequence[float]) -> Point:
    x1, y1, x2, y2 = (float(v) for v in box)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def box_bottom_center(box: Sequence[float]) -> Point:
    """The mid-point of a box's lower edge.

    For an upright object standing on a surface this is the closest cheap
    proxy for where it touches the ground, which makes zone membership behave
    far more sensibly than the centroid does for tall objects. It is still a
    pixel-space heuristic, not a ground-plane projection.
    """
    x1, _y1, x2, y2 = (float(v) for v in box)
    return ((x1 + x2) / 2.0, y2)


def anchor_point(box: Sequence[float], anchor: str = ANCHOR_BOTTOM_CENTER) -> Point:
    if anchor == ANCHOR_CENTER:
        return box_center(box)
    if anchor == ANCHOR_BOTTOM_CENTER:
        return box_bottom_center(box)
    raise ValueError(f"Unknown anchor '{anchor}'. Expected one of {ANCHORS}")


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance between two 2D points, in whatever units they carry.

    Unit-agnostic on purpose: the caller knows which coordinate space its
    points are in, and the named wrappers below make that explicit at the call
    site so no result is ever ambiguous.
    """
    return ((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2) ** 0.5


def pixel_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance **in image pixels**. See the module docstring."""
    return euclidean_distance(a, b)


def ground_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance **in ground-plane metres**.

    Only meaningful for points produced by a
    :class:`~app.calibration.planar.GroundPlaneCalibration`. Passing pixel
    coordinates here produces a number that is wrong in a way nothing can
    detect, which is exactly why the two wrappers are named differently.
    """
    return euclidean_distance(a, b)


def convex_hull(points: Sequence[Point]) -> List[Point]:
    """Convex hull of a point set, counter-clockwise (monotone chain).

    Used to describe the region a calibration was actually fitted over, so a
    mapped point can be reported as inside it or extrapolated beyond it.
    """
    unique = sorted({(float(x), float(y)) for x, y in points})
    if len(unique) <= 2:
        return list(unique)

    def half(source):
        built: List[Point] = []
        for point in source:
            while len(built) >= 2 and _cross(built[-2], built[-1], point) <= 0:
                built.pop()
            built.append(point)
        return built[:-1]

    return half(unique) + half(list(reversed(unique)))


def _cross(o: Point, a: Point, b: Point) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def triangle_sine(a: Point, b: Point, c: Point) -> float:
    """``|sin|`` of the angle at ``a`` between ``ab`` and ``ac``, in ``[0, 1]``.

    A scale-free collinearity measure: 0 means the three points lie on a line.
    Scale-free matters because the same three points must read as equally
    collinear whether they are metres or pixels apart.
    """
    abx, aby = b[0] - a[0], b[1] - a[1]
    acx, acy = c[0] - a[0], c[1] - a[1]
    ab = (abx * abx + aby * aby) ** 0.5
    ac = (acx * acx + acy * acy) ** 0.5
    if ab <= 0 or ac <= 0:
        return 0.0
    return abs(abx * acy - aby * acx) / (ab * ac)


@dataclass(frozen=True)
class Zone:
    """A named polygonal region of the image.

    Parameters
    ----------
    name:
        Stable identifier used in ``entered_zone`` / ``exited_zone`` events.
    polygon:
        Vertices in pixel space, in order. Three points minimum.
    anchor:
        Which point of an entity's bounding box is tested for membership.
    coordinate_space:
        Always :data:`IMAGE_PIXELS` today; carried explicitly so that a
        world-space zone can never be mistaken for a pixel-space one.
    """

    name: str
    polygon: Tuple[Point, ...]
    anchor: str = ANCHOR_BOTTOM_CENTER
    coordinate_space: str = IMAGE_PIXELS
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        polygon = tuple((float(x), float(y)) for x, y in self.polygon)
        if len(polygon) < 3:
            raise ValueError(f"Zone '{self.name}' needs at least 3 points")
        if self.anchor not in ANCHORS:
            raise ValueError(f"Unknown anchor '{self.anchor}'. Expected one of {ANCHORS}")
        if self.coordinate_space != IMAGE_PIXELS:
            raise ValueError(
                f"Zone '{self.name}': only '{IMAGE_PIXELS}' is supported at M0.2"
            )
        object.__setattr__(self, "polygon", polygon)

    # ------------------------------------------------------------------
    @classmethod
    def from_rect(
        cls,
        name: str,
        rect: Sequence[float],
        anchor: str = ANCHOR_BOTTOM_CENTER,
        **kwargs: Any,
    ) -> "Zone":
        """Build an axis-aligned rectangular zone from ``(x1, y1, x2, y2)``."""
        x1, y1, x2, y2 = (float(v) for v in rect)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        return cls(
            name=name,
            polygon=((x1, y1), (x2, y1), (x2, y2), (x1, y2)),
            anchor=anchor,
            **kwargs,
        )

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Zone":
        name = payload["name"]
        anchor = payload.get("anchor", ANCHOR_BOTTOM_CENTER)
        metadata = payload.get("metadata", {})
        if "rect" in payload:
            return cls.from_rect(name, payload["rect"], anchor=anchor, metadata=metadata)
        if "polygon" not in payload:
            raise ValueError(f"Zone '{name}' needs either 'rect' or 'polygon'")
        return cls(
            name=name,
            polygon=tuple(tuple(p) for p in payload["polygon"]),
            anchor=anchor,
            metadata=metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "polygon": [[round(x, 2), round(y, 2)] for x, y in self.polygon],
            "anchor": self.anchor,
            "coordinate_space": self.coordinate_space,
        }

    # ------------------------------------------------------------------
    @property
    def bounds(self) -> Box:
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        return (min(xs), min(ys), max(xs), max(ys))

    def contains_point(self, point: Sequence[float]) -> bool:
        """Ray-casting point-in-polygon test (pixel space).

        Points exactly on an edge are treated as inside, so that an entity
        resting on a zone boundary does not flicker in and out.
        """
        x, y = float(point[0]), float(point[1])
        polygon = self.polygon
        count = len(polygon)

        for i in range(count):
            ax, ay = polygon[i]
            bx, by = polygon[(i + 1) % count]
            if _point_on_segment(x, y, ax, ay, bx, by):
                return True

        inside = False
        for i in range(count):
            ax, ay = polygon[i]
            bx, by = polygon[(i + 1) % count]
            # Does a ray cast in +x from (x, y) cross this edge?
            if (ay > y) != (by > y):
                crossing_x = ax + (y - ay) * (bx - ax) / (by - ay)
                if crossing_x > x:
                    inside = not inside
        return inside

    def contains_box(self, box: Sequence[float]) -> bool:
        """Whether the box's anchor point lies inside the zone."""
        return self.contains_point(anchor_point(box, self.anchor))


def _point_on_segment(
    px: float, py: float, ax: float, ay: float, bx: float, by: float, eps: float = 1e-9
) -> bool:
    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    if abs(cross) > eps:
        return False
    return min(ax, bx) - eps <= px <= max(ax, bx) + eps and (
        min(ay, by) - eps <= py <= max(ay, by) + eps
    )


class ZoneSet:
    """An ordered collection of zones, evaluated together."""

    def __init__(self, zones: Iterable[Zone] = ()) -> None:
        self.zones: List[Zone] = list(zones)
        names = [z.name for z in self.zones]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(f"Duplicate zone names: {sorted(duplicates)}")

    def __len__(self) -> int:
        return len(self.zones)

    def __iter__(self):
        return iter(self.zones)

    def __bool__(self) -> bool:
        return bool(self.zones)

    @property
    def names(self) -> List[str]:
        return [z.name for z in self.zones]

    def get(self, name: str) -> Optional[Zone]:
        for zone in self.zones:
            if zone.name == name:
                return zone
        return None

    def containing_box(self, box: Sequence[float]) -> List[str]:
        """Names of every zone whose region contains the box's anchor point.

        Zones may overlap, so an entity can be in several at once.
        """
        return [z.name for z in self.zones if z.contains_box(box)]

    def containing_point(self, point: Sequence[float]) -> List[str]:
        return [z.name for z in self.zones if z.contains_point(point)]

    # ------------------------------------------------------------------
    @classmethod
    def from_dicts(cls, payloads: Iterable[Dict[str, Any]]) -> "ZoneSet":
        return cls(Zone.from_dict(p) for p in payloads)

    @classmethod
    def from_config(cls, payload: Dict[str, Any]) -> "ZoneSet":
        """Build from a ``{"zones": [...]}`` document."""
        space = payload.get("coordinate_space", IMAGE_PIXELS)
        if space != IMAGE_PIXELS:
            raise ValueError(
                f"Zone config declares coordinate_space '{space}'; "
                f"only '{IMAGE_PIXELS}' is supported at M0.2"
            )
        return cls.from_dicts(payload.get("zones", []))

    @classmethod
    def load_json(cls, path: str) -> "ZoneSet":
        import json

        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_config(json.load(handle))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coordinate_space": IMAGE_PIXELS,
            "zones": [z.to_dict() for z in self.zones],
        }
