"""Tests for image-space geometry and zones."""

import json

import pytest

from app.spatial import (
    ANCHOR_BOTTOM_CENTER,
    ANCHOR_CENTER,
    IMAGE_PIXELS,
    Zone,
    ZoneSet,
    anchor_point,
    box_bottom_center,
    box_center,
    pixel_distance,
)


# ---------------------------------------------------------------------------
# Anchors and distance
# ---------------------------------------------------------------------------
def test_box_center():
    assert box_center((0, 0, 40, 100)) == (20.0, 50.0)


def test_box_bottom_center_is_the_ground_contact_proxy():
    assert box_bottom_center((0, 0, 40, 100)) == (20.0, 100.0)


def test_anchor_point_selects_the_right_policy():
    box = (0, 0, 40, 100)
    assert anchor_point(box, ANCHOR_CENTER) == (20.0, 50.0)
    assert anchor_point(box, ANCHOR_BOTTOM_CENTER) == (20.0, 100.0)


def test_unknown_anchor_raises():
    with pytest.raises(ValueError, match="Unknown anchor"):
        anchor_point((0, 0, 1, 1), "middle_left")


def test_pixel_distance():
    assert pixel_distance((0, 0), (3, 4)) == 5.0


# ---------------------------------------------------------------------------
# Zone construction
# ---------------------------------------------------------------------------
def test_rect_zone_normalises_corner_order():
    zone = Zone.from_rect("bay", (300, 300, 100, 100))
    assert zone.bounds == (100.0, 100.0, 300.0, 300.0)


def test_zone_defaults_to_pixel_space():
    assert Zone.from_rect("bay", (0, 0, 10, 10)).coordinate_space == IMAGE_PIXELS


def test_zone_rejects_a_non_pixel_coordinate_space():
    with pytest.raises(ValueError, match="only 'image_pixels'"):
        Zone(name="bay", polygon=((0, 0), (1, 0), (1, 1)), coordinate_space="world_metres")


def test_zone_needs_three_points():
    with pytest.raises(ValueError, match="at least 3 points"):
        Zone(name="line", polygon=((0, 0), (1, 1)))


def test_zone_rejects_an_unknown_anchor():
    with pytest.raises(ValueError, match="Unknown anchor"):
        Zone.from_rect("bay", (0, 0, 10, 10), anchor="nose")


# ---------------------------------------------------------------------------
# Membership
# ---------------------------------------------------------------------------
def test_rect_membership():
    zone = Zone.from_rect("bay", (100, 100, 300, 300))
    assert zone.contains_point((200, 200))
    assert not zone.contains_point((50, 200))
    assert not zone.contains_point((200, 400))


def test_boundary_points_are_inside():
    """An entity resting on a zone edge must not flicker in and out."""
    zone = Zone.from_rect("bay", (100, 100, 300, 300))
    assert zone.contains_point((100, 200))
    assert zone.contains_point((200, 300))
    assert zone.contains_point((100, 100))


def test_concave_polygon_membership():
    # An L-shape: the notch must read as outside.
    zone = Zone(
        name="L",
        polygon=((0, 0), (100, 0), (100, 40), (40, 40), (40, 100), (0, 100)),
    )
    assert zone.contains_point((20, 20))
    assert zone.contains_point((80, 20))
    assert zone.contains_point((20, 80))
    assert not zone.contains_point((80, 80))


def test_contains_box_uses_the_bottom_edge_by_default():
    zone = Zone.from_rect("bay", (0, 200, 400, 400))
    tall = (100, 50, 140, 250)  # centre is above the zone, feet are inside
    assert zone.contains_box(tall)
    assert not Zone.from_rect("bay", (0, 200, 400, 400), anchor=ANCHOR_CENTER).contains_box(tall)


# ---------------------------------------------------------------------------
# ZoneSet
# ---------------------------------------------------------------------------
def test_zone_set_reports_every_containing_zone():
    zones = ZoneSet(
        [Zone.from_rect("bay", (0, 0, 200, 200)), Zone.from_rect("aisle", (100, 0, 300, 200))]
    )
    assert zones.containing_point((150, 100)) == ["bay", "aisle"]
    assert zones.containing_point((50, 100)) == ["bay"]
    assert zones.containing_point((900, 900)) == []


def test_zone_set_rejects_duplicate_names():
    with pytest.raises(ValueError, match="Duplicate zone names"):
        ZoneSet([Zone.from_rect("bay", (0, 0, 1, 1)), Zone.from_rect("bay", (2, 2, 3, 3))])


def test_empty_zone_set_is_falsy():
    assert not ZoneSet()
    assert ZoneSet([Zone.from_rect("bay", (0, 0, 1, 1))])


def test_zone_set_lookup_by_name():
    zones = ZoneSet([Zone.from_rect("bay", (0, 0, 1, 1))])
    assert zones.get("bay").name == "bay"
    assert zones.get("missing") is None


# ---------------------------------------------------------------------------
# Config round trip
# ---------------------------------------------------------------------------
def test_zone_set_from_config_supports_rect_and_polygon():
    zones = ZoneSet.from_config(
        {
            "coordinate_space": IMAGE_PIXELS,
            "zones": [
                {"name": "bay", "rect": [0, 0, 100, 100]},
                {"name": "ramp", "polygon": [[0, 0], [50, 0], [50, 50]], "anchor": "center"},
            ],
        }
    )
    assert zones.names == ["bay", "ramp"]
    assert zones.get("ramp").anchor == ANCHOR_CENTER


def test_zone_config_rejects_a_world_space_declaration():
    with pytest.raises(ValueError, match="only 'image_pixels'"):
        ZoneSet.from_config({"coordinate_space": "world_metres", "zones": []})


def test_zone_set_json_round_trip(tmp_path):
    zones = ZoneSet([Zone.from_rect("bay", (10, 20, 110, 120))])
    path = tmp_path / "zones.json"
    path.write_text(json.dumps(zones.to_dict()), encoding="utf-8")

    reloaded = ZoneSet.load_json(str(path))
    assert reloaded.names == ["bay"]
    assert reloaded.get("bay").bounds == (10.0, 20.0, 110.0, 120.0)
