"""Tests for the perception data types."""

from app.perception.types import Detection, Track


def test_detection_normalises_inverted_bbox():
    # Backends disagree on corner ordering; the type must not.
    det = Detection(bbox=(50, 90, 10, 10), confidence=0.5, class_id=0, class_name="person")
    assert det.bbox == (10.0, 10.0, 50.0, 90.0)


def test_detection_geometry():
    det = Detection(bbox=(10, 20, 50, 120), confidence=0.5, class_id=0, class_name="person")
    assert det.width == 40
    assert det.height == 100
    assert det.area == 4000
    assert det.center == (30.0, 70.0)


def test_detection_to_dict_is_json_shaped():
    det = Detection(bbox=(1.234, 2, 3, 4), confidence=0.987654, class_id=2, class_name="car")
    payload = det.to_dict()
    assert payload["bbox"] == [1.23, 2.0, 3.0, 4.0]
    assert payload["confidence"] == 0.9877
    assert payload["class_name"] == "car"


def test_track_entity_id_is_class_scoped():
    track = Track(track_id=7, bbox=(0, 0, 10, 10), confidence=0.8, class_id=0, class_name="person")
    assert track.entity_id == "person_7"


def test_track_entity_id_handles_multiword_classes():
    track = Track(track_id=2, bbox=(0, 0, 10, 10), confidence=0.8, class_id=9, class_name="traffic light")
    assert track.entity_id == "traffic_light_2"


def test_track_speed_from_velocity():
    track = Track(
        track_id=1,
        bbox=(0, 0, 10, 10),
        confidence=0.8,
        class_id=0,
        class_name="person",
        velocity=(3.0, 4.0),
    )
    assert track.speed == 5.0
