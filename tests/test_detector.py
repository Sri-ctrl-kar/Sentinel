"""Tests for the detector interface, registry and filtering wrapper."""

import pytest

from app.perception.backends.mock import MockDetector
from app.perception.detector import (
    Detector,
    FilteredDetector,
    available_detectors,
    create_detector,
    register_detector,
)
from app.perception.types import Detection


def test_mock_backend_is_registered():
    assert "mock" in available_detectors()


def test_create_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown detector backend"):
        create_detector("does-not-exist")


def test_create_detector_returns_detector_instance():
    detector = create_detector("mock")
    assert isinstance(detector, Detector)
    assert detector.info().backend == "mock"


def test_scripted_mock_replays_frames():
    a = Detection((0, 0, 10, 10), 0.9, 0, "person")
    b = Detection((5, 5, 15, 15), 0.8, 2, "car")
    detector = MockDetector(script=[[a], [a, b], []])
    assert len(detector.detect(None)) == 1
    assert len(detector.detect(None)) == 2
    assert detector.detect(None) == []
    # Past the end of the script the last entry repeats.
    assert detector.detect(None) == []


def test_synthetic_mock_moves_objects_between_frames():
    detector = MockDetector()
    first = detector.detect(None)
    second = detector.detect(None)
    assert first[0].class_name == "person"
    assert second[0].bbox[0] > first[0].bbox[0]


def test_filtered_detector_drops_unwanted_classes():
    inner = MockDetector(
        script=[
            [
                Detection((0, 0, 10, 10), 0.9, 0, "person"),
                Detection((0, 0, 10, 10), 0.9, 58, "potted plant"),
            ]
        ]
    )
    detector = FilteredDetector(inner, classes=["person"])
    results = detector.detect(None)
    assert [d.class_name for d in results] == ["person"]


def test_filtered_detector_applies_confidence_floor():
    inner = MockDetector(
        script=[
            [
                Detection((0, 0, 10, 10), 0.9, 0, "person"),
                Detection((0, 0, 10, 10), 0.2, 0, "person"),
            ]
        ]
    )
    detector = FilteredDetector(inner, min_confidence=0.5)
    assert len(detector.detect(None)) == 1


def test_filtered_detector_applies_min_area():
    inner = MockDetector(
        script=[
            [
                Detection((0, 0, 100, 100), 0.9, 0, "person"),
                Detection((0, 0, 5, 5), 0.9, 0, "person"),
            ]
        ]
    )
    detector = FilteredDetector(inner, min_area=1000)
    assert len(detector.detect(None)) == 1


def test_create_detector_wraps_with_filter_when_asked():
    detector = create_detector("mock", classes=["person"], min_confidence=0.5)
    assert isinstance(detector, FilteredDetector)
    assert all(d.class_name == "person" for d in detector.detect(None))


def test_custom_backend_can_be_registered():
    class Stub(Detector):
        def detect(self, frame):
            return []

        def info(self):
            from app.perception.types import DetectorInfo

            return DetectorInfo(backend="stub", model="none", device="cpu")

    register_detector("stub-for-test", Stub)
    assert "stub-for-test" in available_detectors()
    assert create_detector("stub-for-test").detect(None) == []


def test_detector_is_a_context_manager():
    with create_detector("mock") as detector:
        assert detector.detect(None) is not None
