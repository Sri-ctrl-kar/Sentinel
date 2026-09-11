"""Tests for video ingestion (requires OpenCV)."""

import pytest

pytest.importorskip("cv2")

from app.perception.video import VideoSource  # noqa: E402


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        VideoSource("/nonexistent/clip.mp4")


def test_invalid_stride_raises(demo_video):
    with pytest.raises(ValueError, match="stride must be"):
        VideoSource(demo_video, stride=0)


def test_metadata_matches_the_generated_clip(demo_video):
    with VideoSource(demo_video) as source:
        assert source.metadata.width == 640
        assert source.metadata.height == 384
        assert source.metadata.fps == pytest.approx(20.0, abs=0.1)
        assert source.metadata.duration == pytest.approx(4.0, abs=0.2)


def test_frames_have_monotonic_timestamps(demo_video):
    with VideoSource(demo_video) as source:
        frames = list(source)

    assert len(frames) > 0
    timestamps = [f.timestamp for f in frames]
    assert timestamps == sorted(timestamps)
    assert frames[0].timestamp < 0.1


def test_frames_carry_image_dimensions(demo_video):
    with VideoSource(demo_video, max_frames=3) as source:
        frames = list(source)
    assert all(f.width == 640 and f.height == 384 for f in frames)
    assert frames[0].image.shape == (384, 640, 3)


def test_max_frames_limits_the_stream(demo_video):
    with VideoSource(demo_video, max_frames=5) as source:
        assert len(list(source)) == 5


def test_stride_skips_frames_but_keeps_real_timestamps(demo_video):
    with VideoSource(demo_video) as source:
        full = list(source)
    with VideoSource(demo_video, stride=4) as source:
        strided = list(source)

    assert len(strided) == pytest.approx(len(full) / 4, abs=1)
    # Timestamps must still reflect real video time, not processed-frame index.
    assert strided[-1].timestamp == pytest.approx(full[-1].timestamp, abs=0.3)
    assert [f.index for f in strided[:3]] == [0, 4, 8]


def test_to_dict_is_json_shaped(demo_video):
    with VideoSource(demo_video) as source:
        payload = source.metadata.to_dict()
    assert set(payload) >= {"path", "width", "height", "fps", "frame_count"}
