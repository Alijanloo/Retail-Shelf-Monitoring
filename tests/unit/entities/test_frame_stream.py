from datetime import datetime

import pytest

from retail_shelf_monitoring.entities.frame import AlignedFrame, Frame
from retail_shelf_monitoring.entities.stream import StreamConfig


class TestFrame:
    def test_frame_creation(self):
        frame = Frame(
            frame_id="test-frame-1",
            stream_id="stream-1",
            timestamp=datetime.now(),
            frame_number=1,
            width=1920,
            height=1080,
            is_keyframe=True,
        )

        assert frame.frame_id == "test-frame-1"
        assert frame.stream_id == "stream-1"
        assert frame.width == 1920
        assert frame.height == 1080
        assert frame.is_keyframe is True

    def test_frame_with_homography(self):
        frame = Frame(
            frame_id="test-frame-2",
            stream_id="stream-1",
            timestamp=datetime.now(),
            frame_number=2,
            width=1920,
            height=1080,
            shelf_id="shelf-1",
            homography_matrix=[1, 0, 0, 0, 1, 0, 0, 0, 1],
            alignment_confidence=0.85,
            inlier_ratio=0.75,
        )

        assert frame.shelf_id == "shelf-1"
        assert frame.alignment_confidence == 0.85
        assert frame.inlier_ratio == 0.75
        assert len(frame.homography_matrix) == 9

    def test_frame_validation(self):
        with pytest.raises(Exception):
            Frame(
                frame_id="test-frame-3",
                stream_id="stream-1",
                timestamp=datetime.now(),
                frame_number=-1,
                width=1920,
                height=1080,
            )


class TestAlignedFrame:
    def test_aligned_frame_creation(self):
        frame = Frame(
            frame_id="test-frame-1",
            stream_id="stream-1",
            timestamp=datetime.now(),
            frame_number=1,
            width=1920,
            height=1080,
        )

        aligned = AlignedFrame(
            frame=frame,
            aligned_image_path="/path/to/aligned.jpg",
            shelf_id="shelf-1",
            confidence=0.92,
        )

        assert aligned.frame.frame_id == "test-frame-1"
        assert aligned.shelf_id == "shelf-1"
        assert aligned.confidence == 0.92


class TestStreamConfig:
    def test_stream_config_creation(self):
        config = StreamConfig(
            stream_id="stream-1",
            source_url="rtsp://example.com/stream",
            fps=30.0,
            process_every_n_frames=15,
        )

        assert config.stream_id == "stream-1"
        assert config.fps == 30.0
        assert config.frame_interval == 0.5

    def test_stream_config_validation(self):
        with pytest.raises(ValueError, match="Source URL cannot be empty"):
            StreamConfig(stream_id="stream-2", source_url="")

    def test_stream_config_defaults(self):
        config = StreamConfig(stream_id="stream-3", source_url="video.mp4")

        assert config.fps == 30.0
        assert config.process_every_n_frames == 30
        assert config.max_width == 1920
        assert config.max_height == 1080
        assert config.active is True
