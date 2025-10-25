import pytest

from retail_shelf_monitoring.entities.stream import StreamConfig
from retail_shelf_monitoring.frameworks.streaming.frame_buffer import FrameBuffer
from retail_shelf_monitoring.frameworks.streaming.stream_manager import StreamManager


class TestFrameBuffer:
    @pytest.fixture
    def buffer(self):
        return FrameBuffer(maxsize=10)

    def test_frame_buffer_creation(self, buffer):
        assert buffer.is_empty() is True
        assert buffer.size() == 0

    def test_put_and_get(self, buffer):
        buffer.put({"data": "test"})
        assert buffer.size() == 1

        item = buffer.get()
        assert item["data"] == "test"
        assert buffer.is_empty() is True

    def test_buffer_stats(self, buffer):
        buffer.put({"frame": 1})
        buffer.put({"frame": 2})

        stats = buffer.get_stats()
        assert stats["total_added"] == 2
        assert stats["current_size"] == 2

    def test_clear_buffer(self, buffer):
        buffer.put({"frame": 1})
        buffer.put({"frame": 2})

        buffer.clear()

        assert buffer.is_empty() is True


class TestStreamManager:
    @pytest.fixture
    def manager(self):
        return StreamManager()

    @pytest.fixture
    def stream_config(self):
        return StreamConfig(
            stream_id="stream-1", source_url="rtsp://example.com/stream"
        )

    def test_register_stream(self, manager, stream_config):
        manager.register_stream(stream_config)

        stream = manager.get_stream("stream-1")
        assert stream is not None
        assert stream.stream_id == "stream-1"

    def test_unregister_stream(self, manager, stream_config):
        manager.register_stream(stream_config)
        manager.unregister_stream("stream-1")

        stream = manager.get_stream("stream-1")
        assert stream is None

    def test_activate_deactivate_stream(self, manager, stream_config):
        manager.register_stream(stream_config)

        manager.deactivate_stream("stream-1")
        active = manager.get_active_streams()
        assert len(active) == 0

        manager.activate_stream("stream-1")
        active = manager.get_active_streams()
        assert len(active) == 1

    def test_get_all_streams(self, manager):
        config1 = StreamConfig(stream_id="stream-1", source_url="url1")
        config2 = StreamConfig(stream_id="stream-2", source_url="url2")

        manager.register_stream(config1)
        manager.register_stream(config2)

        all_streams = manager.get_all_streams()
        assert len(all_streams) == 2
