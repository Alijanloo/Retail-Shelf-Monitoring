from typing import Dict, Optional

from ...entities.stream import StreamConfig
from ..logging_config import get_logger

logger = get_logger(__name__)


class StreamManager:
    def __init__(self):
        self.streams: Dict[str, StreamConfig] = {}
        self.active_streams: Dict[str, bool] = {}

    def register_stream(self, stream_config: StreamConfig):
        self.streams[stream_config.stream_id] = stream_config
        self.active_streams[stream_config.stream_id] = stream_config.active

        logger.info(f"Registered stream: {stream_config.stream_id}")

    def unregister_stream(self, stream_id: str):
        if stream_id in self.streams:
            del self.streams[stream_id]
            del self.active_streams[stream_id]
            logger.info(f"Unregistered stream: {stream_id}")
        else:
            logger.warning(f"Stream {stream_id} not found for unregistration")

    def get_stream(self, stream_id: str) -> Optional[StreamConfig]:
        return self.streams.get(stream_id)

    def get_all_streams(self) -> Dict[str, StreamConfig]:
        return self.streams.copy()

    def get_active_streams(self) -> Dict[str, StreamConfig]:
        return {
            stream_id: config
            for stream_id, config in self.streams.items()
            if self.active_streams.get(stream_id, False)
        }

    def activate_stream(self, stream_id: str):
        if stream_id in self.streams:
            self.active_streams[stream_id] = True
            self.streams[stream_id].active = True
            logger.info(f"Activated stream: {stream_id}")
        else:
            logger.warning(f"Stream {stream_id} not found for activation")

    def deactivate_stream(self, stream_id: str):
        if stream_id in self.streams:
            self.active_streams[stream_id] = False
            self.streams[stream_id].active = False
            logger.info(f"Deactivated stream: {stream_id}")
        else:
            logger.warning(f"Stream {stream_id} not found for deactivation")
