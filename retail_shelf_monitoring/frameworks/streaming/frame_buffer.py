from queue import Empty, Queue
from threading import Lock
from typing import Any, Optional

from ..logging_config import get_logger

logger = get_logger(__name__)


class FrameBuffer:
    def __init__(self, maxsize: int = 100):
        self.queue = Queue(maxsize=maxsize)
        self.lock = Lock()
        self._total_added = 0
        self._total_dropped = 0

    def put(self, item: Any, block: bool = True, timeout: Optional[float] = None):
        try:
            with self.lock:
                self.queue.put(item, block=block, timeout=timeout)
                self._total_added += 1
        except Exception as e:
            with self.lock:
                self._total_dropped += 1
            logger.warning(f"Failed to add item to frame buffer: {e}")

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Any:
        try:
            return self.queue.get(block=block, timeout=timeout)
        except Empty:
            return None

    def clear(self):
        with self.lock:
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except Empty:
                    break

    def size(self) -> int:
        with self.lock:
            return self.queue.qsize()

    def is_empty(self) -> bool:
        with self.lock:
            return self.queue.empty()

    def get_stats(self) -> dict:
        with self.lock:
            return {
                "current_size": self.queue.qsize(),
                "total_added": self._total_added,
                "total_dropped": self._total_dropped,
            }
