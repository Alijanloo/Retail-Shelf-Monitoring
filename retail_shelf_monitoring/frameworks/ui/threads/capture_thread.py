import threading
import time

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from retail_shelf_monitoring.frameworks.logging_config import get_logger

logger = get_logger(__name__)


class CaptureThread(QThread):
    frame_signal = Signal(np.ndarray, str)
    error_signal = Signal(str)
    stopped_signal = Signal()
    fps_signal = Signal(float)

    def __init__(self, source=0, fps_limit=None, shelf_id=None):
        super().__init__()
        self.source = source
        self.fps_limit = fps_limit
        self.shelf_id = shelf_id or "unknown"
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._frame_count = 0
        self._start_time = None

    def run(self):
        cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            error_msg = f"Cannot open video source: {self.source}"
            logger.error(error_msg)
            self.error_signal.emit(error_msg)
            self.stopped_signal.emit()
            return

        logger.info(f"Started capture from {self.source} (shelf: {self.shelf_id})")
        self._start_time = time.time()
        last_frame_time = 0

        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                time.sleep(0.1)
                continue

            ret, frame = cap.read()

            if not ret:
                logger.warning(f"End of stream or read error from {self.source}")
                break

            if self.fps_limit:
                current_time = time.time()
                elapsed_since_last = current_time - last_frame_time
                min_interval = 1.0 / self.fps_limit
                if elapsed_since_last < min_interval:
                    time.sleep(min_interval - elapsed_since_last)
                last_frame_time = current_time

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            self.frame_signal.emit(frame.copy(), timestamp)

            self._frame_count += 1

            if self._frame_count % 30 == 0:
                elapsed = time.time() - self._start_time
                current_fps = self._frame_count / elapsed
                self.fps_signal.emit(current_fps)

        cap.release()
        logger.info(f"Capture stopped for {self.source}")
        self.stopped_signal.emit()

    def stop(self):
        self._stop_event.set()
        self.wait(2000)

    def pause(self):
        self._pause_event.set()

    def resume(self):
        self._pause_event.clear()
