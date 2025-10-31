import asyncio
import queue
import threading
import time
from datetime import datetime, timezone

from PySide6.QtCore import QThread, Signal

from retail_shelf_monitoring.entities.frame import Frame
from retail_shelf_monitoring.frameworks.logging_config import get_logger
from retail_shelf_monitoring.usecases.stream_processing import StreamProcessingUseCase

logger = get_logger(__name__)


class InferenceThread(QThread):
    detection_ready_signal = Signal(list, Frame)
    error_signal = Signal(str)
    latency_signal = Signal(float)

    def __init__(
        self,
        frame_queue: queue.Queue,
        stream_processing_use_case: StreamProcessingUseCase,
    ):
        super().__init__()
        self.frame_queue = frame_queue
        self.stream_processing_use_case = stream_processing_use_case
        self._stop_event = threading.Event()
        self._frame_counter = 0

    def run(self):
        logger.info("Started inference thread")

        while not self._stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                start_time = time.time()

                frame_id = f"frame_{self._frame_counter}"
                self._frame_counter += 1

                result = asyncio.run(
                    self.stream_processing_use_case.process_detections(
                        frame_img=frame,
                        frame_id=frame_id,
                        timestamp=datetime.now(timezone.utc),
                    )
                )

                latency_ms = (time.time() - start_time) * 1000
                self.latency_signal.emit(latency_ms)

                if result.success and result.detections:
                    self.detection_ready_signal.emit(result.detections, result.frame)
                elif result.success:
                    logger.debug(f"No detections for frame {frame_id}")
                else:
                    logger.debug(
                        f"Frame processing failed: {result.reason or 'unknown'}"
                    )

            except Exception as e:
                error_msg = f"Inference error: {str(e)}"
                logger.error(error_msg, exc_info=True)
                self.error_signal.emit(error_msg)

        logger.info("Inference thread stopped")

    def stop(self):
        self._stop_event.set()
        self.wait(2000)
