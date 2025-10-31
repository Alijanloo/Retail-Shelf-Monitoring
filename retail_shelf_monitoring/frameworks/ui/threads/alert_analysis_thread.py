import asyncio
import queue
import threading
import time

from PySide6.QtCore import QThread, Signal

from retail_shelf_monitoring.frameworks.logging_config import get_logger
from retail_shelf_monitoring.usecases.stream_processing import StreamProcessingUseCase

logger = get_logger(__name__)


class AlertAnalysisThread(QThread):
    cell_states_signal = Signal(str, list)
    error_signal = Signal(str)
    analysis_latency_signal = Signal(float)

    def __init__(
        self,
        detection_queue: queue.Queue,
        stream_processing_use_case: StreamProcessingUseCase,
    ):
        super().__init__()
        self.detection_queue = detection_queue
        self.stream_processing_use_case = stream_processing_use_case
        self._stop_event = threading.Event()

    def run(self):
        logger.info("Started alert analysis thread")

        while not self._stop_event.is_set():
            try:
                detection_data = self.detection_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            frame_id = detection_data["frame_id"]
            detections = detection_data["detections"]
            shelf_id = detection_data["shelf_id"]
            timestamp = detection_data["timestamp"]

            try:
                start_time = time.time()

                result = asyncio.run(
                    self.stream_processing_use_case.analyze_compliance(
                        shelf_id=shelf_id,
                        detections=detections,
                        timestamp=timestamp,
                    )
                )

                analysis_latency_ms = (time.time() - start_time) * 1000
                self.analysis_latency_signal.emit(analysis_latency_ms)

                if result.success and result.cell_states:
                    self.cell_states_signal.emit(shelf_id, result.cell_states)
                    logger.debug(
                        f"Analyzed frame {frame_id}: {len(result.cell_states)} "
                        "cell states"
                    )
                else:
                    logger.debug(
                        f"Compliance analysis failed for frame {frame_id}: "
                        f"{result.reason or 'unknown'}"
                    )

            except Exception as e:
                error_msg = f"Alert analysis error: {str(e)}"
                logger.error(error_msg, exc_info=True)
                self.error_signal.emit(error_msg)

        logger.info("Alert analysis thread stopped")

    def stop(self):
        self._stop_event.set()
        self.wait(2000)
