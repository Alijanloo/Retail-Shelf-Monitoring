import queue
import threading
import time
import uuid
from datetime import datetime

from PySide6.QtCore import QThread, Signal

from retail_shelf_monitoring.entities.common import BoundingBox
from retail_shelf_monitoring.entities.detection import Detection
from retail_shelf_monitoring.frameworks.logging_config import get_logger
from retail_shelf_monitoring.usecases.detection_processing import (
    DetectionProcessingUseCase,
)

logger = get_logger(__name__)


class InferenceThread(QThread):
    result_signal = Signal(list, str)
    error_signal = Signal(str)
    latency_signal = Signal(float)

    def __init__(
        self,
        frame_queue: queue.Queue,
        detection_use_case: DetectionProcessingUseCase,
        conf_threshold=0.35,
        shelf_id=None,
    ):
        super().__init__()
        self.frame_queue = frame_queue
        self.detection_use_case = detection_use_case
        self.conf_threshold = conf_threshold
        self.shelf_id = shelf_id or "unknown"
        self._stop_event = threading.Event()

    def run(self):
        logger.info(f"Started inference thread for shelf {self.shelf_id}")

        while not self._stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                start_time = time.time()

                raw_detections = self.detection_use_case.detector.detect(frame)

                if raw_detections:
                    tracker = self.detection_use_case.tracker
                    tracked_detections = tracker.update(raw_detections)

                    detection_objects = []
                    for det in tracked_detections:
                        if det["confidence"] >= self.conf_threshold:
                            x1, y1, x2, y2 = det["bbox"]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(frame.shape[1], x2)
                            y2 = min(frame.shape[0], y2)

                            if x2 > x1 and y2 > y1:
                                cropped_image = frame[y1:y2, x1:x2]
                                sku_detector = self.detection_use_case.sku_detector
                                sku_id = sku_detector.get_sku_id(cropped_image)
                            else:
                                logger.warning(f"Invalid bbox: {det['bbox']}")
                                sku_id = "invalid_bbox"

                            detection = Detection(
                                detection_id=str(uuid.uuid4()),
                                shelf_id=self.shelf_id,
                                frame_timestamp=datetime.utcnow(),
                                bbox=BoundingBox(
                                    x1=det["bbox"][0],
                                    y1=det["bbox"][1],
                                    x2=det["bbox"][2],
                                    y2=det["bbox"][3],
                                ),
                                class_id=det["class_id"],
                                sku_id=sku_id,
                                confidence=det["confidence"],
                                track_id=det.get("track_id"),
                            )
                            detection_objects.append(detection)

                    filtered_detections = detection_objects
                else:
                    filtered_detections = []

                latency_ms = (time.time() - start_time) * 1000
                self.latency_signal.emit(latency_ms)

                self.result_signal.emit(filtered_detections, self.shelf_id)

            except Exception as e:
                error_msg = f"Inference error: {str(e)}"
                logger.error(error_msg, exc_info=True)
                self.error_signal.emit(error_msg)

        logger.info(f"Inference thread stopped for shelf {self.shelf_id}")

    def stop(self):
        self._stop_event.set()
        self.wait(2000)

    def update_confidence(self, conf_threshold: float):
        self.conf_threshold = conf_threshold
        logger.info(f"Updated confidence threshold to {conf_threshold:.2f}")
