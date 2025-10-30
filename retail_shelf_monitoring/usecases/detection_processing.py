import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import List

from ..adaptors.ml.sku_detector import SKUDetector
from ..adaptors.ml.yolo_detector import YOLOv11Detector
from ..entities.common import BoundingBox
from ..entities.detection import Detection
from ..entities.frame import Frame
from ..frameworks.logging_config import get_logger

logger = get_logger(__name__)


class DetectionProcessingUseCase:
    def __init__(
        self,
        detector: YOLOv11Detector,
        sku_detector: SKUDetector,
    ):
        self.detector = detector
        self.sku_detector = sku_detector

    def process_aligned_frame(self, frame: Frame) -> List[Detection]:
        """
        Run detection on an aligned frame
        """
        detections = self.detector.detect(frame.frame_img)

        if not detections:
            logger.debug(f"No detections in frame {frame.frame_id}")
            return []

        def process_detection(det):
            x1, y1, x2, y2 = det["bbox"]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.frame_img.shape[1], x2)
            y2 = min(frame.frame_img.shape[0], y2)

            if x2 > x1 and y2 > y1:
                cropped_image = frame.frame_img[y1:y2, x1:x2]
                sku_id = self.sku_detector.get_sku_id(cropped_image)
            else:
                logger.warning(f"Invalid bbox for detection: {det['bbox']}")
                sku_id = "invalid_bbox"

            return Detection(
                detection_id=str(uuid.uuid4()),
                shelf_id=frame.shelf_id,
                frame_timestamp=frame.timestamp,
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

        with ThreadPoolExecutor() as executor:
            detection_entities = list(executor.map(process_detection, detections))

        return detection_entities
