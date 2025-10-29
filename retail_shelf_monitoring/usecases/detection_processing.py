import uuid
from typing import List, Optional

from ..adaptors.ml.sku_detector import SKUDetector
from ..adaptors.ml.yolo_detector import YOLOv11Detector
from ..entities.common import BoundingBox
from ..entities.detection import Detection
from ..entities.frame import Frame
from ..frameworks.logging_config import get_logger
from .interfaces.tracker_interface import Tracker

logger = get_logger(__name__)


class DetectionProcessingUseCase:
    def __init__(
        self,
        detector: YOLOv11Detector,
        tracker: Tracker,
        sku_detector: SKUDetector,
    ):
        self.detector = detector
        self.tracker = tracker
        self.sku_detector = sku_detector

    def process_aligned_frame(
        self,
        aligned_image,
        frame_metadata: Frame,
        shelf_id: str,
        aligned_frame_path: Optional[str] = None,
    ) -> List[Detection]:
        """
        Run detection on aligned frame, apply tracking, and persist results
        """

        raw_detections = self.detector.detect(aligned_image)

        if not raw_detections:
            logger.debug(f"No detections in frame {frame_metadata.frame_id}")
            return []

        tracked_detections = self.tracker.update(raw_detections)

        detection_entities = []
        for det in tracked_detections:
            x1, y1, x2, y2 = det["bbox"]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(aligned_image.shape[1], x2)
            y2 = min(aligned_image.shape[0], y2)

            if x2 > x1 and y2 > y1:
                cropped_image = aligned_image[y1:y2, x1:x2]
                sku_id = self.sku_detector.get_sku_id(cropped_image)
            else:
                logger.warning(f"Invalid bbox for detection: {det['bbox']}")
                sku_id = "invalid_bbox"

            detection = Detection(
                detection_id=str(uuid.uuid4()),
                shelf_id=shelf_id,
                frame_timestamp=frame_metadata.timestamp,
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
                aligned_frame_path=aligned_frame_path,
            )

            detection_entities.append(detection)

        logger.info(
            f"Processed frame {frame_metadata.frame_id}: "
            f"{len(detection_entities)} detections created"
        )

        return detection_entities
