import uuid
from typing import List, Tuple

import numpy as np

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

    def _extract_cropped_images(
        self, frame: Frame, detections: List[dict]
    ) -> Tuple[List[np.ndarray], List[dict]]:
        """
        Extract cropped images from detections and filter out invalid ones.

        Args:
            frame: Input frame
            detections: List of detection dictionaries

        Returns:
            Tuple of (valid_cropped_images, valid_detections)
        """
        valid_cropped_images = []
        valid_detections = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Clamp coordinates to frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.frame_img.shape[1], x2)
            y2 = min(frame.frame_img.shape[0], y2)

            if x2 > x1 and y2 > y1:
                cropped_image = frame.frame_img[y1:y2, x1:x2]
                valid_cropped_images.append(cropped_image)
                valid_detections.append(det)
            else:
                logger.warning(f"Invalid bbox for detection: {det['bbox']}")
                # Add detection with invalid_bbox SKU
                det["sku_id"] = "invalid_bbox"
                valid_detections.append(det)

        return valid_cropped_images, valid_detections

    def process_aligned_frame(self, frame: Frame) -> List[Detection]:
        """
        Run detection on an aligned frame using batch processing for SKU identification.
        """
        detections = self.detector.detect(frame.frame_img)

        if not detections:
            logger.debug(f"No detections in frame {frame.frame_id}")
            return []

        cropped_images, valid_detections = self._extract_cropped_images(
            frame, detections
        )

        sku_ids = self.sku_detector.batch_identify_skus(cropped_images)

        for idx, sku_id in enumerate(sku_ids):
            valid_detections[idx]["sku_id"] = sku_id

        detection_entities = []
        for det in valid_detections:
            sku_id = det.get("sku_id", "unknown_sku")

            detection_entities.append(
                Detection(
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
            )

        return detection_entities
