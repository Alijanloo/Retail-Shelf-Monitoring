import uuid
from typing import List, Optional

from ..adaptors.ml.sku_mapper import SKUMapper
from ..adaptors.ml.yolo_detector import YOLOv11Detector
from ..entities.common import BoundingBox
from ..entities.detection import Detection
from ..entities.frame import Frame
from ..frameworks.logging_config import get_logger
from .interfaces.repositories import DetectionRepository
from .interfaces.tracker_interface import Tracker

logger = get_logger(__name__)


class DetectionProcessingUseCase:
    def __init__(
        self,
        detector: YOLOv11Detector,
        tracker: Tracker,
        sku_mapper: SKUMapper,
        detection_repository: DetectionRepository,
    ):
        self.detector = detector
        self.tracker = tracker
        self.sku_mapper = sku_mapper
        self.detection_repository = detection_repository

    async def process_aligned_frame(
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
            sku_id = self.sku_mapper.map_class_to_sku(det["class_id"])

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

        saved_detections = await self.detection_repository.create_batch(
            detection_entities
        )

        logger.info(
            f"Processed frame {frame_metadata.frame_id}: "
            f"{len(saved_detections)} detections saved"
        )

        return saved_detections

    async def get_recent_detections_for_cell(
        self, shelf_id: str, row_idx: int, item_idx: int, limit: int = 10
    ) -> List[Detection]:
        """Retrieve recent detections for a specific planogram cell"""
        return await self.detection_repository.get_recent_by_cell(
            shelf_id=shelf_id, row_idx=row_idx, item_idx=item_idx, limit=limit
        )
