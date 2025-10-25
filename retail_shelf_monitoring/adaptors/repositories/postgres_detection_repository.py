from datetime import datetime
from typing import List, Optional

from sqlalchemy import and_, desc
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ...entities.common import BoundingBox
from ...entities.detection import Detection
from ...frameworks.database import DetectionModel
from ...frameworks.exceptions import DatabaseError
from ...frameworks.logging_config import get_logger
from ...usecases.interfaces.repositories import DetectionRepository

logger = get_logger(__name__)


class PostgresDetectionRepository(DetectionRepository):
    def __init__(self, session_factory):
        self.session_factory = session_factory

    async def create(self, detection: Detection) -> Detection:
        session: Session = self.session_factory()
        try:
            db_detection = self._to_db_model(detection)

            session.add(db_detection)
            session.commit()
            session.refresh(db_detection)

            return self._to_entity(db_detection)

        except IntegrityError as e:
            session.rollback()
            raise DatabaseError(f"Failed to create detection: {str(e)}")
        finally:
            session.close()

    async def create_batch(self, detections: List[Detection]) -> List[Detection]:
        """Optimized batch insert"""
        if not detections:
            return []

        session: Session = self.session_factory()
        try:
            db_detections = [self._to_db_model(det) for det in detections]

            # Use add_all instead of bulk_save_objects to maintain ORM tracking
            session.add_all(db_detections)
            session.commit()

            logger.info(f"Batch inserted {len(detections)} detections")

            return [self._to_entity(db_det) for db_det in db_detections]

        except IntegrityError as e:
            session.rollback()
            raise DatabaseError(f"Failed to batch create detections: {str(e)}")
        finally:
            session.close()

    async def get_by_id(self, detection_id: str) -> Optional[Detection]:
        session: Session = self.session_factory()
        try:
            db_detection = (
                session.query(DetectionModel)
                .filter_by(detection_id=detection_id)
                .first()
            )

            return self._to_entity(db_detection) if db_detection else None
        finally:
            session.close()

    async def get_by_shelf(
        self,
        shelf_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Detection]:
        session: Session = self.session_factory()
        try:
            query = session.query(DetectionModel).filter_by(shelf_id=shelf_id)

            if start_time:
                query = query.filter(DetectionModel.frame_timestamp >= start_time)
            if end_time:
                query = query.filter(DetectionModel.frame_timestamp <= end_time)

            query = query.order_by(desc(DetectionModel.frame_timestamp)).limit(limit)

            db_detections = query.all()
            return [self._to_entity(db_det) for db_det in db_detections]
        finally:
            session.close()

    async def get_recent_by_cell(
        self, shelf_id: str, row_idx: int, item_idx: int, limit: int = 10
    ) -> List[Detection]:
        session: Session = self.session_factory()
        try:
            query = (
                session.query(DetectionModel)
                .filter(
                    and_(
                        DetectionModel.shelf_id == shelf_id,
                        DetectionModel.row_idx == row_idx,
                        DetectionModel.item_idx == item_idx,
                    )
                )
                .order_by(desc(DetectionModel.frame_timestamp))
                .limit(limit)
            )

            db_detections = query.all()
            return [self._to_entity(db_det) for db_det in db_detections]
        finally:
            session.close()

    def _to_db_model(self, detection: Detection) -> DetectionModel:
        return DetectionModel(
            detection_id=detection.detection_id,
            shelf_id=detection.shelf_id,
            frame_timestamp=detection.frame_timestamp,
            bbox_x1=detection.bbox.x1,
            bbox_y1=detection.bbox.y1,
            bbox_x2=detection.bbox.x2,
            bbox_y2=detection.bbox.y2,
            class_id=detection.class_id,
            sku_id=detection.sku_id,
            confidence=detection.confidence,
            track_id=detection.track_id,
            row_idx=detection.row_idx,
            item_idx=detection.item_idx,
            aligned_frame_path=detection.aligned_frame_path,
            created_at=detection.created_at,
        )

    def _to_entity(self, db_detection: DetectionModel) -> Detection:
        return Detection(
            detection_id=db_detection.detection_id,
            shelf_id=db_detection.shelf_id,
            frame_timestamp=db_detection.frame_timestamp,
            bbox=BoundingBox(
                x1=db_detection.bbox_x1,
                y1=db_detection.bbox_y1,
                x2=db_detection.bbox_x2,
                y2=db_detection.bbox_y2,
            ),
            class_id=db_detection.class_id,
            sku_id=db_detection.sku_id,
            confidence=db_detection.confidence,
            track_id=db_detection.track_id,
            row_idx=db_detection.row_idx,
            item_idx=db_detection.item_idx,
            aligned_frame_path=db_detection.aligned_frame_path,
            created_at=db_detection.created_at,
        )
