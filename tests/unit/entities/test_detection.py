from datetime import datetime

import pytest

from retail_shelf_monitoring.entities.common import BoundingBox
from retail_shelf_monitoring.entities.detection import Detection


class TestDetection:
    def test_detection_creation(self):
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=100.0, y2=120.0)
        timestamp = datetime.utcnow()
        detection = Detection(
            shelf_id="SHELF-001",
            frame_timestamp=timestamp,
            bbox=bbox,
            class_id=5,
            confidence=0.95,
        )
        assert detection.shelf_id == "SHELF-001"
        assert detection.frame_timestamp == timestamp
        assert detection.bbox == bbox
        assert detection.class_id == 5
        assert detection.confidence == 0.95
        assert detection.detection_id is None

    def test_detection_with_tracking(self):
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=100.0, y2=120.0)
        detection = Detection(
            shelf_id="SHELF-001",
            frame_timestamp=datetime.utcnow(),
            bbox=bbox,
            class_id=5,
            confidence=0.95,
            track_id=123,
        )
        assert detection.track_id == 123

    def test_detection_with_grid_position(self):
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=100.0, y2=120.0)
        detection = Detection(
            shelf_id="SHELF-001",
            frame_timestamp=datetime.utcnow(),
            bbox=bbox,
            class_id=5,
            confidence=0.95,
            row_idx=2,
            item_idx=3,
        )
        assert detection.row_idx == 2
        assert detection.item_idx == 3

    def test_detection_confidence_validation(self):
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=100.0, y2=120.0)

        with pytest.raises(ValueError, match="unreasonably low"):
            Detection(
                shelf_id="SHELF-001",
                frame_timestamp=datetime.utcnow(),
                bbox=bbox,
                class_id=5,
                confidence=0.05,
            )

    def test_detection_confidence_bounds(self):
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=100.0, y2=120.0)

        with pytest.raises(Exception):
            Detection(
                shelf_id="SHELF-001",
                frame_timestamp=datetime.utcnow(),
                bbox=bbox,
                class_id=5,
                confidence=1.5,
            )

        with pytest.raises(Exception):
            Detection(
                shelf_id="SHELF-001",
                frame_timestamp=datetime.utcnow(),
                bbox=bbox,
                class_id=5,
                confidence=-0.1,
            )

    def test_detection_with_sku_mapping(self):
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=100.0, y2=120.0)
        detection = Detection(
            shelf_id="SHELF-001",
            frame_timestamp=datetime.utcnow(),
            bbox=bbox,
            class_id=5,
            confidence=0.95,
            sku_id="SKU-12345",
        )
        assert detection.sku_id == "SKU-12345"
