import pytest

from retail_shelf_monitoring.entities.common import (
    AlertType,
    BoundingBox,
    CellState,
    Priority,
)


class TestBoundingBox:
    def test_bounding_box_creation(self):
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=100.0, y2=200.0)
        assert bbox.x1 == 10.0
        assert bbox.y1 == 20.0
        assert bbox.x2 == 100.0
        assert bbox.y2 == 200.0

    def test_bounding_box_center(self):
        bbox = BoundingBox(x1=0.0, y1=0.0, x2=100.0, y2=100.0)
        assert bbox.center == (50.0, 50.0)

    def test_bounding_box_dimensions(self):
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=110.0, y2=220.0)
        assert bbox.width == 100.0
        assert bbox.height == 200.0
        assert bbox.area == 20000.0

    def test_bounding_box_immutable(self):
        bbox = BoundingBox(x1=0.0, y1=0.0, x2=100.0, y2=100.0)
        with pytest.raises(Exception):
            bbox.x1 = 50.0

    def test_bounding_box_validation(self):
        with pytest.raises(Exception):
            BoundingBox(x1=-10.0, y1=0.0, x2=100.0, y2=100.0)

        with pytest.raises(Exception):
            BoundingBox(x1=0.0, y1=0.0, x2=0.0, y2=100.0)


class TestEnums:
    def test_alert_type_enum(self):
        assert AlertType.OOS == "out_of_stock"
        assert AlertType.MISPLACEMENT == "misplacement"
        assert AlertType.UNKNOWN == "unknown"

    def test_cell_state_enum(self):
        assert CellState.OK == "ok"
        assert CellState.OOS == "out_of_stock"
        assert CellState.MISPLACED == "misplaced"

    def test_priority_enum(self):
        assert Priority.LOW == "low"
        assert Priority.MEDIUM == "medium"
        assert Priority.HIGH == "high"
        assert Priority.CRITICAL == "critical"
