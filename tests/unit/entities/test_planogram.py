import pytest
from retail_shelf_monitoring.entities.planogram import (
    PlanogramItem, PlanogramRow, PlanogramGrid, 
    ClusteringParams, Planogram
)
from retail_shelf_monitoring.entities.common import BoundingBox, Priority


class TestPlanogramItem:
    def test_planogram_item_creation(self):
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=50.0, y2=60.0)
        item = PlanogramItem(
            item_idx=0,
            bbox=bbox,
            sku_id="SKU-123"
        )
        assert item.item_idx == 0
        assert item.bbox == bbox
        assert item.sku_id == "SKU-123"
        assert item.confidence == 1.0

    def test_planogram_item_immutable(self):
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=50.0, y2=60.0)
        item = PlanogramItem(item_idx=0, bbox=bbox, sku_id="SKU-123")
        with pytest.raises(Exception):
            item.item_idx = 5


class TestPlanogramRow:
    def test_planogram_row_creation(self):
        bbox1 = BoundingBox(x1=10.0, y1=20.0, x2=50.0, y2=60.0)
        bbox2 = BoundingBox(x1=60.0, y1=20.0, x2=100.0, y2=60.0)
        
        item1 = PlanogramItem(item_idx=0, bbox=bbox1, sku_id="SKU-1")
        item2 = PlanogramItem(item_idx=1, bbox=bbox2, sku_id="SKU-2")
        
        row = PlanogramRow(
            row_idx=0,
            avg_y=40.0,
            items=[item1, item2]
        )
        assert row.row_idx == 0
        assert row.avg_y == 40.0
        assert len(row.items) == 2

    def test_planogram_row_sorts_by_x(self):
        bbox1 = BoundingBox(x1=60.0, y1=20.0, x2=100.0, y2=60.0)
        bbox2 = BoundingBox(x1=10.0, y1=20.0, x2=50.0, y2=60.0)
        
        item1 = PlanogramItem(item_idx=1, bbox=bbox1, sku_id="SKU-1")
        item2 = PlanogramItem(item_idx=0, bbox=bbox2, sku_id="SKU-2")
        
        row = PlanogramRow(
            row_idx=0,
            avg_y=40.0,
            items=[item1, item2]
        )
        assert row.items[0].bbox.x1 < row.items[1].bbox.x1

    def test_planogram_row_validates_indices(self):
        bbox1 = BoundingBox(x1=10.0, y1=20.0, x2=50.0, y2=60.0)
        bbox2 = BoundingBox(x1=60.0, y1=20.0, x2=100.0, y2=60.0)
        
        item1 = PlanogramItem(item_idx=0, bbox=bbox1, sku_id="SKU-1")
        item2 = PlanogramItem(item_idx=2, bbox=bbox2, sku_id="SKU-2")
        
        with pytest.raises(ValueError, match="sequential"):
            PlanogramRow(row_idx=0, avg_y=40.0, items=[item1, item2])


class TestPlanogramGrid:
    def test_planogram_grid_creation(self):
        bbox1 = BoundingBox(x1=10.0, y1=20.0, x2=50.0, y2=60.0)
        item1 = PlanogramItem(item_idx=0, bbox=bbox1, sku_id="SKU-1")
        row1 = PlanogramRow(row_idx=0, avg_y=40.0, items=[item1])
        
        grid = PlanogramGrid(rows=[row1])
        assert len(grid.rows) == 1
        assert grid.total_items == 1

    def test_planogram_grid_get_cell(self):
        bbox1 = BoundingBox(x1=10.0, y1=20.0, x2=50.0, y2=60.0)
        bbox2 = BoundingBox(x1=60.0, y1=20.0, x2=100.0, y2=60.0)
        
        item1 = PlanogramItem(item_idx=0, bbox=bbox1, sku_id="SKU-1")
        item2 = PlanogramItem(item_idx=1, bbox=bbox2, sku_id="SKU-2")
        row = PlanogramRow(row_idx=0, avg_y=40.0, items=[item1, item2])
        
        grid = PlanogramGrid(rows=[row])
        
        cell = grid.get_cell(0, 1)
        assert cell is not None
        assert cell.sku_id == "SKU-2"
        
        assert grid.get_cell(0, 5) is None
        assert grid.get_cell(5, 0) is None


class TestPlanogram:
    def test_planogram_creation(self):
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=50.0, y2=60.0)
        item = PlanogramItem(item_idx=0, bbox=bbox, sku_id="SKU-1")
        row = PlanogramRow(row_idx=0, avg_y=40.0, items=[item])
        grid = PlanogramGrid(rows=[row])
        params = ClusteringParams()
        
        planogram = Planogram(
            shelf_id="SHELF-001",
            reference_image_path="/path/to/ref.jpg",
            grid=grid,
            clustering_params=params
        )
        assert planogram.shelf_id == "SHELF-001"
        assert planogram.grid.total_items == 1
        assert planogram.priority == Priority.MEDIUM

    def test_planogram_with_metadata(self):
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=50.0, y2=60.0)
        item = PlanogramItem(item_idx=0, bbox=bbox, sku_id="SKU-1")
        row = PlanogramRow(row_idx=0, avg_y=40.0, items=[item])
        grid = PlanogramGrid(rows=[row])
        params = ClusteringParams()
        
        planogram = Planogram(
            shelf_id="SHELF-001",
            reference_image_path="/path/to/ref.jpg",
            grid=grid,
            clustering_params=params,
            meta={"priority": "high"}
        )
        assert planogram.priority == Priority.HIGH
