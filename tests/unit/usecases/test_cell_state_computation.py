from datetime import datetime, timezone

import pytest

from retail_shelf_monitoring.entities.common import BoundingBox, CellState
from retail_shelf_monitoring.entities.detection import Detection
from retail_shelf_monitoring.entities.planogram import (
    Planogram,
    PlanogramGrid,
    PlanogramItem,
    PlanogramRow,
)
from retail_shelf_monitoring.usecases.cell_state_computation import CellStateComputation
from retail_shelf_monitoring.usecases.grid.grid_detector import GridDetector


@pytest.fixture
def sample_planogram():
    rows = [
        PlanogramRow(
            row_idx=0,
            avg_y=100,
            items=[
                PlanogramItem(
                    item_idx=0,
                    bbox=BoundingBox(x1=10, y1=90, x2=50, y2=110),
                    sku_id="sku_cola",
                    confidence=0.9,
                ),
                PlanogramItem(
                    item_idx=1,
                    bbox=BoundingBox(x1=60, y1=90, x2=100, y2=110),
                    sku_id="sku_sprite",
                    confidence=0.9,
                ),
            ],
        )
    ]

    grid = PlanogramGrid(rows=rows)

    return Planogram(
        shelf_id="shelf_001",
        reference_image_path="test.jpg",
        grid=grid,
        clustering_params={},
    )


@pytest.fixture
def grid_detector():
    return GridDetector()


@pytest.fixture
def cell_state_computation(grid_detector):
    return CellStateComputation(
        grid_detector=grid_detector,
        confidence_threshold=0.35,
    )


class TestCellStateComputation:
    def test_all_cells_ok(self, cell_state_computation, sample_planogram):
        detections = [
            Detection(
                detection_id="det1",
                shelf_id="shelf_001",
                frame_timestamp=datetime.now(timezone.utc),
                bbox=BoundingBox(x1=10, y1=90, x2=50, y2=110),
                class_id=0,
                sku_id="sku_cola",
                confidence=0.85,
            ),
            Detection(
                detection_id="det2",
                shelf_id="shelf_001",
                frame_timestamp=datetime.now(timezone.utc),
                bbox=BoundingBox(x1=60, y1=90, x2=100, y2=110),
                class_id=1,
                sku_id="sku_sprite",
                confidence=0.85,
            ),
        ]

        result = cell_state_computation.compute_cell_states(
            planogram=sample_planogram,
            detections=detections,
            frame_timestamp=datetime.now(timezone.utc),
        )

        assert result["summary"]["ok_count"] == 2
        assert result["summary"]["empty_count"] == 0
        assert result["summary"]["misplaced_count"] == 0

    def test_all_cells_empty(self, cell_state_computation, sample_planogram):
        detections = []

        result = cell_state_computation.compute_cell_states(
            planogram=sample_planogram,
            detections=detections,
            frame_timestamp=datetime.now(timezone.utc),
        )

        assert result["summary"]["ok_count"] == 0
        assert result["summary"]["empty_count"] == 2
        assert result["summary"]["misplaced_count"] == 0

        for state in result["cell_states"]:
            assert state["state"] == CellState.EMPTY
            assert state["detected_sku"] is None

    def test_misplaced_items(self, cell_state_computation, sample_planogram):
        # Wrong SKU in position
        detections = [
            Detection(
                detection_id="det1",
                shelf_id="shelf_001",
                frame_timestamp=datetime.now(timezone.utc),
                bbox=BoundingBox(x1=10, y1=90, x2=50, y2=110),
                class_id=1,
                sku_id="sku_sprite",  # Should be sku_cola
                confidence=0.85,
            ),
            Detection(
                detection_id="det2",
                shelf_id="shelf_001",
                frame_timestamp=datetime.now(timezone.utc),
                bbox=BoundingBox(x1=60, y1=90, x2=100, y2=110),
                class_id=0,
                sku_id="sku_cola",  # Should be sku_sprite
                confidence=0.85,
            ),
        ]

        result = cell_state_computation.compute_cell_states(
            planogram=sample_planogram,
            detections=detections,
            frame_timestamp=datetime.now(timezone.utc),
        )

        assert result["summary"]["misplaced_count"] == 2
        assert result["summary"]["ok_count"] == 0

    def test_filters_low_confidence_detections(
        self, cell_state_computation, sample_planogram
    ):
        detections = [
            Detection(
                detection_id="det1",
                shelf_id="shelf_001",
                frame_timestamp=datetime.now(timezone.utc),
                bbox=BoundingBox(x1=10, y1=90, x2=50, y2=110),
                class_id=0,
                sku_id="sku_cola",
                confidence=0.2,  # Below threshold
            ),
        ]

        result = cell_state_computation.compute_cell_states(
            planogram=sample_planogram,
            detections=detections,
            frame_timestamp=datetime.now(timezone.utc),
        )

        # Low confidence detection should be ignored
        assert result["summary"]["empty_count"] == 2
        assert result["summary"]["ok_count"] == 0

    def test_summary_totals(self, cell_state_computation, sample_planogram):
        detections = [
            Detection(
                detection_id="det1",
                shelf_id="shelf_001",
                frame_timestamp=datetime.now(timezone.utc),
                bbox=BoundingBox(x1=10, y1=90, x2=50, y2=110),
                class_id=0,
                sku_id="sku_cola",
                confidence=0.85,
            ),
        ]

        result = cell_state_computation.compute_cell_states(
            planogram=sample_planogram,
            detections=detections,
            frame_timestamp=datetime.now(timezone.utc),
        )

        summary = result["summary"]

        assert summary["total_cells"] == 2
        assert (
            summary["ok_count"] + summary["empty_count"] + summary["misplaced_count"]
        ) <= summary["total_cells"]
