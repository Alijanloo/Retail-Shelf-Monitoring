import pytest

from retail_shelf_monitoring.adaptors.grid.grid_detector import GridDetector
from retail_shelf_monitoring.entities.planogram import PlanogramGrid


class TestGridDetector:
    def test_initialization(self):
        detector = GridDetector(clustering_method="dbscan", eps=20.0, min_samples=3)

        assert detector.clustering_params.row_clustering_method == "dbscan"
        assert detector.clustering_params.eps == 20.0
        assert detector.clustering_params.min_samples == 3

    def test_detect_grid_empty_detections(self):
        detector = GridDetector()

        with pytest.raises(
            ValueError, match="Cannot detect grid from empty detections"
        ):
            detector.detect_grid([])

    def test_detect_grid_single_row(self):
        detections = [
            {"bbox": [10, 50, 30, 70], "class_id": 0, "confidence": 0.9},
            {"bbox": [40, 55, 60, 75], "class_id": 1, "confidence": 0.9},
            {"bbox": [70, 52, 90, 72], "class_id": 2, "confidence": 0.9},
        ]

        detector = GridDetector(clustering_method="dbscan", eps=15.0, min_samples=2)
        grid, params = detector.detect_grid(detections)

        assert isinstance(grid, PlanogramGrid)
        assert len(grid.rows) == 1
        assert len(grid.rows[0].items) == 3
        assert grid.total_items == 3

    def test_detect_grid_multiple_rows(self):
        detections = [
            {"bbox": [10, 50, 30, 70], "class_id": 0, "confidence": 0.9},
            {"bbox": [40, 55, 60, 75], "class_id": 1, "confidence": 0.9},
            {"bbox": [10, 150, 30, 170], "class_id": 2, "confidence": 0.9},
            {"bbox": [40, 155, 60, 175], "class_id": 3, "confidence": 0.9},
        ]

        detector = GridDetector(clustering_method="dbscan", eps=15.0, min_samples=2)
        grid, params = detector.detect_grid(detections)

        assert len(grid.rows) == 2
        assert grid.total_items == 4

    def test_detect_grid_items_sorted_by_x(self):
        detections = [
            {"bbox": [70, 50, 90, 70], "class_id": 2, "confidence": 0.9},
            {"bbox": [10, 50, 30, 70], "class_id": 0, "confidence": 0.9},
            {"bbox": [40, 55, 60, 75], "class_id": 1, "confidence": 0.9},
        ]

        detector = GridDetector(clustering_method="dbscan", eps=15.0, min_samples=2)
        grid, params = detector.detect_grid(detections)

        row = grid.rows[0]
        assert row.items[0].sku_id == "sku_0"
        assert row.items[1].sku_id == "sku_1"
        assert row.items[2].sku_id == "sku_2"

    def test_detect_grid_rows_sorted_by_y(self):
        detections = [
            {"bbox": [10, 150, 30, 170], "class_id": 2, "confidence": 0.9},
            {"bbox": [40, 155, 60, 175], "class_id": 3, "confidence": 0.9},
            {"bbox": [10, 50, 30, 70], "class_id": 0, "confidence": 0.9},
            {"bbox": [40, 55, 60, 75], "class_id": 1, "confidence": 0.9},
        ]

        detector = GridDetector(clustering_method="dbscan", eps=15.0, min_samples=2)
        grid, params = detector.detect_grid(detections)

        assert grid.rows[0].avg_y < grid.rows[1].avg_y

    def test_detect_grid_with_sku_id(self):
        detections = [
            {
                "bbox": [10, 50, 30, 70],
                "class_id": 0,
                "confidence": 0.9,
                "sku_id": "custom_sku_1",
            },
            {
                "bbox": [40, 55, 60, 75],
                "class_id": 1,
                "confidence": 0.9,
                "sku_id": "custom_sku_2",
            },
        ]

        detector = GridDetector(clustering_method="dbscan", eps=15.0, min_samples=2)
        grid, params = detector.detect_grid(detections)

        assert grid.rows[0].items[0].sku_id == "custom_sku_1"
        assert grid.rows[0].items[1].sku_id == "custom_sku_2"

    def test_match_grids_perfect_match(self):
        detections = [
            {"bbox": [10, 50, 30, 70], "class_id": 0, "confidence": 0.9},
            {"bbox": [40, 55, 60, 75], "class_id": 1, "confidence": 0.9},
        ]

        detector = GridDetector(clustering_method="dbscan", eps=15.0, min_samples=2)
        reference_grid, _ = detector.detect_grid(detections)

        result = detector.match_grids(reference_grid, detections)

        assert len(result["matches"]) == 2
        assert len(result["mismatches"]) == 0
        assert len(result["missing"]) == 0
        assert result["reference_total"] == 2
        assert result["current_total"] == 2

    def test_match_grids_with_mismatch(self):
        reference_detections = [
            {"bbox": [10, 50, 30, 70], "class_id": 0, "confidence": 0.9},
            {"bbox": [40, 55, 60, 75], "class_id": 1, "confidence": 0.9},
        ]

        current_detections = [
            {"bbox": [10, 50, 30, 70], "class_id": 0, "confidence": 0.9},
            {"bbox": [40, 55, 60, 75], "class_id": 99, "confidence": 0.9},
        ]

        detector = GridDetector(clustering_method="dbscan", eps=15.0, min_samples=2)
        reference_grid, _ = detector.detect_grid(reference_detections)

        result = detector.match_grids(reference_grid, current_detections)

        assert len(result["matches"]) == 1
        assert len(result["mismatches"]) == 1
        assert result["mismatches"][0]["expected_sku"] == "sku_1"
        assert result["mismatches"][0]["detected_sku"] == "sku_99"

    def test_match_grids_with_missing(self):
        reference_detections = [
            {"bbox": [10, 50, 30, 70], "class_id": 0, "confidence": 0.9},
            {"bbox": [40, 55, 60, 75], "class_id": 1, "confidence": 0.9},
        ]

        current_detections = [
            {"bbox": [10, 50, 30, 70], "class_id": 0, "confidence": 0.9},
        ]

        detector = GridDetector(clustering_method="dbscan", eps=15.0, min_samples=2)
        reference_grid, _ = detector.detect_grid(reference_detections)

        result = detector.match_grids(reference_grid, current_detections)

        assert len(result["missing"]) >= 1
        assert result["reference_total"] == 2
