from datetime import datetime
from typing import Dict, List

from ..adaptors.grid.grid_detector import GridDetector
from ..entities.common import CellState
from ..entities.detection import Detection
from ..entities.planogram import Planogram
from ..frameworks.logging_config import get_logger

logger = get_logger(__name__)


class CellStateComputation:
    def __init__(
        self,
        grid_detector: GridDetector,
        position_tolerance: int = 1,
        confidence_threshold: float = 0.35,
    ):
        self.grid_detector = grid_detector
        self.position_tolerance = position_tolerance
        self.confidence_threshold = confidence_threshold

    def compute_cell_states(
        self,
        planogram: Planogram,
        detections: List[Detection],
        frame_timestamp: datetime,
    ) -> Dict:
        """
        Compare current detections against reference planogram

        Returns dict with:
        - cell_states: List of {row_idx, item_idx, state, expected_sku, detected_sku}
        - summary: {ok_count, oos_count, misplaced_count, unknown_count}
        """

        detection_dicts = [
            {
                "bbox": [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2],
                "sku_id": d.sku_id or f"class_{d.class_id}",
                "confidence": d.confidence,
                "class_id": d.class_id,
            }
            for d in detections
            if d.confidence >= self.confidence_threshold
        ]

        if not detection_dicts:
            return self._all_cells_empty(planogram, frame_timestamp)

        match_result = self.grid_detector.match_grids(
            reference_grid=planogram.grid,
            current_detections=detection_dicts,
            position_tolerance=self.position_tolerance,
        )

        cell_states = []

        for match in match_result["matches"]:
            cell_states.append(
                {
                    "row_idx": match["row_idx"],
                    "item_idx": match["item_idx"],
                    "state": CellState.OK,
                    "expected_sku": match["sku_id"],
                    "detected_sku": match["sku_id"],
                    "confidence": 1.0,
                }
            )

        for mismatch in match_result["mismatches"]:
            cell_states.append(
                {
                    "row_idx": mismatch["row_idx"],
                    "item_idx": mismatch["item_idx"],
                    "state": CellState.MISPLACED,
                    "expected_sku": mismatch["expected_sku"],
                    "detected_sku": mismatch["detected_sku"],
                    "confidence": 1.0,
                }
            )

        for missing in match_result["missing"]:
            cell_states.append(
                {
                    "row_idx": missing["row_idx"],
                    "item_idx": missing["item_idx"],
                    "state": CellState.EMPTY,
                    "expected_sku": missing["expected_sku"],
                    "detected_sku": None,
                    "confidence": 0.0,
                }
            )

        summary = {
            "ok_count": len(match_result["matches"]),
            "oos_count": 0,
            "misplaced_count": len(match_result["mismatches"]),
            "empty_count": len(match_result["missing"]),
            "total_cells": planogram.grid.total_items,
            "timestamp": frame_timestamp,
        }

        logger.info(
            f"Cell state computation for shelf {planogram.shelf_id}: "
            f"{summary['ok_count']} OK, {summary['empty_count']} empty, "
            f"{summary['misplaced_count']} misplaced"
        )

        return {"cell_states": cell_states, "summary": summary}

    def _all_cells_empty(self, planogram: Planogram, frame_timestamp: datetime) -> Dict:
        """Return result when no detections found"""
        cell_states = []

        for row in planogram.grid.rows:
            for item in row.items:
                cell_states.append(
                    {
                        "row_idx": row.row_idx,
                        "item_idx": item.item_idx,
                        "state": CellState.EMPTY,
                        "expected_sku": item.sku_id,
                        "detected_sku": None,
                        "confidence": 0.0,
                    }
                )

        summary = {
            "ok_count": 0,
            "oos_count": 0,
            "misplaced_count": 0,
            "empty_count": planogram.grid.total_items,
            "total_cells": planogram.grid.total_items,
            "timestamp": frame_timestamp,
        }

        return {"cell_states": cell_states, "summary": summary}
