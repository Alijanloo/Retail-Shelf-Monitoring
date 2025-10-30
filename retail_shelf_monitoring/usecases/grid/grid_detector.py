from typing import Dict, List, Tuple

from ...entities.common import BoundingBox
from ...entities.planogram import (
    ClusteringParams,
    PlanogramGrid,
    PlanogramItem,
    PlanogramRow,
)
from ...frameworks.logging_config import get_logger
from .clustering import ClusterItem, ItemSorter, RowClusterer

logger = get_logger(__name__)


class GridDetector:
    def __init__(
        self,
        clustering_method: str = "dbscan",
        eps: float = 15.0,
        min_samples: int = 2,
    ):
        self.clustering_params = ClusteringParams(
            row_clustering_method=clustering_method, eps=eps, min_samples=min_samples
        )
        self.row_clusterer = RowClusterer(
            method=clustering_method, eps=eps, min_samples=min_samples
        )

    def detect_grid(
        self, detections: List[Dict]
    ) -> Tuple[PlanogramGrid, ClusteringParams]:
        if not detections:
            raise ValueError("Cannot detect grid from empty detections")

        cluster_items = self._detections_to_cluster_items(detections)

        row_clusters = self.row_clusterer.cluster_by_y_coordinate(cluster_items)

        if not row_clusters:
            raise ValueError(
                "No valid clusters detected - items may have been filtered as noise"
            )

        planogram_rows = []
        for row_idx, row_items in enumerate(row_clusters):
            avg_y = sum(item.center[1] for item in row_items) / len(row_items)

            indexed_items = ItemSorter.assign_indices(row_items)

            planogram_items = [
                PlanogramItem(
                    item_idx=item_idx,
                    bbox=BoundingBox(
                        x1=item.bbox[0],
                        y1=item.bbox[1],
                        x2=item.bbox[2],
                        y2=item.bbox[3],
                    ),
                    sku_id=item.sku_id,
                    confidence=item.confidence,
                )
                for item_idx, item in indexed_items
            ]

            planogram_row = PlanogramRow(
                row_idx=row_idx, avg_y=avg_y, items=planogram_items
            )
            planogram_rows.append(planogram_row)

        grid = PlanogramGrid(rows=planogram_rows)

        logger.info(
            f"Detected grid with {len(grid.rows)} rows and {grid.total_items} total "
            "items"
        )

        return grid, self.clustering_params

    def _detections_to_cluster_items(self, detections: List[Dict]) -> List[ClusterItem]:
        cluster_items = []
        for det in detections:
            bbox = det["bbox"]
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2

            cluster_items.append(
                ClusterItem(
                    bbox=tuple(bbox),
                    center=(center_x, center_y),
                    sku_id=det.get("sku_id", f"sku_{det.get('class_id', 0)}"),
                    confidence=det.get("confidence", 1.0),
                )
            )

        return cluster_items

    def match_grids(
        self,
        reference_grid: PlanogramGrid,
        current_detections: List[Dict],
        position_tolerance: int = 1,
    ) -> Dict:
        if not current_detections:
            missing = []
            for ref_row in reference_grid.rows:
                for ref_item in ref_row.items:
                    missing.append(
                        {
                            "row_idx": ref_row.row_idx,
                            "item_idx": ref_item.item_idx,
                            "expected_sku": ref_item.sku_id,
                        }
                    )
            return {
                "matches": [],
                "mismatches": [],
                "missing": missing,
                "reference_total": reference_grid.total_items,
                "current_total": 0,
            }

        try:
            current_grid, _ = self.detect_grid(current_detections)
        except ValueError as e:  # noqa: F841
            missing = []
            for ref_row in reference_grid.rows:
                for ref_item in ref_row.items:
                    missing.append(
                        {
                            "row_idx": ref_row.row_idx,
                            "item_idx": ref_item.item_idx,
                            "expected_sku": ref_item.sku_id,
                        }
                    )
            return {
                "matches": [],
                "mismatches": [],
                "missing": missing,
                "reference_total": reference_grid.total_items,
                "current_total": 0,
            }
        matches = []
        mismatches = []
        missing = []

        for ref_row in reference_grid.rows:
            for ref_item in ref_row.items:
                current_item = current_grid.get_cell(ref_row.row_idx, ref_item.item_idx)

                if current_item is None:
                    missing.append(
                        {
                            "row_idx": ref_row.row_idx,
                            "item_idx": ref_item.item_idx,
                            "expected_sku": ref_item.sku_id,
                        }
                    )
                elif current_item.sku_id == ref_item.sku_id:
                    matches.append(
                        {
                            "row_idx": ref_row.row_idx,
                            "item_idx": ref_item.item_idx,
                            "sku_id": ref_item.sku_id,
                        }
                    )
                else:
                    mismatches.append(
                        {
                            "row_idx": ref_row.row_idx,
                            "item_idx": ref_item.item_idx,
                            "expected_sku": ref_item.sku_id,
                            "detected_sku": current_item.sku_id,
                        }
                    )

        return {
            "matches": matches,
            "mismatches": mismatches,
            "missing": missing,
            "reference_total": reference_grid.total_items,
            "current_total": current_grid.total_items,
        }
