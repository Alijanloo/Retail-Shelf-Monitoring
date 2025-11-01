import difflib
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
                    track_id=item.track_id,
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
                    track_id=det.get("track_id", None),
                )
            )

        return cluster_items

    def _row_to_sequence(self, row: PlanogramRow) -> List[str]:
        """Convert a planogram row to a sequence of SKU IDs for sequence matching."""
        return [item.sku_id for item in row.items]

    def _find_best_matching_row(
        self, reference_row: PlanogramRow, current_grid: PlanogramGrid
    ) -> Tuple[PlanogramRow, float]:
        """Find the best matching row in current grid using sequence similarity."""
        ref_sequence = self._row_to_sequence(reference_row)
        best_row = None
        best_ratio = 0.0

        for current_row in current_grid.rows:
            current_sequence = self._row_to_sequence(current_row)
            matcher = difflib.SequenceMatcher(None, ref_sequence, current_sequence)
            ratio = matcher.ratio()

            if ratio > best_ratio:
                best_ratio = ratio
                best_row = current_row

        return best_row, best_ratio

    def _match_rows(
        self, ref_row: PlanogramRow, current_row: PlanogramRow, ref_row_idx: int
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Match two sequences using difflib and return matches, mismatches, and
        missing."""
        ref_sequence = self._row_to_sequence(ref_row)
        current_sequence = self._row_to_sequence(current_row)
        matcher = difflib.SequenceMatcher(None, ref_sequence, current_sequence)
        matches = []
        mismatches = []
        missing = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                # Exact matches
                for k in range(i2 - i1):
                    matches.append(
                        {
                            "row_idx": ref_row_idx,
                            "item_idx": i1 + k,
                            "sku_id": ref_sequence[i1 + k],
                        }
                    )
            elif tag == "replace":
                # Mismatches (items in different positions)
                for k in range(min(i2 - i1, j2 - j1)):
                    mismatches.append(
                        {
                            "row_idx": ref_row_idx,
                            "item_idx": i1 + k,
                            "expected_sku": ref_sequence[i1 + k],
                            "detected_sku": current_sequence[j1 + k],
                            "track_id": current_row.items[j1 + k].track_id,
                        }
                    )
                # Handle extra items in reference (missing in current)
                for k in range(j2 - j1, i2 - i1):
                    missing.append(
                        {
                            "row_idx": ref_row_idx,
                            "item_idx": i1 + k,
                            "expected_sku": ref_sequence[i1 + k],
                        }
                    )
            elif tag == "delete":
                # Items missing in current sequence
                for k in range(i2 - i1):
                    missing.append(
                        {
                            "row_idx": ref_row_idx,
                            "item_idx": i1 + k,
                            "expected_sku": ref_sequence[i1 + k],
                        }
                    )

        return matches, mismatches, missing

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

        for i, ref_row in enumerate(reference_grid.rows):
            # best_current_row, similarity_ratio = self._find_best_matching_row(
            #     ref_row, current_grid
            # )
            best_current_row = (
                current_grid.rows[i] if i < len(current_grid.rows) else None
            )
            similarity_ratio = 1.0

            if best_current_row is None or similarity_ratio < 0.3:
                for ref_item in ref_row.items:
                    missing.append(
                        {
                            "row_idx": ref_row.row_idx,
                            "item_idx": ref_item.item_idx,
                            "expected_sku": ref_item.sku_id,
                        }
                    )
            else:
                row_matches, row_mismatches, row_missing = self._match_rows(
                    ref_row, best_current_row, ref_row.row_idx
                )

                matches.extend(row_matches)
                mismatches.extend(row_mismatches)
                missing.extend(row_missing)

                logger.debug(
                    f"Row {ref_row.row_idx}: matched with similarity "
                    f"{similarity_ratio:.2f}, found {len(row_matches)} matches, "
                    f"{len(row_mismatches)} mismatches, {len(row_missing)} missing"
                )

        return {
            "matches": matches,
            "mismatches": mismatches,
            "missing": missing,
            "reference_total": reference_grid.total_items,
            "current_total": current_grid.total_items,
        }
