from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import DBSCAN, KMeans

from ...frameworks.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ClusterItem:
    bbox: Tuple[float, float, float, float]
    center: Tuple[float, float]
    sku_id: str
    confidence: float
    cluster_id: int = -1


class RowClusterer:
    def __init__(self, method: str = "dbscan", eps: float = 15.0, min_samples: int = 2):
        self.method = method.lower()
        self.eps = eps
        self.min_samples = min_samples

        if self.method not in ["dbscan", "kmeans"]:
            raise ValueError(f"Unsupported clustering method: {method}")

    def cluster_by_y_coordinate(
        self, items: List[ClusterItem]
    ) -> List[List[ClusterItem]]:
        if not items:
            return []

        y_coords = np.array([item.center[1] for item in items]).reshape(-1, 1)

        if self.method == "dbscan":
            clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            labels = clustering.fit_predict(y_coords)
        else:
            n_clusters = self._estimate_n_clusters(y_coords)
            clustering = KMeans(n_clusters=n_clusters, random_state=42)
            labels = clustering.fit_predict(y_coords)

        for item, label in zip(items, labels):
            item.cluster_id = int(label)

        valid_items = [item for item in items if item.cluster_id >= 0]

        clusters: Dict[int, List[ClusterItem]] = {}
        for item in valid_items:
            if item.cluster_id not in clusters:
                clusters[item.cluster_id] = []
            clusters[item.cluster_id].append(item)

        sorted_clusters = sorted(
            clusters.values(),
            key=lambda cluster: np.mean([item.center[1] for item in cluster]),
        )

        logger.info(
            f"Clustered {len(items)} items into {len(sorted_clusters)} rows using "
            f"{self.method}"
        )

        return sorted_clusters

    def _estimate_n_clusters(self, y_coords: np.ndarray) -> int:
        if len(y_coords) < 2:
            return 1

        sorted_y = np.sort(y_coords.flatten())
        gaps = np.diff(sorted_y)

        threshold = np.median(gaps) * 2.0
        n_clusters = np.sum(gaps > threshold) + 1

        return max(1, min(n_clusters, len(y_coords) // 2))


class ItemSorter:
    @staticmethod
    def sort_items_by_x(items: List[ClusterItem]) -> List[ClusterItem]:
        return sorted(items, key=lambda item: item.center[0])

    @staticmethod
    def assign_indices(
        items: List[ClusterItem], start_idx: int = 0
    ) -> List[Tuple[int, ClusterItem]]:
        sorted_items = ItemSorter.sort_items_by_x(items)
        return [(start_idx + idx, item) for idx, item in enumerate(sorted_items)]
