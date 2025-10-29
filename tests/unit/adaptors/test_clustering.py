import pytest

from retail_shelf_monitoring.usecases.grid.clustering import (
    ClusterItem,
    ItemSorter,
    RowClusterer,
)


class TestClusterItem:
    def test_cluster_item_creation(self):
        item = ClusterItem(
            bbox=(10.0, 20.0, 30.0, 40.0),
            center=(20.0, 30.0),
            sku_id="sku_001",
            confidence=0.95,
        )

        assert item.bbox == (10.0, 20.0, 30.0, 40.0)
        assert item.center == (20.0, 30.0)
        assert item.sku_id == "sku_001"
        assert item.confidence == 0.95
        assert item.cluster_id == -1

    def test_cluster_item_with_cluster_id(self):
        item = ClusterItem(
            bbox=(10.0, 20.0, 30.0, 40.0),
            center=(20.0, 30.0),
            sku_id="sku_001",
            confidence=0.95,
            cluster_id=2,
        )

        assert item.cluster_id == 2


class TestRowClusterer:
    def test_initialization_dbscan(self):
        clusterer = RowClusterer(method="dbscan", eps=15.0, min_samples=2)

        assert clusterer.method == "dbscan"
        assert clusterer.eps == 15.0
        assert clusterer.min_samples == 2

    def test_initialization_kmeans(self):
        clusterer = RowClusterer(method="kmeans", eps=15.0, min_samples=2)

        assert clusterer.method == "kmeans"

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Unsupported clustering method"):
            RowClusterer(method="invalid")

    def test_cluster_empty_items(self):
        clusterer = RowClusterer()
        result = clusterer.cluster_by_y_coordinate([])

        assert result == []

    def test_cluster_single_row_dbscan(self):
        items = [
            ClusterItem(
                bbox=(10, 50, 30, 70), center=(20, 60), sku_id="sku_1", confidence=0.9
            ),
            ClusterItem(
                bbox=(40, 55, 60, 75), center=(50, 65), sku_id="sku_2", confidence=0.9
            ),
            ClusterItem(
                bbox=(70, 52, 90, 72), center=(80, 62), sku_id="sku_3", confidence=0.9
            ),
        ]

        clusterer = RowClusterer(method="dbscan", eps=15.0, min_samples=2)
        rows = clusterer.cluster_by_y_coordinate(items)

        assert len(rows) == 1
        assert len(rows[0]) == 3

    def test_cluster_multiple_rows_dbscan(self):
        items = [
            ClusterItem(
                bbox=(10, 50, 30, 70), center=(20, 60), sku_id="sku_1", confidence=0.9
            ),
            ClusterItem(
                bbox=(40, 55, 60, 75), center=(50, 65), sku_id="sku_2", confidence=0.9
            ),
            ClusterItem(
                bbox=(10, 150, 30, 170),
                center=(20, 160),
                sku_id="sku_3",
                confidence=0.9,
            ),
            ClusterItem(
                bbox=(40, 155, 60, 175),
                center=(50, 165),
                sku_id="sku_4",
                confidence=0.9,
            ),
        ]

        clusterer = RowClusterer(method="dbscan", eps=15.0, min_samples=2)
        rows = clusterer.cluster_by_y_coordinate(items)

        assert len(rows) == 2
        assert len(rows[0]) == 2
        assert len(rows[1]) == 2

    def test_cluster_rows_sorted_by_y(self):
        items = [
            ClusterItem(
                bbox=(10, 150, 30, 170),
                center=(20, 160),
                sku_id="sku_3",
                confidence=0.9,
            ),
            ClusterItem(
                bbox=(40, 155, 60, 175),
                center=(50, 165),
                sku_id="sku_4",
                confidence=0.9,
            ),
            ClusterItem(
                bbox=(10, 50, 30, 70), center=(20, 60), sku_id="sku_1", confidence=0.9
            ),
            ClusterItem(
                bbox=(40, 55, 60, 75), center=(50, 65), sku_id="sku_2", confidence=0.9
            ),
        ]

        clusterer = RowClusterer(method="dbscan", eps=15.0, min_samples=2)
        rows = clusterer.cluster_by_y_coordinate(items)

        assert len(rows) == 2
        top_row_avg_y = sum(item.center[1] for item in rows[0]) / len(rows[0])
        bottom_row_avg_y = sum(item.center[1] for item in rows[1]) / len(rows[1])
        assert top_row_avg_y < bottom_row_avg_y

    def test_cluster_multiple_rows_kmeans(self):
        items = [
            ClusterItem(
                bbox=(10, 50, 30, 70), center=(20, 60), sku_id="sku_1", confidence=0.9
            ),
            ClusterItem(
                bbox=(40, 55, 60, 75), center=(50, 65), sku_id="sku_2", confidence=0.9
            ),
            ClusterItem(
                bbox=(10, 150, 30, 170),
                center=(20, 160),
                sku_id="sku_3",
                confidence=0.9,
            ),
            ClusterItem(
                bbox=(40, 155, 60, 175),
                center=(50, 165),
                sku_id="sku_4",
                confidence=0.9,
            ),
        ]

        clusterer = RowClusterer(method="kmeans", eps=15.0, min_samples=2)
        rows = clusterer.cluster_by_y_coordinate(items)

        assert len(rows) == 2

    def test_estimate_n_clusters_single_item(self):
        clusterer = RowClusterer(method="kmeans")
        import numpy as np

        y_coords = np.array([[50.0]])
        n_clusters = clusterer._estimate_n_clusters(y_coords)

        assert n_clusters == 1

    def test_estimate_n_clusters_with_gaps(self):
        clusterer = RowClusterer(method="kmeans")
        import numpy as np

        y_coords = np.array([[50.0], [55.0], [150.0], [155.0], [250.0], [255.0]])
        n_clusters = clusterer._estimate_n_clusters(y_coords)

        assert n_clusters >= 2


class TestItemSorter:
    def test_sort_items_by_x(self):
        items = [
            ClusterItem(
                bbox=(70, 50, 90, 70), center=(80, 60), sku_id="sku_3", confidence=0.9
            ),
            ClusterItem(
                bbox=(10, 50, 30, 70), center=(20, 60), sku_id="sku_1", confidence=0.9
            ),
            ClusterItem(
                bbox=(40, 50, 60, 70), center=(50, 60), sku_id="sku_2", confidence=0.9
            ),
        ]

        sorted_items = ItemSorter.sort_items_by_x(items)

        assert len(sorted_items) == 3
        assert sorted_items[0].sku_id == "sku_1"
        assert sorted_items[1].sku_id == "sku_2"
        assert sorted_items[2].sku_id == "sku_3"

    def test_assign_indices(self):
        items = [
            ClusterItem(
                bbox=(70, 50, 90, 70), center=(80, 60), sku_id="sku_3", confidence=0.9
            ),
            ClusterItem(
                bbox=(10, 50, 30, 70), center=(20, 60), sku_id="sku_1", confidence=0.9
            ),
            ClusterItem(
                bbox=(40, 50, 60, 70), center=(50, 60), sku_id="sku_2", confidence=0.9
            ),
        ]

        indexed_items = ItemSorter.assign_indices(items)

        assert len(indexed_items) == 3
        assert indexed_items[0] == (0, items[1])
        assert indexed_items[1] == (1, items[2])
        assert indexed_items[2] == (2, items[0])

    def test_assign_indices_with_start_idx(self):
        items = [
            ClusterItem(
                bbox=(10, 50, 30, 70), center=(20, 60), sku_id="sku_1", confidence=0.9
            ),
            ClusterItem(
                bbox=(40, 50, 60, 70), center=(50, 60), sku_id="sku_2", confidence=0.9
            ),
        ]

        indexed_items = ItemSorter.assign_indices(items, start_idx=5)

        assert indexed_items[0][0] == 5
        assert indexed_items[1][0] == 6

    def test_empty_items(self):
        sorted_items = ItemSorter.sort_items_by_x([])
        assert sorted_items == []

        indexed_items = ItemSorter.assign_indices([])
        assert indexed_items == []
