from datetime import datetime

import pytest

from retail_shelf_monitoring.entities.sku import SKU


class TestSKU:
    def test_sku_creation(self):
        sku = SKU(sku_id="SKU-12345", name="Coca Cola 330ml")
        assert sku.sku_id == "SKU-12345"
        assert sku.name == "Coca Cola 330ml"
        assert sku.category is None
        assert sku.barcode is None

    def test_sku_with_all_fields(self):
        sku = SKU(
            sku_id="SKU-12345",
            name="Coca Cola 330ml",
            category="Beverages",
            barcode="0123456789012",
        )
        assert sku.category == "Beverages"
        assert sku.barcode == "0123456789012"

    def test_sku_name_validation(self):
        with pytest.raises(Exception):
            SKU(sku_id="SKU-001", name="")

    def test_sku_timestamps(self):
        sku = SKU(sku_id="SKU-001", name="Product A")
        assert isinstance(sku.created_at, datetime)
        assert isinstance(sku.updated_at, datetime)

    def test_sku_mutable(self):
        sku = SKU(sku_id="SKU-001", name="Product A")
        sku.name = "Product B"
        assert sku.name == "Product B"
