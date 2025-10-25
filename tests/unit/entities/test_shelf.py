import pytest
from pydantic import ValidationError
from datetime import datetime
from retail_shelf_monitoring.entities.shelf import Shelf
from retail_shelf_monitoring.entities.common import Priority


class TestShelf:
    def test_shelf_creation(self):
        shelf = Shelf(
            shelf_id="SHELF-001",
            store_id="STORE-001",
            aisle="A1",
            section="Left"
        )
        assert shelf.shelf_id == "SHELF-001"
        assert shelf.store_id == "STORE-001"
        assert shelf.aisle == "A1"
        assert shelf.section == "Left"
        assert shelf.priority == Priority.MEDIUM
        assert shelf.active is True

    def test_shelf_with_custom_priority(self):
        shelf = Shelf(
            shelf_id="SHELF-002",
            store_id="STORE-001",
            priority=Priority.HIGH
        )
        assert shelf.priority == Priority.HIGH

    def test_shelf_id_validation(self):
        with pytest.raises(ValidationError):
            Shelf(shelf_id="", store_id="STORE-001")
        
        with pytest.raises(ValueError, match="shelf_id cannot be empty"):
            Shelf(shelf_id="   ", store_id="STORE-001")

    def test_shelf_id_strip_whitespace(self):
        shelf = Shelf(shelf_id="  SHELF-003  ", store_id="STORE-001")
        assert shelf.shelf_id == "SHELF-003"

    def test_shelf_metadata(self):
        meta = {"location": "front", "special": True}
        shelf = Shelf(
            shelf_id="SHELF-004",
            store_id="STORE-001",
            meta=meta
        )
        assert shelf.meta == meta

    def test_shelf_timestamps(self):
        shelf = Shelf(shelf_id="SHELF-005", store_id="STORE-001")
        assert isinstance(shelf.created_at, datetime)
        assert isinstance(shelf.updated_at, datetime)

    def test_shelf_inactive(self):
        shelf = Shelf(
            shelf_id="SHELF-006",
            store_id="STORE-001",
            active=False
        )
        assert shelf.active is False
