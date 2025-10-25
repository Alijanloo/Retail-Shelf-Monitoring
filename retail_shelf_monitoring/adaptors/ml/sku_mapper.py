import json
from pathlib import Path
from typing import Dict, Optional

from ...frameworks.logging_config import get_logger

logger = get_logger(__name__)


class SKUMapper:
    """Maps YOLO class IDs to SKU identifiers"""

    def __init__(self, mapping_file: Optional[str] = None):
        self.class_to_sku: Dict[int, str] = {}
        self.sku_to_class: Dict[str, int] = {}

        if mapping_file:
            self.load_mapping(mapping_file)

    def load_mapping(self, mapping_file: str):
        """
        Load mapping from JSON file
        Expected format: {"0": "sku_123", "1": "sku_456", ...}
        """
        path = Path(mapping_file)
        if not path.exists():
            raise FileNotFoundError(f"SKU mapping file not found: {mapping_file}")

        with open(path, "r") as f:
            mapping_data = json.load(f)

        self.class_to_sku = {int(k): v for k, v in mapping_data.items()}
        self.sku_to_class = {v: int(k) for k, v in mapping_data.items()}

        logger.info(f"Loaded SKU mapping: {len(self.class_to_sku)} classes")

    def map_class_to_sku(self, class_id: int) -> str:
        """Map class ID to SKU, return generic if not found"""
        return self.class_to_sku.get(class_id, f"unknown_class_{class_id}")

    def map_sku_to_class(self, sku_id: str) -> Optional[int]:
        """Map SKU to class ID"""
        return self.sku_to_class.get(sku_id)

    def add_mapping(self, class_id: int, sku_id: str):
        """Add or update a mapping"""
        # Remove old reverse mapping if updating
        if class_id in self.class_to_sku:
            old_sku = self.class_to_sku[class_id]
            if old_sku in self.sku_to_class:
                del self.sku_to_class[old_sku]

        self.class_to_sku[class_id] = sku_id
        self.sku_to_class[sku_id] = class_id

    def save_mapping(self, output_file: str):
        """Save current mapping to JSON file"""
        mapping_data = {str(k): v for k, v in self.class_to_sku.items()}

        with open(output_file, "w") as f:
            json.dump(mapping_data, f, indent=2)

        logger.info(f"Saved SKU mapping to {output_file}")
