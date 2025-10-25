import json

import pytest

from retail_shelf_monitoring.adaptors.ml.sku_mapper import SKUMapper


class TestSKUMapper:
    def test_initialization_without_file(self):
        mapper = SKUMapper()

        assert len(mapper.class_to_sku) == 0
        assert len(mapper.sku_to_class) == 0

    def test_load_mapping_from_file(self, tmp_path):
        mapping = {"0": "sku_cola", "1": "sku_sprite", "2": "sku_water"}

        mapping_file = tmp_path / "sku_mapping.json"
        with open(mapping_file, "w") as f:
            json.dump(mapping, f)

        mapper = SKUMapper(mapping_file=str(mapping_file))

        assert len(mapper.class_to_sku) == 3
        assert mapper.class_to_sku[0] == "sku_cola"
        assert mapper.class_to_sku[1] == "sku_sprite"
        assert mapper.class_to_sku[2] == "sku_water"

    def test_load_mapping_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            SKUMapper(mapping_file="nonexistent_file.json")

    def test_map_class_to_sku(self):
        mapper = SKUMapper()
        mapper.add_mapping(0, "sku_cola")
        mapper.add_mapping(1, "sku_sprite")

        assert mapper.map_class_to_sku(0) == "sku_cola"
        assert mapper.map_class_to_sku(1) == "sku_sprite"

    def test_map_class_to_sku_unknown(self):
        mapper = SKUMapper()

        result = mapper.map_class_to_sku(99)

        assert result == "unknown_class_99"

    def test_map_sku_to_class(self):
        mapper = SKUMapper()
        mapper.add_mapping(0, "sku_cola")
        mapper.add_mapping(1, "sku_sprite")

        assert mapper.map_sku_to_class("sku_cola") == 0
        assert mapper.map_sku_to_class("sku_sprite") == 1

    def test_map_sku_to_class_unknown(self):
        mapper = SKUMapper()

        result = mapper.map_sku_to_class("unknown_sku")

        assert result is None

    def test_add_mapping(self):
        mapper = SKUMapper()

        mapper.add_mapping(5, "sku_juice")

        assert mapper.class_to_sku[5] == "sku_juice"
        assert mapper.sku_to_class["sku_juice"] == 5

    def test_add_mapping_updates_existing(self):
        mapper = SKUMapper()
        mapper.add_mapping(0, "sku_cola")

        mapper.add_mapping(0, "sku_pepsi")

        assert mapper.class_to_sku[0] == "sku_pepsi"
        assert "sku_cola" not in mapper.sku_to_class
        assert mapper.sku_to_class["sku_pepsi"] == 0

    def test_save_mapping(self, tmp_path):
        mapper = SKUMapper()
        mapper.add_mapping(0, "sku_cola")
        mapper.add_mapping(1, "sku_sprite")
        mapper.add_mapping(2, "sku_water")

        output_file = tmp_path / "output_mapping.json"
        mapper.save_mapping(str(output_file))

        assert output_file.exists()

        with open(output_file, "r") as f:
            saved_mapping = json.load(f)

        assert saved_mapping["0"] == "sku_cola"
        assert saved_mapping["1"] == "sku_sprite"
        assert saved_mapping["2"] == "sku_water"

    def test_bidirectional_mapping(self):
        mapper = SKUMapper()
        mapper.add_mapping(10, "sku_energy_drink")

        # Forward mapping
        assert mapper.map_class_to_sku(10) == "sku_energy_drink"

        # Reverse mapping
        assert mapper.map_sku_to_class("sku_energy_drink") == 10
