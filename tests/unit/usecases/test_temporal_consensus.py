from retail_shelf_monitoring.entities.common import AlertType, CellState
from retail_shelf_monitoring.usecases.temporal_consensus import (
    CellTemporalState,
    TemporalConsensusManager,
)


class TestCellTemporalState:
    def test_initialization(self):
        cell_state = CellTemporalState(
            shelf_id="shelf_001",
            row_idx=0,
            item_idx=0,
            expected_sku="SKU_001",
        )

        assert cell_state.shelf_id == "shelf_001"
        assert cell_state.row_idx == 0
        assert cell_state.item_idx == 0
        assert cell_state.expected_sku == "SKU_001"
        assert cell_state.current_state == CellState.UNKNOWN
        assert cell_state.consecutive_empty_frames == 0
        assert cell_state.consecutive_misplaced_frames == 0
        assert cell_state.consecutive_ok_frames == 0

    def test_update_to_empty_state(self):
        cell_state = CellTemporalState(
            shelf_id="shelf_001", row_idx=0, item_idx=0, expected_sku="SKU_001"
        )

        cell_state.update(CellState.EMPTY)

        assert cell_state.current_state == CellState.EMPTY
        assert cell_state.consecutive_empty_frames == 1
        assert cell_state.consecutive_misplaced_frames == 0
        assert cell_state.consecutive_ok_frames == 0
        assert len(cell_state.state_history) == 1

    def test_update_consecutive_frames(self):
        cell_state = CellTemporalState(
            shelf_id="shelf_001", row_idx=0, item_idx=0, expected_sku="SKU_001"
        )

        cell_state.update(CellState.EMPTY)
        cell_state.update(CellState.EMPTY)
        cell_state.update(CellState.EMPTY)

        assert cell_state.consecutive_empty_frames == 3
        assert cell_state.current_state == CellState.EMPTY

    def test_state_change_resets_counters(self):
        cell_state = CellTemporalState(
            shelf_id="shelf_001", row_idx=0, item_idx=0, expected_sku="SKU_001"
        )

        cell_state.update(CellState.EMPTY)
        cell_state.update(CellState.EMPTY)
        assert cell_state.consecutive_empty_frames == 2

        cell_state.update(CellState.OK)
        assert cell_state.consecutive_empty_frames == 0
        assert cell_state.consecutive_ok_frames == 1

    def test_state_history_tracking(self):
        cell_state = CellTemporalState(
            shelf_id="shelf_001", row_idx=0, item_idx=0, expected_sku="SKU_001"
        )

        cell_state.update(CellState.EMPTY)
        cell_state.update(CellState.OK)
        cell_state.update(CellState.MISPLACED, "SKU_002")

        assert len(cell_state.state_history) == 3
        assert cell_state.state_history == [
            CellState.EMPTY,
            CellState.OK,
            CellState.MISPLACED,
        ]
        assert cell_state.last_detected_sku == "SKU_002"

    def test_oscillation_detection(self):
        cell_state = CellTemporalState(
            shelf_id="shelf_001", row_idx=0, item_idx=0, expected_sku="SKU_001"
        )

        cell_state.update(CellState.EMPTY)
        cell_state.update(CellState.OK)
        cell_state.update(CellState.EMPTY)
        cell_state.update(CellState.OK)
        cell_state.update(CellState.MISPLACED)

        assert cell_state.is_oscillating(window=5)

    def test_no_oscillation_when_stable(self):
        cell_state = CellTemporalState(
            shelf_id="shelf_001", row_idx=0, item_idx=0, expected_sku="SKU_001"
        )

        cell_state.update(CellState.EMPTY)
        cell_state.update(CellState.EMPTY)
        cell_state.update(CellState.EMPTY)
        cell_state.update(CellState.EMPTY)

        assert not cell_state.is_oscillating(window=5)

    def test_should_trigger_oos_alert(self):
        cell_state = CellTemporalState(
            shelf_id="shelf_001", row_idx=0, item_idx=0, expected_sku="SKU_001"
        )

        cell_state.update(CellState.EMPTY)
        assert not cell_state.should_trigger_oos_alert(n_confirm=3)

        cell_state.update(CellState.EMPTY)
        assert not cell_state.should_trigger_oos_alert(n_confirm=3)

        cell_state.update(CellState.EMPTY)
        assert cell_state.should_trigger_oos_alert(n_confirm=3)

    def test_should_trigger_misplacement_alert(self):
        cell_state = CellTemporalState(
            shelf_id="shelf_001", row_idx=0, item_idx=0, expected_sku="SKU_001"
        )

        cell_state.update(CellState.MISPLACED, "SKU_002")
        cell_state.update(CellState.MISPLACED, "SKU_002")
        assert not cell_state.should_trigger_misplacement_alert(n_confirm=3)

        cell_state.update(CellState.MISPLACED, "SKU_002")
        assert cell_state.should_trigger_misplacement_alert(n_confirm=3)

    def test_should_clear_alert(self):
        cell_state = CellTemporalState(
            shelf_id="shelf_001", row_idx=0, item_idx=0, expected_sku="SKU_001"
        )

        cell_state.update(CellState.OK, "SKU_001")
        assert not cell_state.should_clear_alert(n_clear=2)

        cell_state.update(CellState.OK, "SKU_001")
        assert cell_state.should_clear_alert(n_clear=2)


class TestTemporalConsensusManager:
    def test_initialization(self):
        manager = TemporalConsensusManager(n_confirm=3, n_clear=2)

        assert manager.n_confirm == 3
        assert manager.n_clear == 2
        assert len(manager.cell_states) == 0

    def test_update_cell_states_first_observation(self):
        manager = TemporalConsensusManager(n_confirm=3, n_clear=2)

        updates = [
            {
                "row_idx": 0,
                "item_idx": 0,
                "state": CellState.EMPTY,
                "expected_sku": "SKU_001",
            }
        ]

        result = manager.update_cell_states("shelf_001", updates)

        assert len(result["new_alerts"]) == 0
        assert len(result["cleared_alerts"]) == 0
        assert (0, 0) in manager.cell_states["shelf_001"]

    def test_alert_triggered_after_n_confirm(self):
        manager = TemporalConsensusManager(n_confirm=3, n_clear=2)

        updates = [
            {
                "row_idx": 0,
                "item_idx": 0,
                "state": CellState.EMPTY,
                "expected_sku": "SKU_001",
            }
        ]

        manager.update_cell_states("shelf_001", updates)
        manager.update_cell_states("shelf_001", updates)
        result = manager.update_cell_states("shelf_001", updates)

        assert len(result["new_alerts"]) == 1
        alert = result["new_alerts"][0]
        assert alert["alert_type"] == AlertType.OOS
        assert alert["shelf_id"] == "shelf_001"
        assert alert["row_idx"] == 0
        assert alert["item_idx"] == 0
        assert alert["consecutive_frames"] == 3

    def test_misplacement_alert_generation(self):
        manager = TemporalConsensusManager(n_confirm=3, n_clear=2)

        updates = [
            {
                "row_idx": 0,
                "item_idx": 0,
                "state": CellState.MISPLACED,
                "expected_sku": "SKU_001",
                "detected_sku": "SKU_002",
            }
        ]

        manager.update_cell_states("shelf_001", updates)
        manager.update_cell_states("shelf_001", updates)
        result = manager.update_cell_states("shelf_001", updates)

        assert len(result["new_alerts"]) == 1
        alert = result["new_alerts"][0]
        assert alert["alert_type"] == AlertType.MISPLACEMENT
        assert alert["detected_sku"] == "SKU_002"

    def test_alert_clearing(self):
        manager = TemporalConsensusManager(n_confirm=3, n_clear=2)

        empty_updates = [
            {
                "row_idx": 0,
                "item_idx": 0,
                "state": CellState.EMPTY,
                "expected_sku": "SKU_001",
            }
        ]

        manager.update_cell_states("shelf_001", empty_updates)
        manager.update_cell_states("shelf_001", empty_updates)
        manager.update_cell_states("shelf_001", empty_updates)

        ok_updates = [
            {
                "row_idx": 0,
                "item_idx": 0,
                "state": CellState.OK,
                "expected_sku": "SKU_001",
                "detected_sku": "SKU_001",
            }
        ]

        manager.update_cell_states("shelf_001", ok_updates)
        result = manager.update_cell_states("shelf_001", ok_updates)

        assert len(result["cleared_alerts"]) == 1
        assert result["cleared_alerts"][0]["row_idx"] == 0
        assert result["cleared_alerts"][0]["item_idx"] == 0

    def test_multiple_cells_tracking(self):
        manager = TemporalConsensusManager(n_confirm=3, n_clear=2)

        updates = [
            {
                "row_idx": 0,
                "item_idx": 0,
                "state": CellState.EMPTY,
                "expected_sku": "SKU_001",
            },
            {
                "row_idx": 0,
                "item_idx": 1,
                "state": CellState.MISPLACED,
                "expected_sku": "SKU_002",
                "detected_sku": "SKU_003",
            },
            {
                "row_idx": 1,
                "item_idx": 0,
                "state": CellState.OK,
                "expected_sku": "SKU_004",
                "detected_sku": "SKU_004",
            },
        ]

        manager.update_cell_states("shelf_001", updates)
        manager.update_cell_states("shelf_001", updates)
        result = manager.update_cell_states("shelf_001", updates)

        assert len(result["new_alerts"]) == 2
        assert len(manager.cell_states["shelf_001"]) == 3

    def test_get_cell_state(self):
        manager = TemporalConsensusManager(n_confirm=3, n_clear=2)

        updates = [
            {
                "row_idx": 0,
                "item_idx": 0,
                "state": CellState.EMPTY,
                "expected_sku": "SKU_001",
            }
        ]

        manager.update_cell_states("shelf_001", updates)

        cell_state = manager.get_cell_state("shelf_001", 0, 0)
        assert cell_state is not None
        assert cell_state.current_state == CellState.EMPTY
        assert cell_state.consecutive_empty_frames == 1

        non_existent = manager.get_cell_state("shelf_999", 0, 0)
        assert non_existent is None

    def test_oscillation_prevents_alerts(self):
        manager = TemporalConsensusManager(n_confirm=3, n_clear=2)

        manager.update_cell_states(
            "shelf_001",
            [
                {
                    "row_idx": 0,
                    "item_idx": 0,
                    "state": CellState.EMPTY,
                    "expected_sku": "SKU_001",
                }
            ],
        )
        manager.update_cell_states(
            "shelf_001",
            [
                {
                    "row_idx": 0,
                    "item_idx": 0,
                    "state": CellState.OK,
                    "expected_sku": "SKU_001",
                    "detected_sku": "SKU_001",
                }
            ],
        )
        manager.update_cell_states(
            "shelf_001",
            [
                {
                    "row_idx": 0,
                    "item_idx": 0,
                    "state": CellState.EMPTY,
                    "expected_sku": "SKU_001",
                }
            ],
        )

        result = manager.update_cell_states(
            "shelf_001",
            [
                {
                    "row_idx": 0,
                    "item_idx": 0,
                    "state": CellState.MISPLACED,
                    "expected_sku": "SKU_001",
                    "detected_sku": "SKU_002",
                }
            ],
        )

        assert len(result["new_alerts"]) == 0
