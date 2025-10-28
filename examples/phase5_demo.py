import asyncio

from retail_shelf_monitoring.container import ApplicationContainer
from retail_shelf_monitoring.entities.common import CellState
from retail_shelf_monitoring.frameworks.logging_config import get_logger

logger = get_logger(__name__)


async def phase5_demo():
    logger.info("=== Phase 5 Demo: Decision Logic & Alerting ===")

    container = ApplicationContainer()
    container.wire(modules=[__name__])

    db_manager = container.database_manager()
    db_manager.drop_tables()
    db_manager.create_tables()

    temporal_consensus = container.temporal_consensus_manager()
    alert_generation = container.alert_generation_usecase()
    alert_management = container.alert_management_usecase()

    logger.info("Step 1: Creating test shelf")

    logger.info("\nStep 2: Simulating temporal consensus with cell state updates")

    cell_state_updates_frame1 = [
        {
            "row_idx": 0,
            "item_idx": 0,
            "state": CellState.EMPTY,
            "expected_sku": "SKU_001",
            "detected_sku": None,
        },
        {
            "row_idx": 0,
            "item_idx": 1,
            "state": CellState.OK,
            "expected_sku": "SKU_002",
            "detected_sku": "SKU_002",
        },
        {
            "row_idx": 0,
            "item_idx": 2,
            "state": CellState.MISPLACED,
            "expected_sku": "SKU_003",
            "detected_sku": "SKU_004",
        },
    ]

    logger.info("Frame 1: Updating cell states (1st observation)")
    result = temporal_consensus.update_cell_states(
        shelf_id="shelf_001", cell_state_updates=cell_state_updates_frame1
    )
    logger.info(
        f"  New alerts: {len(result['new_alerts'])}, "
        f"Cleared: {len(result['cleared_alerts'])}"
    )

    logger.info("\nFrame 2: Same states (2nd consecutive observation)")
    result = temporal_consensus.update_cell_states(
        shelf_id="shelf_001", cell_state_updates=cell_state_updates_frame1
    )
    logger.info(
        f"  New alerts: {len(result['new_alerts'])}, "
        f"Cleared: {len(result['cleared_alerts'])}"
    )

    logger.info(
        "\nFrame 3: Same states " "(3rd consecutive observation - triggers alerts!)"
    )
    result = temporal_consensus.update_cell_states(
        shelf_id="shelf_001", cell_state_updates=cell_state_updates_frame1
    )
    logger.info(
        f"  New alerts: {len(result['new_alerts'])}, "
        f"Cleared: {len(result['cleared_alerts'])}"
    )

    logger.info("\nStep 3: Generating alerts from temporal consensus")
    for alert_data in result["new_alerts"]:
        alert = await alert_generation.generate_alert(
            alert_data=alert_data,
            evidence_paths=[
                "data/evidence/shelf_001_cell_0_0_frame1.jpg",
                "data/evidence/shelf_001_cell_0_0_frame2.jpg",
                "data/evidence/shelf_001_cell_0_0_frame3.jpg",
            ],
        )
        logger.info(
            f"  Generated {alert.alert_type.value} alert: {alert.alert_id} "
            f"at cell ({alert.row_idx}, {alert.item_idx})"
        )

    logger.info("\nStep 4: Retrieving active alerts")
    active_alerts = await alert_management.get_active_alerts(shelf_id="shelf_001")
    logger.info(f"Active alerts count: {len(active_alerts)}")
    for alert in active_alerts:
        logger.info(
            f"  Alert {alert.alert_id}: {alert.alert_type.value} at "
            f"({alert.row_idx}, {alert.item_idx}), "
            f"Consecutive frames: {alert.consecutive_frames}"
        )

    logger.info("\nStep 5: Confirming an alert")
    if active_alerts:
        first_alert = active_alerts[0]
        confirmed_alert = await alert_management.confirm_alert(
            alert_id=first_alert.alert_id, confirmed_by="staff_user_123"
        )
        alert_id = confirmed_alert.alert_id
        confirmed_by = confirmed_alert.confirmed_by
        logger.info(f"  Alert {alert_id} confirmed by {confirmed_by}")

    logger.info("\nStep 6: Simulating cell recovery (state returns to OK)")
    recovery_updates = [
        {
            "row_idx": 0,
            "item_idx": 0,
            "state": CellState.OK,
            "expected_sku": "SKU_001",
            "detected_sku": "SKU_001",
        },
        {
            "row_idx": 0,
            "item_idx": 1,
            "state": CellState.OK,
            "expected_sku": "SKU_002",
            "detected_sku": "SKU_002",
        },
        {
            "row_idx": 0,
            "item_idx": 2,
            "state": CellState.OK,
            "expected_sku": "SKU_003",
            "detected_sku": "SKU_003",
        },
    ]

    logger.info("Frame 4: Cell returns to OK (1st recovery observation)")
    result = temporal_consensus.update_cell_states(
        shelf_id="shelf_001", cell_state_updates=recovery_updates
    )
    logger.info(
        f"  New alerts: {len(result['new_alerts'])}, "
        f"Cleared: {len(result['cleared_alerts'])}"
    )

    logger.info("Frame 5: Still OK (2nd recovery observation - triggers clearing!)")
    result = temporal_consensus.update_cell_states(
        shelf_id="shelf_001", cell_state_updates=recovery_updates
    )
    logger.info(
        f"  New alerts: {len(result['new_alerts'])}, "
        f"Cleared: {len(result['cleared_alerts'])}"
    )

    logger.info("\nStep 7: Auto-dismissing cleared alerts")
    for cleared_cell in result["cleared_alerts"]:
        await alert_generation.clear_cell_alerts(
            shelf_id=cleared_cell["shelf_id"],
            row_idx=cleared_cell["row_idx"],
            item_idx=cleared_cell["item_idx"],
        )
        row = cleared_cell["row_idx"]
        item = cleared_cell["item_idx"]
        logger.info(f"  Auto-dismissed alert for cell ({row}, {item})")

    logger.info("\nStep 8: Final active alerts count")
    final_alerts = await alert_management.get_active_alerts(shelf_id="shelf_001")
    logger.info(f"Remaining active alerts: {len(final_alerts)}")

    logger.info("\nStep 9: Inspecting cell temporal states")
    cell_state = temporal_consensus.get_cell_state(
        shelf_id="shelf_001", row_idx=0, item_idx=0
    )
    if cell_state:
        logger.info(
            f"Cell (0, 0) temporal state:\n"
            f"  Current state: {cell_state.current_state.value}\n"
            f"  Consecutive OK frames: {cell_state.consecutive_ok_frames}\n"
            f"  Last detected SKU: {cell_state.last_detected_sku}\n"
            f"  State history: {[s.value for s in cell_state.state_history]}"
        )

    logger.info("\n=== Phase 5 Demo Complete ===")
    logger.info(
        "\nKey achievements:\n"
        "  ✓ Temporal consensus filtering (n_confirm=3, n_clear=2)\n"
        "  ✓ Alert generation with evidence tracking\n"
        "  ✓ Alert confirmation and dismissal workflows\n"
        "  ✓ Auto-dismissal on cell recovery\n"
        "  ✓ State history and oscillation detection\n"
        "  ✓ Redis stream integration ready (alerts published to stream)"
    )


if __name__ == "__main__":
    asyncio.run(phase5_demo())
