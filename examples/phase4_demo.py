"""
Phase 4 Demo: Detection, Tracking & Grid Matching

This script demonstrates:
1. Loading a reference shelf and its planogram
2. Running YOLO detection on a test frame
3. Applying object tracking
4. Matching detections against the planogram grid
5. Computing cell states (OK/EMPTY/MISPLACED)
6. Persisting detections to database
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import cv2

from retail_shelf_monitoring.container import ApplicationContainer
from retail_shelf_monitoring.entities.frame import Frame
from retail_shelf_monitoring.frameworks.database import DatabaseManager
from retail_shelf_monitoring.frameworks.logging_config import get_logger
from retail_shelf_monitoring.usecases.detection_processing import (
    DetectionProcessingUseCase,
)

logger = get_logger(__name__)


async def main():
    logger.info("=" * 80)
    logger.info("Phase 4 Demo: Detection, Tracking & Grid Matching")
    logger.info("=" * 80)

    container = ApplicationContainer()

    db_manager: DatabaseManager = container.database_manager()
    db_manager.create_tables()

    tracker = container.tracker()
    sku_detector = container.sku_detector()

    logger.info("\n3. Loading or Creating Planogram for shelf...")
    planogram_repo = container.planogram_repository()

    planograms = await planogram_repo.get_all()
    if not planograms:
        logger.warning("   No planograms found. Creating one from reference image...")

        # Check if shelf exists
        test_shelf_id = "shelf_517"

        # Generate planogram from reference image
        reference_image_path = Path("data/reference_shelves/shelf_517.png")
        if not reference_image_path.exists():
            logger.error(
                "   Reference image not found! Please add a reference shelf image."
            )
            return

        logger.info(f"   Generating planogram from {reference_image_path}...")

        # Get planogram generation use case from container
        planogram_usecase = container.planogram_generation_usecase()

        planogram = await planogram_usecase.generate_planogram_from_reference(
            shelf_id=test_shelf_id,
            reference_image_path=str(reference_image_path),
        )

        logger.info(
            f"   Created planogram with {len(planogram.grid.rows)} rows, "
            f"{planogram.grid.total_items} items"
        )
    else:
        planogram = planograms[0]
    logger.info(f"   Loaded planogram for shelf: {planogram.shelf_id}")
    logger.info(
        f"   Grid: {len(planogram.grid.rows)} rows, "
        f"{planogram.grid.total_items} items"
    )

    logger.info("\n4. Loading Test Frame...")
    test_frames = list(Path("data/aligned_frames").glob("*.jpg"))
    if not test_frames:
        logger.error("   No aligned frames found! Please run Phase 3 demo first.")
        return

    test_frame_path = test_frames[0]
    test_image = cv2.imread(str(test_frame_path))
    logger.info(f"   Loaded frame: {test_frame_path.name}")
    logger.info(f"   Shape: {test_image.shape}")

    logger.info("\n5. Running Detection...")
    detector = container.yolo_detector()
    raw_detections = detector.detect(test_image)
    logger.info(f"   Detected {len(raw_detections)} objects")

    if raw_detections:
        for i, det in enumerate(raw_detections[:5]):
            logger.info(
                f"   Detection {i}: Class {det['class_id']}, "
                f"Confidence {det['confidence']:.2f}"
            )

    logger.info("\n6. Applying Tracking...")
    tracked_detections = tracker.update(raw_detections)
    logger.info(f"   Tracked {len(tracked_detections)} objects")
    logger.info(f"   Active tracks: {len(tracker.tracks)}")

    logger.info("\n7. Creating Detection Entities...")
    detection_repo = container.detection_repository()

    frame_metadata = Frame(
        frame_id=test_frame_path.stem,
        frame_number=0,
        timestamp=datetime.now(timezone.utc),
        stream_id="test_stream",
        width=test_image.shape[1],
        height=test_image.shape[0],
    )

    detection_processing = DetectionProcessingUseCase(
        detector=detector,
        tracker=tracker,
        sku_detector=sku_detector,
        detection_repository=detection_repo,
    )

    saved_detections = await detection_processing.process_aligned_frame(
        aligned_image=test_image,
        frame_metadata=frame_metadata,
        shelf_id=planogram.shelf_id,
        aligned_frame_path=str(test_frame_path),
    )

    logger.info(f"   Saved {len(saved_detections)} detections to database")

    logger.info("\n8. Computing Cell States...")
    cell_state_computation = container.cell_state_computation()

    cell_states_result = cell_state_computation.compute_cell_states(
        planogram=planogram,
        detections=saved_detections,
        frame_timestamp=frame_metadata.timestamp,
    )

    summary = cell_states_result["summary"]
    logger.info("   Cell State Summary:")
    logger.info(f"   - OK: {summary['ok_count']}")
    logger.info(f"   - Empty: {summary['empty_count']}")
    logger.info(f"   - Misplaced: {summary['misplaced_count']}")
    logger.info(f"   - Total Cells: {summary['total_cells']}")

    logger.info("\n9. Cell States Detail:")
    for state in cell_states_result["cell_states"][:10]:
        logger.info(
            f"   Row {state['row_idx']}, Item {state['item_idx']}: "
            f"{state['state']} (Expected: {state['expected_sku']}, "
            f"Detected: {state['detected_sku']})"
        )

    logger.info("\n10. Testing Multiple Frames (Tracking)...")
    for i, frame_path in enumerate(test_frames[1:4], start=2):
        logger.info(f"\n   Processing frame {i}: {frame_path.name}")

        image = cv2.imread(str(frame_path))
        raw_dets = detector.detect(image)
        tracked_dets = tracker.update(raw_dets)

        logger.info(f"   - Detections: {len(tracked_dets)}")
        logger.info(f"   - Active tracks: {len(tracker.tracks)}")

    logger.info("\n" + "=" * 80)
    logger.info("Phase 4 Demo Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
