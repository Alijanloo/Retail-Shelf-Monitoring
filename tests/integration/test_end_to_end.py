import asyncio
import time
from pathlib import Path

import cv2

from retail_shelf_monitoring.container import ApplicationContainer
from retail_shelf_monitoring.frameworks.logging_config import get_logger

logger = get_logger(__name__)


def main():
    container = ApplicationContainer()
    container.init_resources()

    alert_repo = container.alert_repository()
    asyncio.run(alert_repo.delete_index())

    db_mng = container.database_manager()
    db_mng.create_tables()

    stream_processing_usecase = container.stream_processing_usecase()

    video_path = (
        "data/vecteezy_kyiv-ukraine-dec-22-2024-shelves-filled-with-an_54312307.mov"
    )
    if not Path(video_path).exists():
        logger.warning(
            f"Test video not found at {video_path}. "
            "Using the first test set image as fallback."
        )
        test_images = list(Path("data/test_set").rglob("*.jpg"))
        if test_images:
            video_path = str(test_images[0])
        else:
            logger.error("No test data available. Exiting.")
            return

    logger.info(f"Starting stream processing: {video_path}")

    processed_count = 0
    max_frames = 10

    try:
        logger.info(f"Processing up to {max_frames} frames...")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            error_msg = f"Cannot open video source: {video_path}"
            logger.error(error_msg)
            return

        while True:
            ret, frame = cap.read()

            if not ret:
                logger.warning(f"End of stream or read error from {video_path}")
                break

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            asyncio.run(
                stream_processing_usecase.process_frame(
                    frame, f"frame_{processed_count}", timestamp
                )
            )

            processed_count += 1

        logger.info(f"\n✓ Successfully processed {processed_count} frames")
        logger.info("✓ Aligned frames saved to: data/aligned_frames/")

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")

    finally:
        logger.info("Stopping stream processing...")


if __name__ == "__main__":
    main()
