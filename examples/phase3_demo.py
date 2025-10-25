"""
Phase 3 Demo: Camera Ingest & Shelf Localization

This script demonstrates the complete Phase 3 functionality:
1. Loading reference shelf images
2. Processing video streams (file or RTSP)
3. Aligning frames to reference shelves using feature matching
4. Saving aligned frames for further processing

Usage:
    uv run python docs/phase3_demo.py

Requirements:
    - Reference shelf images in data/reference_shelves/
    - Video file or RTSP stream URL
"""

import time
from pathlib import Path

from retail_shelf_monitoring.container import ApplicationContainer
from retail_shelf_monitoring.entities.stream import StreamConfig
from retail_shelf_monitoring.frameworks.logging_config import get_logger

logger = get_logger(__name__)


def main():
    logger.info("=== Phase 3 Demo: Camera Ingest & Shelf Localization ===")

    container = ApplicationContainer()
    container.init_resources()

    shelf_aligner = container.shelf_aligner()
    stream_processing_usecase = container.stream_processing_usecase()

    reference_dir = Path("data/reference_shelves")
    if not reference_dir.exists():
        logger.warning(
            f"Reference directory not found: {reference_dir}. Creating with example..."
        )
        reference_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Please add reference shelf images to data/reference_shelves/ "
            "and run the demo again."
        )
        return

    reference_images = {}
    for pattern in ["*.jpg", "*.png"]:
        for img_path in reference_dir.glob(pattern):
            shelf_id = img_path.stem
            reference_images[shelf_id] = str(img_path)

    if not reference_images:
        logger.error(
            "No reference shelf images found. "
            "Please add .jpg files to data/reference_shelves/"
        )
        return

    logger.info(f"Loading {len(reference_images)} reference shelf images...")
    shelf_aligner.load_reference_shelves(reference_images)

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

    stream_config = StreamConfig(
        stream_id="demo-stream-1",
        source_url=video_path,
        fps=30.0,
        process_every_n_frames=30,
        max_width=1920,
        max_height=1080,
    )

    logger.info(f"Starting stream processing: {stream_config.source_url}")
    stream_processing_usecase.start_stream(stream_config)

    processed_count = 0
    max_frames = 10

    try:
        logger.info(f"Processing up to {max_frames} frames...")

        while processed_count < max_frames:
            result = stream_processing_usecase.process_frame_queue()

            if result is None:
                time.sleep(0.1)
                continue

            processed_count += 1
            logger.info(
                f"[{processed_count}/{max_frames}] Aligned frame to shelf "
                f"{result['shelf_id']} (confidence: "
                f"{result['frame_metadata'].alignment_confidence:.2%})"
            )
            logger.info(f"  Saved to: {result['aligned_image_path']}")

        logger.info(f"\n✓ Successfully processed {processed_count} frames")
        logger.info("✓ Aligned frames saved to: data/aligned_frames/")

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")

    finally:
        logger.info("Stopping stream processing...")
        stream_processing_usecase.stop_all_streams()

    logger.info("\n=== Phase 3 Demo Complete ===")
    logger.info("\nNext Steps:")
    logger.info("  1. Check aligned frames in data/aligned_frames/")
    logger.info("  2. Run Phase 4 to perform SKU detection on aligned frames")
    logger.info("  3. Integrate with real RTSP streams for production")


if __name__ == "__main__":
    main()
