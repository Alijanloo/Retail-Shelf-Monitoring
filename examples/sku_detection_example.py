"""
Example: SKU Detection with MobileNet + FAISS

This example demonstrates:
1. Building a FAISS index from SKU reference images
2. Loading the index for inference
3. Identifying SKUs from cropped product images

Usage:
    # Build index (one-time setup)
    uv run python examples/sku_detection_example.py --build

    # Test SKU identification
    uv run python examples/sku_detection_example.py --test
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from retail_shelf_monitoring.adaptors.ml.sku_detector import SKUDetector
from retail_shelf_monitoring.frameworks.logging_config import get_logger

logger = get_logger(__name__)


def create_sample_dataset(output_dir: Path):
    """
    Create a sample SKU reference dataset for testing.
    In production, replace with real product images.
    """
    logger.info("Creating sample SKU reference dataset...")

    for sku_class in range(5):
        class_dir = output_dir / str(sku_class)
        class_dir.mkdir(parents=True, exist_ok=True)

        for img_idx in range(3):
            color = (
                int(np.random.randint(50, 200)),
                int(np.random.randint(50, 200)),
                int(np.random.randint(50, 200)),
            )

            img = np.ones((224, 224, 3), dtype=np.uint8)
            img[:, :] = color

            cv2.putText(
                img,
                f"SKU {sku_class}",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                img,
                f"Img {img_idx}",
                (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                1,
            )

            img_path = class_dir / f"sample_{img_idx}.jpg"
            cv2.imwrite(str(img_path), img)

    logger.info(f"Created sample dataset in {output_dir}")
    logger.info("Total: 5 SKU classes × 3 images = 15 reference images")


def build_index_example():
    """Build FAISS index from reference images."""
    logger.info("=" * 80)
    logger.info("SKU Detection Example: Building Index")
    logger.info("=" * 80)

    data_dir = Path("data/sku_references_sample")
    model_path = Path("data/mobilenet_sku.xml")
    index_path = Path("data/sku_index_sample.faiss")

    if not data_dir.exists():
        logger.warning(f"Sample dataset not found. Creating at {data_dir}...")
        create_sample_dataset(data_dir)

    if not model_path.exists():
        logger.error(
            f"\nMobileNet model not found at {model_path}\n"
            "Please export your MobileNet model to OpenVINO format:\n"
            "  mo --input_model mobilenet_v2.onnx --output_dir data/ "
            "--model_name mobilenet_sku\n"
        )
        return

    try:
        detector = SKUDetector(
            model_path=str(model_path),
            index_path=None,
            device="CPU",
        )

        num_images, num_classes = detector.build_index(
            data_dir=str(data_dir),
            index_output_path=str(index_path),
        )

        logger.info("\n" + "=" * 80)
        logger.info("Index Built Successfully!")
        logger.info("=" * 80)
        logger.info(f"Images indexed: {num_images}")
        logger.info(f"SKU classes: {num_classes}")
        logger.info(f"Index saved to: {index_path}")
        logger.info(f"Labels saved to: {index_path.with_suffix('.labels.npy')}")

    except Exception as e:
        logger.error(f"Failed to build index: {e}", exc_info=True)


def test_identification_example():
    """Test SKU identification with the built index."""
    logger.info("=" * 80)
    logger.info("SKU Detection Example: Testing Identification")
    logger.info("=" * 80)

    model_path = Path("data/mobilenet_sku.xml")
    index_path = Path("data/sku_index_sample.faiss")

    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.error("Run with --build flag first")
        return

    if not index_path.exists():
        logger.error(f"Index not found: {index_path}")
        logger.error("Run with --build flag first")
        return

    try:
        detector = SKUDetector(
            model_path=str(model_path),
            index_path=str(index_path),
            device="CPU",
            top_k=3,
        )

        logger.info("\nTesting SKU identification on sample images...\n")

        data_dir = Path("data/sku_references_sample")
        test_images = list(data_dir.rglob("*.jpg"))[:5]

        for img_path in test_images:
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            true_sku = int(img_path.parent.name)

            sku_label, confidence, top_k_results = detector.identify_sku(image)

            logger.info(f"Image: {img_path.name}")
            logger.info(f"  True SKU: {true_sku}")
            logger.info(f"  Predicted SKU: {sku_label} (confidence: {confidence:.3f})")
            logger.info(f"  Top-3 matches: {top_k_results}")
            logger.info(f"  Correct: {'✅' if sku_label == true_sku else '❌'}")
            logger.info("")

        logger.info("=" * 80)
        logger.info("Testing Complete!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Failed to test identification: {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(description="SKU Detection Example")
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build FAISS index from reference images",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test SKU identification",
    )

    args = parser.parse_args()

    if not args.build and not args.test:
        parser.print_help()
        logger.info("\nNo action specified. Use --build or --test")
        sys.exit(1)

    if args.build:
        build_index_example()

    if args.test:
        test_identification_example()


if __name__ == "__main__":
    main()
