from pathlib import Path

import cv2
import numpy as np
import pytest

from retail_shelf_monitoring.adaptors.ml.yolo_detector import YOLOv11Detector
from retail_shelf_monitoring.frameworks.config import AppConfig


class TestYOLODetectorIntegration:
    @pytest.fixture(scope="class")
    def config(self):
        """Load the application configuration."""
        return AppConfig.from_yaml_or_default()

    @pytest.fixture(scope="class")
    def detector(self, config):
        """Create YOLOv11Detector instance with real model."""
        model_path = config.ml.model_path
        if not Path(model_path).exists():
            pytest.skip(f"Model not found at {model_path}")

        return YOLOv11Detector(
            model_path=model_path,
            confidence_threshold=config.ml.confidence_threshold,
            nms_threshold=config.ml.nms_threshold,
            device=config.ml.device,
        )

    @pytest.fixture(scope="class")
    def test_image_path(self):
        """Get path to a test image."""
        # Use the first available test image
        test_image_path = Path("data/reference_shelves/517.png")
        if not test_image_path.exists():
            pytest.skip(f"Test image not found at {test_image_path}")
        return test_image_path

    @pytest.fixture(scope="class")
    def test_image(self, test_image_path):
        """Load test image."""
        image = cv2.imread(str(test_image_path))
        if image is None:
            pytest.skip(f"Could not load test image from {test_image_path}")
        return image

    def test_detect_and_visualize(self, detector, test_image, test_image_path):
        """Test YOLO detection on real image and save visualization."""
        detections = detector.detect(test_image)

        assert isinstance(detections, list)

        result_image = self._draw_detections(test_image.copy(), detections)

        output_path = Path("data/output.jpg")
        success = cv2.imwrite(str(output_path), result_image)
        assert success, "Failed to save visualization image"

        assert output_path.exists()
        assert output_path.stat().st_size > 1000  # At least 1KB

        print("\n=== YOLO Detection Results ===")
        print(f"Input image: {test_image_path}")
        print(f"Image shape: {test_image.shape}")
        print(f"Detections found: {len(detections)}")
        print(f"Output saved to: {output_path}")

        if detections:
            print("\nDetection details:")
            for i, detection in enumerate(detections):
                bbox = detection["bbox"]
                class_id = detection["class_id"]
                confidence = detection["confidence"]
                sku_id = detection["sku_id"]
                print(
                    f"  {i+1}. SKU: {sku_id}, Class: {class_id}, "
                    f"Conf: {confidence:.3f}, BBox: [{bbox[0]:.1f}, {bbox[1]:.1f}, "
                    f"{bbox[2]:.1f}, {bbox[3]:.1f}]"
                )
        else:
            print("No objects detected in the image")

        # Clean up temporary file
        output_path.unlink()

    def test_detect_returns_valid_format(self, detector, test_image):
        """Test that detection results have the expected format."""
        detections = detector.detect(test_image)

        assert isinstance(detections, list)

        for detection in detections:
            # Verify required keys exist
            assert "bbox" in detection
            assert "class_id" in detection
            assert "confidence" in detection
            assert "sku_id" in detection

            # Verify data types and ranges
            bbox = detection["bbox"]
            assert len(bbox) == 4
            assert all(isinstance(coord, (int, float)) for coord in bbox)
            assert bbox[0] < bbox[2]  # x1 < x2
            assert bbox[1] < bbox[3]  # y1 < y2

            assert isinstance(detection["class_id"], int)
            assert detection["class_id"] >= 0

            assert isinstance(detection["confidence"], float)
            assert 0.0 <= detection["confidence"] <= 1.0

            assert isinstance(detection["sku_id"], str)
            assert detection["sku_id"].startswith("sku_")

    def test_detect_with_empty_image(self, detector):
        """Test detection on an empty/blank image."""
        # Create a blank image
        blank_image = np.zeros((480, 640, 3), dtype=np.uint8)

        detections = detector.detect(blank_image)

        # Should return empty list for blank image
        assert isinstance(detections, list)
        assert len(detections) == 0

    def test_detect_with_noise_image(self, detector):
        """Test detection on a random noise image."""
        # Create a random noise image
        noise_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        detections = detector.detect(noise_image)

        # Should return a list (might be empty or contain false positives)
        assert isinstance(detections, list)

        # All detections should have valid format if any are returned
        for detection in detections:
            assert "bbox" in detection
            assert "confidence" in detection
            assert detection["confidence"] >= detector.conf_threshold

    def _draw_detections(self, image: np.ndarray, detections: list) -> np.ndarray:
        """Draw bounding boxes and labels on the image."""
        # Define colors for different classes (cycling through colors)
        colors = [
            (0, 255, 0),  # Green
            (255, 0, 0),  # Blue
            (0, 0, 255),  # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]

        for detection in detections:
            bbox = detection["bbox"]
            class_id = detection["class_id"]
            confidence = detection["confidence"]
            sku_id = detection["sku_id"]

            # Get coordinates
            x1, y1, x2, y2 = map(int, bbox)

            # Choose color based on class_id
            color = colors[class_id % len(colors)]

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Create label
            label = f"{sku_id}: {confidence:.2f}"

            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            # Draw background rectangle for text
            cv2.rectangle(
                image,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1,
            )

            # Draw text
            cv2.putText(
                image,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        return image

    def test_multiple_images_batch_processing(self, detector):
        """Test detection on multiple images to verify consistency."""
        # Create a few test images with different characteristics
        test_images = [
            np.zeros((300, 400, 3), dtype=np.uint8),  # Black image
            np.ones((300, 400, 3), dtype=np.uint8) * 255,  # White image
            np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8),  # Random image
        ]

        results = []
        for i, image in enumerate(test_images):
            detections = detector.detect(image)
            results.append(detections)

            # Verify format
            assert isinstance(detections, list)
            print(f"Image {i+1}: {len(detections)} detections")

        # Should process all images without errors
        assert len(results) == len(test_images)


if __name__ == "__main__":
    # Run the test directly for debugging purposes
    test_instance = TestYOLODetectorIntegration()

    # Initialize configuration and dependencies manually
    config = AppConfig.from_yaml_or_default()

    # Create detector
    model_path = config.ml.model_path
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        exit(1)

    detector = YOLOv11Detector(
        model_path=model_path,
        confidence_threshold=config.ml.confidence_threshold,
        nms_threshold=config.ml.nms_threshold,
        device=config.ml.device,
    )

    # Load test image
    test_image_path = Path("data/reference_shelves/517.png")
    if not test_image_path.exists():
        print(f"Test image not found at {test_image_path}")
        exit(1)

    test_image = cv2.imread(str(test_image_path))
    if test_image is None:
        print(f"Could not load test image from {test_image_path}")
        exit(1)

    # Run the test method
    print("Running YOLO detector integration test...")
    try:
        test_instance.test_detect_and_visualize(detector, test_image, test_image_path)
        print("Test completed successfully!")
    except Exception as e:
        print(f"Test failed with error: {e}")
        raise
