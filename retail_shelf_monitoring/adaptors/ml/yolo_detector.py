from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from openvino.runtime import Core

from ...frameworks.logging_config import get_logger

logger = get_logger(__name__)


class YOLOv11Detector:
    """
    YOLOv11 object detection class using OpenVINO runtime for inference.

    This class provides a complete pipeline for object detection including
    model loading, image preprocessing, inference execution, and
    post-processing with NMS filtering.

    Attributes:
        model_path (Path): Path to the ONNX model file
        conf_threshold (float): Minimum confidence threshold for detections
        nms_threshold (float): IoU threshold for Non-Maximum Suppression
        device (str): Target device for inference (CPU, GPU, etc.)
        core (Core): OpenVINO runtime core instance
        model (Model): Loaded OpenVINO model
        compiled_model (CompiledModel): Compiled model for inference
        input_layer: Model input layer information
        output_layer: Model output layer information
        input_shape: Expected input tensor shape
        input_height (int): Model input height
        input_width (int): Model input width
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.35,
        nms_threshold: float = 0.45,
        device: str = "CPU",
    ):
        """
        Initialize YOLOv11 detector with model and inference parameters.

        Args:
            model_path (str): Path to the YOLOv11 ONNX model file
            confidence_threshold (float, optional): Minimum confidence for detections.
                Defaults to 0.35.
            nms_threshold (float, optional): IoU threshold for NMS filtering.
                Defaults to 0.45.
            device (str, optional): OpenVINO target device. Defaults to "CPU".

        Raises:
            FileNotFoundError: If the model file doesn't exist
        """
        self.model_path = Path(model_path)
        self.conf_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.device = device

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info(f"Loading YOLOv11 model from {model_path}")
        self.core = Core()
        self.model = self.core.read_model(str(self.model_path))
        self.compiled_model = self.core.compile_model(self.model, self.device)

        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

        self.input_shape = self.input_layer.shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

        logger.info(f"Model loaded successfully on {device}")
        logger.info(f"Input shape: {self.input_shape}")

    def preprocess_image(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Preprocess input image for YOLOv11 inference.

        Resizes image while maintaining aspect ratio, applies padding, normalizes pixel
        values, and converts to the required tensor format.

        Args:
            image (np.ndarray): Input image in BGR format

        Returns:
            Tuple[np.ndarray, float, Tuple[int, int]]:
                - Preprocessed image blob ready for inference
                - Scale factor used for resizing
                - Original image dimensions (width, height)
        """
        orig_height, orig_width = image.shape[:2]

        scale = min(self.input_width / orig_width, self.input_height / orig_height)
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)

        resized = cv2.resize(image, (new_width, new_height))

        padded = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        padded[:new_height, :new_width] = resized

        blob = padded.transpose(2, 0, 1)
        blob = blob.astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)

        return blob, scale, (orig_width, orig_height)

    def postprocess_detections(
        self, outputs: np.ndarray, scale: float, orig_size: Tuple[int, int]
    ) -> List[Dict]:
        """
        Convert raw model outputs to detection results with coordinates scaled back to
        original image.

        Processes YOLOv11 output tensor, applies confidence filtering, converts
        bounding boxes to original image coordinates, and applies Non-Maximum
        Suppression.

        Args:
            outputs (np.ndarray): Raw model output tensor
            scale (float): Scale factor from preprocessing
            orig_size (Tuple[int, int]): Original image dimensions (width, height)

        Returns:
            List[Dict]: List of detection dictionaries containing:
                - bbox: [x1, y1, x2, y2] coordinates
                - class_id: Predicted class identifier
                - confidence: Detection confidence score
                - sku_id: SKU identifier string
        """
        orig_width, orig_height = orig_size

        # YOLOv11 outputs shape [1, 84, 8400] or [1, num_classes+4, num_boxes]
        # Need to transpose to [num_boxes, num_classes+4]
        predictions = outputs[0]

        # If predictions shape is [84, 8400], transpose to [8400, 84]
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

        detections = []

        for pred in predictions:
            x_center, y_center, width, height = pred[:4]
            class_scores = pred[4:]

            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]

            if confidence < self.conf_threshold:
                continue

            x1 = (x_center - width / 2) / scale
            y1 = (y_center - height / 2) / scale
            x2 = (x_center + width / 2) / scale
            y2 = (y_center + height / 2) / scale

            x1 = max(0, min(x1, orig_width))
            y1 = max(0, min(y1, orig_height))
            x2 = max(0, min(x2, orig_width))
            y2 = max(0, min(y2, orig_height))

            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "class_id": int(class_id),
                    "confidence": float(confidence),
                    "sku_id": f"sku_{class_id}",
                }
            )

        detections = self._apply_nms(detections)

        return detections

    def _apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """
        Apply Non-Maximum Suppression to filter overlapping detections.

        Uses OpenCV's NMSBoxes to remove redundant detections based on IoU threshold.

        Args:
            detections (List[Dict]): List of detection dictionaries

        Returns:
            List[Dict]: Filtered detections after NMS
        """
        if not detections:
            return []

        boxes = np.array([d["bbox"] for d in detections])
        scores = np.array([d["confidence"] for d in detections])

        boxes_xywh = boxes.copy()
        boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]

        indices = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(),
            scores.tolist(),
            self.conf_threshold,
            self.nms_threshold,
        )

        if len(indices) == 0:
            return []

        return [detections[i] for i in indices.flatten()]

    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Perform end-to-end object detection on input image.

        Executes the complete detection pipeline: preprocessing, inference,
        and post-processing.

        Args:
            image (np.ndarray): Input image in BGR format

        Returns:
            List[Dict]: List of detected objects with bounding boxes, classes,
            and confidence scores
        """
        blob, scale, orig_size = self.preprocess_image(image)

        outputs = self.compiled_model([blob])
        output_data = outputs[self.output_layer]

        detections = self.postprocess_detections(output_data, scale, orig_size)

        logger.debug(f"Detected {len(detections)} objects")

        return detections
