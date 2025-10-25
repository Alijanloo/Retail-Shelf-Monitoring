from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from openvino.runtime import Core

from ...frameworks.logging_config import get_logger

logger = get_logger(__name__)


class YOLOv11Detector:
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.35,
        nms_threshold: float = 0.45,
        device: str = "CPU",
    ):
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
        blob, scale, orig_size = self.preprocess_image(image)

        outputs = self.compiled_model([blob])
        output_data = outputs[self.output_layer]

        detections = self.postprocess_detections(output_data, scale, orig_size)

        logger.debug(f"Detected {len(detections)} objects")

        return detections
