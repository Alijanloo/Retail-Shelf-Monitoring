from typing import Optional

import cv2
import numpy as np

from ...frameworks.logging_config import get_logger

logger = get_logger(__name__)


class ImageProcessor:
    def __init__(
        self,
        resize_width: Optional[int] = None,
        resize_height: Optional[int] = None,
        apply_clahe: bool = False,
    ):
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.apply_clahe = apply_clahe

        if apply_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        processed = image.copy()

        if self.resize_width and self.resize_height:
            processed = cv2.resize(
                processed,
                (self.resize_width, self.resize_height),
                interpolation=cv2.INTER_LINEAR,
            )

        if self.apply_clahe:
            if len(processed.shape) == 3:
                lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
                processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                processed = self.clahe.apply(processed)

        return processed

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            return self.clahe.apply(image)

    def denoise(
        self, image: np.ndarray, strength: int = 10, template_window: int = 7
    ) -> np.ndarray:
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(
                image, None, strength, strength, template_window, 21
            )
        else:
            return cv2.fastNlMeansDenoising(image, None, strength, template_window, 21)
