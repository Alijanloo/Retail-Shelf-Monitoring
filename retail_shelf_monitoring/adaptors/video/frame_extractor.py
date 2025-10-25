from typing import Optional

import numpy as np

from ...frameworks.logging_config import get_logger

logger = get_logger(__name__)


class FrameExtractor:
    def __init__(self, target_size: Optional[tuple] = None):
        self.target_size = target_size

    def extract_and_preprocess(self, frame: np.ndarray) -> np.ndarray:
        if self.target_size:
            frame = self._resize(frame)

        return frame

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        return cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)
