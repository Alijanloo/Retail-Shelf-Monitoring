import numpy as np

from ..frameworks.logging_config import get_logger

logger = get_logger(__name__)


class KeyframeSelector:
    def __init__(self, diff_threshold: float = 0.1):
        self.diff_threshold = diff_threshold
        self.last_keyframe: np.ndarray = None

    def is_keyframe(self, frame: np.ndarray) -> bool:
        if self.last_keyframe is None:
            self.last_keyframe = frame.copy()
            return True

        diff = self._compute_frame_difference(frame, self.last_keyframe)

        if diff > self.diff_threshold:
            self.last_keyframe = frame.copy()
            return True

        return False

    def _compute_frame_difference(
        self, frame1: np.ndarray, frame2: np.ndarray
    ) -> float:
        import cv2

        gray1 = (
            cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            if len(frame1.shape) == 3
            else frame1
        )
        gray2 = (
            cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            if len(frame2.shape) == 3
            else frame2
        )

        diff = cv2.absdiff(gray1, gray2)
        return np.mean(diff) / 255.0
