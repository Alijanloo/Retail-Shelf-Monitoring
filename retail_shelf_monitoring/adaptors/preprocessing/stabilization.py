from collections import deque
from typing import Optional

import cv2
import numpy as np

from ...frameworks.logging_config import get_logger

logger = get_logger(__name__)


class MotionStabilizer:
    def __init__(self, smoothing_radius: int = 30):
        self.smoothing_radius = smoothing_radius
        self.prev_gray: Optional[np.ndarray] = None
        self.transforms = deque(maxlen=smoothing_radius)

    def stabilize(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return frame

        prev_pts = cv2.goodFeaturesToTrack(
            self.prev_gray,
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=30,
            blockSize=3,
        )

        if prev_pts is None or len(prev_pts) < 4:
            self.prev_gray = gray
            return frame

        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_pts, None
        )

        if curr_pts is None or status is None:
            self.prev_gray = gray
            return frame

        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        if len(prev_pts) < 4:
            self.prev_gray = gray
            return frame

        m = cv2.estimateAffinePartial2D(prev_pts, curr_pts)[0]

        if m is None:
            self.prev_gray = gray
            return frame

        dx = m[0, 2]
        dy = m[1, 2]
        da = np.arctan2(m[1, 0], m[0, 0])

        self.transforms.append([dx, dy, da])

        trajectory = np.cumsum(self.transforms, axis=0)
        smoothed_trajectory = self._smooth(trajectory)

        difference = smoothed_trajectory - trajectory
        if len(difference) > 0:
            dx_smooth = difference[-1, 0]
            dy_smooth = difference[-1, 1]
            da_smooth = difference[-1, 2]

            m_smooth = np.zeros((2, 3))
            m_smooth[0, 0] = np.cos(da_smooth)
            m_smooth[0, 1] = -np.sin(da_smooth)
            m_smooth[1, 0] = np.sin(da_smooth)
            m_smooth[1, 1] = np.cos(da_smooth)
            m_smooth[0, 2] = dx_smooth
            m_smooth[1, 2] = dy_smooth

            stabilized = cv2.warpAffine(
                frame, m_smooth, (frame.shape[1], frame.shape[0])
            )
        else:
            stabilized = frame

        self.prev_gray = gray

        return stabilized

    def _smooth(self, trajectory: np.ndarray) -> np.ndarray:
        smoothed = np.copy(trajectory)
        for i in range(len(trajectory)):
            start = max(0, i - self.smoothing_radius)
            end = min(len(trajectory), i + self.smoothing_radius + 1)
            smoothed[i] = np.mean(trajectory[start:end], axis=0)
        return smoothed

    def reset(self):
        self.prev_gray = None
        self.transforms.clear()
