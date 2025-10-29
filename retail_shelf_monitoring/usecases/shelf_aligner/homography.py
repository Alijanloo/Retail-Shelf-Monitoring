from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from ...frameworks.logging_config import get_logger
from .feature_matcher import FeatureMatchResult

logger = get_logger(__name__)


@dataclass
class HomographyResult:
    matrix: Optional[np.ndarray]
    inlier_mask: Optional[np.ndarray]
    inlier_ratio: float
    num_inliers: int
    is_valid: bool


class HomographyEstimator:
    def __init__(
        self,
        ransac_reproj_threshold: float = 5.0,
        min_inlier_ratio: float = 0.3,
        min_inliers: int = 10,
        max_iterations: int = 2000,
    ):
        self.ransac_threshold = ransac_reproj_threshold
        self.min_inlier_ratio = min_inlier_ratio
        self.min_inliers = min_inliers
        self.max_iterations = max_iterations

    def estimate_homography(self, match_result: FeatureMatchResult) -> HomographyResult:
        if match_result.num_matches < 4:
            logger.warning("Insufficient matches for homography estimation (< 4)")
            return HomographyResult(
                matrix=None,
                inlier_mask=None,
                inlier_ratio=0.0,
                num_inliers=0,
                is_valid=False,
            )

        query_pts = np.float32(
            [match_result.keypoints_query[m.queryIdx].pt for m in match_result.matches]
        ).reshape(-1, 1, 2)

        ref_pts = np.float32(
            [match_result.keypoints_ref[m.trainIdx].pt for m in match_result.matches]
        ).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(
            query_pts,
            ref_pts,
            cv2.RANSAC,
            self.ransac_threshold,
            maxIters=self.max_iterations,
        )

        if H is None:
            logger.warning("Homography computation failed")
            return HomographyResult(
                matrix=None,
                inlier_mask=None,
                inlier_ratio=0.0,
                num_inliers=0,
                is_valid=False,
            )

        mask = mask.ravel()
        num_inliers = int(np.sum(mask))
        inlier_ratio = num_inliers / len(match_result.matches)

        is_valid = (
            num_inliers >= self.min_inliers
            and inlier_ratio >= self.min_inlier_ratio
            and self._is_homography_valid(H)
        )

        logger.debug(
            f"Homography: {num_inliers}/{len(match_result.matches)} inliers "
            f"({inlier_ratio:.2%}), valid={is_valid}"
        )

        return HomographyResult(
            matrix=H if is_valid else None,
            inlier_mask=mask,
            inlier_ratio=inlier_ratio,
            num_inliers=num_inliers,
            is_valid=is_valid,
        )

    def _is_homography_valid(self, H: np.ndarray) -> bool:
        det = np.linalg.det(H[:2, :2])
        if det <= 0 or det > 10 or det < 0.1:
            logger.debug(f"Invalid homography: bad determinant {det}")
            return False

        if abs(H[2, 0]) > 0.002 or abs(H[2, 1]) > 0.002:
            logger.debug("Invalid homography: excessive perspective warping")
            return False

        try:
            U, S, Vt = np.linalg.svd(H[:2, :2])
            if S[0] / S[1] > 10:
                logger.debug("Invalid homography: condition number too high")
                return False
        except Exception:
            return False

        return True

    def warp_image(
        self, image: np.ndarray, H: np.ndarray, output_shape: Tuple[int, int]
    ) -> np.ndarray:
        warped = cv2.warpPerspective(
            image,
            H,
            output_shape,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        return warped
