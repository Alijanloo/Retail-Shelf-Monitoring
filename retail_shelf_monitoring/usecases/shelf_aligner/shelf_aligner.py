from pathlib import Path
from typing import Dict, Optional

import cv2

from ...entities.frame import Frame
from ...frameworks.logging_config import get_logger
from .feature_matcher import FeatureMatcher
from .homography import HomographyEstimator

logger = get_logger(__name__)


class ShelfAligner:
    def __init__(
        self,
        reference_dir: str,
        feature_matcher: FeatureMatcher,
        homography_estimator: HomographyEstimator,
        min_alignment_confidence: float = 0.3,
    ):
        self.feature_matcher = feature_matcher
        self.homography_estimator = homography_estimator
        self.min_confidence = min_alignment_confidence
        self.reference_dir = reference_dir

        self.reference_features: Dict = {}
        self.load_reference_shelves_from_dir()

    def load_reference_shelves_from_dir(self):
        reference_path = Path(self.reference_dir)
        if not reference_path.exists() or not reference_path.is_dir():
            logger.error(f"Reference directory not found: {self.reference_dir}")
            return

        images = {}
        for image_file in reference_path.glob("*.*"):
            shelf_id = image_file.stem
            image = cv2.imread(str(image_file))
            if image is None:
                logger.warning(
                    f"Failed to load reference image for shelf {shelf_id}: {image_file}"
                )
                continue
            images[shelf_id] = image

        self.reference_features = self.feature_matcher.precompute_reference_features(
            images
        )
        logger.info(
            f"Loaded {len(self.reference_features)} reference shelves from directory"
        )

    def align_to_best_reference(
        self,
        frame: Frame,
    ) -> Optional[Frame]:
        best_match = None
        best_confidence = 0.0

        for shelf_id, ref_data in self.reference_features.items():
            match_result = self.feature_matcher.match_features(
                query_image=frame.frame_img,
                ref_image=ref_data["image"],
                ref_keypoints=ref_data["keypoints"],
                ref_descriptors=ref_data["descriptors"],
            )

            if match_result.num_matches < 10:
                logger.warning(
                    f"Insufficient matches for shelf {shelf_id}: "
                    f"{match_result.num_matches}"
                )
                continue

            homography_result = self.homography_estimator.estimate_homography(
                match_result
            )

            if not homography_result.is_valid:
                logger.warning(f"Invalid homography for shelf {shelf_id}")
                continue

            confidence = homography_result.inlier_ratio

            if confidence > best_confidence:
                best_confidence = confidence
                best_match = {
                    "shelf_id": shelf_id,
                    "homography": homography_result.matrix,
                    "confidence": confidence,
                    "num_inliers": homography_result.num_inliers,
                    "ref_shape": ref_data["image"].shape,
                }

        if best_match is None or best_confidence < self.min_confidence:
            logger.debug(
                f"No valid shelf alignment found "
                f"(best confidence: {best_confidence:.2%})"
            )
            return frame

        ref_height, ref_width = best_match["ref_shape"][:2]

        frame.frame_img = self.homography_estimator.warp_image(
            frame.frame_img, best_match["homography"], (ref_width, ref_height)
        )

        frame.shelf_id = best_match["shelf_id"]
        frame.homography_matrix = best_match["homography"].flatten().tolist()
        frame.alignment_confidence = best_match["confidence"]
        frame.inlier_ratio = best_match["confidence"]

        logger.info(
            f"Aligned frame to shelf {best_match['shelf_id']} "
            f"(confidence: {best_match['confidence']:.2%}, "
            f"inliers: {best_match['num_inliers']})"
        )

        return frame
