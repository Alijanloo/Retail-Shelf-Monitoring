from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ...entities.frame import Frame
from ...frameworks.logging_config import get_logger
from .feature_matcher import FeatureMatcher
from .homography import HomographyEstimator, HomographyResult

logger = get_logger(__name__)


class ShelfAligner:
    def __init__(
        self,
        feature_matcher: FeatureMatcher,
        homography_estimator: HomographyEstimator,
        min_alignment_confidence: float = 0.3,
    ):
        self.feature_matcher = feature_matcher
        self.homography_estimator = homography_estimator
        self.min_confidence = min_alignment_confidence

        self.reference_features: Dict = {}

    def load_reference_shelves(self, reference_images: Dict[str, str]):
        logger.info(f"Loading {len(reference_images)} reference shelf images")

        images = {}
        for shelf_id, image_path in reference_images.items():
            path = Path(image_path)
            if not path.exists():
                logger.warning(
                    f"Reference image not found for shelf {shelf_id}: {image_path}"
                )
                continue

            image = cv2.imread(str(path))
            if image is None:
                logger.warning(f"Failed to load reference image for shelf {shelf_id}")
                continue

            images[shelf_id] = image

        self.reference_features = self.feature_matcher.precompute_reference_features(
            images
        )

        logger.info(f"Loaded {len(self.reference_features)} reference shelves")

    def align_to_best_reference(
        self,
        frame: np.ndarray,
        frame_metadata: Frame,
        candidate_shelves: Optional[List[str]] = None,
    ) -> Optional[Tuple[str, Frame, np.ndarray]]:
        if not self.reference_features:
            logger.warning("No reference shelves loaded")
            return None

        shelf_ids = (
            candidate_shelves
            if candidate_shelves
            else list(self.reference_features.keys())
        )

        best_match = None
        best_confidence = 0.0

        for shelf_id in shelf_ids:
            ref_data, homography_result = self.align_to_specific_shelf(frame, shelf_id)

            if not homography_result:
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
            return None

        ref_height, ref_width = best_match["ref_shape"][:2]

        aligned_image = self.homography_estimator.warp_image(
            frame, best_match["homography"], (ref_width, ref_height)
        )

        frame_metadata.shelf_id = best_match["shelf_id"]
        frame_metadata.homography_matrix = best_match["homography"].flatten().tolist()
        frame_metadata.alignment_confidence = best_match["confidence"]
        frame_metadata.inlier_ratio = best_match["confidence"]

        logger.info(
            f"Aligned frame to shelf {best_match['shelf_id']} "
            f"(confidence: {best_match['confidence']:.2%}, "
            f"inliers: {best_match['num_inliers']})"
        )

        return best_match["shelf_id"], frame_metadata, aligned_image

    def align_to_specific_shelf(
        self, frame: np.ndarray, shelf_id: str
    ) -> Tuple[Dict, HomographyResult]:
        if shelf_id not in self.reference_features:
            logger.warning(f"Reference shelf not loaded: {shelf_id}")
            return None, None

        ref_data = self.reference_features[shelf_id]

        match_result = self.feature_matcher.match_features(
            query_image=frame,
            ref_image=ref_data["image"],
            ref_keypoints=ref_data["keypoints"],
            ref_descriptors=ref_data["descriptors"],
        )

        if match_result.num_matches < 10:
            logger.warning(
                f"Insufficient matches for shelf {shelf_id}: {match_result.num_matches}"
            )
            return None

        homography_result = self.homography_estimator.estimate_homography(match_result)

        if not homography_result.is_valid:
            logger.warning(f"Invalid homography for shelf {shelf_id}")
            return None

        return ref_data, homography_result
