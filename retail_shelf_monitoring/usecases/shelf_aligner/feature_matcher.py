from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ...frameworks.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureMatchResult:
    matches: List[cv2.DMatch]
    keypoints_query: List[cv2.KeyPoint]
    keypoints_ref: List[cv2.KeyPoint]
    inlier_mask: Optional[np.ndarray] = None
    inlier_ratio: float = 0.0
    num_matches: int = 0


class FeatureMatcher:
    def __init__(
        self,
        feature_type: str = "orb",
        max_features: int = 5000,
        match_threshold: float = 0.75,
        min_matches: int = 10,
    ):
        self.feature_type = feature_type.lower()
        self.max_features = max_features
        self.match_threshold = match_threshold
        self.min_matches = min_matches

        if self.feature_type == "orb":
            self.detector = cv2.ORB_create(nfeatures=max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        elif self.feature_type == "sift":
            self.detector = cv2.SIFT_create(nfeatures=max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")

        logger.info(f"Initialized FeatureMatcher with {feature_type.upper()}")

    def extract_features(
        self, image: np.ndarray
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        keypoints, descriptors = self.detector.detectAndCompute(gray, None)

        return keypoints, descriptors

    def match_features(
        self,
        query_image: np.ndarray,
        ref_image: np.ndarray,
        ref_keypoints: Optional[List[cv2.KeyPoint]] = None,
        ref_descriptors: Optional[np.ndarray] = None,
    ) -> FeatureMatchResult:
        query_kp, query_desc = self.extract_features(query_image)

        if query_desc is None or len(query_kp) < self.min_matches:
            kp_count = len(query_kp) if query_kp else 0
            logger.warning(f"Insufficient keypoints in query image: {kp_count}")
            return FeatureMatchResult(
                matches=[],
                keypoints_query=query_kp,
                keypoints_ref=[],
                num_matches=0,
            )

        if ref_keypoints is None or ref_descriptors is None:
            ref_kp, ref_desc = self.extract_features(ref_image)
        else:
            ref_kp, ref_desc = ref_keypoints, ref_descriptors

        if ref_desc is None or len(ref_kp) < self.min_matches:
            kp_count = len(ref_kp) if ref_kp else 0
            logger.warning(f"Insufficient keypoints in reference image: {kp_count}")
            return FeatureMatchResult(
                matches=[],
                keypoints_query=query_kp,
                keypoints_ref=ref_kp,
                num_matches=0,
            )

        knn_matches = self.matcher.knnMatch(query_desc, ref_desc, k=2)

        good_matches = []
        for match_pair in knn_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.match_threshold * n.distance:
                    good_matches.append(m)

        logger.debug(
            f"Feature matching: {len(query_kp)} query kp, {len(ref_kp)} ref kp, "
            f"{len(good_matches)} good matches"
        )

        return FeatureMatchResult(
            matches=good_matches,
            keypoints_query=query_kp,
            keypoints_ref=ref_kp,
            num_matches=len(good_matches),
        )

    def precompute_reference_features(self, reference_images: dict) -> dict:
        precomputed = {}

        for shelf_id, ref_image in reference_images.items():
            keypoints, descriptors = self.extract_features(ref_image)
            precomputed[shelf_id] = {
                "keypoints": keypoints,
                "descriptors": descriptors,
                "image": ref_image,
            }
            logger.info(
                f"Precomputed features for shelf {shelf_id}: {len(keypoints)} keypoints"
            )

        return precomputed
