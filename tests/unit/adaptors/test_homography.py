import cv2
import numpy as np
import pytest

from retail_shelf_monitoring.usecases.shelf_aligner.feature_matcher import (
    FeatureMatcher,
    FeatureMatchResult,
)
from retail_shelf_monitoring.usecases.shelf_aligner.homography import (
    HomographyEstimator,
)


class TestHomographyEstimator:
    @pytest.fixture
    def estimator(self):
        return HomographyEstimator(
            ransac_reproj_threshold=5.0,
            min_inlier_ratio=0.3,
            min_inliers=10,
            max_iterations=2000,
        )

    @pytest.fixture
    def feature_matcher(self):
        return FeatureMatcher(feature_type="orb", max_features=1000)

    @pytest.fixture
    def sample_images(self):
        img1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img2 = img1.copy()
        return img1, img2

    def test_homography_estimator_init(self, estimator):
        assert estimator.ransac_threshold == 5.0
        assert estimator.min_inlier_ratio == 0.3
        assert estimator.min_inliers == 10

    def test_insufficient_matches(self, estimator):
        kp = [cv2.KeyPoint(10, 10, 1)]
        match_result = FeatureMatchResult(
            matches=[], keypoints_query=kp, keypoints_ref=kp, num_matches=2
        )

        result = estimator.estimate_homography(match_result)

        assert result.is_valid is False
        assert result.matrix is None

    def test_warp_image(self, estimator):
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        H = np.eye(3, dtype=np.float32)

        warped = estimator.warp_image(image, H, (640, 480))

        assert warped.shape == (480, 640, 3)
