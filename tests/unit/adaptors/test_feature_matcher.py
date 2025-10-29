import numpy as np
import pytest

from retail_shelf_monitoring.usecases.shelf_aligner.feature_matcher import (
    FeatureMatcher,
)


class TestFeatureMatcher:
    @pytest.fixture
    def orb_matcher(self):
        return FeatureMatcher(
            feature_type="orb", max_features=1000, match_threshold=0.75, min_matches=10
        )

    @pytest.fixture
    def sample_image(self):
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_orb_initialization(self, orb_matcher):
        assert orb_matcher.feature_type == "orb"
        assert orb_matcher.max_features == 1000
        assert orb_matcher.match_threshold == 0.75

    def test_extract_features(self, orb_matcher, sample_image):
        keypoints, descriptors = orb_matcher.extract_features(sample_image)

        assert keypoints is not None
        assert isinstance(keypoints, (list, tuple))

    def test_match_features(self, orb_matcher, sample_image):
        image1 = sample_image.copy()
        image2 = sample_image.copy()

        result = orb_matcher.match_features(image1, image2)

        assert result is not None
        assert hasattr(result, "matches")
        assert hasattr(result, "num_matches")

    def test_invalid_feature_type(self):
        with pytest.raises(ValueError, match="Unsupported feature type"):
            FeatureMatcher(feature_type="invalid")

    def test_precompute_reference_features(self, orb_matcher, sample_image):
        reference_images = {"shelf-1": sample_image, "shelf-2": sample_image.copy()}

        precomputed = orb_matcher.precompute_reference_features(reference_images)

        assert len(precomputed) == 2
        assert "shelf-1" in precomputed
        assert "shelf-2" in precomputed
        assert "keypoints" in precomputed["shelf-1"]
        assert "descriptors" in precomputed["shelf-1"]
