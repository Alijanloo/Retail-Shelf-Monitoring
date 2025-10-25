from typing import Dict, Optional

import numpy as np

from ..adaptors.vision.image_aligner import ShelfAligner
from ..entities.frame import Frame
from ..frameworks.logging_config import get_logger

logger = get_logger(__name__)


class ShelfLocalizationUseCase:
    def __init__(self, shelf_aligner: ShelfAligner):
        self.shelf_aligner = shelf_aligner

    def load_reference_shelves(self, reference_images: Dict[str, str]):
        self.shelf_aligner.load_reference_shelves(reference_images)
        logger.info(
            f"Loaded {len(reference_images)} reference shelf images for localization"
        )

    def localize_frame(
        self, frame: np.ndarray, frame_metadata: Frame
    ) -> Optional[tuple]:
        alignment_result = self.shelf_aligner.align_to_best_reference(
            frame=frame, frame_metadata=frame_metadata
        )

        if alignment_result is None:
            logger.debug(f"Failed to localize frame {frame_metadata.frame_id}")
            return None

        shelf_id, updated_metadata, aligned_image = alignment_result

        logger.info(
            f"Localized frame {frame_metadata.frame_id} to shelf {shelf_id} "
            f"(confidence: {updated_metadata.alignment_confidence:.2%})"
        )

        return shelf_id, updated_metadata, aligned_image

    def localize_to_specific_shelf(
        self, frame: np.ndarray, shelf_id: str
    ) -> Optional[tuple]:
        result = self.shelf_aligner.align_to_specific_shelf(
            frame=frame, shelf_id=shelf_id
        )

        if result is None:
            logger.warning(f"Failed to align frame to shelf {shelf_id}")
            return None

        aligned_image, homography_result = result

        logger.info(
            f"Aligned frame to shelf {shelf_id} "
            f"(inliers: {homography_result.num_inliers}, "
            f"ratio: {homography_result.inlier_ratio:.2%})"
        )

        return aligned_image, homography_result
