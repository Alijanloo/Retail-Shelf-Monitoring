from pathlib import Path
from typing import Optional

import cv2
from dependency_injector.wiring import Provide, inject

from ..adaptors.grid.grid_detector import GridDetector
from ..adaptors.ml.yolo_detector import YOLOv11Detector
from ..container import ApplicationContainer
from ..entities.planogram import Planogram
from ..frameworks.exceptions import EntityNotFoundError, ValidationError
from ..frameworks.logging_config import get_logger
from .interfaces.repositories import PlanogramRepository, ShelfRepository

logger = get_logger(__name__)


class PlanogramGenerationUseCase:
    @inject
    def __init__(
        self,
        shelf_repository: ShelfRepository = Provide[
            ApplicationContainer.shelf_repository
        ],
        planogram_repository: PlanogramRepository = Provide[
            ApplicationContainer.planogram_repository
        ],
        detector: YOLOv11Detector = Provide[ApplicationContainer.yolo_detector],
        grid_detector: GridDetector = Provide[ApplicationContainer.grid_detector],
    ):
        self.shelf_repository = shelf_repository
        self.planogram_repository = planogram_repository
        self.detector = detector
        self.grid_detector = grid_detector

    async def generate_planogram_from_reference(
        self,
        shelf_id: str,
        reference_image_path: str,
        clustering_method: str = "dbscan",
        eps: float = 15.0,
        min_samples: int = 2,
    ) -> Planogram:
        shelf = await self.shelf_repository.get_by_id(shelf_id)
        if not shelf:
            raise EntityNotFoundError("Shelf", shelf_id)

        image_path = Path(reference_image_path)
        if not image_path.exists():
            raise FileNotFoundError(
                f"Reference image not found: {reference_image_path}"
            )

        logger.info(f"Loading reference image: {reference_image_path}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValidationError(f"Failed to load image: {reference_image_path}")

        logger.info(f"Running SKU detection on reference image for shelf {shelf_id}")
        detections = self.detector.detect(image)

        if not detections:
            raise ValidationError(
                f"No SKUs detected in reference image for shelf {shelf_id}"
            )

        logger.info(f"Detected {len(detections)} SKUs in reference image")

        grid_detector = GridDetector(
            clustering_method=clustering_method, eps=eps, min_samples=min_samples
        )

        logger.info("Generating planogram grid structure")
        grid, clustering_params = grid_detector.detect_grid(detections)

        planogram = Planogram(
            shelf_id=shelf_id,
            reference_image_path=reference_image_path,
            grid=grid,
            clustering_params=clustering_params,
            meta={
                "store": shelf.store_id,
                "aisle": shelf.aisle,
                "section": shelf.section,
                "total_items": grid.total_items,
                "total_rows": len(grid.rows),
            },
        )

        logger.info(f"Persisting planogram for shelf {shelf_id}")
        saved_planogram = await self.planogram_repository.create(planogram)

        logger.info(
            f"Successfully generated planogram for shelf {shelf_id}: "
            f"{len(grid.rows)} rows, {grid.total_items} items"
        )

        return saved_planogram

    async def regenerate_planogram(
        self, shelf_id: str, new_reference_image_path: Optional[str] = None
    ) -> Planogram:
        existing = await self.planogram_repository.get_by_shelf_id(shelf_id)
        if not existing:
            raise EntityNotFoundError("Planogram", shelf_id)

        ref_image_path = new_reference_image_path or existing.reference_image_path

        return await self.generate_planogram_from_reference(
            shelf_id=shelf_id,
            reference_image_path=ref_image_path,
            clustering_method=existing.clustering_params.row_clustering_method,
            eps=existing.clustering_params.eps,
            min_samples=existing.clustering_params.min_samples,
        )
