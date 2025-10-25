from dependency_injector import containers, providers

from .adaptors.grid.grid_detector import GridDetector
from .adaptors.ml.yolo_detector import YOLOv11Detector
from .adaptors.repositories.postgres_planogram_repository import (
    PostgresPlanogramRepository,
)
from .adaptors.repositories.postgres_shelf_repository import PostgresShelfRepository
from .frameworks.config import AppConfig
from .frameworks.database import DatabaseManager
from .frameworks.logging_config import get_logger
from .usecases.planogram_generation import PlanogramGenerationUseCase
from .usecases.shelf_management import ShelfManagementUseCase


class ApplicationContainer(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(
        modules=[
            "retail_shelf_monitoring.__main__",
            "retail_shelf_monitoring.usecases.planogram_generation",
            "retail_shelf_monitoring.usecases.shelf_management",
        ]
    )

    config = providers.Singleton(AppConfig.from_yaml_or_default)

    logger = providers.Singleton(get_logger, name="retail_shelf_monitoring")

    database_manager = providers.Singleton(
        DatabaseManager, database_url=config.provided.database.url
    )

    shelf_repository = providers.Factory(
        PostgresShelfRepository,
        session_factory=database_manager.provided.get_session,
    )

    planogram_repository = providers.Factory(
        PostgresPlanogramRepository,
        session_factory=database_manager.provided.get_session,
    )

    yolo_detector = providers.Singleton(
        YOLOv11Detector,
        model_path=config.provided.ml.model_path,
        confidence_threshold=config.provided.ml.confidence_threshold,
        nms_threshold=config.provided.ml.nms_threshold,
        device=config.provided.ml.device,
    )

    grid_detector = providers.Factory(
        GridDetector,
        clustering_method=config.provided.grid.clustering_method,
        eps=config.provided.grid.eps,
        min_samples=config.provided.grid.min_samples,
    )

    shelf_management_usecase = providers.Factory(
        ShelfManagementUseCase,
        shelf_repository=shelf_repository,
    )

    planogram_generation_usecase = providers.Factory(
        PlanogramGenerationUseCase,
        shelf_repository=shelf_repository,
        planogram_repository=planogram_repository,
        detector=yolo_detector,
        grid_detector=grid_detector,
    )


class TestContainer(containers.DeclarativeContainer):
    """Test-specific dependency injection container with mocked dependencies."""

    # Configuration with test defaults
    config = providers.Configuration()

    # Mock dependencies for testing
    # Add your mock providers here
    # Example:
    # mock_database_client = providers.Factory(
    #     lambda: None,  # Mock database client for testing
    # )

    # mock_user_repository = providers.Factory(
    #     lambda: None,  # Mock repository for testing
    # )

    # mock_user_service = providers.Factory(
    #     lambda: None,  # Mock service for testing
    # )
