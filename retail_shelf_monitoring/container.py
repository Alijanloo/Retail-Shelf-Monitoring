from datetime import timedelta

from dependency_injector import containers, providers
from redis import Redis

from .adaptors.keyframe_selector import KeyframeSelector
from .adaptors.messaging.alert_publisher import AlertPublisher
from .adaptors.messaging.redis_stream import RedisStreamClient
from .adaptors.ml.sku_detector import SKUDetector
from .adaptors.ml.yolo_detector import YOLOv11Detector
from .adaptors.repositories.postgres_alert_repository import PostgresAlertRepository
from .adaptors.repositories.postgres_planogram_repository import (
    PostgresPlanogramRepository,
)
from .adaptors.tracking.sort import SortTracker
from .frameworks.config import AppConfig
from .frameworks.database import DatabaseManager
from .frameworks.logging_config import get_logger
from .usecases.alert_generation import AlertGenerationUseCase, AlertManagementUseCase
from .usecases.cell_state_computation import CellStateComputation
from .usecases.detection_processing import DetectionProcessingUseCase
from .usecases.grid.grid_detector import GridDetector
from .usecases.planogram_generation import PlanogramGenerationUseCase
from .usecases.shelf_aligner.feature_matcher import FeatureMatcher
from .usecases.shelf_aligner.homography import HomographyEstimator
from .usecases.shelf_aligner.shelf_aligner import ShelfAligner
from .usecases.stream_processing import StreamProcessingUseCase
from .usecases.temporal_consensus import TemporalConsensusManager


class ApplicationContainer(containers.DeclarativeContainer):
    config = providers.Singleton(AppConfig.from_yaml_or_default)

    logger = providers.Singleton(get_logger, name="retail_shelf_monitoring")

    database_manager = providers.Singleton(
        DatabaseManager, database_url=config.provided.database.url
    )

    planogram_repository = providers.Factory(
        PostgresPlanogramRepository,
        session_factory=database_manager.provided.get_session,
    )

    tracker = providers.Singleton(
        SortTracker,
        max_age=config.provided.tracking.max_age,
        min_hits=config.provided.tracking.min_hits,
        iou_threshold=config.provided.tracking.iou_threshold,
    )

    sku_detector = providers.Singleton(
        SKUDetector,
        model_path=config.provided.sku_detection.model_path,
        index_path=config.provided.sku_detection.index_path,
        device=config.provided.sku_detection.device,
        top_k=config.provided.sku_detection.top_k,
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

    keyframe_selector = providers.Factory(KeyframeSelector, diff_threshold=0.1)

    feature_matcher = providers.Singleton(
        FeatureMatcher,
        feature_type=config.provided.feature_matching.feature_type,
        max_features=config.provided.feature_matching.max_features,
        match_threshold=config.provided.feature_matching.match_threshold,
        min_matches=config.provided.feature_matching.min_matches,
    )

    homography_estimator = providers.Singleton(
        HomographyEstimator,
        ransac_reproj_threshold=config.provided.homography.ransac_reproj_threshold,
        min_inlier_ratio=config.provided.homography.min_inlier_ratio,
        min_inliers=config.provided.homography.min_inliers,
        max_iterations=config.provided.homography.max_iterations,
    )

    shelf_aligner = providers.Singleton(
        ShelfAligner,
        reference_dir=config.provided.alignment.reference_dir,
        feature_matcher=feature_matcher,
        homography_estimator=homography_estimator,
        min_alignment_confidence=config.provided.homography.min_alignment_confidence,
    )

    planogram_generation_usecase = providers.Factory(
        PlanogramGenerationUseCase,
        planogram_repository=planogram_repository,
        detector=yolo_detector,
        sku_detector=sku_detector,
        grid_detector=grid_detector,
    )

    detection_processing_usecase = providers.Factory(
        DetectionProcessingUseCase,
        detector=yolo_detector,
        tracker=tracker,
        sku_detector=sku_detector,
    )

    cell_state_computation = providers.Factory(
        CellStateComputation,
        grid_detector=grid_detector,
        position_tolerance=config.provided.grid.position_tolerance,
        confidence_threshold=config.provided.ml.confidence_threshold,
    )

    redis_client = providers.Singleton(
        Redis,
        host=config.provided.redis.host,
        port=config.provided.redis.port,
        db=config.provided.redis.db,
        password=config.provided.redis.password,
        decode_responses=False,
    )

    alert_publisher = providers.Singleton(
        AlertPublisher,
        redis_client=redis_client,
        stream_name=config.provided.alerting.stream_name,
    )

    redis_stream_client = providers.Singleton(
        RedisStreamClient,
        redis_client=redis_client,
        stream_name=config.provided.alerting.stream_name,
        consumer_group=config.provided.alerting.consumer_group,
        consumer_name=config.provided.alerting.consumer_name,
    )

    alert_repository = providers.Factory(
        PostgresAlertRepository,
        session_factory=database_manager.provided.get_session,
    )

    temporal_consensus_manager = providers.Singleton(
        TemporalConsensusManager,
        n_confirm=config.provided.alerting.n_confirm,
        n_clear=config.provided.alerting.n_clear,
        state_timeout=providers.Factory(
            timedelta, seconds=config.provided.alerting.state_timeout
        ),
    )

    alert_generation_usecase = providers.Factory(
        AlertGenerationUseCase,
        alert_repository=alert_repository,
        planogram_repository=planogram_repository,
        alert_publisher=alert_publisher,
    )

    alert_management_usecase = providers.Factory(
        AlertManagementUseCase,
        alert_repository=alert_repository,
        alert_publisher=alert_publisher,
    )

    stream_processing_usecase = providers.Factory(
        StreamProcessingUseCase,
        shelf_aligner=shelf_aligner,
        detection_processing=detection_processing_usecase,
        planogram_repository=planogram_repository,
        tracker=tracker,
        cell_state_computation=cell_state_computation,
        temporal_consensus=temporal_consensus_manager,
        alert_generation=alert_generation_usecase,
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
