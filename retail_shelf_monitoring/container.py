from datetime import timedelta

from dependency_injector import containers, providers
from redis import Redis

from .adaptors.grid.grid_detector import GridDetector
from .adaptors.messaging.alert_publisher import AlertPublisher
from .adaptors.messaging.redis_stream import RedisStreamClient
from .adaptors.ml.sku_mapper import SKUMapper
from .adaptors.ml.yolo_detector import YOLOv11Detector
from .adaptors.preprocessing.image_processing import ImageProcessor
from .adaptors.preprocessing.stabilization import MotionStabilizer
from .adaptors.repositories.postgres_alert_repository import PostgresAlertRepository
from .adaptors.repositories.postgres_detection_repository import (
    PostgresDetectionRepository,
)
from .adaptors.repositories.postgres_planogram_repository import (
    PostgresPlanogramRepository,
)
from .adaptors.repositories.postgres_shelf_repository import PostgresShelfRepository
from .adaptors.tracking.bytetrack import SimpleTracker
from .adaptors.video.frame_extractor import FrameExtractor
from .adaptors.video.keyframe_selector import KeyframeSelector
from .adaptors.vision.feature_matcher import FeatureMatcher
from .adaptors.vision.homography import HomographyEstimator
from .adaptors.vision.image_aligner import ShelfAligner
from .frameworks.config import AppConfig
from .frameworks.database import DatabaseManager
from .frameworks.logging_config import get_logger
from .frameworks.streaming.frame_buffer import FrameBuffer
from .frameworks.streaming.stream_manager import StreamManager
from .usecases.alert_generation import AlertGenerationUseCase, AlertManagementUseCase
from .usecases.cell_state_computation import CellStateComputation
from .usecases.detection_processing import DetectionProcessingUseCase
from .usecases.planogram_generation import PlanogramGenerationUseCase
from .usecases.shelf_localization import ShelfLocalizationUseCase
from .usecases.shelf_management import ShelfManagementUseCase
from .usecases.stream_processing import StreamProcessingUseCase
from .usecases.temporal_consensus import TemporalConsensusManager


class ApplicationContainer(containers.DeclarativeContainer):
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

    detection_repository = providers.Factory(
        PostgresDetectionRepository,
        session_factory=database_manager.provided.get_session,
    )

    tracker = providers.Singleton(
        SimpleTracker,
        track_thresh=config.provided.tracking.track_thresh,
        match_thresh=config.provided.tracking.match_thresh,
        max_age=config.provided.tracking.max_age,
    )

    sku_mapper = providers.Singleton(
        SKUMapper, mapping_file=config.provided.tracking.sku_mapping_file
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

    frame_buffer = providers.Singleton(
        FrameBuffer, maxsize=config.provided.streaming.frame_buffer_size
    )

    stream_manager = providers.Singleton(StreamManager)

    image_processor = providers.Factory(
        ImageProcessor,
        resize_width=config.provided.streaming.max_width,
        resize_height=config.provided.streaming.max_height,
        apply_clahe=False,
    )

    motion_stabilizer = providers.Factory(MotionStabilizer, smoothing_radius=30)

    frame_extractor = providers.Factory(FrameExtractor, target_size=None)

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
        feature_matcher=feature_matcher,
        homography_estimator=homography_estimator,
        min_alignment_confidence=config.provided.homography.min_alignment_confidence,
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

    stream_processing_usecase = providers.Factory(
        StreamProcessingUseCase,
        shelf_aligner=shelf_aligner,
        output_dir=config.provided.alignment.output_dir,
    )

    shelf_localization_usecase = providers.Factory(
        ShelfLocalizationUseCase, shelf_aligner=shelf_aligner
    )

    detection_processing_usecase = providers.Factory(
        DetectionProcessingUseCase,
        detector=yolo_detector,
        tracker=tracker,
        sku_mapper=sku_mapper,
        detection_repository=detection_repository,
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
        shelf_repository=shelf_repository,
        alert_publisher=alert_publisher,
    )

    alert_management_usecase = providers.Factory(
        AlertManagementUseCase,
        alert_repository=alert_repository,
        alert_publisher=alert_publisher,
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
