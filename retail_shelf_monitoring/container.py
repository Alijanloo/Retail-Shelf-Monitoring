from datetime import timedelta

from dependency_injector import containers, providers
from redis import Redis

from .adaptors.keyframe_selector import KeyframeSelector
from .adaptors.ml.sku_detector import SKUDetector
from .adaptors.ml.yolo_detector import YOLOv11Detector
from .adaptors.repositories.postgres_planogram_repository import (
    PostgresPlanogramRepository,
)
from .adaptors.repositories.redis_alert_repository import RedisAlertRepository
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


def _create_inference_model(
    model_path: str, pytorch_model: str, device: str, engine_type: str = "openvino"
):
    if engine_type.lower() == "onnx_runtime":
        from .frameworks.inference_engines.onnx_runtime import ONNXRuntimeModel

        return ONNXRuntimeModel(model_path=model_path, device=device.lower())
    elif engine_type.lower() == "pytorch_tensorrt":
        from .frameworks.inference_engines.pytorch_tensorrt import PyTorchTensorRTModel

        return PyTorchTensorRTModel(
            model_path=model_path,
            pytorch_model=pytorch_model,
            device=device.lower(),
            optimize_for_inference=True,
        )
    elif engine_type.lower() == "openvino":
        from .frameworks.inference_engines.openvino_model import OpenVINOModel

        return OpenVINOModel(model_path=model_path, device=device)
    elif engine_type.lower() == "tensorrt":
        from .frameworks.inference_engines.tensor_rt import TensorRTModel

        return TensorRTModel(onnx_path=model_path)
    else:
        raise ValueError(f"Unsupported inference engine type: {engine_type}")


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
        max_bbox_width=config.provided.tracking.max_bbox_width,
        max_bbox_height=config.provided.tracking.max_bbox_height,
    )

    # Inference Models
    yolo_inference_model = providers.Singleton(
        _create_inference_model,
        model_path=config.provided.ml.model_path,
        pytorch_model=config.provided.ml.pytorch_model,
        device=config.provided.ml.device,
        engine_type=config.provided.ml.inference_engine,
    )

    sku_inference_model = providers.Singleton(
        _create_inference_model,
        model_path=config.provided.sku_detection.model_path,
        pytorch_model=config.provided.sku_detection.pytorch_model,
        device=config.provided.sku_detection.device,
        engine_type=config.provided.sku_detection.inference_engine,
    )

    sku_detector = providers.Singleton(
        SKUDetector,
        inference_model=sku_inference_model,
        index_path=config.provided.sku_detection.index_path,
        top_k=config.provided.sku_detection.top_k,
        use_gpu=config.provided.sku_detection.use_gpu,
        gpu_id=config.provided.sku_detection.gpu_id,
    )

    yolo_detector = providers.Singleton(
        YOLOv11Detector,
        inference_model=yolo_inference_model,
        confidence_threshold=config.provided.ml.confidence_threshold,
        nms_threshold=config.provided.ml.nms_threshold,
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

    alert_repository = providers.Factory(
        RedisAlertRepository,
        redis_client=redis_client,
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
    )

    alert_management_usecase = providers.Factory(
        AlertManagementUseCase,
        alert_repository=alert_repository,
    )

    stream_processing_usecase = providers.Factory(
        StreamProcessingUseCase,
        shelf_aligner=shelf_aligner,
        detection_processing=detection_processing_usecase,
        planogram_repository=planogram_repository,
        tracker=tracker,
        keyframe_selector=keyframe_selector,
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
