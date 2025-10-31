from ast import Dict
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import numpy as np

from ..adaptors.keyframe_selector import KeyframeSelector
from ..entities.alert import Alert
from ..entities.detection import Detection
from ..entities.frame import Frame
from ..entities.planogram import Planogram
from ..frameworks.logging_config import get_logger
from .alert_generation import AlertGenerationUseCase
from .cell_state_computation import CellStateComputation
from .detection_processing import DetectionProcessingUseCase
from .interfaces.repositories import PlanogramRepository
from .interfaces.tracker_interface import Tracker
from .shelf_aligner.shelf_aligner import ShelfAligner
from .temporal_consensus import TemporalConsensusManager

logger = get_logger(__name__)


@dataclass
class DetectionResult:
    success: bool
    frame: Frame = None
    detections: List[Detection] = None
    reason: Optional[str] = None

    def __post_init__(self):
        if self.detections is None:
            self.detections = []


@dataclass
class ComplianceAnalysisResult:
    success: bool
    shelf_id: str = None
    cell_states: List[dict] = None
    alerts: List[Alert] = None
    summary: Optional[dict] = None
    reason: Optional[str] = None

    def __post_init__(self):
        if self.cell_states is None:
            self.cell_states = []
        if self.alerts is None:
            self.alerts = []


@dataclass
class StreamProcessingResult:
    success: bool
    frame: Frame = None
    detections: List[Detection] = None
    alerts: List[Alert] = None
    cell_states: List[dict] = None
    summary: Optional[dict] = None
    reason: Optional[str] = None

    def __post_init__(self):
        if self.detections is None:
            self.detections = []
        if self.alerts is None:
            self.alerts = []
        if self.cell_states is None:
            self.cell_states = []


class StreamProcessingUseCase:
    def __init__(
        self,
        shelf_aligner: ShelfAligner,
        detection_processing: DetectionProcessingUseCase,
        planogram_repository: PlanogramRepository,
        tracker: Tracker,
        keyframe_selector: KeyframeSelector,
        cell_state_computation: CellStateComputation,
        temporal_consensus: TemporalConsensusManager,
        alert_generation: AlertGenerationUseCase,
    ):
        self.shelf_aligner = shelf_aligner
        self.detection_processing = detection_processing
        self.planogram_repository = planogram_repository
        self.cell_state_computation = cell_state_computation
        self.temporal_consensus = temporal_consensus
        self.alert_generation = alert_generation
        self.tracker = tracker
        self.keyframe_selector = keyframe_selector
        self.frame_index = 0

        self._last_frame: Optional[Frame] = None
        self._planograms: Dict[str, Planogram] = {}

    async def process_detections(
        self,
        frame_img: np.ndarray,
        frame_id: str,
        timestamp: datetime,
    ) -> DetectionResult:
        frame = Frame(
            frame_id=frame_id,
            frame_img=frame_img,
            timestamp=timestamp,
        )

        # frame = self.keyframe_selector.is_keyframe(frame)
        frame.is_keyframe = self.frame_index % 30 == 0
        self.frame_index += 1
        if frame.is_keyframe:
            # frame = self.shelf_aligner.align_to_best_reference(frame)
            frame.shelf_id = "shelf_517"
            frame.alignment_confidence = 0.95
            frame.inlier_ratio = 0.95
            if not frame.shelf_id:
                logger.debug("No shelf alignment found for frame")
                self._last_frame = frame
                return DetectionResult(
                    success=False,
                    frame=frame,
                    reason="no_alignment",
                )
            detections = self.detection_processing.process_aligned_frame(frame)

            if self.tracker and detections:
                self.tracker.update(detections)
        else:
            frame = self._last_frame
            if not frame.shelf_id:
                logger.debug("No shelf alignment found for frame")
                self._last_frame = frame
                return DetectionResult(
                    success=False,
                    frame=frame,
                    reason="no_alignment",
                )
            detections = self.tracker.predict()

        if not detections:
            logger.debug(f"No detections for shelf {frame.shelf_id}")
            self._last_frame = frame
            return DetectionResult(success=True, frame=frame)

        # logger.debug(f"Detected {len(detections)} objects for shelf {frame.shelf_id}")
        self._last_frame = frame
        return DetectionResult(
            success=True,
            frame=frame,
            detections=detections,
        )

    async def analyze_compliance(
        self,
        shelf_id: str,
        detections: List[Detection],
        timestamp: datetime,
    ) -> ComplianceAnalysisResult:
        if not detections:
            logger.debug(f"No detections to analyze for shelf {shelf_id}")
            return ComplianceAnalysisResult(
                success=False,
                shelf_id=shelf_id,
                reason="no_detections",
            )

        if shelf_id not in self._planograms:
            self._planograms[
                shelf_id
            ] = await self.planogram_repository.get_by_shelf_id(shelf_id)
            if not self._planograms[shelf_id]:
                logger.debug(f"No planogram found for shelf {shelf_id}")
                return ComplianceAnalysisResult(
                    success=False,
                    shelf_id=shelf_id,
                    reason="no_planogram",
                )

        cell_state_result = self.cell_state_computation.compute_cell_states(
            planogram=self._planograms[shelf_id],
            detections=detections,
            frame_timestamp=timestamp,
        )

        consensus_result = self.temporal_consensus.update_cell_states(
            shelf_id=shelf_id,
            cell_state_updates=cell_state_result["cell_states"],
        )

        generated_alerts = []
        for alert_data in consensus_result["new_alerts"]:
            try:
                alert = await self.alert_generation.generate_alert(alert_data)
                generated_alerts.append(alert)
            except Exception as e:
                logger.error(f"Failed to generate alert: {e}", exc_info=True)

        for cell_info in consensus_result["cleared_alerts"]:
            try:
                await self.alert_generation.clear_cell_alerts(
                    shelf_id=cell_info["shelf_id"],
                    row_idx=cell_info["row_idx"],
                    item_idx=cell_info["item_idx"],
                )
            except Exception as e:
                logger.error(f"Failed to clear alert: {e}", exc_info=True)

        logger.info(
            f"Compliance analysis for shelf {shelf_id}: "
            f"{len(generated_alerts)} new alerts, "
            f"{len(consensus_result['cleared_alerts'])} cleared"
        )

        return ComplianceAnalysisResult(
            success=True,
            shelf_id=shelf_id,
            cell_states=cell_state_result["cell_states"],
            alerts=generated_alerts,
            summary=cell_state_result["summary"],
        )

    async def process_frame(
        self,
        frame_img: np.ndarray,
        frame_id: str,
        timestamp: datetime,
    ) -> StreamProcessingResult:
        detection_result = await self.process_detections(frame_img, frame_id, timestamp)

        if not detection_result.success or not detection_result.detections:
            return StreamProcessingResult(
                success=detection_result.success,
                frame=detection_result.frame,
                reason=detection_result.reason,
            )

        compliance_result = await self.analyze_compliance(
            shelf_id=detection_result.frame.shelf_id,
            detections=detection_result.detections,
            timestamp=timestamp,
        )

        return StreamProcessingResult(
            success=True,
            frame=detection_result.frame,
            detections=detection_result.detections,
            alerts=compliance_result.alerts if compliance_result.success else [],
            cell_states=compliance_result.cell_states
            if compliance_result.success
            else [],
            summary=compliance_result.summary if compliance_result.success else None,
        )
