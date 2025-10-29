from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import numpy as np

from ..adaptors.keyframe_selector import KeyframeSelector
from ..entities.alert import Alert
from ..entities.detection import Detection
from ..entities.frame import Frame
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

    async def process_frame(
        self,
        frame_img: np.ndarray,
        frame_id: str,
        timestamp: datetime,
    ) -> StreamProcessingResult:
        frame = Frame(
            frame_id=frame_id,
            frame_img=frame_img,
            timestamp=timestamp,
        )

        frame = self.keyframe_selector.is_keyframe(frame)
        if not frame.is_keyframe:
            return StreamProcessingResult(
                success=False,
                frame=frame,
                reason="not_keyframe",
            )

        frame = self.shelf_aligner.align_to_best_reference(frame)

        if not frame.shelf_id:
            logger.debug("No shelf alignment found for frame")
            return StreamProcessingResult(
                success=False,
                frame=frame,
                reason="no_alignment",
            )

        detections = self.detection_processing.process_aligned_frame(frame)

        if self.tracker and detections:
            self.tracker.update(detections)

        if not detections:
            logger.debug(f"No detections for shelf {frame.shelf_id}")
            return StreamProcessingResult(
                success=True,
                frame=frame,
                shelf_id=frame.shelf_id,
            )

        planogram = await self.planogram_repository.get_by_shelf_id(frame.shelf_id)
        if not planogram:
            logger.debug(
                f"No planogram found for shelf {frame.shelf_id}, "
                "returning detections only!"
            )
            return StreamProcessingResult(
                success=True,
                frame=frame,
                detections=detections,
                reason="no_planogram",
            )

        cell_state_result = self.cell_state_computation.compute_cell_states(
            planogram=planogram,
            detections=detections,
            frame_timestamp=timestamp,
        )

        consensus_result = self.temporal_consensus.update_cell_states(
            shelf_id=frame.shelf_id,
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
            f"Processed frame {frame_id}: {len(detections)} detections, "
            f"{len(generated_alerts)} new alerts"
        )

        return StreamProcessingResult(
            success=True,
            frame=frame,
            detections=detections,
            alerts=generated_alerts,
            cell_states=cell_state_result["cell_states"],
            summary=cell_state_result["summary"],
        )
