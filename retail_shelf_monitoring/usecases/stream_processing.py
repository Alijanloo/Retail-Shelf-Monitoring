from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import numpy as np

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
    shelf_id: Optional[str] = None
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

    async def process_frame(
        self,
        frame: np.ndarray,
        frame_id: str,
        timestamp: datetime,
    ) -> StreamProcessingResult:
        frame_metadata = Frame(
            frame_id=frame_id,
            timestamp=timestamp,
            source_id="camera",
        )

        alignment_result = self.shelf_aligner.align_to_best_reference(
            frame, frame_metadata
        )

        if not alignment_result:
            logger.debug("No shelf alignment found for frame")
            return StreamProcessingResult(
                success=False,
                reason="no_alignment",
            )

        shelf_id, aligned_metadata, aligned_image = alignment_result

        detections = await self.detection_processing.process_aligned_frame(
            aligned_image=aligned_image,
            frame_metadata=aligned_metadata,
            shelf_id=shelf_id,
        )

        if self.tracker and detections:
            self.tracker.update(detections)

        if not detections:
            logger.debug(f"No detections for shelf {shelf_id}")
            return StreamProcessingResult(
                success=True,
                shelf_id=shelf_id,
            )

        planogram = await self.planogram_repository.get_by_shelf_id(shelf_id)
        if not planogram:
            logger.debug(
                f"No planogram found for shelf {shelf_id}, returning detections only"
            )
            return StreamProcessingResult(
                success=True,
                shelf_id=shelf_id,
                detections=detections,
                reason="no_planogram",
            )

        cell_state_result = self.cell_state_computation.compute_cell_states(
            planogram=planogram,
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
            f"Processed frame {frame_id}: {len(detections)} detections, "
            f"{len(generated_alerts)} new alerts"
        )

        return StreamProcessingResult(
            success=True,
            shelf_id=shelf_id,
            detections=detections,
            alerts=generated_alerts,
            cell_states=cell_state_result["cell_states"],
            summary=cell_state_result["summary"],
        )
