from typing import List, Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QFrame, QLabel

from retail_shelf_monitoring.entities.detection import Detection
from retail_shelf_monitoring.frameworks.logging_config import get_logger

logger = get_logger(__name__)


class VideoWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.setStyleSheet("background-color: #1e1e1e; border: 2px solid #3d3d3d;")

        self.current_frame: Optional[np.ndarray] = None
        self.detections: List[Detection] = []
        self.cell_states: List[dict] = []
        self.current_shelf_id: Optional[str] = None
        self.show_detections = True
        self.show_grid = False
        self.planogram_grid = None
        self.homography_matrix = None

        self.update_timer = QTimer()
        self.update_timer.setInterval(33)
        self.update_timer.timeout.connect(self.render_frame)

    def start_rendering(self):
        self.update_timer.start()

    def stop_rendering(self):
        self.update_timer.stop()

    @Slot(np.ndarray, str)
    def update_frame(self, frame_bgr: np.ndarray, timestamp: str = None):
        self.current_frame = frame_bgr.copy()

    @Slot(list, object)
    def update_detections(self, detections: List[Detection], homography_matrix=None):
        self.detections = detections
        self.homography_matrix = (
            np.array(homography_matrix).reshape(3, 3)
            if homography_matrix is not None
            else None
        )

    @Slot(str, list)
    def update_cell_states(self, shelf_id: str, cell_states: List[dict]):
        self.current_shelf_id = shelf_id
        self.cell_states = cell_states
        logger.debug(
            f"Updated cell states for shelf {shelf_id}: {len(cell_states)} cells"
        )

    def render_frame(self):
        if self.current_frame is None:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                placeholder,
                "No Video Feed",
                (200, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (128, 128, 128),
                2,
            )
            self._display_frame(placeholder)
            return

        frame = self.current_frame.copy()

        if self.cell_states and self.planogram_grid:
            frame = self._draw_cell_state_overlays(frame, self.cell_states)

        if self.show_detections and self.detections:
            frame = self._draw_detections(frame, self.detections)

        if self.show_grid and self.planogram_grid:
            frame = self._draw_planogram_grid(frame, self.planogram_grid)

        self._display_frame(frame)

    def _draw_detections(
        self, frame: np.ndarray, detections: List[Detection]
    ) -> np.ndarray:
        # If detections are produced in aligned (reference) coordinates we
        # receive a homography mapping original->reference. To draw boxes on
        # the raw frame, map bbox corners from reference back to original by
        # applying the inverse homography.
        H_inv = None
        if self.homography_matrix is not None:
            try:
                H_inv = np.linalg.inv(self.homography_matrix)
            except Exception:
                H_inv = None

        for detection in detections:
            bbox = detection.bbox

            if H_inv is not None:
                # map bbox corners from reference to original frame coords
                pts_ref = np.array(
                    [
                        [bbox.x1, bbox.y1, 1.0],
                        [bbox.x2, bbox.y1, 1.0],
                        [bbox.x2, bbox.y2, 1.0],
                        [bbox.x1, bbox.y2, 1.0],
                    ]
                )

                pts_orig = (H_inv @ pts_ref.T).T
                pts_orig = pts_orig[:, :2] / pts_orig[:, 2, None]

                xs = pts_orig[:, 0]
                ys = pts_orig[:, 1]

                x1, x2 = int(max(0, np.min(xs))), int(min(frame.shape[1], np.max(xs)))
                y1, y2 = int(max(0, np.min(ys))), int(min(frame.shape[0], np.max(ys)))
            else:
                x1, y1 = int(bbox.x1), int(bbox.y1)
                x2, y2 = int(bbox.x2), int(bbox.y2)

            color = (0, 255, 0)
            try:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            except Exception as e:
                logger.error(f"Failed to draw bbox: {e}", exc_info=True)

            label = f"{detection.sku_id or 'Unknown'} {detection.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                frame,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1,
            )
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

        return frame

    def _draw_cell_state_overlays(
        self, frame: np.ndarray, cell_states: List[dict]
    ) -> np.ndarray:
        if not self.planogram_grid or not hasattr(self.planogram_grid, "rows"):
            return frame

        for cell_state in cell_states:
            state = cell_state.get("state")
            row_idx = cell_state.get("row_idx")
            item_idx = cell_state.get("item_idx")

            if state == "ok":
                continue

            cell_bbox = self._get_cell_bbox(row_idx, item_idx)
            if not cell_bbox:
                continue

            if state == "misplaced":
                color = (0, 0, 255)
                label = f"Misplaced: {cell_state.get('detected_sku', 'Unknown')}"
            elif state == "empty":
                color = (0, 255, 255)
                label = f"Empty: Expected {cell_state.get('expected_sku', 'Unknown')}"
            else:
                continue

            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (cell_bbox["x1"], cell_bbox["y1"]),
                (cell_bbox["x2"], cell_bbox["y2"]),
                color,
                -1,
            )
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

            cv2.rectangle(
                frame,
                (cell_bbox["x1"], cell_bbox["y1"]),
                (cell_bbox["x2"], cell_bbox["y2"]),
                color,
                3,
            )

            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                frame,
                (cell_bbox["x1"], cell_bbox["y1"] - label_size[1] - 10),
                (cell_bbox["x1"] + label_size[0], cell_bbox["y1"]),
                color,
                -1,
            )
            cv2.putText(
                frame,
                label,
                (cell_bbox["x1"], cell_bbox["y1"] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        return frame

    def _get_cell_bbox(self, row_idx: int, item_idx: int) -> Optional[dict]:
        if not self.planogram_grid or not hasattr(self.planogram_grid, "rows"):
            return None

        for row in self.planogram_grid.rows:
            if row.row_idx == row_idx:
                for item in row.items:
                    if item.item_idx == item_idx:
                        bbox = item.bounding_box
                        return {
                            "x1": int(bbox.x1),
                            "y1": int(bbox.y1),
                            "x2": int(bbox.x2),
                            "y2": int(bbox.y2),
                        }
        return None

    def _draw_planogram_grid(self, frame: np.ndarray, grid) -> np.ndarray:
        if not hasattr(grid, "cells"):
            return frame

        for cell in grid.cells:
            if hasattr(cell, "bounding_box"):
                bbox = cell.bounding_box
                x1, y1 = int(bbox.x_min), int(bbox.y_min)
                x2, y2 = int(bbox.x_max), int(bbox.y_max)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)

        return frame

    def _display_frame(self, frame_bgr: np.ndarray):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setPixmap(scaled_pixmap)

    def toggle_detections(self, show: bool):
        self.show_detections = show

    def toggle_grid(self, show: bool):
        self.show_grid = show

    def set_planogram(self, planogram):
        self.planogram_grid = planogram
        logger.debug("Planogram grid updated for video widget")
