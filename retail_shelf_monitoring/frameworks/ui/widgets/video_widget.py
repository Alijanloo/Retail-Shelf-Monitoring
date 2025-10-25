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
        self.show_detections = True
        self.show_grid = False
        self.planogram_grid = None

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

    @Slot(list)
    def update_detections(self, detections: List[Detection]):
        self.detections = detections

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

        if self.show_detections and self.detections:
            frame = self._draw_detections(frame, self.detections)

        if self.show_grid and self.planogram_grid:
            frame = self._draw_planogram_grid(frame, self.planogram_grid)

        self._display_frame(frame)

    def _draw_detections(
        self, frame: np.ndarray, detections: List[Detection]
    ) -> np.ndarray:
        for detection in detections:
            bbox = detection.bbox
            x1, y1 = int(bbox.x1), int(bbox.y1)
            x2, y2 = int(bbox.x2), int(bbox.y2)

            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

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
