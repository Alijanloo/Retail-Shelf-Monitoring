import asyncio
import queue
import sys

import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from retail_shelf_monitoring.container import ApplicationContainer
from retail_shelf_monitoring.frameworks.logging_config import get_logger

from .threads.alert_thread import AlertThread
from .threads.capture_thread import CaptureThread
from .threads.inference_thread import InferenceThread
from .widgets.alert_panel import AlertPanel
from .widgets.video_widget import VideoWidget

logger = get_logger(__name__)


class MainWindow(QMainWindow):
    def __init__(self, container: ApplicationContainer):
        super().__init__()
        self.container = container
        self.setWindowTitle("Retail Shelf Monitoring System")

        self.capture_thread = None
        self.inference_thread = None
        self.alert_thread = None

        self.frame_queue = queue.Queue(maxsize=4)

        self.current_source = 0

        self._init_ui()
        self._init_threads()

    def _init_ui(self):
        central = QWidget()
        main_layout = QHBoxLayout()

        left_panel = QWidget()
        left_layout = QVBoxLayout()

        self.video_widget = VideoWidget()
        left_layout.addWidget(self.video_widget, stretch=4)

        controls_group = self._create_controls()
        left_layout.addWidget(controls_group, stretch=1)

        left_panel.setLayout(left_layout)

        self.alert_panel = AlertPanel()
        self.alert_panel.setMinimumWidth(400)
        self.alert_panel.setMaximumWidth(500)

        self.alert_panel.alert_confirmed_signal.connect(self._on_alert_confirmed)
        self.alert_panel.alert_dismissed_signal.connect(self._on_alert_dismissed)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(self.alert_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)
        central.setLayout(main_layout)
        self.setCentralWidget(central)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        self.fps_label = QLabel("FPS: 0.0")
        self.status_bar.addPermanentWidget(self.fps_label)

        self.latency_label = QLabel("Latency: 0ms")
        self.status_bar.addPermanentWidget(self.latency_label)

    def _create_controls(self) -> QGroupBox:
        group = QGroupBox("Controls")
        layout = QFormLayout()

        self.shelf_combo = QComboBox()
        self.shelf_combo.addItem("Camera 0", 0)
        self.shelf_combo.addItem("Camera 1", 1)
        video_path = (
            "data/vecteezy_kyiv-ukraine-dec-22-2024-shelves"
            "-filled-with-an_54312307.mov"
        )
        self.shelf_combo.addItem("Test Video", video_path)
        layout.addRow("Source:", self.shelf_combo)

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(30)
        self.fps_spin.setSuffix(" fps")
        layout.addRow("FPS Limit:", self.fps_spin)

        self.show_detections_check = QCheckBox("Show Detections")
        self.show_detections_check.setChecked(True)
        self.show_detections_check.toggled.connect(self.video_widget.toggle_detections)
        layout.addRow("", self.show_detections_check)

        self.show_grid_check = QCheckBox("Show Grid")
        self.show_grid_check.setChecked(False)
        self.show_grid_check.toggled.connect(self.video_widget.toggle_grid)
        layout.addRow("", self.show_grid_check)

        btn_layout = QHBoxLayout()

        self.start_btn = QPushButton("▶ Start")
        self.start_btn.clicked.connect(self._on_start)
        self.start_btn.setStyleSheet(
            "background-color: #28a745; color: white; font-weight: bold;"
        )

        self.stop_btn = QPushButton("■ Stop")
        self.stop_btn.clicked.connect(self._on_stop)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(
            "background-color: #dc3545; color: white; font-weight: bold;"
        )

        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)

        layout.addRow("", btn_layout)

        group.setLayout(layout)
        return group

    def _init_threads(self):
        alert_use_case = self.container.alert_management_usecase()
        self.alert_thread = AlertThread(alert_use_case, poll_interval=5.0)
        self.alert_thread.new_alert_signal.connect(self.alert_panel.add_alert)
        self.alert_thread.error_signal.connect(self._on_thread_error)

    @Slot()
    def _on_start(self):
        self.current_source = self.shelf_combo.currentData()

        logger.info(f"Starting monitoring for {self.current_source}")

        fps_limit = self.fps_spin.value()
        self.capture_thread = CaptureThread(
            source=self.current_source,
            fps_limit=fps_limit,
        )
        self.capture_thread.frame_signal.connect(self.video_widget.update_frame)
        self.capture_thread.frame_signal.connect(self._on_frame_captured)
        self.capture_thread.fps_signal.connect(self._on_fps_update)
        self.capture_thread.error_signal.connect(self._on_thread_error)
        self.capture_thread.start()

        stream_processing_use_case = self.container.stream_processing_usecase()

        self.inference_thread = InferenceThread(
            frame_queue=self.frame_queue,
            stream_processing_use_case=stream_processing_use_case,
        )
        self.inference_thread.result_signal.connect(self._on_inference_result)
        self.inference_thread.latency_signal.connect(self._on_latency_update)
        self.inference_thread.error_signal.connect(self._on_thread_error)
        self.inference_thread.start()

        if not self.alert_thread.isRunning():
            self.alert_thread.start()

        self.video_widget.start_rendering()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.shelf_combo.setEnabled(False)
        self.status_bar.showMessage(f"Monitoring {self.current_source}...")

    @Slot()
    def _on_stop(self):
        logger.info("Stopping monitoring")

        if self.capture_thread and self.capture_thread.isRunning():
            self.capture_thread.stop()

        if self.inference_thread and self.inference_thread.isRunning():
            self.inference_thread.stop()

        self.video_widget.stop_rendering()

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.shelf_combo.setEnabled(True)
        self.status_bar.showMessage("Stopped")

    @Slot(np.ndarray, str)
    def _on_frame_captured(self, frame, timestamp):
        try:
            self.frame_queue.put_nowait(frame.copy())
        except queue.Full:
            pass

    @Slot(list, str)
    def _on_inference_result(self, detections, shelf_id):
        self.video_widget.update_detections(detections)

    @Slot(float)
    def _on_fps_update(self, fps):
        self.fps_label.setText(f"FPS: {fps:.1f}")

    @Slot(float)
    def _on_latency_update(self, latency_ms):
        self.latency_label.setText(f"Latency: {latency_ms:.0f}ms")

    @Slot(str, str)
    def _on_alert_confirmed(self, alert_id, staff_id):
        logger.info(f"Confirming alert {alert_id} by {staff_id}")

        alert_use_case = self.container.alert_management_usecase()
        asyncio.run(alert_use_case.confirm_alert(alert_id, staff_id))

        self.alert_panel.remove_alert(alert_id)
        self.status_bar.showMessage(f"Alert {alert_id} confirmed", 3000)

    @Slot(str)
    def _on_alert_dismissed(self, alert_id):
        logger.info(f"Dismissing alert {alert_id}")

        alert_use_case = self.container.alert_management_usecase()
        asyncio.run(alert_use_case.dismiss_alert(alert_id))

        self.alert_panel.remove_alert(alert_id)
        self.status_bar.showMessage(f"Alert {alert_id} dismissed", 3000)

    @Slot(str)
    def _on_thread_error(self, error_msg):
        logger.error(f"Thread error: {error_msg}")
        self.status_bar.showMessage(f"Error: {error_msg}", 5000)

    def closeEvent(self, event):
        logger.info("Closing application")

        if self.capture_thread:
            self.capture_thread.stop()

        if self.inference_thread:
            self.inference_thread.stop()

        if self.alert_thread:
            self.alert_thread.stop()

        event.accept()


def main():
    app = QApplication(sys.argv)

    container = ApplicationContainer()
    container.init_resources()

    window = MainWindow(container)
    window.resize(1400, 900)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
