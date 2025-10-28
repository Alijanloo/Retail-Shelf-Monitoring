from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from retail_shelf_monitoring.entities.alert import Alert
from retail_shelf_monitoring.entities.common import AlertType
from retail_shelf_monitoring.frameworks.logging_config import get_logger

logger = get_logger(__name__)


class AlertPanel(QWidget):
    alert_confirmed_signal = Signal(str, str)
    alert_dismissed_signal = Signal(str)
    alert_selected_signal = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.alerts = {}
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()

        header = QLabel("Active Alerts")
        header.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(header)

        filter_layout = QHBoxLayout()

        self.type_filter = QComboBox()
        self.type_filter.addItem("All Types", None)
        for alert_type in AlertType:
            self.type_filter.addItem(alert_type.value.title(), alert_type)
        self.type_filter.currentIndexChanged.connect(self._apply_filters)

        filter_layout.addWidget(QLabel("Type:"))
        filter_layout.addWidget(self.type_filter)

        layout.addLayout(filter_layout)

        self.alert_list = QListWidget()
        self.alert_list.itemClicked.connect(self._on_alert_clicked)
        layout.addWidget(self.alert_list)

        action_layout = QHBoxLayout()

        self.confirm_btn = QPushButton("✓ Confirm")
        self.confirm_btn.clicked.connect(self._on_confirm_clicked)
        self.confirm_btn.setStyleSheet("background-color: #28a745; color: white;")

        self.dismiss_btn = QPushButton("✗ Dismiss")
        self.dismiss_btn.clicked.connect(self._on_dismiss_clicked)
        self.dismiss_btn.setStyleSheet("background-color: #dc3545; color: white;")

        self.details_btn = QPushButton("📋 Details")
        self.details_btn.clicked.connect(self._on_details_clicked)

        action_layout.addWidget(self.confirm_btn)
        action_layout.addWidget(self.dismiss_btn)
        action_layout.addWidget(self.details_btn)

        layout.addLayout(action_layout)

        self.stats_label = QLabel("Total: 0 | Critical: 0 | High: 0")
        self.stats_label.setStyleSheet("color: #888;")
        layout.addWidget(self.stats_label)

        self.setLayout(layout)

    @Slot(object)
    def add_alert(self, alert: Alert):
        if alert.alert_id not in self.alerts:
            self.alerts[alert.alert_id] = alert
            self._refresh_list()
            logger.info(f"Added alert {alert.alert_id} to panel")

    @Slot(str)
    def remove_alert(self, alert_id: str):
        if alert_id in self.alerts:
            del self.alerts[alert_id]
            self._refresh_list()

    def _refresh_list(self):
        self.alert_list.clear()

        filtered_alerts = self._get_filtered_alerts()

        for alert in sorted(
            filtered_alerts,
            key=lambda a: (a.priority.value, a.first_seen),
            reverse=True,
        ):
            item = QListWidgetItem()
            priority_str = alert.priority.value.upper()
            alert_str = alert.alert_type.value
            text = f"[{priority_str}] {alert_str}: {alert.message}"
            item.setText(text)
            item.setData(Qt.UserRole, alert.alert_id)

            self.alert_list.addItem(item)

        self._update_stats()

    def _get_filtered_alerts(self):
        alerts = list(self.alerts.values())

        type_filter = self.type_filter.currentData()
        if type_filter:
            alerts = [a for a in alerts if a.alert_type == type_filter]

        priority_filter = self.priority_filter.currentData()
        if priority_filter:
            alerts = [a for a in alerts if a.priority == priority_filter]

        return alerts

    def _update_stats(self):
        total = len(self.alerts)
        self.stats_label.setText(f"Total: {total}")

    @Slot()
    def _apply_filters(self):
        self._refresh_list()

    @Slot()
    def _on_alert_clicked(self, item: QListWidgetItem):
        alert_id = item.data(Qt.UserRole)
        alert = self.alerts.get(alert_id)
        if alert:
            self.alert_selected_signal.emit(alert)

    @Slot()
    def _on_confirm_clicked(self):
        current = self.alert_list.currentItem()
        if current:
            alert_id = current.data(Qt.UserRole)
            staff_id = "staff_001"
            self.alert_confirmed_signal.emit(alert_id, staff_id)

    @Slot()
    def _on_dismiss_clicked(self):
        current = self.alert_list.currentItem()
        if current:
            alert_id = current.data(Qt.UserRole)
            self.alert_dismissed_signal.emit(alert_id)

    @Slot()
    def _on_details_clicked(self):
        current = self.alert_list.currentItem()
        if current:
            alert_id = current.data(Qt.UserRole)
            alert = self.alerts.get(alert_id)
            if alert:
                logger.info(f"Showing details for alert {alert_id}: {alert}")
