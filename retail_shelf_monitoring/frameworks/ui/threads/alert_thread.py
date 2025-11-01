import asyncio
import threading

from PySide6.QtCore import QThread, Signal

from retail_shelf_monitoring.frameworks.logging_config import get_logger
from retail_shelf_monitoring.usecases.alert_generation import AlertManagementUseCase

logger = get_logger(__name__)


class AlertThread(QThread):
    new_alert_signal = Signal(object)
    alert_update_signal = Signal(object)
    error_signal = Signal(str)

    def __init__(self, alert_use_case: AlertManagementUseCase, poll_interval=5.0):
        super().__init__()
        self.alert_use_case = alert_use_case
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._last_alert_ids = set()

    def run(self):
        logger.info("Started alert processing thread")

        while not self._stop_event.is_set():
            try:
                alerts = asyncio.run(self.alert_use_case.get_active_alerts())

                current_alert_ids = {alert.alert_id for alert in alerts}

                new_alerts = [
                    alert
                    for alert in alerts
                    if alert.alert_id not in self._last_alert_ids
                ]

                for alert in new_alerts:
                    self.new_alert_signal.emit(alert)

                self._last_alert_ids = current_alert_ids

            except Exception as e:
                error_msg = f"Alert processing error: {str(e)}"
                logger.error(error_msg, exc_info=True)
                self.error_signal.emit(error_msg)

            self._stop_event.wait(self.poll_interval)

        logger.info("Alert processing thread stopped")

    def stop(self):
        self._stop_event.set()
        self.wait(2000)
