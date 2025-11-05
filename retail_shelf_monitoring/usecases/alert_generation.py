import uuid
from datetime import datetime, timezone
from typing import List, Optional

from ..entities.alert import Alert
from ..frameworks.logging_config import get_logger
from ..usecases.interfaces.repositories import AlertRepository, PlanogramRepository

logger = get_logger(__name__)


class AlertGenerationUseCase:
    def __init__(
        self,
        alert_repository: AlertRepository,
        planogram_repository: PlanogramRepository,
        alert_publisher=None,
    ):
        self.alert_repository = alert_repository
        self.planogram_repository = planogram_repository
        self.alert_publisher = alert_publisher

    async def generate_alert(
        self, alert_data: dict, evidence_paths: Optional[List[str]] = None
    ) -> Alert:
        shelf_id = alert_data["shelf_id"]
        row_idx = alert_data["row_idx"]
        item_idx = alert_data["item_idx"]

        existing_alert = await self.alert_repository.get_by_cell(
            shelf_id=shelf_id, row_idx=row_idx, item_idx=item_idx
        )

        if (
            existing_alert
            and not existing_alert.confirmed
            and not existing_alert.dismissed
        ):
            existing_alert.last_seen = datetime.now(timezone.utc)
            existing_alert.consecutive_frames = alert_data["consecutive_frames"]
            if evidence_paths:
                existing_alert.evidence_paths.extend(evidence_paths)

            updated_alert = await self.alert_repository.update(existing_alert)
            return updated_alert

        alert = Alert(
            alert_id=str(uuid.uuid4()),
            shelf_id=shelf_id,
            row_idx=row_idx,
            item_idx=item_idx,
            alert_type=alert_data["alert_type"],
            expected_sku=alert_data["expected_sku"],
            detected_sku=alert_data.get("detected_sku"),
            first_seen=datetime.now(timezone.utc),
            last_seen=datetime.now(timezone.utc),
            evidence_paths=evidence_paths or [],
            consecutive_frames=alert_data["consecutive_frames"],
        )

        saved_alert = await self.alert_repository.create(alert)

        if self.alert_publisher:
            await self.alert_publisher.publish_alert(saved_alert)

        # logger.info(
        #     f"Generated new {alert.alert_type.value} alert: {saved_alert.alert_id} "
        #     f"for shelf {shelf_id} cell ({row_idx}, {item_idx})"
        # )

        return saved_alert

    async def clear_cell_alerts(self, shelf_id: str, row_idx: int, item_idx: int):
        alert = await self.alert_repository.get_by_cell(
            shelf_id=shelf_id, row_idx=row_idx, item_idx=item_idx
        )

        if alert and not alert.confirmed:
            await self.alert_repository.dismiss_alert(alert.alert_id)
            logger.info(f"Auto-dismissed alert {alert.alert_id} (cell returned to OK)")


class AlertManagementUseCase:
    def __init__(self, alert_repository: AlertRepository, alert_publisher=None):
        self.alert_repository = alert_repository
        self.alert_publisher = alert_publisher

    async def get_active_alerts(self, shelf_id: Optional[str] = None) -> List[Alert]:
        return await self.alert_repository.get_active_alerts(shelf_id)

    async def get_alert_by_id(self, alert_id: str) -> Optional[Alert]:
        return await self.alert_repository.get_by_id(alert_id)

    async def confirm_alert(self, alert_id: str, confirmed_by: str) -> Alert:
        alert = await self.alert_repository.confirm_alert(alert_id, confirmed_by)

        if self.alert_publisher:
            await self.alert_publisher.publish_alert_update(
                alert_id=alert_id, update_type="confirmed", updated_by=confirmed_by
            )

        logger.info(f"Alert {alert_id} confirmed by {confirmed_by}")
        return alert

    async def dismiss_alert(self, alert_id: str) -> Alert:
        alert = await self.alert_repository.dismiss_alert(alert_id)

        if self.alert_publisher:
            await self.alert_publisher.publish_alert_update(
                alert_id=alert_id, update_type="dismissed"
            )

        logger.info(f"Alert {alert_id} dismissed")
        return alert
