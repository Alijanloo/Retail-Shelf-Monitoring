import json
from datetime import datetime, timezone
from typing import List, Optional

from redis import Redis

from ...entities.alert import Alert
from ...entities.common import AlertType
from ...frameworks.exceptions import DatabaseError, EntityNotFoundError
from ...frameworks.logging_config import get_logger
from ...usecases.interfaces.repositories import AlertRepository

logger = get_logger(__name__)


class RedisAlertRepository(AlertRepository):
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.alerts_key_prefix = "alert:"
        self.shelf_alerts_key_prefix = "shelf_alerts:"
        self.cell_alerts_key_prefix = "cell_alert:"
        self.active_alerts_key = "active_alerts"

    async def create(self, alert: Alert) -> Alert:
        try:
            alert_data = self._to_dict(alert)
            alert_key = f"{self.alerts_key_prefix}{alert.alert_id}"

            self.redis.set(alert_key, json.dumps(alert_data))

            if not alert.dismissed:
                self.redis.sadd(self.active_alerts_key, alert.alert_id)

            shelf_key = f"{self.shelf_alerts_key_prefix}{alert.shelf_id}"
            self.redis.sadd(shelf_key, alert.alert_id)

            cell_key = (
                f"{self.cell_alerts_key_prefix}{alert.shelf_id}:{alert.row_idx}:"
                f"{alert.item_idx}"
            )
            self.redis.set(cell_key, alert.alert_id)

            logger.info(f"Alert {alert.alert_id} created in Redis")
            return alert

        except Exception as e:
            raise DatabaseError(f"Failed to create alert: {str(e)}")

    async def get_by_id(self, alert_id: str) -> Optional[Alert]:
        try:
            alert_key = f"{self.alerts_key_prefix}{alert_id}"
            alert_data = self.redis.get(alert_key)

            if not alert_data:
                return None

            return self._to_entity(json.loads(alert_data))
        except Exception as e:
            logger.error(f"Failed to get alert {alert_id}: {str(e)}")
            return None

    async def get_active_alerts(self, shelf_id: Optional[str] = None) -> List[Alert]:
        try:
            if shelf_id:
                shelf_key = f"{self.shelf_alerts_key_prefix}{shelf_id}"
                alert_ids = self.redis.smembers(shelf_key)
            else:
                alert_ids = self.redis.smembers(self.active_alerts_key)

            alerts = []
            for alert_id in alert_ids:
                if isinstance(alert_id, bytes):
                    alert_id = alert_id.decode("utf-8")

                alert = await self.get_by_id(alert_id)
                if alert and not alert.confirmed and not alert.dismissed:
                    alerts.append(alert)

            return alerts
        except Exception as e:
            logger.error(f"Failed to get active alerts: {str(e)}")
            return []

    async def get_by_cell(
        self, shelf_id: str, row_idx: int, item_idx: int
    ) -> Optional[Alert]:
        try:
            cell_key = f"{self.cell_alerts_key_prefix}{shelf_id}:{row_idx}:{item_idx}"
            alert_id = self.redis.get(cell_key)

            if not alert_id:
                return None

            if isinstance(alert_id, bytes):
                alert_id = alert_id.decode("utf-8")

            alert = await self.get_by_id(alert_id)

            if alert and alert.dismissed:
                return None

            return alert
        except Exception as e:
            logger.error(f"Failed to get alert by cell: {str(e)}")
            return None

    async def update(self, alert: Alert) -> Alert:
        try:
            existing = await self.get_by_id(alert.alert_id)
            if not existing:
                raise EntityNotFoundError("Alert", alert.alert_id)

            alert.updated_at = datetime.now(timezone.utc)
            alert_data = self._to_dict(alert)
            alert_key = f"{self.alerts_key_prefix}{alert.alert_id}"

            self.redis.set(alert_key, json.dumps(alert_data))

            logger.info(f"Alert {alert.alert_id} updated in Redis")
            return alert

        except EntityNotFoundError:
            raise
        except Exception as e:
            raise DatabaseError(f"Failed to update alert: {str(e)}")

    async def confirm_alert(self, alert_id: str, confirmed_by: str) -> Alert:
        try:
            alert = await self.get_by_id(alert_id)
            if not alert:
                raise EntityNotFoundError("Alert", alert_id)

            alert.confirmed = True
            alert.confirmed_by = confirmed_by
            alert.confirmed_at = datetime.now(timezone.utc)
            alert.updated_at = datetime.now(timezone.utc)

            alert_key = f"{self.alerts_key_prefix}{alert_id}"
            alert_data = self._to_dict(alert)
            self.redis.set(alert_key, json.dumps(alert_data))

            self.redis.srem(self.active_alerts_key, alert_id)

            logger.info(f"Alert {alert_id} confirmed by {confirmed_by}")
            return alert

        except EntityNotFoundError:
            raise
        except Exception as e:
            raise DatabaseError(f"Failed to confirm alert: {str(e)}")

    async def dismiss_alert(self, alert_id: str) -> Alert:
        try:
            alert = await self.get_by_id(alert_id)
            if not alert:
                raise EntityNotFoundError("Alert", alert_id)

            alert.dismissed = True
            alert.updated_at = datetime.now(timezone.utc)

            alert_key = f"{self.alerts_key_prefix}{alert_id}"
            alert_data = self._to_dict(alert)
            self.redis.set(alert_key, json.dumps(alert_data))

            self.redis.srem(self.active_alerts_key, alert_id)

            cell_key = (
                f"{self.cell_alerts_key_prefix}{alert.shelf_id}:{alert.row_idx}"
                f":{alert.item_idx}"
            )
            self.redis.delete(cell_key)

            logger.info(f"Alert {alert_id} dismissed")
            return alert

        except EntityNotFoundError:
            raise
        except Exception as e:
            raise DatabaseError(f"Failed to dismiss alert: {str(e)}")

    async def delete_index(self) -> bool:
        try:
            pattern = f"{self.alerts_key_prefix}*"
            keys = self.redis.keys(pattern)

            if keys:
                self.redis.delete(*keys)

            shelf_pattern = f"{self.shelf_alerts_key_prefix}*"
            shelf_keys = self.redis.keys(shelf_pattern)
            if shelf_keys:
                self.redis.delete(*shelf_keys)

            cell_pattern = f"{self.cell_alerts_key_prefix}*"
            cell_keys = self.redis.keys(cell_pattern)
            if cell_keys:
                self.redis.delete(*cell_keys)

            self.redis.delete(self.active_alerts_key)

            logger.info("All alert data deleted from Redis successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to delete alert data from Redis: {str(e)}")
            return False

    def _to_dict(self, alert: Alert) -> dict:
        return {
            "alert_id": alert.alert_id,
            "shelf_id": alert.shelf_id,
            "row_idx": alert.row_idx,
            "item_idx": alert.item_idx,
            "alert_type": alert.alert_type.name,
            "expected_sku": alert.expected_sku,
            "detected_sku": alert.detected_sku,
            "first_seen": alert.first_seen.isoformat(),
            "last_seen": alert.last_seen.isoformat(),
            "confirmed": alert.confirmed,
            "confirmed_by": alert.confirmed_by,
            "confirmed_at": (
                alert.confirmed_at.isoformat() if alert.confirmed_at else None
            ),
            "dismissed": alert.dismissed,
            "evidence_paths": alert.evidence_paths,
            "consecutive_frames": alert.consecutive_frames,
            "created_at": alert.created_at.isoformat(),
            "updated_at": alert.updated_at.isoformat(),
        }

    def _to_entity(self, data: dict) -> Alert:
        return Alert(
            alert_id=data["alert_id"],
            shelf_id=data["shelf_id"],
            row_idx=data["row_idx"],
            item_idx=data["item_idx"],
            alert_type=AlertType[data["alert_type"]],
            expected_sku=data.get("expected_sku"),
            detected_sku=data.get("detected_sku"),
            first_seen=datetime.fromisoformat(data["first_seen"]),
            last_seen=datetime.fromisoformat(data["last_seen"]),
            confirmed=data["confirmed"],
            confirmed_by=data.get("confirmed_by"),
            confirmed_at=(
                datetime.fromisoformat(data["confirmed_at"])
                if data.get("confirmed_at")
                else None
            ),
            dismissed=data["dismissed"],
            evidence_paths=data.get("evidence_paths", []),
            consecutive_frames=data["consecutive_frames"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )
