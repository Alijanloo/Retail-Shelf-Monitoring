import json
from datetime import datetime, timezone
from typing import Optional

import redis

from ...entities.alert import Alert
from ...frameworks.logging_config import get_logger

logger = get_logger(__name__)


class AlertPublisher:
    def __init__(self, redis_client: redis.Redis, stream_name: str = "alerts"):
        self.redis_client = redis_client
        self.stream_name = stream_name

    async def publish_alert(self, alert: Alert) -> str:
        alert_data = {
            "alert_id": alert.alert_id,
            "shelf_id": alert.shelf_id,
            "row_idx": alert.row_idx,
            "item_idx": alert.item_idx,
            "alert_type": alert.alert_type.value,
            "expected_sku": alert.expected_sku or "",
            "detected_sku": alert.detected_sku or "",
            "first_seen": alert.first_seen.isoformat(),
            "last_seen": alert.last_seen.isoformat(),
            "consecutive_frames": alert.consecutive_frames,
            "evidence_paths": json.dumps(alert.evidence_paths),
        }

        message_id = self.redis_client.xadd(self.stream_name, alert_data, maxlen=10000)

        logger.info(
            f"Published alert {alert.alert_id} to stream {self.stream_name} "
            f"(message_id: {message_id.decode()})"
        )

        return message_id.decode()

    async def publish_alert_update(
        self, alert_id: str, update_type: str, updated_by: Optional[str] = None
    ):
        update_data = {
            "alert_id": alert_id,
            "update_type": update_type,
            "updated_by": updated_by or "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        message_id = self.redis_client.xadd(
            f"{self.stream_name}:updates", update_data, maxlen=5000
        )

        logger.info(f"Published alert update: {alert_id} - {update_type}")

        return message_id.decode()
