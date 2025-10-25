from typing import Dict, List

import redis

from ...frameworks.logging_config import get_logger

logger = get_logger(__name__)


class RedisStreamClient:
    def __init__(
        self,
        redis_client: redis.Redis,
        stream_name: str = "alerts",
        consumer_group: str = "alert_processors",
        consumer_name: str = "processor_1",
    ):
        self.redis_client = redis_client
        self.stream_name = stream_name
        self.consumer_group = consumer_group
        self.consumer_name = consumer_name

        self._ensure_consumer_group()

    def _ensure_consumer_group(self):
        try:
            self.redis_client.xgroup_create(
                self.stream_name, self.consumer_group, id="0", mkstream=True
            )
            logger.info(f"Created consumer group: {self.consumer_group}")
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.debug(f"Consumer group {self.consumer_group} already exists")
            else:
                raise

    def read_messages(self, count: int = 10, block: int = 1000) -> List[Dict]:
        messages = self.redis_client.xreadgroup(
            self.consumer_group,
            self.consumer_name,
            {self.stream_name: ">"},
            count=count,
            block=block,
        )

        parsed_messages = []

        for stream_name, stream_messages in messages:
            for msg_id, msg_data in stream_messages:
                decoded_data = {
                    k.decode()
                    if isinstance(k, bytes)
                    else k: v.decode()
                    if isinstance(v, bytes)
                    else v
                    for k, v in msg_data.items()
                }

                parsed_messages.append(
                    {
                        "id": msg_id.decode() if isinstance(msg_id, bytes) else msg_id,
                        "data": decoded_data,
                    }
                )

        return parsed_messages

    def acknowledge_message(self, message_id: str):
        self.redis_client.xack(self.stream_name, self.consumer_group, message_id)
