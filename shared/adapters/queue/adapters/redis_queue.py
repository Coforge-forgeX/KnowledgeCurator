"""Redis Queue adapter"""

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, List, Optional


def get_logger(name: str):
    return logging.getLogger(name)

from ..models import QueueMessage
from ..protocols import QueueAdapter

logger = get_logger(__name__)


class RedisQueueAdapter(QueueAdapter):
    """Redis-backed queue implementation using list operations"""

    def __init__(
        self,
        connection_string: Optional[str] = None,
        queue_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Redis queue adapter.

        Args:
            connection_string: Redis URL (redis://...)
            queue_name: Redis list key used as queue
        """
        try:
            import redis
        except ImportError as exc:
            raise ImportError("redis package is required for RedisQueueAdapter") from exc

        self._queue_name = queue_name or "indexing-jobs"

        redis_url = connection_string or kwargs.get("redis_url")
        if not redis_url:
            host = kwargs.get("redis_host") or "localhost"
            port = int(kwargs.get("redis_port") or 6379)
            db = int(kwargs.get("redis_db") or 0)
            password = kwargs.get("redis_password")
            if password:
                redis_url = f"redis://:{password}@{host}:{port}/{db}"
            else:
                redis_url = f"redis://{host}:{port}/{db}"

        self._redis_client = redis.Redis.from_url(redis_url, decode_responses=True)

        logger.info(f"Redis queue adapter initialized (queue={self._queue_name})")

    async def send_message(
        self, message: Dict[str, Any], delay_seconds: int = 0
    ) -> str:
        """Push message onto Redis list queue"""
        message_id = uuid.uuid4().hex
        payload = {
            "message_id": message_id,
            "content": message,
        }

        if delay_seconds > 0:
            # Keep behavior simple and explicit for now.
            await asyncio.to_thread(self._redis_client.rpush, self._queue_name, json.dumps(payload))
            logger.warning("RedisQueueAdapter does not support delayed delivery; message sent immediately")
        else:
            await asyncio.to_thread(self._redis_client.rpush, self._queue_name, json.dumps(payload))

        return message_id

    async def receive_messages(
        self,
        max_messages: int = 1,
        visibility_timeout: int = 30,
        wait_time_seconds: int = 0,
    ) -> List[QueueMessage]:
        """Pop messages from Redis list queue (at-most-once delivery)"""
        del visibility_timeout  # Not supported for list-based semantics.

        messages: List[QueueMessage] = []
        count = max(1, max_messages)

        for _ in range(count):
            if wait_time_seconds > 0:
                item = await asyncio.to_thread(
                    self._redis_client.blpop,
                    self._queue_name,
                    wait_time_seconds,
                )
                if not item:
                    break
                _, payload_str = item
            else:
                payload_str = await asyncio.to_thread(self._redis_client.lpop, self._queue_name)
                if payload_str is None:
                    break

            try:
                payload = json.loads(payload_str)
                messages.append(
                    QueueMessage(
                        content=payload.get("content", {}),
                        message_id=payload.get("message_id"),
                        receipt_handle=payload.get("message_id"),
                    )
                )
            except json.JSONDecodeError:
                logger.warning("Skipping malformed Redis queue payload")
                continue

        return messages

    async def delete_message(self, receipt_handle: str) -> bool:
        """Deletion is implicit on pop for list-based queue"""
        return bool(receipt_handle)

    async def get_queue_size(self) -> int:
        """Get queue length"""
        size = await asyncio.to_thread(self._redis_client.llen, self._queue_name)
        return int(size)

    async def purge_queue(self) -> bool:
        """Delete queue key and all pending messages"""
        try:
            await asyncio.to_thread(self._redis_client.delete, self._queue_name)
            return True
        except Exception as exc:
            logger.error(f"Failed to purge Redis queue: {str(exc)}")
            return False

    @property
    def provider_name(self) -> str:
        return "redis"

    @property
    def queue_name(self) -> str:
        return self._queue_name
