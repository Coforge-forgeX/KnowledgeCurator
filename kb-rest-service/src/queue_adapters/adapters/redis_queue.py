"""Redis Queue adapter (using Redis lists as queue)"""

import json
import uuid
from typing import Any, Dict, List, Optional

from core.exceptions import ConfigurationException
from core.logging import get_logger

from ..models import QueueMessage
from ..protocols import QueueAdapter

logger = get_logger(__name__)


class RedisQueueAdapter(QueueAdapter):
    """Redis-based queue implementation using lists"""

    def __init__(self) -> None:
        """
        Initialize Redis queue adapter.

        Required environment variables:
            REDIS_HOST: Redis host
            REDIS_PORT: Redis port (default: 6379)
            REDIS_PASSWORD: Redis password (optional)
            REDIS_DB: Redis database number (default: 0)
            REDIS_QUEUE_NAME: Queue name/key (default: indexing-jobs)
        """
        try:
            import redis.asyncio as redis
        except ImportError:
            raise ConfigurationException(
                "redis not installed. Install with: pip install redis",
                config_key="redis",
            )

        from core.config import settings

        redis_host = settings.database.REDIS_HOST
        if not redis_host:
            raise ConfigurationException(
                "REDIS_HOST not configured",
                config_key="REDIS_HOST",
            )

        redis_port = settings.database.REDIS_PORT or 6379
        redis_password = settings.database.REDIS_PASSWORD
        redis_db = settings.database.REDIS_DB or 0

        self._queue_name = getattr(settings, 'REDIS_QUEUE_NAME', None) or "indexing-jobs"
        self._processing_set = f"{self._queue_name}:processing"

        # Create Redis client
        self._redis = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            db=redis_db,
            decode_responses=True,
        )

        logger.info(
            "Redis Queue adapter initialized",
            queue_name=self._queue_name,
            host=redis_host,
            port=redis_port,
        )

    @property
    def provider_name(self) -> str:
        return "redis"

    @property
    def queue_name(self) -> str:
        return self._queue_name

    async def send_message(
        self, message: Dict[str, Any], delay_seconds: int = 0
    ) -> str:
        """
        Send message to Redis queue.

        Note: Redis lists don't natively support message delay.
        This implementation ignores delay_seconds for simplicity.
        """
        try:
            message_id = str(uuid.uuid4())

            # Wrap message with metadata
            message_wrapper = {
                "message_id": message_id,
                "content": message,
            }

            message_str = json.dumps(message_wrapper)

            # Push to right end of list (RPUSH)
            await self._redis.rpush(self._queue_name, message_str)

            logger.info(
                "Message sent to Redis queue",
                message_id=message_id,
                queue=self._queue_name,
            )

            return message_id

        except Exception as e:
            logger.error(f"Failed to send message to Redis: {e}")
            raise

    async def receive_messages(
        self,
        max_messages: int = 1,
        visibility_timeout: int = 30,
        wait_time_seconds: int = 0,
    ) -> List[QueueMessage]:
        """
        Receive messages from Redis queue.

        Uses BLPOP for blocking pop from left end of list.
        Messages are moved to a processing set temporarily.
        """
        try:
            messages = []

            for _ in range(max_messages):
                # BLPOP blocks for wait_time_seconds, returns None if timeout
                result = await self._redis.blpop(
                    self._queue_name, timeout=wait_time_seconds
                )

                if result is None:
                    break

                queue_name, message_str = result

                try:
                    message_wrapper = json.loads(message_str)
                    message_id = message_wrapper.get("message_id", str(uuid.uuid4()))
                    content = message_wrapper.get("content", {})
                except json.JSONDecodeError:
                    message_id = str(uuid.uuid4())
                    content = {"raw": message_str}

                # Store in processing set with visibility timeout
                receipt_handle = f"{message_id}:{uuid.uuid4()}"
                await self._redis.setex(
                    f"{self._processing_set}:{receipt_handle}",
                    visibility_timeout,
                    message_str,
                )

                messages.append(
                    QueueMessage(
                        content=content,
                        message_id=message_id,
                        receipt_handle=receipt_handle,
                    )
                )

            logger.info(
                f"Received {len(messages)} message(s) from Redis queue",
                queue=self._queue_name,
            )

            return messages

        except Exception as e:
            logger.error(f"Failed to receive messages from Redis: {e}")
            raise

    async def delete_message(self, receipt_handle: str) -> bool:
        """Delete message from Redis processing set"""
        try:
            # Delete from processing set
            result = await self._redis.delete(f"{self._processing_set}:{receipt_handle}")

            logger.info(
                "Message deleted from Redis queue",
                receipt_handle=receipt_handle,
                queue=self._queue_name,
            )

            return result > 0

        except Exception as e:
            logger.error(f"Failed to delete message from Redis: {e}")
            return False

    async def get_queue_size(self) -> int:
        """Get queue size from Redis"""
        try:
            return await self._redis.llen(self._queue_name)
        except Exception as e:
            logger.error(f"Failed to get queue size: {e}")
            return 0

    async def purge_queue(self) -> bool:
        """Purge all messages from Redis queue"""
        try:
            # Delete the list
            await self._redis.delete(self._queue_name)

            # Also clean up processing set keys
            processing_keys = await self._redis.keys(f"{self._processing_set}:*")
            if processing_keys:
                await self._redis.delete(*processing_keys)

            logger.info(f"Purged Redis queue: {self._queue_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to purge queue: {e}")
            return False

    async def close(self) -> None:
        """Close Redis connection"""
        try:
            await self._redis.close()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Failed to close Redis connection: {e}")
