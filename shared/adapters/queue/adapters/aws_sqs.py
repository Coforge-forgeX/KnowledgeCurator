"""AWS SQS Queue adapter"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional


def get_logger(name: str):
    return logging.getLogger(name)

from ..models import QueueMessage
from ..protocols import QueueAdapter

logger = get_logger(__name__)


class AWSSQSAdapter(QueueAdapter):
    """AWS SQS queue implementation"""

    def __init__(
        self,
        connection_string: Optional[str] = None,
        queue_name: Optional[str] = None,
        queue_url: Optional[str] = None,
        region_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize AWS SQS adapter.

        Args:
            connection_string: Optional queue URL fallback
            queue_name: SQS queue name (used to resolve URL)
            queue_url: Full SQS queue URL
            region_name: AWS region
        """
        try:
            import boto3
        except ImportError as exc:
            raise ImportError("boto3 is required for AWS SQS adapter") from exc

        self._boto3 = boto3
        self._region_name = region_name or kwargs.get("aws_region") or "us-east-1"
        self._queue_name = queue_name or "indexing-jobs"
        self._queue_url = queue_url or connection_string

        self._sqs_client = self._boto3.client("sqs", region_name=self._region_name)

        if not self._queue_url:
            response = self._sqs_client.get_queue_url(QueueName=self._queue_name)
            self._queue_url = response["QueueUrl"]
        else:
            # Derive queue name from URL when queue_name not provided.
            self._queue_name = self._queue_url.rstrip("/").split("/")[-1]

        logger.info(
            f"AWS SQS adapter initialized (queue={self._queue_name}, region={self._region_name})"
        )

    async def send_message(
        self, message: Dict[str, Any], delay_seconds: int = 0
    ) -> str:
        """Send message to AWS SQS"""
        message_body = json.dumps(message)

        result = await asyncio.to_thread(
            self._sqs_client.send_message,
            QueueUrl=self._queue_url,
            MessageBody=message_body,
            DelaySeconds=max(delay_seconds, 0),
        )

        message_id = result.get("MessageId", "")
        logger.info(f"Message sent to AWS SQS (message_id={message_id}, queue={self._queue_name})")
        return message_id

    async def receive_messages(
        self,
        max_messages: int = 1,
        visibility_timeout: int = 30,
        wait_time_seconds: int = 0,
    ) -> List[QueueMessage]:
        """Receive messages from AWS SQS"""
        result = await asyncio.to_thread(
            self._sqs_client.receive_message,
            QueueUrl=self._queue_url,
            MaxNumberOfMessages=max(1, min(max_messages, 10)),
            VisibilityTimeout=max(0, visibility_timeout),
            WaitTimeSeconds=max(0, min(wait_time_seconds, 20)),
        )

        messages: List[QueueMessage] = []
        for msg in result.get("Messages", []):
            body = msg.get("Body", "{}")
            try:
                content = json.loads(body)
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed SQS message (message_id={msg.get('MessageId')})")
                continue

            messages.append(
                QueueMessage(
                    content=content,
                    message_id=msg.get("MessageId"),
                    receipt_handle=msg.get("ReceiptHandle"),
                )
            )

        return messages

    async def delete_message(self, receipt_handle: str) -> bool:
        """Delete message from AWS SQS"""
        if not receipt_handle:
            return False

        try:
            await asyncio.to_thread(
                self._sqs_client.delete_message,
                QueueUrl=self._queue_url,
                ReceiptHandle=receipt_handle,
            )
            return True
        except Exception as exc:
            logger.error(f"Failed to delete SQS message: {str(exc)}")
            return False

    async def get_queue_size(self) -> int:
        """Get approximate queue size"""
        try:
            attrs = await asyncio.to_thread(
                self._sqs_client.get_queue_attributes,
                QueueUrl=self._queue_url,
                AttributeNames=["ApproximateNumberOfMessages"],
            )
            return int(attrs.get("Attributes", {}).get("ApproximateNumberOfMessages", "0"))
        except Exception as exc:
            logger.error(f"Failed to fetch SQS queue size: {str(exc)}")
            return 0

    async def purge_queue(self) -> bool:
        """Purge AWS SQS queue"""
        try:
            await asyncio.to_thread(self._sqs_client.purge_queue, QueueUrl=self._queue_url)
            return True
        except Exception as exc:
            logger.error(f"Failed to purge SQS queue: {str(exc)}")
            return False

    @property
    def provider_name(self) -> str:
        return "aws"

    @property
    def queue_name(self) -> str:
        return self._queue_name
