"""AWS SQS adapter"""

import json
from typing import Any, Dict, List

from core.exceptions import ConfigurationException
from core.logging import get_logger

from ..models import QueueMessage
from ..protocols import QueueAdapter

logger = get_logger(__name__)


class AWSSQSAdapter(QueueAdapter):
    """AWS SQS implementation"""

    def __init__(self) -> None:
        """
        Initialize AWS SQS adapter.

        Required settings:
            AWS_ACCESS_KEY_ID: AWS access key
            AWS_SECRET_ACCESS_KEY: AWS secret key
            AWS_REGION: AWS region
            SQS_QUEUE_NAME: SQS queue name
        """
        try:
            import boto3
        except ImportError:
            raise ConfigurationException(
                "boto3 not installed. Install with: pip install boto3",
                config_key="boto3",
            )

        from core.config import settings

        access_key = settings.storage.AWS_ACCESS_KEY_ID
        secret_key = settings.storage.AWS_SECRET_ACCESS_KEY
        region = getattr(settings.storage, 'AWS_REGION', None) or "us-east-1"

        if not access_key or not secret_key:
            raise ConfigurationException(
                "AWS credentials not configured",
                config_key="AWS_ACCESS_KEY_ID",
            )

        self._queue_name = getattr(settings, 'SQS_QUEUE_NAME', None) or "indexing-jobs"
        self._region = region

        # Create SQS client
        self._sqs_client = boto3.client(
            "sqs",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )

        # Get or create queue URL
        try:
            response = self._sqs_client.get_queue_url(QueueName=self._queue_name)
            self._queue_url = response["QueueUrl"]
        except self._sqs_client.exceptions.QueueDoesNotExist:
            response = self._sqs_client.create_queue(QueueName=self._queue_name)
            self._queue_url = response["QueueUrl"]

        logger.info(
            "AWS SQS adapter initialized",
            queue_name=self._queue_name,
            queue_url=self._queue_url,
            region=region,
        )

    @property
    def provider_name(self) -> str:
        return "aws"

    @property
    def queue_name(self) -> str:
        return self._queue_name

    async def send_message(
        self, message: Dict[str, Any], delay_seconds: int = 0
    ) -> str:
        """Send message to AWS SQS"""
        import asyncio

        try:
            message_body = json.dumps(message)

            response = await asyncio.to_thread(
                self._sqs_client.send_message,
                QueueUrl=self._queue_url,
                MessageBody=message_body,
                DelaySeconds=delay_seconds,
            )

            message_id = response["MessageId"]

            logger.info(
                "Message sent to SQS",
                message_id=message_id,
                queue=self._queue_name,
            )

            return message_id

        except Exception as e:
            logger.error(f"Failed to send message to SQS: {e}")
            raise

    async def receive_messages(
        self,
        max_messages: int = 1,
        visibility_timeout: int = 30,
        wait_time_seconds: int = 0,
    ) -> List[QueueMessage]:
        """Receive messages from AWS SQS"""
        import asyncio

        try:
            response = await asyncio.to_thread(
                self._sqs_client.receive_message,
                QueueUrl=self._queue_url,
                MaxNumberOfMessages=min(max_messages, 10),
                VisibilityTimeout=visibility_timeout,
                WaitTimeSeconds=wait_time_seconds,
            )

            messages = []
            for msg in response.get("Messages", []):
                try:
                    content = json.loads(msg["Body"])
                except json.JSONDecodeError:
                    content = {"raw": msg["Body"]}

                messages.append(
                    QueueMessage(
                        content=content,
                        message_id=msg["MessageId"],
                        receipt_handle=msg["ReceiptHandle"],
                    )
                )

            logger.info(
                f"Received {len(messages)} message(s) from SQS",
                queue=self._queue_name,
            )

            return messages

        except Exception as e:
            logger.error(f"Failed to receive messages from SQS: {e}")
            raise

    async def delete_message(self, receipt_handle: str) -> bool:
        """Delete message from AWS SQS"""
        import asyncio

        try:
            await asyncio.to_thread(
                self._sqs_client.delete_message,
                QueueUrl=self._queue_url,
                ReceiptHandle=receipt_handle,
            )

            logger.info(
                "Message deleted from SQS",
                queue=self._queue_name,
            )

            return True

        except Exception as e:
            logger.error(f"Failed to delete message from SQS: {e}")
            return False

    async def get_queue_size(self) -> int:
        """Get approximate message count from SQS"""
        import asyncio

        try:
            response = await asyncio.to_thread(
                self._sqs_client.get_queue_attributes,
                QueueUrl=self._queue_url,
                AttributeNames=["ApproximateNumberOfMessages"],
            )

            return int(response["Attributes"].get("ApproximateNumberOfMessages", 0))

        except Exception as e:
            logger.error(f"Failed to get queue size: {e}")
            return 0

    async def purge_queue(self) -> bool:
        """Purge all messages from SQS"""
        import asyncio

        try:
            await asyncio.to_thread(
                self._sqs_client.purge_queue,
                QueueUrl=self._queue_url,
            )

            logger.info(f"Purged queue: {self._queue_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to purge queue: {e}")
            return False
