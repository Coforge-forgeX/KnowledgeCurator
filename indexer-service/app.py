"""
KB Indexer Service - Azure WebApp Background Worker

Polls Azure Storage Queue for indexing jobs from kb-rest service.
Processes documents and updates Neo4j + PostgreSQL.
"""
import asyncio
import json
import os
import sys

# Add src and shared folders to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))

from azure.storage.queue.aio import QueueServiceClient

from core.config import settings
from core.database import db_manager
from core.logging import bind_context, clear_context, get_logger
from shared.models import IndexingJob
from src.workers.document_processor_with_retry import get_document_processor_with_retry

logger = get_logger(__name__)

# Get processor with retry capability
processor = get_document_processor_with_retry()


async def poll_queue():
    """Poll Azure Storage Queue for indexing jobs"""
    logger.info(
        "Starting indexer worker",
        queue=settings.azure.INDEXING_QUEUE_NAME,
        environment=settings.ENVIRONMENT,
    )

    if not settings.azure.AZURE_STORAGE_CONNECTION_STRING:
        logger.error("AZURE_STORAGE_CONNECTION_STRING not set")
        return

    # Initialize database connections
    await db_manager.initialize()

    queue_service = QueueServiceClient.from_connection_string(
        settings.azure.AZURE_STORAGE_CONNECTION_STRING
    )
    queue_client = queue_service.get_queue_client(settings.azure.INDEXING_QUEUE_NAME)

    # Create queue if not exists
    try:
        await queue_client.create_queue()
        logger.info("Queue created or already exists")
    except Exception:
        pass

    while True:
        try:
            # Get messages from queue
            messages = []
            async for msg in queue_client.receive_messages(
                messages_per_page=settings.MAX_CONCURRENT_JOBS,
                visibility_timeout=settings.MESSAGE_VISIBILITY_TIMEOUT,
            ):
                messages.append(msg)

            if not messages:
                logger.debug(
                    "No messages in queue",
                    poll_interval=settings.azure.QUEUE_POLL_INTERVAL,
                )
                await asyncio.sleep(settings.azure.QUEUE_POLL_INTERVAL)
                continue

            logger.info("Received messages from queue", count=len(messages))

            # Process messages concurrently
            tasks = []
            for msg in messages:
                tasks.append(process_message(queue_client, msg))

            await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            logger.error("Queue polling error", error_msg=str(e), exc_info=True)
            await asyncio.sleep(settings.azure.QUEUE_POLL_INTERVAL)


async def process_message(queue_client, message):
    """Process a single indexing job"""
    try:
        body = message.content
        data = json.loads(body)

        # Check if this is the new job format (from upload_and_index API)
        if "task_id" in data and "file_path" in data:
            # New format - use the new handler
            from src.workers.indexing_job_handler import process_indexing_job

            task_id = data.get("task_id")
            bind_context(task_id=task_id, workspace_id=data.get("workspace_id"))

            logger.info(
                "Processing new format indexing job",
                message_id=message.id,
                task_id=task_id,
                file_path=data.get("file_path"),
            )

            result = await process_indexing_job(data)

            if result.get("success"):
                logger.info(
                    "Indexing job completed successfully",
                    task_id=task_id,
                    chunk_count=result.get("chunk_count"),
                )
                # Delete message from queue
                await queue_client.delete_message(message)
            else:
                error = result.get("error", "Unknown error")
                logger.error("Indexing job failed", task_id=task_id, error_msg=error)
                # Delete message (status already updated to failed in DB)
                await queue_client.delete_message(message)

        else:
            # Legacy format - use existing processor
            # Validate message payload with Pydantic
            job = IndexingJob(**data)

            # Bind job context for structured logging
            bind_context(job_id=job.job_id, workspace_id=job.workspace_id)

            logger.info(
                "Processing legacy format message",
                message_id=message.id,
                document_url=job.document_url[:100],
            )

            # Process document with retry and resume capability
            result = await processor.process_document(
                job_id=job.job_id,
                workspace_id=job.workspace_id,
                document_url=job.document_url,
                kb_id=job.kb_id,
                max_retries=settings.MAX_RETRIES,
            )

            if result.get("success"):
                logger.info(
                    "Job completed successfully",
                    duration=result.get("duration_seconds"),
                    chunks=result.get("chunks_processed"),
                    retry_count=result.get("retry_count", 0),
                )
                # Delete message from queue
                await queue_client.delete_message(message)

            elif result.get("retry_scheduled"):
                # Retry scheduled with exponential backoff
                retry_delay = result.get("retry_delay_seconds", 60)
                logger.info(
                    "Job failed, retry scheduled",
                    error_msg=result.get("error"),
                    retry_count=result.get("retry_count"),
                    retry_delay_seconds=retry_delay,
                )
                # Update message visibility to delay retry
                await queue_client.update_message(
                    message,
                    visibility_timeout=retry_delay,
                )

            elif result.get("max_retries_exceeded"):
                # Max retries exceeded, move to dead letter
                logger.error(
                    "Job failed permanently, max retries exceeded",
                    error_msg=result.get("error"),
                    retry_count=result.get("retry_count"),
                )
                # Delete from queue (Azure will move to poison queue if configured)
                await queue_client.delete_message(message)

            else:
                # Unexpected failure
                error = result.get("error", "Unknown error")
                logger.error("Job failed", error_msg=error)
                # Leave in queue for automatic retry

    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in message", error_msg=str(e))
        await queue_client.delete_message(message)  # Delete invalid message
    except Exception as e:
        logger.error("Message processing error", error_msg=str(e), exc_info=True)
        # Don't delete message - let it retry
    finally:
        clear_context()


async def main():
    """Main entry point"""
    logger.info(
        "KB Indexer Service starting",
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
    )
    try:
        await poll_queue()
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully")
    finally:
        await db_manager.close()


if __name__ == "__main__":
    asyncio.run(main())
