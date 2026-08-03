"""
Indexer Service - Background Worker for Document Indexing

This service processes document indexing jobs from a message queue.
It's a long-running worker (not serverless) that:
- Polls message queue for indexing jobs
- Downloads documents from cloud storage
- Processes documents with LightRAG
- Updates task status in database
- Deletes processed messages from queue

Multi-cloud deployment support: Azure, AWS, GCP, Docker
"""
import asyncio
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI

# Add src/ to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.core.logging import get_logger, setup_logging

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Worker state
worker_task = None
shutdown_event = asyncio.Event()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup/shutdown tasks.

    Startup: Start background indexing worker
    Shutdown: Stop background worker gracefully
    """
    global worker_task

    # Startup
    logger.info(
        "Indexer Service starting",
        version="2.0.0",
        environment=os.getenv("ENVIRONMENT", "development"),
        cloud_provider=os.getenv("CLOUD_PROVIDER", "unknown"),
        storage_provider=os.getenv("STORAGE_PROVIDER", os.getenv("CLOUD_PROVIDER", "azure")),
        queue_provider=os.getenv("QUEUE_PROVIDER", os.getenv("CLOUD_PROVIDER", "azure")),
    )

    # Start background worker
    worker_task = asyncio.create_task(indexing_worker())
    logger.info("Background indexing worker started")

    yield

    # Shutdown
    logger.info("Indexer Service shutting down")
    shutdown_event.set()

    if worker_task:
        # Wait for worker to finish gracefully (max 30 seconds)
        try:
            await asyncio.wait_for(worker_task, timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning("Worker did not shut down gracefully, canceling")
            worker_task.cancel()
        except Exception as e:
            logger.error(f"Error during worker shutdown: {e}")


# Create FastAPI app
app = FastAPI(
    title="Indexer Service",
    description="Background worker for document indexing with LightRAG",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs" if os.getenv("DEBUG", "false").lower() == "true" else None,
    redoc_url="/redoc" if os.getenv("DEBUG", "false").lower() == "true" else None,
)


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "service": "indexer-service",
        "version": "2.0.0",
        "cloud_provider": os.getenv("CLOUD_PROVIDER", "unknown"),
        "storage_provider": os.getenv("STORAGE_PROVIDER", os.getenv("CLOUD_PROVIDER", "azure")),
        "queue_provider": os.getenv("QUEUE_PROVIDER", os.getenv("CLOUD_PROVIDER", "azure")),
        "worker_running": worker_task is not None and not worker_task.done(),
    }


async def indexing_worker():
    """
    Background worker that processes indexing jobs from queue.

    This function:
    1. Polls message queue for new indexing jobs
    2. Downloads documents from storage
    3. Processes with LightRAG indexer
    4. Updates database task status
    5. Deletes processed messages from queue
    """
    # Lazy imports to avoid loading heavy dependencies at startup
    from queue_adapter import get_queue_adapter
    from storage_adapter import get_storage_adapter
    from src.workers.indexing_job_handler import process_indexing_job

    queue = get_queue_adapter()
    storage = get_storage_adapter()

    logger.info(
        "Indexing worker initialized",
        queue_provider=queue.provider_name,
        storage_provider=storage.provider_name,
    )

    # Worker loop
    while not shutdown_event.is_set():
        try:
            # Poll queue for messages (long polling with 20 second wait)
            messages = await queue.receive_messages(
                max_messages=10,
                visibility_timeout=300,  # 5 minutes to process
                wait_time_seconds=20,     # Long polling
            )

            if not messages:
                # No messages, continue polling (silent - no log spam)
                continue

            # Only log when we actually receive jobs
            logger.info(f"Received {len(messages)} indexing job(s)")

            # Process each message
            for message in messages:
                try:
                    job_data = message.content
                    task_id = job_data.get("task_id")
                    job_id = job_data.get("job_id", str(task_id))
                    file_path = job_data.get("file_path")
                    workspace_id = job_data.get("workspace_id")

                    # Get retry count from message (dequeue_count - 1 because first dequeue is not a retry)
                    retry_count = max(0, getattr(message, 'dequeue_count', 1) - 1)

                    logger.info(
                        "Processing indexing job",
                        job_id=job_id,
                        task_id=task_id,
                        file_path=file_path,
                        workspace_id=workspace_id,
                        retry_count=retry_count,
                    )

                    # Process the indexing job with retry count
                    result = await process_indexing_job(job_data, retry_count=retry_count)
                    success = result.get("success", False)

                    if success:
                        # Delete message from queue - job completed successfully
                        await queue.delete_message(message.receipt_handle)
                        logger.info(
                            "Successfully processed and removed from queue",
                            job_id=job_id,
                            task_id=task_id,
                            retry_count=retry_count
                        )
                    else:
                        error_msg = result.get("error", "Unknown error")
                        logger.error(
                            "Job failed - will retry after visibility timeout",
                            job_id=job_id,
                            task_id=task_id,
                            retry_count=retry_count,
                            error=error_msg
                        )
                        # Message will become visible again after visibility timeout for automatic retry

                except Exception as e:
                    logger.error(
                        "Error processing indexing job",
                        error=e,
                        message_id=message.message_id,
                        exc_info=True,
                    )
                    # Message will become visible again for retry

        except asyncio.CancelledError:
            logger.info("Worker canceled")
            break
        except Exception as e:
            logger.error(f"Error in worker loop: {e}", exc_info=True)
            # Sleep before retrying
            await asyncio.sleep(5)

    logger.info("Indexing worker stopped")


# Application entry point
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8081))

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("DEBUG", "false").lower() == "true",
        log_level="info",
        access_log=True,
    )
