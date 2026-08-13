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
import signal
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response

# Put the app root on the import path so `src.*` resolves when the process is
# started from another working directory. The `shared` package is resolved
# normally: installed (pip install -e ..) for local dev, vendored at the app
# root by the deploy pipeline in production.
_service_dir = os.path.dirname(os.path.abspath(__file__))
if _service_dir not in sys.path:
    sys.path.insert(0, _service_dir)

# Configure Windows console for UTF-8 encoding (prevents Unicode crashes)
from shared.windows_encoding import configure_windows_console_encoding
configure_windows_console_encoding()

from src.core.logging import get_logger, setup_logging
from src.core.config import settings

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Worker state
worker_task = None
shutdown_event = asyncio.Event()
_received_signal = None


def _install_signal_logging() -> None:
    """Install lightweight signal hooks to explain why the process is stopping."""
    global _received_signal
    for sig_name in ("SIGINT", "SIGTERM", "SIGHUP", "SIGBREAK"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            previous = signal.getsignal(sig)

            def _handler(signum, frame, *, _previous=previous):
                nonlocal sig_name
                global _received_signal
                _received_signal = sig_name
                logger.warning("Shutdown signal received", signal=sig_name)
                if callable(_previous):
                    _previous(signum, frame)

            signal.signal(sig, _handler)
        except Exception:
            # Signal registration may fail in some hosted environments.
            continue


_install_signal_logging()


def _log_worker_result(task: asyncio.Task) -> None:
    """Log unhandled worker task failures so they are not silent."""
    try:
        task.result()
    except asyncio.CancelledError:
        logger.info("Background indexing worker canceled")
    except Exception as exc:
        logger.error("Background indexing worker crashed", error=str(exc), exc_info=True)


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
        environment=settings.ENVIRONMENT,
        cloud_provider=settings.CLOUD_PROVIDER,
        storage_provider=settings.active_storage_provider,
        queue_provider=settings.active_queue_provider,
    )

    # Start background worker
    worker_task = asyncio.create_task(indexing_worker())
    worker_task.add_done_callback(_log_worker_result)
    logger.info("Background indexing worker started")

    yield

    # Shutdown
    logger.info("Indexer Service shutting down")
    if _received_signal:
        logger.warning("Lifecycle shutdown initiated after signal", signal=_received_signal)
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
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)


@app.get("/health")
async def health_check(response: Response):
    """Health check endpoint for monitoring. Probes Postgres and Neo4j."""
    from src.core.health import run_health_checks

    checks, overall_status = await run_health_checks()
    worker_running = worker_task is not None and not worker_task.done()
    if not worker_running:
        overall_status = "unhealthy"
        checks["worker"] = {"status": "unhealthy", "error": "background worker not running"}
    else:
        checks["worker"] = {"status": "healthy"}

    response.status_code = 200 if overall_status == "healthy" else 503

    return {
        "status": overall_status,
        "service": "indexer-service",
        "version": "2.0.0",
        "cloud_provider": settings.CLOUD_PROVIDER,
        "storage_provider": settings.active_storage_provider,
        "queue_provider": settings.active_queue_provider,
        "worker_running": worker_running,
        "checks": checks,
    }


async def indexing_worker():
    """
    Background worker that processes indexing jobs from queue.

    This function:
    1. Continuously polls message queue for new indexing jobs
    2. Spawns concurrent tasks (up to MAX_CONCURRENT_JOBS)
    3. Each task: downloads, processes with LightRAG, updates status
    4. Deletes processed messages from queue

    Uses producer-consumer pattern to pick up new jobs immediately
    without waiting for current jobs to complete.
    """
    # Lazy imports to avoid loading heavy dependencies at startup.
    from src.queue_adapters import get_queue_adapter
    from src.workers.indexing_job_handler import process_indexing_job

    queue = get_queue_adapter()
    storage_provider_name = settings.active_storage_provider
    max_concurrent_jobs = max(1, int(settings.MAX_CONCURRENT_JOBS))

    # Track active job tasks (not count - actual task objects)
    active_tasks: set = set()

    # Avoid probing storage adapter on every poll cycle; job handler manages per-job adapter creation.
    storage = None

    logger.info(
        "Indexing worker initialized",
        queue_provider=queue.provider_name,
        storage_provider=storage_provider_name,
        max_concurrent_jobs=max_concurrent_jobs,
    )

    async def _process_single_message(message):
        """Process a single message and handle cleanup."""
        try:
            job_data = message.content
            task_id = job_data.get("task_id")
            job_id = job_data.get("job_id", str(task_id))
            file_path = job_data.get("file_path")
            workspace_id = job_data.get("workspace_id")

            # Get retry count from message (dequeue_count - 1 because first dequeue is not a retry)
            retry_count = max(0, getattr(message, 'dequeue_count', 1) - 1)

            # Check if max retries exceeded
            if retry_count >= settings.MAX_RETRIES:
                error_msg = f"Max retries ({settings.MAX_RETRIES}) exceeded"
                logger.error(
                    "Moving message to dead letter queue",
                    job_id=job_id,
                    task_id=task_id,
                    file_path=file_path,
                    retry_count=retry_count,
                    max_retries=settings.MAX_RETRIES,
                    error_msg=error_msg,
                )
                # Move to dead letter queue
                await queue.move_to_dead_letter(
                    message,
                    reason="MaxRetriesExceeded",
                    error_description=f"Job failed after {retry_count} retries: {file_path}"
                )
                # Update task status to failed
                if task_id:
                    from src.workers.indexing_job_handler import update_file_task_status
                    await update_file_task_status(task_id, "failed", error_msg)
                return  # Skip processing this message

            logger.info(
                "Processing indexing job",
                job_id=job_id,
                task_id=task_id,
                file_path=file_path,
                workspace_id=workspace_id,
                retry_count=retry_count,
                max_retries=settings.MAX_RETRIES,
            )

            # Process the indexing job with retry count
            result = await process_indexing_job(job_data, retry_count=retry_count)
            success = result.get("success", False)

            if success:
                # Delete message from queue - job completed successfully
                await queue.delete_message(message)
                logger.info(
                    "Successfully processed and removed from queue",
                    job_id=job_id,
                    task_id=task_id,
                    retry_count=retry_count
                )
            else:
                error_msg = result.get("error", "Unknown error")
                if result.get("non_retryable", False):
                    await queue.delete_message(message)
                    logger.warning(
                        "Dropped non-retryable indexing message",
                        job_id=job_id,
                        task_id=task_id,
                        retry_count=retry_count,
                        error_msg=error_msg,
                    )
                else:
                    logger.error(
                        "Job failed - will retry after visibility timeout",
                        job_id=job_id,
                        task_id=task_id,
                        retry_count=retry_count,
                        error_msg=error_msg,
                    )
                    # Message will become visible again after visibility timeout for automatic retry

        except Exception as e:
            logger.error(
                "Error processing indexing job",
                message_id=message.message_id,
                error_msg=str(e),
                exc_info=True,
            )
            # Message will become visible again for retry

    try:
        # Producer-consumer loop: continuously poll and spawn tasks
        while not shutdown_event.is_set():
            try:
                # Clean up completed tasks
                done_tasks = {task for task in active_tasks if task.done()}
                for task in done_tasks:
                    try:
                        # Retrieve exception if any (prevents "Task exception was never retrieved")
                        task.result()
                    except Exception as e:
                        logger.error("Task failed with exception", error_msg=str(e), exc_info=True)
                active_tasks -= done_tasks

                # Calculate available slots
                available_slots = max_concurrent_jobs - len(active_tasks)

                if available_slots <= 0:
                    # At capacity, wait briefly before checking again
                    await asyncio.sleep(1)
                    continue

                # Poll queue for messages (short wait since we're continuously polling)
                # logger.info(
                #     "Polling for messages",
                #     available_slots=available_slots,
                #     active_tasks=len(active_tasks)
                # )
                messages = await queue.receive_messages(
                    max_messages=available_slots,
                    visibility_timeout=300,  # 5 minutes to process
                    wait_time_seconds=5,     # Shorter wait for responsive polling
                )

                if not messages:
                    # No messages, continue polling
                    logger.info("No messages in queue, continuing to poll")
                    await asyncio.sleep(5)
                    continue

                # Spawn tasks for new messages immediately (non-blocking)
                logger.info(
                    "Received indexing jobs",
                    count=len(messages),
                    in_flight_jobs=len(active_tasks),
                    max_concurrent_jobs=max_concurrent_jobs,
                )

                for message in messages:
                    task = asyncio.create_task(_process_single_message(message))
                    active_tasks.add(task)

                # Continue polling immediately for more messages
                # (don't wait for these tasks to complete)

            except asyncio.CancelledError:
                logger.info("Worker canceled")
                break
            except Exception as e:
                logger.error(f"Error in worker loop: {e}", exc_info=True)
                # Sleep before retrying
                await asyncio.sleep(5)
    finally:
        # Wait for active tasks to complete during shutdown
        if active_tasks:
            logger.info(
                "Waiting for active jobs to complete",
                active_jobs=len(active_tasks),
            )
            await asyncio.gather(*active_tasks, return_exceptions=True)
            logger.info("All active jobs completed")

        close_tasks = []
        queue_close = getattr(queue, "close", None)
        if callable(queue_close):
            close_tasks.append(queue_close())

        storage_obj = locals().get("storage")
        storage_close = getattr(storage_obj, "close", None)
        if callable(storage_close):
            close_tasks.append(storage_close())

        if close_tasks:
            results = await asyncio.gather(*close_tasks, return_exceptions=True)
            for close_result in results:
                if isinstance(close_result, Exception):
                    logger.warning(
                        "Failed to close worker adapter cleanly",
                        error_msg=str(close_result),
                    )

    logger.info("Indexing worker stopped")


# Application entry point
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
        access_log=True,
    )
