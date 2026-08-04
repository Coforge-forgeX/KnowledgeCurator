"""Enqueue KB indexing job"""
import json
import uuid

from src.core.abstractions import AbstractContext, AbstractRequest, AbstractResponse
from src.core.logging import get_logger
from src.queue_adapters import get_queue_adapter
from src.shared.payloads import parse_request
from src.shared.response_utils import create_error_response, create_success_response

from .payloads import KBIndexRequest

logger = get_logger(__name__)


async def main(
    req: AbstractRequest, context: AbstractContext
) -> AbstractResponse:
    """
    Enqueue indexing job to background worker.

    POST /api/kb/index
    Body: {
        "workspace_id": 1,
        "document_url": "https://...",
        "kb_id": 1
    }

    Response: {
        "job_id": "...",
        "status": "queued"
    }
    """
    # Validate request payload
    payload, error = parse_request(req, KBIndexRequest)
    if error:
        return error

    try:
        # Generate job ID
        job_id = str(uuid.uuid4())

        # Enqueue job using provider-agnostic queue adapter
        job_payload = {
            "job_id": job_id,
            "workspace_id": payload.workspace_id,
            "document_url": payload.document_url,
            "kb_id": payload.kb_id,
        }

        queue = get_queue_adapter()
        await queue.send_message(job_payload)

        logger.info(
            f"Enqueued indexing job {job_id} for workspace {payload.workspace_id}",
            extra={"job_id": job_id, "workspace_id": payload.workspace_id},
        )

        return create_success_response(
            message="Indexing job queued successfully",
            data={"job_id": job_id, "status": "queued"},
            status_code=202,
            correlation_id=context.correlation_id,
        )

    except Exception as e:
        logger.error(
            f"KB index error: {e}",
            exc_info=True,
            extra={"error_type": type(e).__name__},
        )
        return create_error_response(
            message="Failed to enqueue indexing job",
            error_code="QUEUE_ERROR",
            status_code=500,
            correlation_id=context.correlation_id,
        )
