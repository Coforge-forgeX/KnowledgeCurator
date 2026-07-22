"""Enqueue KB indexing job"""
import json
import logging
import uuid

import azure.functions as func

from shared.payloads import parse_request
from functions.api.kb_index.payloads import KBIndexRequest

logger = logging.getLogger(__name__)


async def main(
    req: func.HttpRequest, context: func.Context, msg: func.Out[str]
) -> func.HttpResponse:
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

        # Enqueue job
        job_payload = {
            "job_id": job_id,
            "workspace_id": payload.workspace_id,
            "document_url": payload.document_url,
            "kb_id": payload.kb_id,
        }
        msg.set(json.dumps(job_payload))

        logger.info(f"Enqueued indexing job {job_id} for workspace {payload.workspace_id}")

        return func.HttpResponse(
            json.dumps({"job_id": job_id, "status": "queued"}),
            status_code=202,
            mimetype="application/json",
        )

    except Exception as e:
        logger.error(f"KB index error: {e}", exc_info=True)
        return func.HttpResponse(
            json.dumps({"error": "Failed to enqueue indexing job"}),
            status_code=500,
            mimetype="application/json",
        )
