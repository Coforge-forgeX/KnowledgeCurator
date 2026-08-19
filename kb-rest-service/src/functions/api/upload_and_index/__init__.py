"""
Upload and Index API - Delegates to RAG Service

This endpoint:
1. Validates payload and user permissions
2. Checks workspace access and can_curate_kb permission
3. Delegates upload/indexing to RAG service
4. Returns immediately with task IDs (non-blocking)
"""

import asyncio
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from sqlalchemy import select

from src.core import (
    get_logger,
    get_user_id,
    require_auth,
)
from src.core.abstractions import AbstractContext, AbstractRequest, AbstractResponse
from src.core.exceptions import APIException
from src.core.idempotency import check_idempotency, store_idempotency_result
from src.services.rag_service import get_rag_service
from src.helpers.workspace_helpers import get_workspace_storage_paths
from src.helpers.workspace_permissions import require_workspace_admin_curator
from src.common import (
    create_error_response,
    create_success_response,
    parse_request,
)

from .payloads import FileTaskResponse, UploadAndIndexRequest, UploadAndIndexResponse

logger = get_logger(__name__)


@require_auth()
async def main(req: AbstractRequest, context: AbstractContext) -> AbstractResponse:
    """
    Upload and index documents endpoint.

    POST /api/upload-and-index
    Headers:
        Authorization: Bearer <token>
        Idempotency-Key: <unique-key> (optional, recommended)
    Body:
        {
            "workspace_id": 1,
            "files": [
                {
                    "file_name": "document.pdf",
                    "file_content": "<base64>",
                    "file_size": 12345
                }
            ],
            "idempotency_key": "<unique-key>" (optional, can be in header or body)
        }

    Domain, KB name, and container are derived from workspace context.
    Supported file types: .pdf, .docx, .doc, .txt, .md

    Returns:
        202 Accepted: Files queued for indexing
        400 Bad Request: Invalid payload
        403 Forbidden: No permission
        404 Not Found: Workspace not found
        500 Server Error
    """
    # Parse request
    payload, error_response = parse_request(req, UploadAndIndexRequest)

    if error_response:
        return error_response

    user_id = get_user_id(req)
    # user_workspaces = get_workspace_ids(req)

    workspace_id = payload.workspace_id

    # Check for idempotency key (from header or body)
    idempotency_key = req.get_header("Idempotency-Key") or getattr(payload, "idempotency_key", None)

    if idempotency_key:
        # Check if this request was already processed
        cached_response = await check_idempotency(
            idempotency_key=idempotency_key,
            workspace_id=workspace_id,
            user_id=user_id,
            endpoint="/api/v2/documents/upload",
            request_body=payload.model_dump(),
        )

        if cached_response:
            logger.info(
                "Returning cached response for duplicate upload request",
                idempotency_key=idempotency_key,
                workspace_id=workspace_id,
                user_id=user_id,
            )
            return cached_response

    # # Check workspace access
    # if workspace_id not in user_workspaces:
    #     logger.warning(
    #         "Unauthorized workspace access",
    #         user_id=user_id,
    #         workspace_id=workspace_id,
    #     )
    #     raise AuthorizationException(
    #         message="You do not have access to this workspace"
    #     )

    try:
        await require_workspace_admin_curator(
            user_id=user_id,
            workspace_id=workspace_id,
            action_description="upload and index documents",
        )

        # Get workspace storage paths (container, upload_path, domain, kb_name)
        workspace_paths = await get_workspace_storage_paths(workspace_id)

        if not workspace_paths:
            logger.error("Failed to get workspace storage paths", workspace_id=workspace_id)
            return create_error_response(
                message="Failed to retrieve workspace information",
                error_code="WORKSPACE_NOT_FOUND",
                status_code=404,
                correlation_id=context.correlation_id,
            )

        container_name = workspace_paths["container"]
        upload_path_base = workspace_paths["upload_path"]
        domain = workspace_paths["domain"]
        kb_name = workspace_paths["kb_name"]
        is_kg = workspace_paths["is_kg"]
        kb_ids = workspace_paths["all_kb_ids"]

        # Initialize RAG service and delegate upload/indexing
        rag_service = get_rag_service()

        # Call RAG service to handle upload and indexing
        rag_result = await rag_service.upload_and_index_tool(
            files=payload.files,
            workspace_id=workspace_id,
            user_id=user_id,
            upload_path=upload_path_base,
            kb_ids=kb_ids,
            domain=domain,
            kb_name=kb_name,
            container_name=container_name,
            is_kg=is_kg,
        )

        task_ids = rag_result.get("task_ids", [])
        uploaded_files = rag_result.get("uploaded_files", [])
        failed_files = rag_result.get("failed_files", [])

        # Build response
        if not task_ids and failed_files:
            # All files failed
            return create_error_response(
                message="All files failed to upload and index",
                error_code="UPLOAD_FAILED",
                details={"failed_files": failed_files},
                status_code=500,
                correlation_id=context.correlation_id,
            )

        tasks: List[FileTaskResponse] = []
        for task_id, file_name in zip(task_ids, uploaded_files):
            tasks.append(
                FileTaskResponse(
                    task_id=task_id,
                    file_name=file_name,
                    file_path=file_name,
                    status="pending",
                )
            )

        response_data = UploadAndIndexResponse(
            success=True,
            message=f"Successfully queued {len(tasks)} file(s) for indexing",
            workspace_id=workspace_id,
            total_files=len(payload.files),
            tasks=tasks,
            failed_files=failed_files,
        )

        response = create_success_response(
            message=response_data.message,
            data=response_data.model_dump(),
            status_code=202,  # Accepted
            correlation_id=context.request_id,
        )

        # Store idempotency result for future duplicate requests
        if idempotency_key:
            await store_idempotency_result(
                idempotency_key=idempotency_key,
                workspace_id=workspace_id,
                user_id=user_id,
                endpoint="/api/v2/documents/upload",
                request_body=payload.model_dump(),
                response_status=response.status_code,
                response_body=response.body if isinstance(response.body, dict) else {},
            )

        return response

    except APIException:
        raise
    except Exception as e:
        logger.exception(
            "Unexpected error in upload_and_index",
            error=str(e),
            workspace_id=workspace_id,
            user_id=user_id,
            correlation_id=getattr(context, "correlation_id", None),
        )
        return create_error_response(
            message="An unexpected error occurred while processing your request",
            error_code="INTERNAL_ERROR",
            status_code=500,
            correlation_id=getattr(context, "correlation_id", None),
        )
