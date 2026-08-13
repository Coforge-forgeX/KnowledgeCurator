"""
Delete All Indexed Documents API

Backend service endpoint to delete all indexed documents in a workspace.
Requires secret key authentication for service-to-service calls.
Uses shared deletion utilities from src.helpers.deletion.
"""
from typing import Any, Dict

from src.core.abstractions import AbstractContext, AbstractRequest, AbstractResponse
from src.core.config import settings
from src.core.exceptions import AuthenticationException, ValidationException
from src.core.logging import get_logger
from src.common import create_error_response, create_internal_error_response, create_success_response, parse_request
from src.helpers.deletion import (
    clear_redis_file_cache,
    delete_orm_metadata_for_workspace,
    delete_single_document,
    get_all_workspace_documents,
    run_with_db_retry,
    summarize_lightrag_counts,
)

from .payloads import DeleteAllIndexedDocumentsRequest

logger = get_logger(__name__)


def _validate_backend_service_auth(req: AbstractRequest) -> None:
    """
    Validate backend service secret key from X-Backend-Secret header.

    Args:
        req: AbstractRequest with headers

    Raises:
        AuthenticationException: If secret key is missing or invalid
    """
    expected_secret = settings.security.BACKEND_SERVICE_SECRET_KEY
    if not expected_secret:
        logger.error("BACKEND_SERVICE_SECRET_KEY not configured in environment")
        raise AuthenticationException(
            message="Backend service authentication is not configured"
        )

    provided_secret = req.get_header("X-Backend-Secret")
    if not provided_secret:
        raise AuthenticationException(
            message="Missing X-Backend-Secret header"
        )

    if provided_secret != expected_secret:
        raise AuthenticationException(
            message="Invalid backend service secret key"
        )


async def main(req: AbstractRequest, context: AbstractContext) -> AbstractResponse:
    """
    Delete all indexed documents in a workspace.

    Requires X-Backend-Secret header for authentication.
    This endpoint is intended for backend service-to-service calls only.

    Args:
        req: AbstractRequest with workspace_id in body and X-Backend-Secret in headers
        context: AbstractContext with correlation_id

    Returns:
        AbstractResponse with deletion summary
    """
    correlation_id = context.correlation_id

    try:
        # Validate backend service authentication
        _validate_backend_service_auth(req)

        # Parse request payload
        payload, error_response = parse_request(req, DeleteAllIndexedDocumentsRequest)
        if error_response:
            return error_response

        workspace_id = int(payload.workspace_id)

        logger.info(
            "Starting delete all indexed documents operation",
            workspace_id=workspace_id,
            correlation_id=correlation_id,
        )

        # Fetch all documents for the workspace using shared utility
        documents = await get_all_workspace_documents(workspace_id)

        if not documents:
            logger.info(
                "No documents found for workspace",
                workspace_id=workspace_id,
            )
            return create_success_response(
                message="No indexed documents found in workspace",
                data={
                    "workspace_id": workspace_id,
                    "deleted_count": 0,
                    "failed_count": 0,
                    "cleanup_summary": {},
                },
                correlation_id=correlation_id,
            )

        deleted_count = 0
        failed_count = 0
        total_lightrag_deleted = {
            "lightrag_vdb_chunks": 0,
            "lightrag_vdb_relation": 0,
            "lightrag_vdb_entity": 0,
        }
        total_graph_deleted = {
            "nodes_deleted": 0,
            "relations_deleted": 0,
        }
        total_storage_deleted = 0

        # Delete each document using shared deletion utility
        for doc in documents:
            file_path = doc["file_path"]
            file_name = doc["file_name"]
            try:
                # Use shared delete_single_document utility
                result = await delete_single_document(
                    workspace_id=workspace_id,
                    file_path=file_path,
                    file_name=file_name,
                    workspace_labels=[],
                )

                # Accumulate totals
                cleanup = result.get("cleanup", {})
                if cleanup.get("storage_deleted"):
                    total_storage_deleted += 1

                lightrag_deleted = cleanup.get("lightrag_deleted", {})
                for key in ["chunks", "relations", "entities"]:
                    table_key = f"lightrag_vdb_{key.rstrip('s')}"  # Convert to table name
                    if key == "chunks":
                        total_lightrag_deleted["lightrag_vdb_chunks"] += lightrag_deleted.get(key, 0)
                    elif key == "relations":
                        total_lightrag_deleted["lightrag_vdb_relation"] += lightrag_deleted.get(key, 0)
                    elif key == "entities":
                        total_lightrag_deleted["lightrag_vdb_entity"] += lightrag_deleted.get(key, 0)

                neo4j_deleted = cleanup.get("neo4j_deleted", {})
                total_graph_deleted["nodes_deleted"] += neo4j_deleted.get("nodes", 0)
                total_graph_deleted["relations_deleted"] += neo4j_deleted.get("relations", 0)

                deleted_count += 1

            except Exception as e:
                logger.error(
                    f"Failed to delete document: {file_path}",
                    error=e,
                    workspace_id=workspace_id,
                )
                failed_count += 1

        # Delete ORM metadata for workspace using shared utility
        orm_deleted = await run_with_db_retry(
            lambda: delete_orm_metadata_for_workspace(workspace_id=workspace_id),
            operation_name="delete_orm_metadata_for_workspace",
        )

        # Clear Redis cache using shared utility
        cleared_keys = clear_redis_file_cache(clear_all=True)
        if cleared_keys > 0:
            logger.info(f"Cleared {cleared_keys} Redis cache entries")

        # Summarize totals using shared utility
        lightrag_summary = summarize_lightrag_counts(total_lightrag_deleted)

        cleanup_summary = {
            "storage_deleted": total_storage_deleted,
            "lightrag_deleted": lightrag_summary,
            "neo4j_deleted": total_graph_deleted,
            "orm_deleted": orm_deleted,
        }

        logger.info(
            "Delete all indexed documents completed",
            workspace_id=workspace_id,
            deleted_count=deleted_count,
            failed_count=failed_count,
            cleanup_summary=cleanup_summary,
        )

        return create_success_response(
            message=f"Successfully deleted {deleted_count} documents from workspace",
            data={
                "workspace_id": workspace_id,
                "deleted_count": deleted_count,
                "failed_count": failed_count,
                "cleanup_summary": cleanup_summary,
            },
            correlation_id=correlation_id,
        )

    except AuthenticationException as e:
        return create_error_response(
            message=e.message,
            error_code="AUTHENTICATION_ERROR",
            status_code=401,
            correlation_id=correlation_id,
        )
    except ValidationException as e:
        return create_error_response(
            message=e.message,
            error_code="VALIDATION_ERROR",
            status_code=400,
            correlation_id=correlation_id,
        )
    except Exception as e:
        logger.error("Delete all indexed documents failed", error=e)
        return create_internal_error_response(
            message="Failed to delete all indexed documents",
            error=e,
            error_code="DELETE_ALL_DOCUMENTS_FAILED",
            correlation_id=correlation_id,
        )
