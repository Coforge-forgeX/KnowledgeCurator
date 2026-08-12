"""
Query RAG API Endpoint

Thin HTTP layer over the shared `services.query_rag_executor`:
- authenticates the caller and validates workspace membership
- resolves domain / kb_name / additional KBs from the database (never from the UI)
- delegates the cached RAG execution to `execute_query_rag`
- maps the result (or exception) onto an HTTP response

The execution itself lives in the service layer so `message_gpt` (chat) runs the
exact same path, cache included. Anything changed there applies to both callers.
"""
import time
from typing import Optional

from src.core.abstractions import AbstractContext, AbstractRequest, AbstractResponse
from src.core.auth import get_user_id, require_auth
from src.core.exceptions import AuthorizationException, ValidationException
from src.core.logging import get_logger
from src.helpers.workspace_helpers import get_workspace_storage_paths
from src.models.query_rag_models import QueryRAGRequest
from src.services.query_rag_executor import execute_query_rag
from src.services.workspace_service import get_workspace_service
from src.common import create_error_response, create_internal_error_response, create_success_response, parse_request

logger = get_logger(__name__)


@require_auth()
async def main(req: AbstractRequest, context: AbstractContext) -> AbstractResponse:
    """
    Query RAG endpoint.

    POST /api/query-rag
    Headers: Authorization: Bearer <token>
    Body: {
        "query": "What is asset management?",
        "workspace_id": 123,
        "mode": "hybrid",
        "history": [...],
        "agent_id": 1
    }

    Security:
    1. Validates user is authenticated (via @require_auth decorator)
    2. Validates user is member of workspace (database check)
    3. Fetches domain and kb_name from database (not from UI)
    4. Validates workspace exists and is active

    Returns:
        200: QueryRAGResponse with answer and sources
        400: Validation error
        403: Not authorized for workspace
        500: Server error
    """
    correlation_id = context.correlation_id
    user_id = get_user_id(req)
    start_time = time.time()
    payload: Optional[QueryRAGRequest] = None

    logger.info(
        "Query RAG request received",
        correlation_id=correlation_id,
        user_id=user_id
    )

    try:
        payload, error_response = parse_request(req, QueryRAGRequest)
        if error_response:
            return error_response

        workspace_id = payload.workspace_id

        # SECURITY: membership check runs before any cache lookup, so a cached
        # answer can never be served to a user who lost workspace access.
        workspace_service = get_workspace_service()
        is_authorized, role_id = await workspace_service.validate_user_workspace_access(
            user_id=user_id,
            workspace_id=workspace_id
        )

        if not is_authorized:
            logger.warning(
                "User not authorized for workspace",
                user_id=user_id,
                workspace_id=workspace_id,
                correlation_id=correlation_id
            )
            raise AuthorizationException(
                message=f"You are not authorized to access workspace {workspace_id}"
            )

        storage_paths = await get_workspace_storage_paths(workspace_id)

        if not storage_paths:
            logger.error(
                "Failed to retrieve workspace storage paths",
                workspace_id=workspace_id,
                correlation_id=correlation_id
            )
            raise ValidationException(
                message=f"Failed to retrieve workspace configuration for workspace {workspace_id}"
            )

        domain = storage_paths.get("domain", "")
        kb_name = storage_paths.get("kb_name", "")
        all_kb_titles = storage_paths.get("all_kb_titles", [])

        # For non-KG workspaces with multiple KBs, pass additional KB titles for querying.
        # The primary kb_name is the base; all_kb_titles provides the rest.
        additional_kbs = all_kb_titles[1:] if len(all_kb_titles) > 1 else None

        logger.info(
            "Workspace storage paths retrieved",
            workspace_id=workspace_id,
            domain=domain,
            kb_name=kb_name,
            container=storage_paths.get("container"),
            is_kg=storage_paths.get("is_kg"),
            kb_count=len(all_kb_titles),
            role_id=role_id,
            correlation_id=correlation_id
        )

        response_dict, cache_hit = await execute_query_rag(
            query=payload.query,
            workspace_id=workspace_id,
            role_id=role_id,
            domain=domain,
            kb_name=kb_name,
            mode=payload.mode,
            history=payload.history,
            additional_kbs=additional_kbs,
            agent_id=payload.agent_id,
            is_kg=storage_paths.get("is_kg"),
            correlation_id=correlation_id,
        )

        logger.info(
            "Query RAG completed successfully",
            correlation_id=correlation_id,
            workspace_id=workspace_id,
            cache_hit=cache_hit,
            total_time_ms=round((time.time() - start_time) * 1000, 2),
        )

        return create_success_response(
            message="Query processed successfully (cached)" if cache_hit else "Query processed successfully",
            data=response_dict,
            status_code=200,
            correlation_id=correlation_id
        )

    except ValidationException as e:
        logger.warning(
            "Validation error",
            error=e.message,
            correlation_id=correlation_id
        )
        return create_error_response(
            message=e.message,
            error_code="VALIDATION_ERROR",
            status_code=400,
            correlation_id=correlation_id
        )

    except AuthorizationException as e:
        logger.warning(
            "Authorization error",
            error=e.message,
            user_id=user_id,
            workspace_id=payload.workspace_id if payload else None,
            correlation_id=correlation_id
        )
        return create_error_response(
            message=e.message,
            error_code="AUTHORIZATION_ERROR",
            status_code=403,
            correlation_id=correlation_id
        )

    except Exception as e:
        logger.error(
            "Query RAG failed",
            error=e,
            correlation_id=correlation_id
        )
        return create_internal_error_response(
            message="An error occurred while processing your query",
            error=e,
            error_code="INTERNAL_ERROR",
            correlation_id=correlation_id
        )
