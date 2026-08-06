"""
Query RAG API Endpoint - UPDATED

REST API endpoint with proper security:
- User-workspace membership validation
- Domain and KB name fetched from database (not from UI)
- Follows SOLID principles and security best practices
"""
from src.core.abstractions import AbstractContext, AbstractRequest, AbstractResponse
from src.core.auth import get_user_id, require_auth
from src.core.exceptions import AuthorizationException, ValidationException
from src.core.logging import get_logger
from src.functions.api.query_rag.payloads import (
    ErrorResponse,
    QueryRAGRequest,
    QueryRAGResponse,
    RetrievedChunkInfo,
    SourceInfo,
)
from src.services.rag_query_service import get_rag_query_service
from src.services.workspace_service import get_workspace_service
from src.shared import create_error_response, create_success_response, parse_request

logger = get_logger(__name__)


@require_auth()
async def main(req: AbstractRequest, context: AbstractContext) -> AbstractResponse:
    """
    Query RAG endpoint with proper security.

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
        200: QueryRAGResponse with answer, sources, and chunks
        400: Validation error
        403: Not authorized for workspace
        500: Server error
    """
    correlation_id = context.correlation_id
    user_id = get_user_id(req)

    logger.info(
        "Query RAG request received",
        correlation_id=correlation_id,
        user_id=user_id
    )

    try:
        # Parse and validate request payload
        payload, error_response = parse_request(req, QueryRAGRequest)
        if error_response:
            return error_response

        workspace_id = payload.workspace_id

        # SECURITY: Validate user-workspace membership
        # This ensures user is actually part of the workspace
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

        # RELIABILITY: Fetch workspace config from database
        # This ensures domain and kb_name are consistent and tamper-proof
        workspace_config = await workspace_service.get_workspace_config(workspace_id)

        logger.info(
            "Workspace config retrieved",
            workspace_id=workspace_id,
            domain=workspace_config.domain,
            kb_name=workspace_config.kb_name,
            role_id=role_id,
            correlation_id=correlation_id
        )

        # Execute query via service layer
        rag_service = get_rag_query_service()
        result = await rag_service.query(
            query=payload.query,
            workspace_id=workspace_id,
            role_id=role_id,
            domain=workspace_config.domain,  # From database, not UI
            kb_name=workspace_config.kb_name,  # From database, not UI
            mode=payload.mode,
            history=payload.history,
            knowledge_bases=None,  # Derived internally based on workspace
            agent_id=payload.agent_id
        )

        # Convert to response model
        response_data = _build_response(result, workspace_config)

        logger.info(
            "Query RAG completed successfully",
            correlation_id=correlation_id,
            workspace_id=workspace_id,
            answer_length=len(response_data.response),
            source_count=len(response_data.sources),
            chunk_count=len(response_data.retrieved_chunks)
        )

        return create_success_response(
            data=response_data.dict(),
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
        return create_error_response(
            message="An error occurred while processing your query",
            error_code="INTERNAL_ERROR",
            details={"error": str(e)},
            status_code=500,
            correlation_id=correlation_id
        )


def _build_response(result, workspace_config) -> QueryRAGResponse:
    """
    Build API response from service result.

    Design: Data transformation in controller layer
    - Converts domain models to API models
    - Adds workspace metadata
    - Adds legacy compatibility fields
    """
    # Convert sources
    sources = [
        SourceInfo(
            file_name=src.file_name,
            download_url=src.download_url,
            container_name=src.container_name,
            blob_path=src.blob_path,
            download_name=src.download_name,
            citation=src.citation
        )
        for src in result.sources
    ]

    # Convert chunks
    chunks = [
        RetrievedChunkInfo(
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            score=chunk.score,
            source=chunk.source,
            metadata=chunk.metadata
        )
        for chunk in result.retrieved_chunks
    ]

    # Build metadata with workspace info
    metadata = dict(result.metadata)
    metadata.update({
        "workspace_id": workspace_config.workspace_id,
        "domain": workspace_config.domain,
        "kb_name": workspace_config.kb_name,
    })

    # Build response
    response = QueryRAGResponse(
        response=result.answer,
        sources=sources,
        retrieved_chunks=chunks,
        metadata=metadata,
        LightRAG=result.answer,  # Legacy compatibility
        task_ids=[]  # Legacy compatibility
    )

    return response
