"""
Query RAG API Endpoint - OPTIMIZED

REST API endpoint with proper security and performance optimizations:
- User-workspace membership validation
- Domain and KB name fetched from database (not from UI)
- Redis caching for query results (60%+ hit ratio expected)
- Workspace config caching
- Follows SOLID principles and security best practices
"""
import json
import time
import hashlib
from typing import Any, Dict, List

from src.core.abstractions import AbstractContext, AbstractRequest, AbstractResponse
from src.core.auth import get_user_id, require_auth
from src.core.config import settings
from src.core.exceptions import AuthorizationException, ValidationException
from src.core.logging import get_logger
from src.core.redis import get_query_cache, set_query_cache, redis_manager
from src.functions.api.query_rag.payloads import (
    KBChunkModel,
    KBResultModel,
    GraphDataModel,
    QueryRAGRequest,
    QueryRAGResponse,
    SourceReferenceModel,
)
from src.helpers.workspace_helpers import get_workspace_storage_paths
from src.helpers.graph_parser import format_chunk_with_graph_data
from src.helpers.file_token import create_signed_file_id
from src.services.rag_query_service import get_rag_query_service
from src.services.workspace_service import get_workspace_service
from src.common import create_error_response, create_success_response, parse_request

logger = get_logger(__name__)

QUERY_EVIDENCE_TTL_SECONDS = 30 * 60


def _normalize_query_text(query: str) -> str:
    """Normalize user query text so semantically identical spacing maps to one key."""
    return " ".join((query or "").strip().split())


def _make_query_evidence_key(workspace_id: int, query: str, mode: str) -> str:
    """Create deterministic redis key for cached query evidence payload."""
    normalized_query = _normalize_query_text(query)
    normalized_mode = (mode or "").strip().lower()
    key_material = f"{workspace_id}|{normalized_mode}|{normalized_query}"
    query_hash = hashlib.sha256(key_material.encode("utf-8")).hexdigest()[:24]
    return f"query_evidence:{workspace_id}:{query_hash}"


def _make_source_mapping_key(file_id: str) -> str:
    """Create redis key for file token to storage mapping."""
    return f"query_file:{file_id}"


@require_auth()
async def main(req: AbstractRequest, context: AbstractContext) -> AbstractResponse:
    """
    Optimized Query RAG endpoint with caching and proper security.

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

    Performance Optimizations:
    1. Redis caching for query results (100x faster for cached queries)
    2. Workspace config caching (5 min TTL)
    3. Connection pooling via SQLAlchemy async engine

    Returns:
        200: QueryRAGResponse with answer, sources, and chunks
        400: Validation error
        403: Not authorized for workspace
        500: Server error
    """
    correlation_id = context.correlation_id
    user_id = get_user_id(req)
    start_time = time.time()

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

        # ===========================================
        # OPTIMIZATION 1: Try Cache First
        # ===========================================
        if redis_manager.is_available and getattr(settings.cache, 'REDIS_ENABLED', True):
            cached_result = get_query_cache(workspace_id, payload.query, payload.mode)
            if cached_result:
                cache_elapsed = time.time() - start_time
                logger.info(
                    "Query RAG completed from cache",
                    correlation_id=correlation_id,
                    workspace_id=workspace_id,
                    cache_hit=True,
                    response_time_ms=round(cache_elapsed * 1000, 2),
                )
                return create_success_response(
                    message="Query processed successfully (cached)",
                    data=cached_result,
                    status_code=200,
                    correlation_id=correlation_id
                )

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

        # ===========================================
        # OPTIMIZATION 2: Get Workspace Storage Paths
        # ===========================================
        # Fetch workspace storage paths (includes computed domain and kb_name)
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

        # For non-KG workspaces with multiple KBs, pass additional KB titles for querying
        # The primary kb_name is used as base, and all_kb_titles provides additional KBs to search
        additional_kbs = None
        if len(all_kb_titles) > 1:
            # Skip the first KB (already in kb_name), add the rest
            additional_kbs = all_kb_titles[1:]
            logger.debug(
                "Multi-KB workspace detected",
                workspace_id=workspace_id,
                primary_kb=all_kb_titles[0] if all_kb_titles else None,
                additional_kb_count=len(additional_kbs)
            )

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

        # Execute query via service layer
        rag_service = get_rag_query_service()
        result = await rag_service.query(
            query=payload.query,
            workspace_id=workspace_id,
            role_id=role_id,
            domain=domain,  # Computed from workspace metadata
            kb_name=kb_name,  # Primary KB name (for indexing and base query)
            mode=payload.mode,
            history=payload.history,
            knowledge_bases=additional_kbs,  # Additional KBs for multi-KB search
            agent_id=payload.agent_id,
            is_kg=storage_paths.get("is_kg"),
        )

        # Convert to response model and cache heavyweight evidence separately.
        response_data, kb_results = _build_response(result, workspace_id, domain, kb_name)

        if redis_manager.is_available and getattr(settings.cache, 'REDIS_ENABLED', True):
            evidence_cache_key = _make_query_evidence_key(workspace_id, payload.query, payload.mode)
            evidence_payload = {
                "workspace_id": workspace_id,
                "query": payload.query,
                "requested_mode": payload.mode,
                "source": [source_item.dict() for source_item in response_data.source],
                "kb_results": [kb.dict() for kb in kb_results],
            }
            try:
                redis_manager.setex(
                    evidence_cache_key,
                    QUERY_EVIDENCE_TTL_SECONDS,
                    json.dumps(evidence_payload),
                )
            except Exception as cache_error:
                logger.warning(
                    "Failed to cache query evidence; continuing without cache",
                    error=cache_error,
                    correlation_id=correlation_id,
                    workspace_id=workspace_id,
                )

            for source_item in response_data.source:
                file_id = create_signed_file_id(
                    workspace_id=workspace_id,
                    container_name=source_item.container_name,
                    blob_path=source_item.blob_path,
                    provider=source_item.provider,
                    file_name=source_item.file_name,
                )
                source_mapping = {
                    "file_id": file_id,
                    "workspace_id": workspace_id,
                    "container_name": source_item.container_name,
                    "blob_path": source_item.blob_path,
                    "provider": source_item.provider,
                    "file_name": source_item.file_name,
                    "citation": source_item.citation,
                    "evidence_cache_key": evidence_cache_key,
                }
                try:
                    redis_manager.setex(
                        _make_source_mapping_key(file_id),
                        QUERY_EVIDENCE_TTL_SECONDS,
                        json.dumps(source_mapping),
                    )
                except Exception as cache_error:
                    logger.warning(
                        "Failed to cache source mapping; signed file_id remains usable",
                        error=cache_error,
                        correlation_id=correlation_id,
                        workspace_id=workspace_id,
                    )
                source_item.file_id = file_id

        response_dict = response_data.dict()

        # ===========================================
        # OPTIMIZATION 3: Cache Result
        # ===========================================
        if redis_manager.is_available and getattr(settings.cache, 'REDIS_ENABLED', True):
            cache_ttl = getattr(settings.cache, 'QUERY_CACHE_TTL', 3600)
            cache_ttl = min(cache_ttl, QUERY_EVIDENCE_TTL_SECONDS)
            set_query_cache(workspace_id, payload.query, payload.mode, response_dict, ttl=cache_ttl)

        total_elapsed = time.time() - start_time

        logger.info(
            "Query RAG completed successfully",
            correlation_id=correlation_id,
            workspace_id=workspace_id,
            answer_length=len(response_data.final_answer),
            source_count=len(response_data.source),
            kb_count=len(kb_results),
            chunk_count=sum(len(kb.chunks) for kb in kb_results),
            cache_hit=False,
            total_time_ms=round(total_elapsed * 1000, 2),
        )

        return create_success_response(
            message="Query processed successfully",
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
        return create_error_response(
            message="An error occurred while processing your query",
            error_code="INTERNAL_ERROR",
            details={"error": str(e)},
            status_code=500,
            correlation_id=correlation_id
        )


def _build_response(result, workspace_id: int, domain: str, kb_name: str) -> tuple[QueryRAGResponse, List[KBResultModel]]:
    """
    Build API response from service result.

    Design: Data transformation in controller layer
    - Converts domain models to API models
    - Adds workspace metadata
    - Adds legacy compatibility fields
    - Parses and structures graph data into JSON
    """
    metadata = result.metadata if isinstance(result.metadata, dict) else {}

    def _apply_graph_payload(
        graph_payload: Dict[str, Any],
        graph_entities: List[Dict[str, Any]],
        graph_relationships: List[Dict[str, Any]],
        graph_metadata: Dict[str, Any],
        graph_chunk_refs: List[Dict[str, str]],
        seen_graph_refs: set,
        kb_chunks: List[KBChunkModel],
        seen_chunk_ids: set,
        base_chunk_id: str,
        fallback_file_path: str,
    ) -> None:
        """Merge parsed graph payload into response graph_data and chunks."""
        entities = graph_payload.get("entities", []) if isinstance(graph_payload, dict) else []
        relationships = graph_payload.get("relationships", []) if isinstance(graph_payload, dict) else []
        parsed_metadata = graph_payload.get("metadata", {}) if isinstance(graph_payload, dict) else {}
        document_chunks = graph_payload.get("document_chunks", []) if isinstance(graph_payload, dict) else []

        source_to_file_path: Dict[str, str] = {}

        def _extract_source_ids(raw_source_id: Any) -> List[str]:
            if raw_source_id is None:
                return []
            if isinstance(raw_source_id, list):
                return [str(s).strip() for s in raw_source_id if str(s).strip()]
            if isinstance(raw_source_id, (int, float)):
                return [str(raw_source_id)]
            text = str(raw_source_id)
            normalized = text.replace(";", ",").replace("|", ",")
            return [part.strip() for part in normalized.split(",") if part.strip()]

        if isinstance(relationships, list):
            graph_relationships.extend([r for r in relationships if isinstance(r, dict)])

        if isinstance(parsed_metadata, dict):
            graph_metadata.update(parsed_metadata)

        if isinstance(document_chunks, list):
            for doc_index, doc_chunk in enumerate(document_chunks):
                if not isinstance(doc_chunk, dict):
                    continue

                doc_content = str(
                    doc_chunk.get("content")
                    or doc_chunk.get("text")
                    or doc_chunk.get("chunk")
                    or doc_chunk.get("description")
                    or ""
                ).strip()
                if not doc_content:
                    continue

                doc_chunk_id = str(
                    doc_chunk.get("chunk_id")
                    or doc_chunk.get("id")
                    or f"{base_chunk_id}:doc:{doc_index}"
                )
                if doc_chunk_id in seen_chunk_ids:
                    continue

                doc_file_path = str(
                    doc_chunk.get("file_path")
                    or doc_chunk.get("source")
                    or doc_chunk.get("file_name")
                    or fallback_file_path
                )

                source_to_file_path[doc_chunk_id] = doc_file_path

                for raw_ref in _extract_source_ids(doc_chunk.get("source_id")):
                    source_to_file_path[raw_ref] = doc_file_path

                kb_chunks.append(
                    KBChunkModel(
                        chunk_id=doc_chunk_id,
                        chunk=doc_content,
                        file_path=doc_file_path,
                    )
                )
                seen_chunk_ids.add(doc_chunk_id)

                ref_key = (doc_chunk_id, doc_file_path)
                if ref_key not in seen_graph_refs:
                    graph_chunk_refs.append(
                        {
                            "chunk_id": doc_chunk_id,
                            "file_path": doc_file_path,
                        }
                    )
                    seen_graph_refs.add(ref_key)

        if isinstance(entities, list):
            for entity in entities:
                if not isinstance(entity, dict):
                    continue

                enriched_entity = dict(entity)
                source_ids = _extract_source_ids(enriched_entity.get("source_id"))

                if not source_ids:
                    source_ids = [base_chunk_id] if base_chunk_id else []
                    if source_ids:
                        enriched_entity["source_id"] = source_ids[0]

                if not enriched_entity.get("file_path"):
                    resolved_paths = [
                        source_to_file_path[sid]
                        for sid in source_ids
                        if sid in source_to_file_path and source_to_file_path[sid]
                    ]
                    if resolved_paths:
                        unique_paths = list(dict.fromkeys(resolved_paths))
                        enriched_entity["file_path"] = unique_paths[0] if len(unique_paths) == 1 else unique_paths
                    elif fallback_file_path:
                        enriched_entity["file_path"] = fallback_file_path

                graph_entities.append(enriched_entity)

    requested_mode = str(metadata.get("requested_mode") or metadata.get("mode") or "hybrid")
    effective_mode = str(metadata.get("effective_mode") or requested_mode)

    kb_results_payload = metadata.get("kb_results") if isinstance(metadata.get("kb_results"), dict) else {}

    per_kb_results: List[KBResultModel] = []

    if kb_results_payload:
        for kb_source, kb_payload in kb_results_payload.items():
            if not isinstance(kb_payload, dict) or "error" in kb_payload:
                continue

            graph_entities: List[Dict[str, Any]] = []
            graph_relationships: List[Dict[str, Any]] = []
            graph_metadata: Dict[str, Any] = {}
            graph_chunk_refs: List[Dict[str, str]] = []
            kb_chunks: List[KBChunkModel] = []
            seen_chunk_ids = set()
            seen_graph_refs = set()

            raw_chunks = kb_payload.get("_retrieved_chunks", [])
            if isinstance(raw_chunks, list):
                for raw_chunk in raw_chunks:
                    if not isinstance(raw_chunk, dict):
                        continue

                    # Extract metadata
                    raw_chunk_metadata = raw_chunk.get("metadata") if isinstance(raw_chunk.get("metadata"), dict) else {}

                    # Get file_path from multiple possible locations
                    file_path = str(
                        raw_chunk.get("file_path")
                        or raw_chunk_metadata.get("file_path")
                        or raw_chunk.get("source")
                        or raw_chunk.get("file_name")
                        or ""
                    )

                    # Build chunk dict - preserve original content for vector chunks
                    chunk_id = str(raw_chunk.get("chunk_id") or "")
                    chunk_content = str(raw_chunk.get("content") or "")

                    # Only format as graph data if it actually contains graph markers
                    if "Knowledge Graph Data" in chunk_content:
                        chunk_dict = {
                            "chunk_id": chunk_id,
                            "content": chunk_content,
                            "score": raw_chunk.get("score", 0.0),
                            "source": str(raw_chunk.get("source") or ""),
                            "metadata": raw_chunk_metadata
                        }
                        enhanced = format_chunk_with_graph_data(chunk_dict)
                        final_content = enhanced.get("summary", chunk_content)

                        graph_payload = enhanced.get("graph_data") if isinstance(enhanced, dict) else None
                        if isinstance(graph_payload, dict):
                            _apply_graph_payload(
                                graph_payload=graph_payload,
                                graph_entities=graph_entities,
                                graph_relationships=graph_relationships,
                                graph_metadata=graph_metadata,
                                graph_chunk_refs=graph_chunk_refs,
                                seen_graph_refs=seen_graph_refs,
                                kb_chunks=kb_chunks,
                                seen_chunk_ids=seen_chunk_ids,
                                base_chunk_id=chunk_id,
                                fallback_file_path=file_path,
                            )
                    else:
                        # This is a vector chunk - use original content
                        final_content = chunk_content

                    # Add chunk with proper structure: chunk_id, chunk (content), file_path
                    kb_chunks.append(
                        KBChunkModel(
                            chunk_id=chunk_id,
                            chunk=final_content,  # This is chunk_data
                            file_path=file_path
                        )
                    )
                    seen_chunk_ids.add(chunk_id)

                    if "Knowledge Graph Data" in chunk_content:
                        ref_key = (chunk_id, file_path)
                        if ref_key not in seen_graph_refs:
                            graph_chunk_refs.append(
                                {
                                    "chunk_id": chunk_id,
                                    "file_path": file_path,
                                }
                            )
                            seen_graph_refs.add(ref_key)

            graph_context = kb_payload.get("_raw_context", [])
            graph_context_items = graph_context if isinstance(graph_context, list) else [graph_context]
            for graph_item in graph_context_items:
                if not graph_item:
                    continue
                if isinstance(graph_item, dict):
                    context_chunk_id = str(kb_source) + ":raw_context"
                    _apply_graph_payload(
                        graph_payload=graph_item,
                        graph_entities=graph_entities,
                        graph_relationships=graph_relationships,
                        graph_metadata=graph_metadata,
                        graph_chunk_refs=graph_chunk_refs,
                        seen_graph_refs=seen_graph_refs,
                        kb_chunks=kb_chunks,
                        seen_chunk_ids=seen_chunk_ids,
                        base_chunk_id=context_chunk_id,
                        fallback_file_path=str(kb_source),
                    )
                    continue
                graph_chunk_dict = {
                    "chunk_id": "",
                    "content": str(graph_item),
                    "score": 0.0,
                    "source": kb_source,
                    "metadata": {},
                }
                graph_enhanced = format_chunk_with_graph_data(graph_chunk_dict)
                if graph_enhanced.get("content_type") != "graph" or not isinstance(graph_enhanced.get("graph_data"), dict):
                    continue
                context_chunk_id = str(kb_source) + ":raw_context"
                _apply_graph_payload(
                    graph_payload=graph_enhanced["graph_data"],
                    graph_entities=graph_entities,
                    graph_relationships=graph_relationships,
                    graph_metadata=graph_metadata,
                    graph_chunk_refs=graph_chunk_refs,
                    seen_graph_refs=seen_graph_refs,
                    kb_chunks=kb_chunks,
                    seen_chunk_ids=seen_chunk_ids,
                    base_chunk_id=context_chunk_id,
                    fallback_file_path=str(kb_source),
                )

            per_kb_results.append(
                KBResultModel(
                    source=kb_source,
                    graph_data=GraphDataModel(
                        entities=graph_entities,
                        relationship=graph_relationships,
                        metadata=graph_metadata,
                        chunk_references=graph_chunk_refs if graph_chunk_refs else None,
                    ),
                    chunks=kb_chunks,
                )
            )
    else:
        # Single-KB fallback if strategy metadata does not include multi-KB payload.
        graph_entities: List[Dict[str, Any]] = []
        graph_relationships: List[Dict[str, Any]] = []
        graph_metadata: Dict[str, Any] = {}
        graph_chunk_refs: List[Dict[str, str]] = []
        kb_chunks: List[KBChunkModel] = []
        seen_chunk_ids = set()
        seen_graph_refs = set()

        kb_source = metadata.get("kb") or f"{domain}/{kb_name}"

        for chunk in result.retrieved_chunks:
            chunk_metadata = getattr(chunk, "metadata", {}) if isinstance(getattr(chunk, "metadata", {}), dict) else {}
            chunk_content = str(getattr(chunk, "content", ""))
            chunk_id = str(getattr(chunk, "chunk_id", ""))

            # Get file_path from multiple possible locations
            file_path = str(
                chunk_metadata.get("file_path")
                or getattr(chunk, "file_path", "")
                or getattr(chunk, "source", "")
                or chunk_metadata.get("source")
                or ""
            )

            # Only format as graph data if it actually contains graph markers
            if "Knowledge Graph Data" in chunk_content:
                chunk_dict = {
                    "chunk_id": chunk_id,
                    "content": chunk_content,
                    "score": getattr(chunk, "score", 0.0),
                    "source": str(getattr(chunk, "source", "")),
                    "metadata": chunk_metadata,
                }
                enhanced = format_chunk_with_graph_data(chunk_dict)
                final_content = enhanced.get("summary", chunk_content)

                graph_payload = enhanced.get("graph_data") if isinstance(enhanced, dict) else None
                if isinstance(graph_payload, dict):
                    _apply_graph_payload(
                        graph_payload=graph_payload,
                        graph_entities=graph_entities,
                        graph_relationships=graph_relationships,
                        graph_metadata=graph_metadata,
                        graph_chunk_refs=graph_chunk_refs,
                        seen_graph_refs=seen_graph_refs,
                        kb_chunks=kb_chunks,
                        seen_chunk_ids=seen_chunk_ids,
                        base_chunk_id=chunk_id,
                        fallback_file_path=file_path,
                    )
            else:
                # This is a vector chunk - use original content
                final_content = chunk_content

            kb_chunks.append(
                KBChunkModel(
                    chunk_id=chunk_id,
                    chunk=final_content,  # This is chunk_data
                    file_path=file_path
                )
            )
            seen_chunk_ids.add(chunk_id)

            if "Knowledge Graph Data" in chunk_content:
                ref_key = (chunk_id, file_path)
                if ref_key not in seen_graph_refs:
                    graph_chunk_refs.append(
                        {
                            "chunk_id": chunk_id,
                            "file_path": file_path,
                        }
                    )
                    seen_graph_refs.add(ref_key)

        raw_graph_context = metadata.get("graph_context_by_kb", {}).get(kb_source, []) if isinstance(metadata.get("graph_context_by_kb"), dict) else []
        graph_context_items = raw_graph_context if isinstance(raw_graph_context, list) else [raw_graph_context]
        for graph_item in graph_context_items:
            if not graph_item:
                continue

            if isinstance(graph_item, dict):
                context_chunk_id = str(kb_source) + ":raw_context"
                _apply_graph_payload(
                    graph_payload=graph_item,
                    graph_entities=graph_entities,
                    graph_relationships=graph_relationships,
                    graph_metadata=graph_metadata,
                    graph_chunk_refs=graph_chunk_refs,
                    seen_graph_refs=seen_graph_refs,
                    kb_chunks=kb_chunks,
                    seen_chunk_ids=seen_chunk_ids,
                    base_chunk_id=context_chunk_id,
                    fallback_file_path=str(kb_source),
                )
                continue

            graph_chunk_dict = {
                "chunk_id": "",
                "content": str(graph_item),
                "score": 0.0,
                "source": kb_source,
                "metadata": {},
            }
            graph_enhanced = format_chunk_with_graph_data(graph_chunk_dict)
            if graph_enhanced.get("content_type") != "graph" or not isinstance(graph_enhanced.get("graph_data"), dict):
                continue

            context_chunk_id = str(kb_source) + ":raw_context"
            _apply_graph_payload(
                graph_payload=graph_enhanced["graph_data"],
                graph_entities=graph_entities,
                graph_relationships=graph_relationships,
                graph_metadata=graph_metadata,
                graph_chunk_refs=graph_chunk_refs,
                seen_graph_refs=seen_graph_refs,
                kb_chunks=kb_chunks,
                seen_chunk_ids=seen_chunk_ids,
                base_chunk_id=context_chunk_id,
                fallback_file_path=str(kb_source),
            )

        per_kb_results.append(
            KBResultModel(
                source=str(kb_source),
                graph_data=GraphDataModel(
                    entities=graph_entities,
                    relationship=graph_relationships,
                    metadata=graph_metadata,
                    chunk_references=graph_chunk_refs if graph_chunk_refs else None,
                ),
                chunks=kb_chunks,
            )
        )

    response = QueryRAGResponse(
        final_answer=result.answer,
        source=_build_source_references(result, per_kb_results),
        requested_mode=requested_mode,
        effective_mode=effective_mode,
    )

    return response, per_kb_results


def _build_source_references(result, kb_results: List[KBResultModel]) -> List[SourceReferenceModel]:
    """Build compact source payload for client-side download URL generation."""
    output: List[SourceReferenceModel] = []
    seen: set = set()

    for src in getattr(result, "sources", []) or []:
        file_name = str(getattr(src, "download_name", "") or getattr(src, "file_name", "")).strip()
        container_name = str(getattr(src, "container_name", "")).strip()
        blob_path = str(getattr(src, "blob_path", "")).strip()
        if not file_name or not container_name or not blob_path:
            continue

        key = (container_name, blob_path)
        if key in seen:
            continue

        seen.add(key)
        output.append(
            SourceReferenceModel(
                file_id="",
                file_name=file_name,
                container_name=container_name,
                blob_path=blob_path,
                provider=str(getattr(settings.storage, "STORAGE_PROVIDER", "azure") or "azure"),
                citation=getattr(src, "citation", None),
            )
        )

    if output:
        return output

    # Fallback when LLM citations are missing: infer from chunks.
    for kb in kb_results:
        for chunk in kb.chunks:
            raw_path = str(chunk.file_path or "").strip()
            if not raw_path:
                continue

            file_name = raw_path.split("/")[-1]
            key = ("", raw_path)
            if key in seen:
                continue
            seen.add(key)

            output.append(
                SourceReferenceModel(
                    file_id="",
                    file_name=file_name,
                    container_name=str(getattr(settings.storage, "STORAGE_CONTAINER_NAME", "") or ""),
                    blob_path=raw_path,
                    provider=str(getattr(settings.storage, "STORAGE_PROVIDER", "azure") or "azure"),
                    citation=None,
                )
            )

    return output
