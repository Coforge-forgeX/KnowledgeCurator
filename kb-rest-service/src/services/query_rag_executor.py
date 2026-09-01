"""
Shared query_rag execution path.

This module owns everything the `query_rag` endpoint used to do inline after
authentication and workspace resolution: the Redis result cache, the RAG call,
the response/evidence assembly and the source-token caching.

It exists so `message_gpt` (chat) and `POST /query-rag` run the SAME code.
Previously the chat path called `RAGQueryService` directly and therefore
bypassed the Redis cache entirely - every chat turn paid full RAG cost, and any
fix made to query_rag did not reach chat. Both callers now go through
`execute_query_rag`, so behaviour and caching stay in lockstep.

Callers supply an already-resolved workspace context (domain, kb_name, role_id,
additional KBs). Neither authentication nor membership validation happens here:
`query_rag` validates via `workspace_service`, chat validates via
`ChatAccessValidator`, and duplicating either check would cost an extra DB
round-trip per request.
"""
import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Tuple

from src.core.config import settings
from src.core.logging import get_logger
from src.core.redis import get_query_cache, redis_manager, set_query_cache
from src.helpers.file_token import create_signed_file_id
from src.helpers.graph_parser import format_chunk_with_graph_data
from src.helpers.source_extractor import extract_sources
from src.models.query_rag_models import (
    GraphDataModel,
    KBChunkModel,
    KBResultModel,
    QueryRAGResponse,
    SourceReferenceModel,
)
from src.services.rag_query_service import get_rag_query_service

logger = get_logger(__name__)

QUERY_EVIDENCE_TTL_SECONDS = 30 * 60
QUERY_RESULT_CACHE_VERSION = "source-map-v4"


def _cache_enabled() -> bool:
    return bool(redis_manager.is_available and getattr(settings.cache, "REDIS_ENABLED", True))


def _normalize_query_text(query: str) -> str:
    """Normalize user query text so semantically identical spacing maps to one key."""
    return " ".join((query or "").strip().split())


def _make_query_evidence_key(workspace_id: int, query: str, mode: str) -> str:
    """Create deterministic redis key for cached query evidence payload.

    Deliberately keyed on the RAW query only (no conversation history):
    `fetch_graph` recomputes this exact key from `(workspace_id, query, mode)`
    to look the evidence back up, so folding history in here would orphan it.
    """
    normalized_query = _normalize_query_text(query)
    normalized_mode = (mode or "").strip().lower()
    key_material = f"{workspace_id}|{normalized_mode}|{normalized_query}"
    query_hash = hashlib.sha256(key_material.encode("utf-8")).hexdigest()[:24]
    return f"query_evidence:{workspace_id}:{query_hash}"


def _make_source_mapping_key(file_id: str) -> str:
    """Create redis key for file token to storage mapping."""
    return f"query_file:{file_id}"


def _result_cache_query(query: str, history: Optional[List[dict]]) -> str:
    """
    Build the query component of the RESULT cache key.

    The answer is conditioned on the conversation history, so two turns asking
    the same words after different exchanges are different questions and must
    not share a cached answer. A short digest of the history is folded into the
    key; with no history the key is the plain query, which keeps existing
    single-shot query_rag cache entries valid.
    """
    normalized_query = f"{QUERY_RESULT_CACHE_VERSION}|{_normalize_query_text(query)}"
    if not history:
        return normalized_query

    try:
        history_material = json.dumps(
            [
                {"role": str(m.get("role") or ""), "content": str(m.get("content") or "")}
                for m in history
                if isinstance(m, dict)
            ],
            sort_keys=True,
        )
    except (TypeError, ValueError):
        # Unhashable history: use a per-object marker so this request simply
        # misses the cache rather than colliding with another conversation.
        return f"{normalized_query}||h:unhashable:{id(history)}"

    history_hash = hashlib.sha256(history_material.encode("utf-8")).hexdigest()[:16]
    return f"{normalized_query}||h:{history_hash}"


async def execute_query_rag(
    *,
    query: str,
    workspace_id: int,
    role_id: int,
    domain: str,
    kb_name: str,
    mode: str = "hybrid",
    history: Optional[List[dict]] = None,
    additional_kbs: Optional[List[str]] = None,
    agent_id: Optional[int] = None,
    is_kg: Optional[bool] = None,
    container_name: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> Tuple[Dict[str, Any], bool]:
    """
    Run a RAG query through the shared cached path.

    Returns `(response_dict, cache_hit)` where `response_dict` is a serialized
    `QueryRAGResponse`: `final_answer`, `source` (each with `file_id`,
    `file_name`, `citation`), `requested_mode`, `effective_mode`.
    """
    start_time = time.time()
    cache_query = _result_cache_query(query, history)

    if _cache_enabled():
        cached_result = get_query_cache(workspace_id, cache_query, mode)
        if cached_result:
            logger.info(
                "Query RAG served from cache",
                correlation_id=correlation_id,
                workspace_id=workspace_id,
                cache_hit=True,
                has_history=bool(history),
                response_time_ms=round((time.time() - start_time) * 1000, 2),
            )
            return cached_result, True

    rag_service = get_rag_query_service()
    result = await rag_service.query(
        query=query,
        workspace_id=workspace_id,
        role_id=role_id,
        domain=domain,
        kb_name=kb_name,
        mode=mode,
        history=history,
        knowledge_bases=additional_kbs,
        agent_id=agent_id,
        is_kg=is_kg,
        container_name=container_name,
    )

    response_data, kb_results = build_query_rag_response(
        result,
        workspace_id,
        domain,
        kb_name,
        default_container=container_name,
    )

    if _cache_enabled():
        _cache_evidence_and_sources(
            response_data=response_data,
            kb_results=kb_results,
            workspace_id=workspace_id,
            query=query,
            mode=mode,
            correlation_id=correlation_id,
        )
    else:
        # Without Redis the signed token is still the client's handle on the
        # file, so mint it regardless of whether the mapping could be cached.
        for source_item in response_data.source:
            source_item.file_id = create_signed_file_id(
                workspace_id=workspace_id,
                container_name=source_item.container_name,
                blob_path=source_item.blob_path,
                provider=source_item.provider,
                file_name=source_item.file_name,
            )

    response_dict = response_data.dict()

    if _cache_enabled():
        cache_ttl = getattr(settings.cache, "QUERY_CACHE_TTL", 3600)
        cache_ttl = min(cache_ttl, QUERY_EVIDENCE_TTL_SECONDS)
        set_query_cache(workspace_id, cache_query, mode, response_dict, ttl=cache_ttl)

    logger.info(
        "Query RAG executed",
        correlation_id=correlation_id,
        workspace_id=workspace_id,
        answer_length=len(response_data.final_answer),
        source_count=len(response_data.source),
        kb_count=len(kb_results),
        chunk_count=sum(len(kb.chunks) for kb in kb_results),
        cache_hit=False,
        has_history=bool(history),
        total_time_ms=round((time.time() - start_time) * 1000, 2),
    )

    return response_dict, False


def _cache_evidence_and_sources(
    *,
    response_data: QueryRAGResponse,
    kb_results: List[KBResultModel],
    workspace_id: int,
    query: str,
    mode: str,
    correlation_id: Optional[str],
) -> None:
    """Cache the heavyweight evidence payload and per-source file_id mappings.

    Mutates `response_data.source` in place to attach the signed `file_id`.
    Cache failures are logged and swallowed - a signed file_id stays usable on
    its own, so a Redis outage must not fail the query.
    """
    evidence_cache_key = _make_query_evidence_key(workspace_id, query, mode)
    evidence_payload = {
        "workspace_id": workspace_id,
        "query": query,
        "requested_mode": mode,
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


def build_query_rag_response(
    result,
    workspace_id: int,
    domain: str,
    kb_name: str,
    default_container: Optional[str] = None,
) -> tuple[QueryRAGResponse, List[KBResultModel]]:

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

        source=_build_source_references(result, per_kb_results, default_container),

        requested_mode=requested_mode,

        effective_mode=effective_mode,

    )



    return response, per_kb_results





def _build_source_references(
    result,
    kb_results: List[KBResultModel],
    workspace_container: Optional[str] = None,
) -> List[SourceReferenceModel]:

    """

    Build compact source payload for client-side download URL generation.



    Sources come from the retrieved chunks (see `helpers.source_extractor`),

    not from parsing a `## References` section out of the answer — the answer's

    formatting must never decide whether the caller gets sources. LLM

    citations, when present, only add citation numbers and a storage-verified

    blob path on top.

    """

    provider = str(getattr(settings.storage, "STORAGE_PROVIDER", "azure") or "azure")

    default_container = str(
        workspace_container
        or getattr(settings.storage, "STORAGE_CONTAINER_NAME", "")
        or ""
    )



    # kb_results carry the graph-derived chunk paths assembled in build_query_rag_response,

    # which are not all present on result.retrieved_chunks.

    extra_paths = [chunk.file_path for kb in kb_results for chunk in kb.chunks if chunk.file_path]



    return [

        SourceReferenceModel(

            file_id="",

            file_name=ref.file_name,

            container_name=ref.container_name or default_container,

            blob_path=ref.blob_path,

            provider=provider,

            citation=ref.citation or None,

        )

        for ref in extract_sources(

            result,

            extra_paths=extra_paths,

            default_container=default_container,

        )

        if ref.blob_path

    ]

