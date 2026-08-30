"""
Query Strategies - Strategy Pattern

Different strategies for executing RAG queries.
Follows Open/Closed Principle - extensible without modification.
"""
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.core.config import settings
from src.core.lightrag_pool import get_pooled_lightrag_service, invalidate_pooled_service
from src.core.lightrag_service import LightRAGService
from src.core.logging import get_logger
from src.core.prompt_builder import get_prompt_builder
from src.core.reference_parser import clean_response, parse_references
from src.helpers.workspace_resolver import WorkspaceResolver
from src.models.rag_models import (
    EnrichedSource,
    KnowledgeBase,
    MultiKBResult,
    QueryContext,
    QueryMode,
    RAGQueryResult,
    RetrievedChunk,
)

logger = get_logger(__name__)


class QueryStrategy(ABC):
    """
    Abstract base class for query strategies.

    Design: Strategy Pattern
    - Common interface (execute method)
    - Different implementations for different scenarios
    """

    def __init__(self):
        # No LightRAG instance is held on the strategy: services are resolved
        # per (KB, workspace, agent) from `lightrag_pool`. The previous global
        # singleton had to be re-pointed at a different working_dir/workspace
        # on every call, which invalidated its runtime signature and forced a
        # full re-initialization each time.
        self.prompt_builder = get_prompt_builder("rag")

    @abstractmethod
    async def execute(
        self,
        context: QueryContext,
        knowledge_bases: List[KnowledgeBase]
    ) -> RAGQueryResult:
        """
        Execute query strategy.

        Args:
            context: Query context
            knowledge_bases: List of KBs to query

        Returns:
            RAGQueryResult with answer, sources, and chunks
        """
        pass

    async def _resolve_service(
        self,
        kb: KnowledgeBase,
        context: QueryContext
    ) -> LightRAGService:
        """
        Get the pooled, initialized LightRAG service for this KB.

        Strategies hold no LightRAG instance of their own: the previous global
        singleton had to be re-pointed at a different working_dir/workspace on
        every call, which changed its runtime signature and forced a full
        re-initialization each time.
        """
        return await get_pooled_lightrag_service(
            working_dir=self._get_working_dir(kb),
            workspace_label=WorkspaceResolver.build_workspace_name(kb.domain, kb.name),
            workspace_id=context.workspace_id,
            agent_id=context.agent_id,
        )

    def _invalidate_service(self, kb: KnowledgeBase, context: QueryContext) -> None:
        """Retire a pooled service after a query failure so it gets rebuilt.

        A failure may mean the instance is holding a database connection the
        server has since closed (common when a serverless worker is frozen
        between invocations). Without this, that broken instance would be
        handed to every subsequent request for the same KB.
        """
        invalidate_pooled_service(
            working_dir=self._get_working_dir(kb),
            workspace_label=WorkspaceResolver.build_workspace_name(kb.domain, kb.name),
            workspace_id=context.workspace_id,
            agent_id=context.agent_id,
        )

    async def _query_lightrag(
        self,
        kb: KnowledgeBase,
        context: QueryContext,
        prompt: str
    ) -> Dict[str, Any]:
        """
        Execute LightRAG query.

        Args:
            kb: Knowledge base to query
            context: Query context
            prompt: Formatted prompt

        Returns:
            Normalized LightRAG response payload

        Raises:
            Exception: If query fails
        """
        try:
            service = await self._resolve_service(kb, context)

            requested_mode = context.mode.value
            if requested_mode == QueryMode.HYBRID.value:
                requested_mode = QueryMode.MIX.value

            # Execute one unified query for answer + structured retrieval data.
            unified = await service.query_llm(
                query=prompt,
                mode=requested_mode,
                conversation_history=context.history,
                top_k=5,
                chunk_top_k=5,
                stream=False
            )
            effective_mode = requested_mode

            llm_response = unified.get("llm_response", {}) if isinstance(unified, dict) else {}
            data_payload = unified.get("data", {}) if isinstance(unified, dict) else {}

            answer_text = ""
            if isinstance(llm_response, dict):
                answer_text = str(llm_response.get("content") or "")

            chunks_payload = data_payload.get("chunks", []) if isinstance(data_payload, dict) else []
            entities = data_payload.get("entities", []) if isinstance(data_payload, dict) else []
            relationships = data_payload.get("relationships", []) if isinstance(data_payload, dict) else []

            normalized_response = {
                "answer": answer_text,
                "retrieved_chunks": chunks_payload if isinstance(chunks_payload, list) else [],
                "sources": [],
                "mode": requested_mode,
                "_raw_context": [
                    {
                        "entities": entities if isinstance(entities, list) else [],
                        "relationships": relationships if isinstance(relationships, list) else [],
                        "metadata": unified.get("metadata", {}) if isinstance(unified, dict) else {},
                    }
                ],
            }

            normalized_response.setdefault("mode", effective_mode)
            normalized_response["_requested_mode"] = requested_mode
            normalized_response["_effective_mode"] = effective_mode

            logger.info(
                f"LightRAG query completed for KB {kb.full_name}",
                response_length=len(answer_text)
            )

            return normalized_response

        except Exception as e:
            # The pooled instance may be holding a dead connection; retire it so
            # the retry below (and any later request) gets a fresh one.
            self._invalidate_service(kb, context)

            if context.mode != QueryMode.HYBRID:
                logger.warning(
                    "Query failed, retrying in hybrid mode",
                    kb=kb.full_name,
                    error=str(e),
                    requested_mode=context.mode.value,
                )
                fallback_service = await self._resolve_service(kb, context)
                fallback_response = await fallback_service.query(
                    query=prompt,
                    mode=QueryMode.HYBRID.value,
                    conversation_history=context.history,
                    top_k=5,
                    stream=False,
                )
                if isinstance(fallback_response, dict):
                    fallback_response.setdefault("mode", QueryMode.HYBRID.value)
                    fallback_response["_requested_mode"] = context.mode.value
                    fallback_response["_effective_mode"] = QueryMode.HYBRID.value
                    return fallback_response
                return {
                    "answer": str(fallback_response) if fallback_response is not None else "",
                    "retrieved_chunks": [],
                    "sources": [],
                    "mode": QueryMode.HYBRID.value,
                    "_requested_mode": context.mode.value,
                    "_effective_mode": QueryMode.HYBRID.value,
                }

            logger.error(
                f"LightRAG query failed for KB {kb.full_name}",
                error=e
            )
            raise

    async def _retrieve_evidence(
        self,
        kb: KnowledgeBase,
        context: QueryContext,
        prompt: str
    ) -> Dict[str, Any]:
        """
        Retrieve structured evidence (chunks + graph) in a single LightRAG call.

        Args:
            kb: Knowledge base
            context: Query context
            prompt: Query prompt

        Returns:
            Dict containing serialized chunks and raw graph data
        """
        try:
            # In unified mode, evidence is already included in _query_lightrag
            # response, which resolves its own pooled service for this KB.
            response_payload = await self._query_lightrag(kb, context, prompt)
            chunks = self._parse_chunks(response_payload, kb)
            graph_payload = response_payload.get("_raw_context", []) if isinstance(response_payload, dict) else []

            logger.info(
                "Retrieved structured evidence",
                kb=kb.full_name,
                chunk_count=len(chunks),
                graph_entity_count=len(graph_payload[0].get("entities", [])) if graph_payload else 0,
                graph_relationship_count=len(graph_payload[0].get("relationships", [])) if graph_payload else 0,
            )

            return {
                "chunks": self._serialize_chunks(chunks),
                "graph": graph_payload if isinstance(graph_payload, list) else [],
            }

        except Exception as e:
            if context.mode != QueryMode.HYBRID:
                logger.warning(
                    "Structured evidence retrieval failed, retrying in hybrid mode",
                    kb=kb.full_name,
                    error=str(e),
                    requested_mode=context.mode.value,
                )
                try:
                    fallback_service = await self._resolve_service(kb, context)
                    data_response = await fallback_service.query_llm(
                        query=prompt,
                        mode=QueryMode.HYBRID.value,
                        conversation_history=context.history,
                        chunk_top_k=5,
                        top_k=5,
                        stream=False,
                    )
                    data = data_response.get("data", {}) if isinstance(data_response, dict) else {}
                    chunks_payload = data.get("chunks", []) if isinstance(data.get("chunks"), list) else []
                    entities = data.get("entities", []) if isinstance(data.get("entities"), list) else []
                    relationships = data.get("relationships", []) if isinstance(data.get("relationships"), list) else []

                    chunks = self._parse_chunks({"retrieved_chunks": chunks_payload}, kb)
                    logger.info(
                        f"Retrieved {len(chunks)} chunks from KB {kb.full_name} via hybrid fallback"
                    )
                    return {
                        "chunks": self._serialize_chunks(chunks),
                        "graph": [
                            {
                                "entities": entities,
                                "relationships": relationships,
                                "metadata": data_response.get("metadata", {}) if isinstance(data_response, dict) else {},
                            }
                        ],
                    }
                except Exception as fallback_error:
                    logger.warning(
                        f"Hybrid fallback evidence retrieval failed for KB {kb.full_name}",
                        error=fallback_error,
                    )

            logger.warning(
                f"Evidence retrieval failed for KB {kb.full_name}",
                error=e
            )
            return {"chunks": [], "graph": []}

    def _parse_chunks(
        self,
        context_response: Dict[str, Any],
        kb: KnowledgeBase
    ) -> List[RetrievedChunk]:
        """Parse chunks from LightRAG context response into RetrievedChunk models."""
        raw_chunks: List[Any] = []

        if isinstance(context_response, dict):
            candidate = context_response.get("retrieved_chunks", [])
            if isinstance(candidate, list):
                raw_chunks = candidate
            elif candidate is not None:
                raw_chunks = [candidate]
        elif isinstance(context_response, list):
            raw_chunks = context_response
        elif context_response is not None:
            raw_chunks = [context_response]

        parsed_chunks: List[RetrievedChunk] = []
        for idx, item in enumerate(raw_chunks):
            if isinstance(item, RetrievedChunk):
                parsed_chunks.append(item)
                continue

            if isinstance(item, dict):
                content = str(
                    item.get("content")
                    or item.get("text")
                    or item.get("chunk")
                    or item.get("value")
                    or ""
                ).strip()
                score_raw = item.get("score", item.get("similarity", item.get("cosine", 0.0)))
                try:
                    score = float(score_raw)
                except (TypeError, ValueError):
                    score = 0.0

                source = str(
                    item.get("source")
                    or item.get("file_path")
                    or item.get("file_name")
                    or item.get("doc_id")
                    or kb.full_name
                )
                chunk_id = str(item.get("chunk_id") or item.get("id") or f"{kb.full_name}:{idx}")

                metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
                inferred_file_path = str(
                    item.get("file_path")
                    or metadata.get("file_path")
                    or item.get("source")
                    or item.get("file_name")
                    or ""
                )
                metadata = {
                    **metadata,
                    "kb": kb.full_name,
                    "file_path": inferred_file_path,
                    "raw_chunk": item,
                }

                if not content and item:
                    content = str(item)
            else:
                content = str(item).strip()
                score = 0.0
                source = kb.full_name
                chunk_id = f"{kb.full_name}:{idx}"
                metadata = {"kb": kb.full_name}

            if not content:
                continue

            parsed_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    content=content,
                    score=score,
                    source=source,
                    metadata=metadata,
                )
            )

        return parsed_chunks

    def _serialize_chunks(self, chunks: List[RetrievedChunk]) -> List[Dict[str, Any]]:
        """Serialize RetrievedChunk objects for response metadata caching."""
        return [
            {
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "score": chunk.score,
                "source": chunk.source,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ]

    def _deserialize_chunks(self, serialized_chunks: Any) -> List[RetrievedChunk]:
        """Deserialize cached chunk dictionaries into RetrievedChunk models."""
        if not isinstance(serialized_chunks, list):
            return []

        chunks: List[RetrievedChunk] = []
        for idx, item in enumerate(serialized_chunks):
            if isinstance(item, RetrievedChunk):
                chunks.append(item)
                continue

            if not isinstance(item, dict):
                continue

            chunk_id = str(item.get("chunk_id") or item.get("id") or f"chunk:{idx}")
            content = str(item.get("content") or "").strip()
            if not content:
                continue

            score_raw = item.get("score", 0.0)
            try:
                score = float(score_raw)
            except (TypeError, ValueError):
                score = 0.0

            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    content=content,
                    score=score,
                    source=str(item.get("source") or "unknown"),
                    metadata=item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
                )
            )

        return chunks

    def _extract_answer_text(self, response: Any) -> str:
        """Extract answer text from LightRAG response payload safely."""
        if isinstance(response, dict):
            answer = response.get("answer", "")
            if answer is None:
                return ""
            return answer if isinstance(answer, str) else str(answer)
        if response is None:
            return ""
        return response if isinstance(response, str) else str(response)

    def _get_working_dir(self, kb: KnowledgeBase) -> str:
        """
        Get working directory for knowledge base.

        Uses the shared workspace_helpers to build the working directory
        consistently with other services.

        Args:
            kb: Knowledge base with domain, name, and workspace_id

        Returns:
            Full working directory path for LightRAG
        """
        base_dir = settings.lightrag.LIGHTRAG_WORKING_DIR

        # Keep query-time working_dir aligned with indexer-service initialization.
        # LightRAG manages workspace separation via the `workspace` label.
        working_dir = base_dir

        logger.debug(
            "Built working directory for KB",
            domain=kb.domain,
            kb_name=kb.name,
            workspace_id=kb.workspace_id,
            working_dir=working_dir
        )

        return working_dir


class SingleKBStrategy(QueryStrategy):
    """
    Strategy for querying a single knowledge base.

    Design:
    - Simple, focused responsibility
    - Clear sequential flow
    - Comprehensive error handling
    """

    async def execute(
        self,
        context: QueryContext,
        knowledge_bases: List[KnowledgeBase]
    ) -> RAGQueryResult:
        """Execute query against single knowledge base"""

        if not knowledge_bases or len(knowledge_bases) == 0:
            raise ValueError("No knowledge bases provided")

        kb = knowledge_bases[0]  # Use first KB
        logger.info(f"Executing single KB query: {kb.full_name}")

        try:
            # Build prompt
            prompt = self.prompt_builder.build(context.query, context.history)

            # Execute one unified query call (answer + evidence)
            response_payload = await self._query_lightrag(kb, context, prompt)
            response_text = self._extract_answer_text(response_payload)

            # Parse references
            references = parse_references(response_text)

            # Clean response (remove references section)
            clean_answer = clean_response(response_text)

            chunks = self._parse_chunks(response_payload, kb)
            if not references and not chunks:
                clean_answer = "No relevant information found in the knowledge base."
            evidence = {
                "chunks": self._serialize_chunks(chunks),
                "graph": response_payload.get("_raw_context", []) if isinstance(response_payload, dict) else [],
            }

            # Note: Sources enrichment (URL generation) happens in service layer
            # This keeps strategy focused on query execution

            result = RAGQueryResult(
                answer=clean_answer,
                sources=[],  # Enriched by service layer
                retrieved_chunks=chunks,
                metadata={
                    "kb": kb.full_name,
                    "mode": context.mode.value,
                    "available_modes": [m.value for m in QueryMode],
                    "requested_mode": context.mode.value,
                    "effective_mode": response_payload.get("_effective_mode", context.mode.value),
                    "reference_count": len(references),
                }
            )

            # Store references in metadata for later enrichment
            result.metadata["raw_references"] = references
            result.metadata["chunks_by_kb"] = {kb.full_name: evidence.get("chunks", [])}
            result.metadata["graph_context_by_kb"] = {kb.full_name: evidence.get("graph", [])}

            logger.info(
                f"Single KB query completed",
                answer_length=len(clean_answer),
                reference_count=len(references),
                chunk_count=len(chunks)
            )

            return result

        except Exception as e:
            logger.error(f"Single KB query failed", error=e, kb=kb.full_name)
            raise


class MultiKBStrategy(QueryStrategy):
    """
    Strategy for querying multiple knowledge bases in parallel.

    Design:
    - Parallel execution for performance
    - Graceful degradation (some KBs can fail)
    - Result aggregation via LLM
    """

    def __init__(self):
        super().__init__()
        self.summary_prompt_builder = get_prompt_builder("multi_kb")

    async def execute(
        self,
        context: QueryContext,
        knowledge_bases: List[KnowledgeBase]
    ) -> RAGQueryResult:
        """Execute query across multiple knowledge bases"""

        if not knowledge_bases or len(knowledge_bases) == 0:
            raise ValueError("No knowledge bases provided")

        logger.info(
            f"Executing multi-KB query across {len(knowledge_bases)} KBs"
        )

        try:
            # Build prompt
            prompt = self.prompt_builder.build(context.query, context.history)

            # Query all KBs in parallel
            kb_results = await self._query_all_kbs(knowledge_bases, context, prompt)

            if not any(
                parse_references(self._extract_answer_text(result))
                or self._deserialize_chunks(result.get("_retrieved_chunks", []))
                for result in kb_results.values()
                if isinstance(result, dict) and "error" not in result
            ):
                return RAGQueryResult(
                    answer="No relevant information found in the knowledge base.",
                    metadata={
                        "kbs": [kb.full_name for kb in knowledge_bases],
                        "mode": context.mode.value,
                        "requested_mode": context.mode.value,
                        "effective_mode": context.mode.value,
                        "kb_count": len(knowledge_bases),
                        "successful_kbs": len(kb_results),
                        "refrences_count": 0,
                        "raw_refrences": [],
                        "kb_results": kb_results,
                        "chunks_by_kb": {},
                        "graph_context_by_kb": {}
                    },
                )

            # Aggregate results using LLM
            aggregated_answer = await self._aggregate_results(
                context,
                kb_results,
                knowledge_bases
            )

            # Clean aggregated answer
            clean_answer = clean_response(aggregated_answer)

            # Collect all references from all KBs
            all_references = []
            for kb_name, result in kb_results.items():
                if not isinstance(result, dict) or "error" not in result:
                    refs = parse_references(self._extract_answer_text(result))
                    all_references.extend(refs)

            # Collect chunks from all successful KBs so eval cache has full context.
            chunks: List[RetrievedChunk] = []
            chunks_by_kb: Dict[str, List[Dict[str, Any]]] = {}
            graph_context_by_kb: Dict[str, Any] = {}
            for kb_name, result_payload in kb_results.items():
                if isinstance(result_payload, dict) and "error" not in result_payload:
                    serialized_chunks = result_payload.get("_retrieved_chunks", [])
                    kb_chunks = self._deserialize_chunks(serialized_chunks)
                    chunks.extend(kb_chunks)
                    chunks_by_kb[kb_name] = serialized_chunks
                    graph_context_by_kb[kb_name] = result_payload.get("_raw_context", [])

            result = RAGQueryResult(
                answer=clean_answer,
                sources=[],  # Enriched by service layer
                retrieved_chunks=chunks,
                metadata={
                    "kbs": [kb.full_name for kb in knowledge_bases],
                    "mode": context.mode.value,
                    "available_modes": [m.value for m in QueryMode],
                    "requested_mode": context.mode.value,
                    "effective_mode": self._resolve_multi_kb_effective_mode(context, kb_results),
                    "kb_count": len(knowledge_bases),
                    "successful_kbs": len([r for r in kb_results.values() if not isinstance(r, dict) or "error" not in r]),
                    "reference_count": len(all_references),
                }
            )

            # Store references and KB results for later enrichment
            result.metadata["raw_references"] = all_references
            result.metadata["kb_results"] = kb_results
            result.metadata["chunks_by_kb"] = chunks_by_kb
            result.metadata["graph_context_by_kb"] = graph_context_by_kb
            result.metadata["effective_mode_by_kb"] = {
                kb_name: (
                    result_payload.get("_effective_mode", context.mode.value)
                    if isinstance(result_payload, dict) and "error" not in result_payload
                    else "error"
                )
                for kb_name, result_payload in kb_results.items()
            }

            logger.info(
                f"Multi-KB query completed",
                kb_count=len(knowledge_bases),
                answer_length=len(clean_answer),
                reference_count=len(all_references)
            )

            return result

        except Exception as e:
            logger.error(f"Multi-KB query failed", error=e)
            raise

    async def _query_all_kbs(
        self,
        knowledge_bases: List[KnowledgeBase],
        context: QueryContext,
        prompt: str
    ) -> Dict[str, Any]:
        """Query all KBs using isolated per-KB services.

        Service resolution stays sequential to avoid global env races
        (Neo4j/Postgres env wiring happens inside `initialize()`), while query
        execution remains parallel for performance. Services come from
        `lightrag_pool`, so this loop is a no-op after the first request for a
        given (KB, workspace, agent) — the ~8s-per-KB initialization it used to
        pay on every request is now paid once per worker.
        """

        kb_results: Dict[str, Any] = {}
        kb_services: Dict[str, LightRAGService] = {}

        for kb in knowledge_bases:
            try:
                kb_services[kb.full_name] = await self._resolve_service(kb, context)
                logger.debug("Prepared KB LightRAG service", kb=kb.full_name)
            except Exception as e:
                logger.warning("KB initialization failed", kb=kb.full_name, error=e)
                kb_results[kb.full_name] = {"error": str(e)}

        async def query_single_kb(kb: KnowledgeBase):
            """Query single KB with error handling"""
            try:
                service = kb_services[kb.full_name]
                requested_mode = context.mode.value
                if requested_mode == QueryMode.HYBRID.value:
                    requested_mode = QueryMode.MIX.value
                effective_mode = requested_mode

                # Retrieve answer + evidence in one call.
                unified_response = await service.query_llm(
                    query=prompt,
                    mode=requested_mode,
                    conversation_history=context.history,
                    top_k=5,
                    chunk_top_k=5,
                    stream=False,
                )

                llm_response = unified_response.get("llm_response", {}) if isinstance(unified_response, dict) else {}
                answer_text = str(llm_response.get("content") or "") if isinstance(llm_response, dict) else ""

                data = unified_response.get("data", {}) if isinstance(unified_response, dict) else {}
                chunks_payload = data.get("chunks", []) if isinstance(data.get("chunks"), list) else []
                entities = data.get("entities", []) if isinstance(data.get("entities"), list) else []
                relationships = data.get("relationships", []) if isinstance(data.get("relationships"), list) else []

                if not chunks_payload:
                    hybrid_data_response = await service.query_llm(
                        query=prompt,
                        mode=QueryMode.HYBRID.value,
                        conversation_history=context.history,
                        top_k=5,
                        chunk_top_k=5,
                        stream=False,
                    )

                    hybrid_llm_response = hybrid_data_response.get("llm_response", {}) if isinstance(hybrid_data_response, dict) else {}
                    if not answer_text and isinstance(hybrid_llm_response, dict):
                        answer_text = str(hybrid_llm_response.get("content") or "")

                    hybrid_data = hybrid_data_response.get("data", {}) if isinstance(hybrid_data_response, dict) else {}
                    chunks_payload = hybrid_data.get("chunks", []) if isinstance(hybrid_data.get("chunks"), list) else []
                    entities = hybrid_data.get("entities", []) if isinstance(hybrid_data.get("entities"), list) else []
                    relationships = hybrid_data.get("relationships", []) if isinstance(hybrid_data.get("relationships"), list) else []

                response = {
                    "answer": answer_text,
                    "retrieved_chunks": chunks_payload if isinstance(chunks_payload, list) else [],
                    "sources": [],
                    "mode": effective_mode,
                    "_requested_mode": requested_mode,
                    "_effective_mode": effective_mode,
                }

                parsed_chunks = self._parse_chunks({"retrieved_chunks": chunks_payload}, kb)
                response["_retrieved_chunks"] = self._serialize_chunks(parsed_chunks)
                response["_raw_context"] = [
                    {
                        "entities": entities,
                        "relationships": relationships,
                        "metadata": unified_response.get("metadata", {}) if isinstance(unified_response, dict) else {},
                    }
                ]
                return (kb.full_name, response, None)
            except Exception as e:
                # A failure here may mean this pooled instance is holding a dead
                # connection; retire it so the retry and later requests rebuild.
                self._invalidate_service(kb, context)

                if context.mode != QueryMode.HYBRID:
                    logger.warning(
                        "Query failed, retrying in hybrid mode",
                        kb=kb.full_name,
                        error=str(e),
                        requested_mode=context.mode.value,
                    )
                    try:
                        service = await self._resolve_service(kb, context)
                        fallback_unified = await service.query_llm(
                            query=prompt,
                            mode=QueryMode.HYBRID.value,
                            conversation_history=context.history,
                            top_k=5,
                            chunk_top_k=5,
                            stream=False,
                        )

                        llm_response = fallback_unified.get("llm_response", {}) if isinstance(fallback_unified, dict) else {}
                        answer_text = str(llm_response.get("content") or "") if isinstance(llm_response, dict) else ""
                        data = fallback_unified.get("data", {}) if isinstance(fallback_unified, dict) else {}
                        chunks_payload = data.get("chunks", []) if isinstance(data.get("chunks"), list) else []
                        entities = data.get("entities", []) if isinstance(data.get("entities"), list) else []
                        relationships = data.get("relationships", []) if isinstance(data.get("relationships"), list) else []

                        response = {
                            "answer": answer_text,
                            "retrieved_chunks": chunks_payload if isinstance(chunks_payload, list) else [],
                            "sources": [],
                            "mode": QueryMode.HYBRID.value,
                            "_requested_mode": context.mode.value,
                            "_effective_mode": QueryMode.HYBRID.value,
                        }
                        parsed_chunks = self._parse_chunks({"retrieved_chunks": chunks_payload}, kb)
                        response["_retrieved_chunks"] = self._serialize_chunks(parsed_chunks)
                        response["_raw_context"] = [
                            {
                                "entities": entities,
                                "relationships": relationships,
                                "metadata": fallback_unified.get("metadata", {}) if isinstance(fallback_unified, dict) else {},
                            }
                        ]
                        return (kb.full_name, response, None)
                    except Exception as fallback_error:
                        logger.warning(
                            "Hybrid fallback query failed",
                            kb=kb.full_name,
                            error=fallback_error,
                        )
                        return (kb.full_name, None, str(fallback_error))

                logger.warning(f"KB query failed: {kb.full_name}", error=e)
                return (kb.full_name, None, str(e))

        # Execute query tasks in parallel for successfully initialized KBs only.
        query_candidates = [
            kb for kb in knowledge_bases
            if kb.full_name in kb_services
        ]
        tasks = [query_single_kb(kb) for kb in query_candidates]
        task_results = await asyncio.gather(*tasks)

        # Build results dict
        for kb_name, response, error in task_results:
            if error:
                kb_results[kb_name] = {"error": error}
            else:
                kb_results[kb_name] = response

        # Services are pooled and shared — deliberately NOT closed here.
        # Closing would reset `_initialized`/`_rag` and force the next request
        # to pay full initialization again.

        return kb_results

    async def _aggregate_results(
        self,
        context: QueryContext,
        kb_results: Dict[str, Any],
        knowledge_bases: List[KnowledgeBase]
    ) -> str:
        """
        Aggregate KB results intelligently.

        Uses LLM to synthesize multiple answers into one coherent response,
        or returns single result directly if only one KB succeeded.

        `knowledge_bases` is needed only to borrow an initialized service for
        the bypass-mode LLM call below — aggregation runs no retrieval, so any
        KB's service will do.
        """
        # Collect successful results
        successful_results = []
        for kb_name, result in kb_results.items():
            if not isinstance(result, dict) or "error" not in result:
                answer = self._extract_answer_text(result)
                if answer and answer.strip() and not answer.startswith("Sorry, I'm not able"):
                    successful_results.append({
                        "kb": kb_name,
                        "answer": answer.strip()
                    })

        # If only one successful result, return it directly
        if len(successful_results) == 1:
            logger.info(
                "Single KB result, returning directly",
                kb=successful_results[0]["kb"]
            )
            return successful_results[0]["answer"]

        # If no successful results, return error message
        if not successful_results:
            logger.warning("No successful results found in any knowledge base")
            return "Sorry, I'm not able to provide an answer to that question based on the available knowledge bases."

        # Multiple successful results - aggregate via LLM
        logger.info(
            "Aggregating results from multiple KBs via LLM",
            kb_count=len(successful_results)
        )

        # Build aggregation prompt
        aggregation_prompt = f"""You are aggregating answers from multiple knowledge bases to provide a single, coherent response.

**User Query:** {context.query}

**Answers from different knowledge bases:**

"""
        for idx, result in enumerate(successful_results, 1):
            aggregation_prompt += f"\n### Source {idx}: {result['kb']}\n{result['answer']}\n"

        aggregation_prompt += """

**Task:**
Please synthesize these answers into a single, coherent response following these rules:
1. Combine complementary information from different sources
2. Remove duplicate information - don't repeat the same facts multiple times
3. If sources provide different perspectives, include both but note the source
4. If sources contradict, briefly note the discrepancy
5. Keep the response clear, well-structured, and comprehensive
6. Maintain all important details from each source
7. Use markdown formatting for better readability

**Synthesized Answer:**"""

        try:
            # Call LLM for aggregation in bypass mode so it does not run retrieval again.
            aggregation_service = await self._resolve_service(knowledge_bases[0], context)
            aggregated = await aggregation_service.query(
                query=aggregation_prompt,
                mode="bypass",
                stream=False
            )

            answer_text = self._extract_answer_text(aggregated)

            logger.info(
                "Multi-KB aggregation completed via LLM",
                source_count=len(successful_results),
                original_total_length=sum(len(r["answer"]) for r in successful_results),
                aggregated_length=len(answer_text)
            )

            return answer_text

        except Exception as e:
            logger.warning(
                "LLM aggregation failed, falling back to structured concatenation",
                error=str(e)
            )

            # Fallback: concatenate with clear separation
            parts = []
            for result in successful_results:
                kb_name = result['kb'].split('/')[-1]  # Use just the last part of KB name
                parts.append(f"**From {kb_name}:**\n\n{result['answer']}")

            fallback_result = "\n\n---\n\n".join(parts)

            logger.info(
                "Multi-KB aggregation completed via fallback concatenation",
                source_count=len(successful_results)
            )

            return fallback_result

    def _resolve_multi_kb_effective_mode(
        self,
        context: QueryContext,
        kb_results: Dict[str, Any],
    ) -> str:
        """Summarize effective mode across KBs for response metadata."""
        effective_modes = []
        for result in kb_results.values():
            if isinstance(result, dict) and "error" not in result:
                effective_modes.append(result.get("_effective_mode", context.mode.value))

        if not effective_modes:
            return context.mode.value
        if all(mode == effective_modes[0] for mode in effective_modes):
            return effective_modes[0]
        return "mixed"

    async def _get_first_kb_chunks(
        self,
        knowledge_bases: List[KnowledgeBase],
        context: QueryContext,
        prompt: str
    ) -> List[RetrievedChunk]:
        """Get chunks from first KB for evaluation"""
        if knowledge_bases:
            return await self._retrieve_chunks(
                knowledge_bases[0],
                context,
                prompt
            )
        return []


class QueryStrategyFactory:
    """
    Factory for creating query strategies.

    Design: Factory Pattern
    - Encapsulates strategy selection logic
    - Easy to add new strategies
    """

    @staticmethod
    def create(knowledge_base_count: int) -> QueryStrategy:
        """
        Create appropriate strategy based on KB count.

        Args:
            knowledge_base_count: Number of knowledge bases to query

        Returns:
            QueryStrategy instance

        Examples:
            >>> factory = QueryStrategyFactory()
            >>> strategy = factory.create(1)  # Returns SingleKBStrategy
            >>> strategy = factory.create(3)  # Returns MultiKBStrategy
        """
        if knowledge_base_count == 1:
            logger.debug("Selected SingleKBStrategy")
            return SingleKBStrategy()
        else:
            logger.debug("Selected MultiKBStrategy")
            return MultiKBStrategy()
