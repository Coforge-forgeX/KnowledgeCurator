"""
Query Strategies - Strategy Pattern

Different strategies for executing RAG queries.
Follows Open/Closed Principle - extensible without modification.
"""
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.core.config import settings
from src.core.lightrag_service import LightRAGService, QueryParam, get_lightrag_service
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
        self.lightrag = get_lightrag_service()
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
            # Set working directory for this KB
            working_dir = self._get_working_dir(kb)
            self.lightrag.working_dir = working_dir
            # Keep LightRAG workspace in sync with the KB-specific label namespace.
            self.lightrag.workspace = WorkspaceResolver.build_workspace_name(
                kb.domain,
                kb.name,
            )
            self.lightrag.set_runtime_context(
                workspace_id=context.workspace_id,
                agent_id=context.agent_id,
            )

            requested_mode = context.mode.value

            # Execute query
            response = await self.lightrag.query(
                query=prompt,
                mode=requested_mode,
                conversation_history=context.history,
                top_k=2,
                stream=False
            )
            effective_mode = requested_mode

            # Normalize result to a dict payload for downstream callers.
            if isinstance(response, dict):
                normalized_response = response
            else:
                normalized_response = {
                    "answer": str(response) if response is not None else "",
                    "retrieved_chunks": [],
                    "sources": [],
                    "mode": context.mode.value,
                }

            normalized_response.setdefault("mode", effective_mode)
            normalized_response["_requested_mode"] = requested_mode
            normalized_response["_effective_mode"] = effective_mode

            answer_text = self._extract_answer_text(normalized_response)

            logger.info(
                f"LightRAG query completed for KB {kb.full_name}",
                response_length=len(answer_text)
            )

            return normalized_response

        except Exception as e:
            if context.mode != QueryMode.GLOBAL:
                logger.warning(
                    "Query failed, retrying in global mode",
                    kb=kb.full_name,
                    error=str(e),
                    requested_mode=context.mode.value,
                )
                fallback_response = await self.lightrag.query(
                    query=prompt,
                    mode=QueryMode.GLOBAL.value,
                    conversation_history=context.history,
                    top_k=2,
                    stream=False,
                )
                if isinstance(fallback_response, dict):
                    fallback_response.setdefault("mode", QueryMode.GLOBAL.value)
                    fallback_response["_requested_mode"] = context.mode.value
                    fallback_response["_effective_mode"] = QueryMode.GLOBAL.value
                    return fallback_response
                return {
                    "answer": str(fallback_response) if fallback_response is not None else "",
                    "retrieved_chunks": [],
                    "sources": [],
                    "mode": QueryMode.GLOBAL.value,
                    "_requested_mode": context.mode.value,
                    "_effective_mode": QueryMode.GLOBAL.value,
                }

            logger.error(
                f"LightRAG query failed for KB {kb.full_name}",
                error=e
            )
            raise

    async def _retrieve_chunks(
        self,
        kb: KnowledgeBase,
        context: QueryContext,
        prompt: str
    ) -> List[RetrievedChunk]:
        """
        Retrieve document chunks for evaluation.

        Args:
            kb: Knowledge base
            context: Query context
            prompt: Query prompt

        Returns:
            List of retrieved chunks
        """
        try:
            # Query for context only
            working_dir = self._get_working_dir(kb)
            self.lightrag.working_dir = working_dir
            # Keep LightRAG workspace in sync with the KB-specific label namespace.
            self.lightrag.workspace = WorkspaceResolver.build_workspace_name(
                kb.domain,
                kb.name,
            )
            self.lightrag.set_runtime_context(
                workspace_id=context.workspace_id,
                agent_id=context.agent_id,
            )

            requested_mode = context.mode.value
            context_response = await self.lightrag.query(
                query=prompt,
                mode=requested_mode,
                only_need_context=True,
                chunk_top_k=20,
                top_k=2,
                stream=False
            )

            # Parse chunks from context
            chunks = self._parse_chunks(context_response, kb)

            # No-answer is different from failure, but empty context is not useful for eval.
            # Retry context retrieval in global mode to collect graph evidence when available.
            if context.mode != QueryMode.GLOBAL and not chunks:
                logger.warning(
                    "No chunks in requested mode, retrying context retrieval in global mode",
                    kb=kb.full_name,
                    requested_mode=requested_mode,
                )
                context_response = await self.lightrag.query(
                    query=prompt,
                    mode=QueryMode.GLOBAL.value,
                    only_need_context=True,
                    chunk_top_k=20,
                    top_k=2,
                    stream=False,
                )
                chunks = self._parse_chunks(context_response, kb)

            logger.info(
                f"Retrieved {len(chunks)} chunks from KB {kb.full_name}"
            )

            return chunks

        except Exception as e:
            if context.mode != QueryMode.GLOBAL:
                logger.warning(
                    "Chunk retrieval failed, retrying context retrieval in global mode",
                    kb=kb.full_name,
                    error=str(e),
                    requested_mode=context.mode.value,
                )
                try:
                    context_response = await self.lightrag.query(
                        query=prompt,
                        mode=QueryMode.GLOBAL.value,
                        only_need_context=True,
                        chunk_top_k=20,
                        top_k=2,
                        stream=False,
                    )
                    chunks = self._parse_chunks(context_response, kb)
                    logger.info(
                        f"Retrieved {len(chunks)} chunks from KB {kb.full_name} via global fallback"
                    )
                    return chunks
                except Exception as fallback_error:
                    logger.warning(
                        f"Global fallback chunk retrieval failed for KB {kb.full_name}",
                        error=fallback_error,
                    )

            logger.warning(
                f"Chunk retrieval failed for KB {kb.full_name}",
                error=e
            )
            return []

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
                    or item.get("doc_id")
                    or kb.full_name
                )
                chunk_id = str(item.get("chunk_id") or item.get("id") or f"{kb.full_name}:{idx}")

                metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
                metadata = {
                    **metadata,
                    "kb": kb.full_name,
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

            # Execute query
            response_payload = await self._query_lightrag(kb, context, prompt)
            response_text = self._extract_answer_text(response_payload)

            # Parse references
            references = parse_references(response_text)

            # Clean response (remove references section)
            clean_answer = clean_response(response_text)

            # Retrieve chunks for evaluation
            chunks = await self._retrieve_chunks(kb, context, prompt)

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

            # Aggregate results using LLM
            aggregated_answer = await self._aggregate_results(
                context,
                kb_results
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

        Initialization is sequential to avoid global env races (Neo4j/Postgres env wiring),
        while query execution remains parallel for performance.
        """

        kb_results: Dict[str, Any] = {}
        kb_services: Dict[str, LightRAGService] = {}

        # Prepare each KB service sequentially to avoid env races during initialize().
        for kb in knowledge_bases:
            try:
                working_dir = self._get_working_dir(kb)
                workspace_label = WorkspaceResolver.build_workspace_name(
                    kb.domain,
                    kb.name,
                )
                service = LightRAGService(
                    working_dir=working_dir,
                    workspace=workspace_label,
                )
                service.set_runtime_context(
                    workspace_id=context.workspace_id,
                    agent_id=context.agent_id,
                )
                await service.initialize()
                kb_services[kb.full_name] = service

                logger.debug(
                    "Prepared KB LightRAG service",
                    kb=kb.full_name,
                    working_dir=working_dir,
                    workspace_label=workspace_label,
                )
            except Exception as e:
                logger.warning("KB initialization failed", kb=kb.full_name, error=e)
                kb_results[kb.full_name] = {"error": str(e)}

        async def query_single_kb(kb: KnowledgeBase):
            """Query single KB with error handling"""
            try:
                service = kb_services[kb.full_name]
                requested_mode = context.mode.value
                response = await service.query(
                    query=prompt,
                    mode=requested_mode,
                    conversation_history=context.history,
                    top_k=2,
                    stream=False,
                )
                effective_mode = requested_mode

                if isinstance(response, dict):
                    response.setdefault("mode", effective_mode)
                    response["_requested_mode"] = requested_mode
                    response["_effective_mode"] = effective_mode
                else:
                    response = {
                        "answer": str(response) if response is not None else "",
                        "retrieved_chunks": [],
                        "sources": [],
                        "mode": effective_mode,
                        "_requested_mode": requested_mode,
                        "_effective_mode": effective_mode,
                    }

                # Capture raw context and parsed chunks for caching/evaluation.
                try:
                    context_response = await service.query(
                        query=prompt,
                        mode=effective_mode,
                        only_need_context=True,
                        chunk_top_k=20,
                        top_k=2,
                        stream=False,
                    )
                except Exception as context_error:
                    logger.warning(
                        "Context retrieval failed for KB after answer query",
                        kb=kb.full_name,
                        error=context_error,
                    )
                    context_response = {}

                parsed_chunks = self._parse_chunks(context_response, kb)
                response["_retrieved_chunks"] = self._serialize_chunks(parsed_chunks)
                response["_raw_context"] = (
                    context_response.get("retrieved_chunks", [])
                    if isinstance(context_response, dict)
                    else context_response
                )
                return (kb.full_name, response, None)
            except Exception as e:
                if context.mode != QueryMode.GLOBAL:
                    logger.warning(
                        "Query failed, retrying in global mode",
                        kb=kb.full_name,
                        error=str(e),
                        requested_mode=context.mode.value,
                    )
                    try:
                        service = kb_services[kb.full_name]
                        response = await service.query(
                            query=prompt,
                            mode=QueryMode.GLOBAL.value,
                            conversation_history=context.history,
                            top_k=2,
                            stream=False,
                        )
                        if isinstance(response, dict):
                            response.setdefault("mode", QueryMode.GLOBAL.value)
                            response["_requested_mode"] = context.mode.value
                            response["_effective_mode"] = QueryMode.GLOBAL.value
                        else:
                            response = {
                                "answer": str(response) if response is not None else "",
                                "retrieved_chunks": [],
                                "sources": [],
                                "mode": QueryMode.GLOBAL.value,
                                "_requested_mode": context.mode.value,
                                "_effective_mode": QueryMode.GLOBAL.value,
                            }
                        return (kb.full_name, response, None)
                    except Exception as fallback_error:
                        logger.warning(
                            "Global fallback query failed",
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

        # Best-effort close of isolated services.
        for service in kb_services.values():
            try:
                await service.close()
            except Exception:
                pass

        return kb_results

    async def _aggregate_results(
        self,
        context: QueryContext,
        kb_results: Dict[str, Any]
    ) -> str:
        """
        Aggregate KB results intelligently.

        Uses LLM to synthesize multiple answers into one coherent response,
        or returns single result directly if only one KB succeeded.
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
            # Call LLM for aggregation using naive mode (no RAG, just text generation)
            aggregated = await self.lightrag.query(
                query=aggregation_prompt,
                mode="naive",
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
