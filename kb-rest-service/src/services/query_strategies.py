"""
Query Strategies - Strategy Pattern

Different strategies for executing RAG queries.
Follows Open/Closed Principle - extensible without modification.
"""
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from src.core.lightrag_service import get_lightrag_service, QueryParam
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
    ) -> str:
        """
        Execute LightRAG query.

        Args:
            kb: Knowledge base to query
            context: Query context
            prompt: Formatted prompt

        Returns:
            RAG response text

        Raises:
            Exception: If query fails
        """
        try:
            # Set working directory for this KB
            working_dir = self._get_working_dir(kb)
            self.lightrag.working_dir = working_dir

            # Execute query
            response = await self.lightrag.query(
                query=prompt,
                mode=context.mode.value,
                conversation_history=context.history,
                top_k=2,
                stream=False
            )

            logger.info(
                f"LightRAG query completed for KB {kb.full_name}",
                response_length=len(response)
            )

            return response

        except Exception as e:
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

            context_response = await self.lightrag.query(
                query=prompt,
                mode=context.mode.value,
                only_need_context=True,
                chunk_top_k=20,
                top_k=2,
                stream=False
            )

            # Parse chunks from context
            chunks = self._parse_chunks(context_response, kb)

            logger.info(
                f"Retrieved {len(chunks)} chunks from KB {kb.full_name}"
            )

            return chunks

        except Exception as e:
            logger.warning(
                f"Chunk retrieval failed for KB {kb.full_name}",
                error=e
            )
            return []

    def _parse_chunks(
        self,
        context_response: str,
        kb: KnowledgeBase
    ) -> List[RetrievedChunk]:
        """
        Parse chunks from LightRAG context response.

        TODO: Implement actual parsing logic based on LightRAG response format
        """
        # Placeholder - implement based on actual LightRAG context format
        return []

    def _get_working_dir(self, kb: KnowledgeBase) -> str:
        """Get working directory for knowledge base"""
        # TODO: Implement based on actual directory structure
        # This should match the workspace_working_dir logic from KnowledgeCurator
        base_dir = "/path/to/lightrag"  # Get from config
        workspace_name = WorkspaceResolver.build_workspace_name(
            kb.domain,
            kb.name,
            WorkspaceResolver.workspace_id_to_alpha(kb.workspace_id) if kb.workspace_id else ""
        )
        return f"{base_dir}/{workspace_name}"


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
            response = await self._query_lightrag(kb, context, prompt)

            # Parse references
            references = parse_references(response)

            # Clean response (remove references section)
            clean_answer = clean_response(response)

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
                    refs = parse_references(result)
                    all_references.extend(refs)

            # Collect chunks from first successful KB
            chunks = await self._get_first_kb_chunks(
                knowledge_bases,
                context,
                prompt
            )

            result = RAGQueryResult(
                answer=clean_answer,
                sources=[],  # Enriched by service layer
                retrieved_chunks=chunks,
                metadata={
                    "kbs": [kb.full_name for kb in knowledge_bases],
                    "mode": context.mode.value,
                    "kb_count": len(knowledge_bases),
                    "successful_kbs": len([r for r in kb_results.values() if not isinstance(r, dict) or "error" not in r]),
                    "reference_count": len(all_references),
                }
            )

            # Store references and KB results for later enrichment
            result.metadata["raw_references"] = all_references
            result.metadata["kb_results"] = kb_results

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
    ) -> Dict[str, any]:
        """Query all KBs in parallel"""

        async def query_single_kb(kb: KnowledgeBase):
            """Query single KB with error handling"""
            try:
                response = await self._query_lightrag(kb, context, prompt)
                return (kb.full_name, response, None)
            except Exception as e:
                logger.warning(f"KB query failed: {kb.full_name}", error=e)
                return (kb.full_name, None, str(e))

        # Execute in parallel
        tasks = [query_single_kb(kb) for kb in knowledge_bases]
        task_results = await asyncio.gather(*tasks)

        # Build results dict
        kb_results = {}
        for kb_name, response, error in task_results:
            if error:
                kb_results[kb_name] = {"error": error}
            else:
                kb_results[kb_name] = response

        return kb_results

    async def _aggregate_results(
        self,
        context: QueryContext,
        kb_results: Dict[str, any]
    ) -> str:
        """Aggregate KB results using LLM"""

        # Build aggregation prompt
        summary_prompt = self.summary_prompt_builder.build(
            context.query,
            context.history
        )

        # Add KB results to prompt
        summary_prompt += "\n\n---Knowledge Base Results---\n"
        for kb_name, result in kb_results.items():
            summary_prompt += f"\n### Knowledge Base: {kb_name}\n"
            if isinstance(result, dict) and "error" in result:
                summary_prompt += f"Error: {result['error']}\n"
            else:
                summary_prompt += f"{result}\n"

        # TODO: Call LLM router service for aggregation
        # For now, return a placeholder
        logger.warning("LLM aggregation not implemented - using first KB result")

        # Fallback: return first successful result
        for kb_name, result in kb_results.items():
            if not isinstance(result, dict) or "error" not in result:
                return result

        return "No successful results to aggregate"

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
