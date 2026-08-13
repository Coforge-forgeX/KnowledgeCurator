"""
RAG Query Service - Main Orchestrator

Coordinates RAG query execution with clean separation of concerns.
Follows SOLID principles and uses dependency injection.
"""
import os
from datetime import datetime, timedelta
from typing import List, Optional

from src.core.config import settings
from src.core.exceptions import LightRAGException, ValidationException
from src.core.logging import get_logger
from src.helpers.workspace_resolver import WorkspaceResolver
from src.models.rag_models import (
    DocumentReference,
    EnrichedSource,
    KnowledgeBase,
    QueryContext,
    QueryMode,
    RAGQueryResult,
)
from src.services.query_strategies import QueryStrategyFactory
from src.storage import get_storage_adapter

logger = get_logger(__name__)


class SourceEnricher:
    """
    Enriches document references with download URLs.

    Design:
    - Single Responsibility: Only generates URLs
    - Dependency Injection: Storage adapter injected
    - No side effects: Pure transformation
    - Provider Agnostic: Works with Azure, AWS, GCP
    """

    def __init__(self):
        """Initialize with provider-agnostic storage adapter"""
        self._storage = None

    @property
    def storage(self):
        """Lazy initialization of storage adapter"""
        if self._storage is None:
            self._storage = get_storage_adapter()
        return self._storage

    async def enrich_reference(
        self,
        reference: DocumentReference,
        domain: str,
        kb_name: str,
        workspace_id: Optional[int] = None,
        role_id: Optional[int] = None
    ) -> Optional[EnrichedSource]:
        """
        Enrich reference with download URL.

        Args:
            reference: Parsed document reference
            domain: Domain name
            kb_name: Knowledge base name
            workspace_id: Optional workspace ID
            role_id: Optional role ID

        Returns:
            EnrichedSource with download URL, or None if file not found

        Design:
        - Graceful degradation: Returns None if file not found
        - Comprehensive logging: Warns on missing files
        """
        try:
            # Build blob path
            blob_path = self._build_blob_path(
                domain,
                kb_name,
                reference.file_path,
                workspace_id,
                role_id
            )

            # Check if file exists using storage adapter
            exists = await self.storage.blob_exists(blob_path)
            if not exists:
                logger.warning(
                    f"File not found for reference: {reference.file_path}",
                    blob_path=blob_path,
                    provider=self.storage.provider_name,
                )
                return None

            # Generate download URL using storage adapter
            download_url = await self._generate_download_url(blob_path)

            enriched = EnrichedSource(
                file_name=f"{reference.citation_number} {reference.file_name}",
                download_url=download_url,
                container_name=self.storage.container_name,
                blob_path=blob_path,
                download_name=reference.file_name,
                citation=reference.citation_number
            )

            logger.debug(
                f"Enriched reference: {reference.citation_number} -> {blob_path}"
            )

            return enriched

        except Exception as e:
            logger.error(
                f"Failed to enrich reference: {reference.citation_number}",
                error=e
            )
            return None

    def _build_blob_path(
        self,
        domain: str,
        kb_name: str,
        file_path: str,
        workspace_id: Optional[int],
        role_id: Optional[int]
    ) -> str:
        """Build blob storage path for file"""
        # Extract original KB name (without workspace suffix)
        original_kb_name = kb_name.split('/')[0] if '/' in kb_name else kb_name

        parts = [domain, original_kb_name]

        # Add workspace ID for non-SME users
        if workspace_id and role_id != 34:
            parts.append(str(workspace_id))

        # Handle file_path (might be just filename or include path)
        if '/' in file_path:
            # Full path provided - use as is
            parts.append(file_path)
        else:
            # Just filename - append directly
            parts.append(file_path)

        return '/'.join(parts)

    async def _generate_download_url(self, blob_path: str) -> str:
        """Generate signed download URL using storage adapter (provider-agnostic)"""
        try:
            # Use storage adapter to generate download URL
            # Works for Azure SAS, AWS presigned, GCP signed URLs
            url = await self.storage.generate_download_url(
                filename=blob_path,
                expiry_minutes=525600  # 1 year (365 * 24 * 60)
            )

            logger.debug(
                "Generated download URL",
                blob_path=blob_path,
                provider=self.storage.provider_name,
            )

            return url

        except Exception as e:
            logger.error(f"Failed to generate download URL", error=e, blob_path=blob_path)
            raise


class RAGQueryService:
    """
    Main orchestrator for RAG query operations.

    Design:
    - Facade Pattern: Simplifies complex subsystem interactions
    - Dependency Injection: All dependencies injected
    - Clear flow: Setup -> Execute -> Enrich -> Return
    """

    def __init__(
        self,
        blob_connection_string: Optional[str] = None,  # Deprecated - kept for backwards compatibility
        blob_container_name: Optional[str] = None  # Deprecated - kept for backwards compatibility
    ):
        """
        Initialize RAG Query Service with provider-agnostic storage.

        Args:
            blob_connection_string: (Deprecated) Kept for backwards compatibility
            blob_container_name: (Deprecated) Kept for backwards compatibility
        """
        # Initialize source enricher with storage adapter (provider-agnostic)
        try:
            self.source_enricher = SourceEnricher()
            logger.info(
                "Source enricher initialized",
                provider=self.source_enricher.storage.provider_name,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize source enricher: {e} - source enrichment disabled")
            self.source_enricher = None

    async def query(
        self,
        query: str,
        workspace_id: int,
        role_id: int,
        domain: str,
        kb_name: str,
        mode: str = "hybrid",
        history: Optional[List[dict]] = None,
        knowledge_bases: Optional[List[str]] = None,
        agent_id: Optional[int] = None,
        is_kg: Optional[bool] = None,
    ) -> RAGQueryResult:
        """
        Execute RAG query with full orchestration.

        Args:
            query: User query string
            workspace_id: Workspace identifier
            role_id: User role ID
            domain: Domain name
            kb_name: Base knowledge base name
            mode: Query mode (naive, local, global, hybrid, mix)
            history: Optional conversation history
            knowledge_bases: Optional list of additional KB names
            agent_id: Optional agent ID for LLM routing

        Returns:
            RAGQueryResult with answer, sources, and chunks

        Raises:
            ValidationException: If input validation fails
            LightRAGException: If query execution fails
        """
        try:
            # Validate input
            self._validate_input(query, workspace_id, role_id)

            # Build query context
            context = self._build_query_context(
                query,
                workspace_id,
                role_id,
                mode,
                history,
                agent_id
            )

            # Resolve knowledge bases
            kbs = self._resolve_knowledge_bases(
                domain,
                kb_name,
                workspace_id,
                knowledge_bases,
                is_kg=is_kg,
            )

            logger.info(
                f"Executing RAG query",
                workspace_id=workspace_id,
                kb_count=len(kbs),
                mode=mode
            )

            # Select and execute strategy
            strategy = QueryStrategyFactory.create(len(kbs))
            result = await strategy.execute(context, kbs)

            # Enrich sources with download URLs
            if self.source_enricher and "raw_references" in result.metadata:
                result.sources = await self._enrich_sources(
                    result.metadata["raw_references"],
                    domain,
                    kb_name,
                    workspace_id,
                    role_id
                )

            logger.info(
                f"RAG query completed",
                answer_length=len(result.answer),
                source_count=len(result.sources),
                chunk_count=len(result.retrieved_chunks)
            )

            return result

        except ValidationException:
            raise
        except Exception as e:
            logger.error("RAG query failed", error=e)
            raise LightRAGException(
                message=f"Failed to execute RAG query: {str(e)}",
                operation="query_rag"
            )

    def _validate_input(
        self,
        query: str,
        workspace_id: int,
        role_id: int
    ) -> None:
        """Validate query input"""
        if not query or not query.strip():
            raise ValidationException(message="Query cannot be empty")

        if workspace_id is None or workspace_id < 0:
            raise ValidationException(message="Invalid workspace_id")

        if role_id is None or role_id < 0:
            raise ValidationException(message="Invalid role_id")

    def _build_query_context(
        self,
        query: str,
        workspace_id: int,
        role_id: int,
        mode: str,
        history: Optional[List[dict]],
        agent_id: Optional[int]
    ) -> QueryContext:
        """Build query context from parameters"""
        normalized_mode = (mode or "").lower().strip()
        if normalized_mode == QueryMode.HYBRID.value:
            normalized_mode = QueryMode.MIX.value

        # Convert mode string to QueryMode enum
        try:
            query_mode = QueryMode(normalized_mode)
        except ValueError:
            logger.warning(f"Unknown query mode '{mode}', defaulting to MIX")
            query_mode = QueryMode.MIX

        return QueryContext(
            query=query.strip(),
            workspace_id=workspace_id,
            role_id=role_id,
            mode=query_mode,
            history=history or [],
            agent_id=agent_id
        )

    def _resolve_knowledge_bases(
        self,
        domain: str,
        kb_name: str,
        workspace_id: int,
        knowledge_bases: Optional[List[str]],
        is_kg: Optional[bool] = None,
    ) -> List[KnowledgeBase]:
        """Resolve list of knowledge bases to query.

        Rules:
        - Non-KG workspace: one workspace-level graph + domain-level graphs.
        - KG workspace: only one domain-level graph.
        """
        kb_list: List[KnowledgeBase] = []
        seen_names = set()

        def normalize_kb_name(raw_name: str) -> str:
            """Normalize KB name and strip domain prefix if present."""
            normalized = (raw_name or "").strip().strip("/")
            if not normalized:
                return ""
            domain_prefix = f"{domain}/"
            if normalized.startswith(domain_prefix):
                normalized = normalized[len(domain_prefix):]
            return normalized

        def add_kb_if_new(raw_name: str) -> None:
            normalized_name = normalize_kb_name(raw_name)
            if not normalized_name:
                return
            dedupe_key = normalized_name.lower()
            if dedupe_key in seen_names:
                return
            seen_names.add(dedupe_key)
            kb_list.append(
                KnowledgeBase(
                    domain=domain,
                    name=normalized_name,
                    workspace_id=workspace_id,
                )
            )

        normalized_primary = normalize_kb_name(kb_name)
        primary_parts = [part for part in normalized_primary.split("/") if part]
        subindustry = primary_parts[0] if primary_parts else ""

        # Always include the primary graph first.
        add_kb_if_new(normalized_primary)

        # KG workspace has exactly one domain-level graph.
        if is_kg:
            logger.debug("KG workspace detected; skipping additional KB graphs")
            logger.debug(f"Resolved {len(kb_list)} knowledge bases for query")
            return kb_list

        # Non-KG workspace: add domain-level graphs for additional KB titles.
        if knowledge_bases:
            for kb_entry in knowledge_bases:
                title = (kb_entry or "").strip().strip("/")
                if not title:
                    continue

                # If caller already passed a scoped path, keep it.
                if "/" in title:
                    add_kb_if_new(title)
                    continue

                if subindustry:
                    add_kb_if_new(f"{subindustry}/{title}")
                else:
                    add_kb_if_new(title)

        logger.debug(f"Resolved {len(kb_list)} knowledge bases for query")
        return kb_list

    async def _enrich_sources(
        self,
        references: List[DocumentReference],
        domain: str,
        kb_name: str,
        workspace_id: int,
        role_id: int
    ) -> List[EnrichedSource]:
        """Enrich references with download URLs"""
        enriched_sources = []

        for ref in references:
            enriched = await self.source_enricher.enrich_reference(
                ref,
                domain,
                kb_name,
                workspace_id,
                role_id
            )
            if enriched:
                enriched_sources.append(enriched)

        logger.info(
            f"Enriched {len(enriched_sources)} / {len(references)} references"
        )

        return enriched_sources


# ============================================================================
# Singleton Instance
# ============================================================================

_rag_query_service_instance: Optional[RAGQueryService] = None


def get_rag_query_service() -> RAGQueryService:
    """Get or create singleton RAG query service instance"""
    global _rag_query_service_instance
    if _rag_query_service_instance is None:
        _rag_query_service_instance = RAGQueryService()
    return _rag_query_service_instance
