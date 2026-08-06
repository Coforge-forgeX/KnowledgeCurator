"""
RAG Query Service - Main Orchestrator

Coordinates RAG query execution with clean separation of concerns.
Follows SOLID principles and uses dependency injection.
"""
import os
from datetime import datetime, timedelta
from typing import List, Optional

from azure.storage.blob import BlobSasPermissions, BlobServiceClient, generate_blob_sas

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

logger = get_logger(__name__)


class SourceEnricher:
    """
    Enriches document references with download URLs.

    Design:
    - Single Responsibility: Only generates URLs
    - Dependency Injection: Blob client injected
    - No side effects: Pure transformation
    """

    def __init__(self, blob_connection_string: str, container_name: str):
        self.blob_connection_string = blob_connection_string
        self.container_name = container_name
        self._blob_service_client = None

    @property
    def blob_service_client(self) -> BlobServiceClient:
        """Lazy initialization of blob service client"""
        if not self._blob_service_client:
            self._blob_service_client = BlobServiceClient.from_connection_string(
                self.blob_connection_string
            )
        return self._blob_service_client

    def enrich_reference(
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

            # Check if blob exists
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_path
            )

            if not blob_client.exists():
                logger.warning(
                    f"Blob not found for reference: {reference.file_path}",
                    blob_path=blob_path
                )
                return None

            # Generate SAS URL
            download_url = self._generate_sas_url(blob_path)

            enriched = EnrichedSource(
                file_name=f"{reference.citation_number} {reference.file_name}",
                download_url=download_url,
                container_name=self.container_name,
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

    def _generate_sas_url(self, blob_path: str) -> str:
        """Generate SAS URL for blob"""
        try:
            # Get blob client
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_path
            )

            # Extract account name and key from connection string
            account_name = self.blob_service_client.account_name
            account_key = self._extract_account_key()

            if not account_key:
                raise ValueError("Could not extract account key from connection string")

            # Generate SAS token
            sas_token = generate_blob_sas(
                account_name=account_name,
                container_name=self.container_name,
                blob_name=blob_path,
                account_key=account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.utcnow() + timedelta(days=365 * 10)  # 10 years
            )

            # Build URL
            url = f"{blob_client.url}?{sas_token}"
            return url

        except Exception as e:
            logger.error(f"Failed to generate SAS URL", error=e, blob_path=blob_path)
            raise

    def _extract_account_key(self) -> Optional[str]:
        """Extract account key from connection string"""
        try:
            parts = self.blob_connection_string.split(';')
            for part in parts:
                if part.startswith('AccountKey='):
                    return part.split('=', 1)[1]
            return None
        except Exception:
            return None


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
        blob_connection_string: Optional[str] = None,
        blob_container_name: Optional[str] = None
    ):
        """
        Initialize RAG Query Service.

        Args:
            blob_connection_string: Azure Blob Storage connection string
            blob_container_name: Blob storage container name
        """
        self.blob_connection_string = blob_connection_string or \
            settings.storage.AZURE_BLOB_STORAGE_CONNECTION_STRING
        self.blob_container_name = blob_container_name or \
            settings.storage.AZURE_BLOB_STORAGE_CONTAINER_NAME

        if not self.blob_connection_string:
            logger.warning("Blob storage not configured - source enrichment disabled")
            self.source_enricher = None
        else:
            self.source_enricher = SourceEnricher(
                self.blob_connection_string,
                self.blob_container_name
            )

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
        agent_id: Optional[int] = None
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
                knowledge_bases
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
        # Convert mode string to QueryMode enum
        try:
            query_mode = QueryMode(mode.lower())
        except ValueError:
            logger.warning(f"Unknown query mode '{mode}', defaulting to HYBRID")
            query_mode = QueryMode.HYBRID

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
        knowledge_bases: Optional[List[str]]
    ) -> List[KnowledgeBase]:
        """Resolve list of knowledge bases to query"""
        kb_list = []

        # Add workspace-scoped KB
        scoped_kb_name = WorkspaceResolver.build_kb_name(kb_name, workspace_id)
        kb_list.append(KnowledgeBase(
            domain=domain,
            name=scoped_kb_name,
            workspace_id=workspace_id
        ))

        # Add additional KBs if provided
        if knowledge_bases:
            for kb_suffix in knowledge_bases:
                if kb_suffix and kb_suffix.strip():
                    additional_kb_name = f"{kb_name}/{kb_suffix.strip()}"
                    kb_list.append(KnowledgeBase(
                        domain=domain,
                        name=additional_kb_name,
                        workspace_id=workspace_id
                    ))

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
            enriched = self.source_enricher.enrich_reference(
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
