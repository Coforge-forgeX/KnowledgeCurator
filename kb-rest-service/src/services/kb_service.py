"""
Knowledge Base Service - Business Logic Layer

Following SOLID principles:
- Single Responsibility: Each method has one clear purpose
- Open/Closed: Extensible without modification
- Liskov Substitution: Consistent interfaces
- Interface Segregation: Minimal dependencies
- Dependency Inversion: Depends on abstractions
"""
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.core.config import settings
from src.core.exceptions import LightRAGException, ValidationException
from src.core.lightrag_service import get_lightrag_service
from src.core.logging import get_logger
from src.helpers.queue_helpers import get_indexing_queue_helper

logger = get_logger(__name__)


class KnowledgeBaseService:
    """
    Service layer for Knowledge Base operations.
    Encapsulates business logic, keeping controllers thin.
    """

    def __init__(self):
        self.lightrag_service = get_lightrag_service()
        self._queue_helper = None  # Lazy-loaded only when needed

    @property
    def queue_helper(self):
        """Lazy-load queue helper only when needed (for indexing operations)"""
        if self._queue_helper is None:
            self._queue_helper = get_indexing_queue_helper()
        return self._queue_helper

    async def queue_document_for_indexing(
        self,
        document_text: str,
        workspace_id: int,
        file_name: str,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Queue a document for background indexing.

        Args:
            document_text: Document content
            workspace_id: Workspace identifier
            file_name: Original file name
            metadata: Optional metadata

        Returns:
            Queue message ID

        Raises:
            QueueException: If queueing fails
        """
        try:
            logger.info(
                "Queueing document for indexing",
                workspace_id=workspace_id,
                file_name=file_name,
                content_length=len(document_text),
            )

            # Prepare queue message
            message = {
                "workspace_id": workspace_id,
                "file_name": file_name,
                "document_text": document_text,
                "metadata": metadata or {},
                "queued_at": str(datetime.utcnow()),
            }

            # Send to queue
            message_id = await self.queue_helper.send_message_async(message)

            logger.info(
                "Document queued successfully",
                workspace_id=workspace_id,
                message_id=message_id,
            )

            return message_id

        except Exception as e:
            logger.error(
                "Failed to queue document",
                error=e,
                workspace_id=workspace_id,
            )
            raise

    async def get_indexed_documents(
        self,
        workspace_id: int,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        Get paginated list of indexed documents for a workspace.

        Args:
            workspace_id: Workspace identifier
            limit: Maximum number of documents to return per page (default: 100, max: 1000)
            offset: Number of documents to skip for pagination (default: 0)

        Returns:
            Dict containing:
            - documents: List of indexed document metadata
            - total: Total count of documents
            - limit: Limit used in query
            - offset: Offset used in query
            - has_more: Boolean indicating if more documents exist
            - page: Current page number
            - total_pages: Total number of pages
        """
        try:
            logger.info(
                "Fetching indexed documents",
                workspace_id=workspace_id,
                limit=limit,
                offset=offset,
            )

            working_dir = await self._get_workspace_working_dir(workspace_id)
            self.lightrag_service.working_dir = working_dir

            # Get paginated documents from LightRAG service with workspace filter
            result = await self.lightrag_service.get_indexed_documents(
                workspace_id=workspace_id,
                limit=limit,
                offset=offset,
            )

            logger.info(
                "Fetched indexed documents",
                workspace_id=workspace_id,
                count=len(result["documents"]),
                total=result["total"],
                page=result["page"],
            )

            return result

        except Exception as e:
            logger.error(
                "Failed to fetch indexed documents",
                error=e,
                workspace_id=workspace_id,
            )
            raise

    async def delete_documents(
        self,
        doc_ids: List[str],
        workspace_id: int,
    ) -> Dict[str, Any]:
        """
        Delete multiple documents from the knowledge base.

        Args:
            doc_ids: List of document IDs to delete
            workspace_id: Workspace identifier

        Returns:
            Dict with deletion summary

        Raises:
            LightRAGException: If deletion fails
        """
        try:
            logger.info(
                "Deleting documents",
                workspace_id=workspace_id,
                count=len(doc_ids),
            )

            working_dir = await self._get_workspace_working_dir(workspace_id)
            self.lightrag_service.working_dir = working_dir

            # Delete documents
            successful = 0
            failed = 0
            errors = []

            for doc_id in doc_ids:
                try:
                    await self.lightrag_service.delete_by_doc_id(doc_id)
                    successful += 1
                except Exception as e:
                    failed += 1
                    errors.append({"doc_id": doc_id, "error": str(e)})
                    logger.warning(
                        "Failed to delete document",
                        doc_id=doc_id,
                        error=e,
                    )

            result = {
                "total": len(doc_ids),
                "successful": successful,
                "failed": failed,
                "errors": errors if errors else None,
            }

            logger.info(
                "Document deletion completed",
                workspace_id=workspace_id,
                **result,
            )

            return result

        except Exception as e:
            logger.error(
                "Document deletion failed",
                error=e,
                workspace_id=workspace_id,
            )
            raise LightRAGException(
                message=f"Failed to delete documents: {str(e)}",
                operation="delete",
            )

    async def get_knowledge_graph(
        self,
        workspace_id: int,
    ) -> Dict[str, Any]:
        """
        Get the knowledge graph for a workspace.

        Args:
            workspace_id: Workspace identifier

        Returns:
            Dict with nodes and edges

        Raises:
            LightRAGException: If retrieval fails
        """
        try:
            logger.info(
                "Fetching knowledge graph",
                workspace_id=workspace_id,
            )

            working_dir = await self._get_workspace_working_dir(workspace_id)
            self.lightrag_service.working_dir = working_dir

            # Get KG from LightRAG
            kg = await self.lightrag_service.get_knowledge_graph()

            logger.info(
                "Fetched knowledge graph",
                workspace_id=workspace_id,
                node_count=kg.get("node_count", 0),
                edge_count=kg.get("edge_count", 0),
            )

            return kg

        except Exception as e:
            logger.error(
                "Failed to fetch knowledge graph",
                error=e,
                workspace_id=workspace_id,
            )
            raise LightRAGException(
                message=f"Failed to fetch knowledge graph: {str(e)}",
                operation="get_kg",
            )

    async def check_indexing_status(
        self,
        task_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Check indexing status for specific task IDs.

        Args:
            task_ids: List of task IDs to check

        Returns:
            List of task status dictionaries
        """
        try:
            logger.info(
                "Checking indexing status",
                task_count=len(task_ids),
            )

            # Query PostgreSQL for task status using SQLAlchemy
            from src.core.database import get_async_session, FileTask
            from sqlalchemy import select

            statuses = []

            async with get_async_session() as session:
                for task_id in task_ids:
                    result = await session.execute(
                        select(FileTask).where(FileTask.id == int(task_id))
                    )
                    task = result.scalar_one_or_none()

                    if task:
                        # Extract file_name from file_path
                        file_name = os.path.basename(task.file_path) if task.file_path else None

                        statuses.append({
                            "task_id": task.id,
                            "status": task.status,
                            "file_name": file_name,
                            "workspace_id": task.workspace_id,
                            "error_message": task.error_message,
                            "created_at": str(task.created_at) if task.created_at else None,
                            "updated_at": str(task.updated_at) if task.updated_at else None,
                        })

            logger.info(
                "Checked indexing status",
                found_count=len(statuses),
            )

            return statuses

        except Exception as e:
            logger.error(
                "Failed to check indexing status",
                error=e,
            )
            raise

    async def _get_workspace_working_dir(
        self,
        workspace_id: int,
        domain: Optional[str] = None,
        kb_name: Optional[str] = None
    ) -> str:
        """
        Get LightRAG working directory for a specific workspace.

        DRY principle: Centralize workspace directory logic.
        If domain/kb_name not provided, fetches them from database.

        Args:
            workspace_id: Workspace identifier
            domain: Optional domain (fetched from DB if not provided)
            kb_name: Optional KB name (fetched from DB if not provided)

        Returns:
            Full working directory path for LightRAG

        Raises:
            Exception: If workspace storage paths cannot be retrieved
        """
        from src.helpers.workspace_helpers import get_workspace_storage_paths
        from shared.workspace_helpers import get_workspace_working_dir

        base_dir = settings.lightrag.LIGHTRAG_WORKING_DIR

        # Fetch domain/kb_name from database if not provided
        if domain is None or kb_name is None:
            storage_paths = await get_workspace_storage_paths(workspace_id)
            if not storage_paths:
                logger.error(
                    "Failed to get workspace storage paths for working directory",
                    workspace_id=workspace_id
                )
                raise ValidationException(
                    message=f"Cannot determine working directory for workspace {workspace_id}"
                )

            domain = storage_paths.get("domain")
            kb_name = storage_paths.get("kb_name")

            logger.debug(
                "Retrieved domain/kb_name from database",
                workspace_id=workspace_id,
                domain=domain,
                kb_name=kb_name
            )

        # Build working directory with domain/kb_name for proper scoping
        working_dir = get_workspace_working_dir(
            workspace_id=workspace_id,
            base_dir=base_dir,
            domain=domain,
            kb_name=kb_name
        )

        logger.debug(
            "Built workspace working directory",
            workspace_id=workspace_id,
            domain=domain,
            kb_name=kb_name,
            working_dir=working_dir
        )

        return working_dir


# Singleton instance
_kb_service_instance: Optional[KnowledgeBaseService] = None


def get_kb_service() -> KnowledgeBaseService:
    """Get or create singleton KB service instance."""
    global _kb_service_instance
    if _kb_service_instance is None:
        _kb_service_instance = KnowledgeBaseService()
    return _kb_service_instance
