"""
Chat Service - Conversation & Chatbot Management

Handles chatbot conversations, message processing, and session management.
Integrates with RAG service for knowledge-based responses.
"""
import asyncio
from typing import Any, Dict, Optional
import uuid

from src.core.config import settings
from src.core.exceptions import (
    APIException,
    DatabaseException,
    NotFoundException,
    ValidationException,
)
from src.core.logging import get_logger
# from src.helpers.workspace_helpers import get_workspace_storage_paths
from src.services.mongodb_service import get_mongodb_service
# from src.services.rag_service import get_rag_service
# from src.services.rag_query_service import get_rag_query_service

logger = get_logger(__name__)


class ChatService:
    """
    Service layer for chatbot and conversation operations.

    Provides:
    - Conversation session management
    - Message handling and history
    - Integration with RAG for knowledge-based responses
    """

    def __init__(self):
        self.mongo_service = get_mongodb_service()
        # self.rag_service = get_rag_service()  # For document operations
        # self.rag_query_service = get_rag_query_service()  # For optimized queries

    async def initialize(self) -> None:
        """Initialize chat service dependencies"""
        await self.mongo_service.initialize()

    # ========================================================================
    # Session Management
    # ========================================================================

    async def get_conversation_history(
        self,
        workspace_id: int,
        user_id: int,
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        """
        Get one page of a user's conversation sessions.

        Args:
            workspace_id: Workspace identifier
            user_id: User identifier
            page: 1-indexed page number
            page_size: Sessions per page

        Returns:
            Dict with `items` (session summaries for this page), `page`,
            `page_size` and `total_count` — enough for the caller to build the
            has_next/has_previous envelope.
        """
        try:
            page = max(int(page), 1)
            page_size = max(int(page_size), 1)
            skip = (page - 1) * page_size

            logger.info(
                "Fetching conversation history",
                workspace_id=workspace_id,
                user_id=user_id,
                page=page,
                page_size=page_size,
            )

            sessions, total_count = await asyncio.gather(
                self.mongo_service.list_sessions(
                    workspace_id=workspace_id,
                    user_id=user_id,
                    limit=page_size,
                    skip=skip,
                ),
                self.mongo_service.count_sessions(
                    workspace_id=workspace_id,
                    user_id=user_id,
                ),
            )

            logger.info(
                "Retrieved conversation history",
                session_count=len(sessions),
                total_count=total_count,
            )

            return {
                "items": sessions,
                "page": page,
                "page_size": page_size,
                "total_count": total_count,
            }

        except APIException:
            raise
        except Exception as e:
            logger.error("Failed to get conversation history", error=e)
            raise DatabaseException(
                message=f"Failed to get conversation history: {str(e)}",
                operation="get_conversation_history"
            )

    async def load_conversation(
        self,
        session_id: str,
        workspace_id: int,
        user_id: int,
        page: int = 1,
        page_size: int = 50,
        newest_first: bool = True,
    ) -> Dict[str, Any]:
        """
        Load a specific conversation with one page of its messages.

        Args:
            session_id: Session identifier
            workspace_id: Workspace identifier
            user_id: User identifier
            page: 1-indexed page number
            page_size: Messages per page
            newest_first: page 1 is the newest slice of the transcript (see
                `MongoDBService.get_messages_page`)

        Returns:
            Dict with session metadata, this page of messages, and the paging
            counters (`page`, `page_size`, `total_count`). `message_count` is the
            size of this page; `total_count` is the whole conversation.
        """
        try:
            logger.info(
                "Loading conversation",
                session_id=session_id,
                workspace_id=workspace_id,
                page=page,
                page_size=page_size,
            )

            # Get session metadata
            session_meta = await self.mongo_service.get_session(
                session_id=session_id,
                workspace_id=workspace_id,
                user_id=user_id,
            )

            if not session_meta:
                raise NotFoundException(
                    message=f"Conversation not found: {session_id}",
                    resource_type="conversation",
                )

            messages, total_count = await self.mongo_service.get_messages_page(
                session_id=session_id,
                workspace_id=workspace_id,
                user_id=user_id,
                page=page,
                page_size=page_size,
                newest_first=newest_first,
            )

            logger.info(
                "Loaded conversation",
                session_id=session_id,
                message_count=len(messages),
                total_count=total_count,
            )

            return {
                "session_id": session_id,
                "session_metadata": session_meta,
                "messages": messages,
                "message_count": len(messages),
                "page": max(int(page), 1),
                "page_size": max(int(page_size), 1),
                "total_count": total_count,
                "newest_first": newest_first,
            }

        except APIException:
            # Preserve 404/400 semantics — wrapping these in DatabaseException
            # turned "no such conversation" into a 500.
            raise
        except Exception as e:
            logger.error("Failed to load conversation", error=e)
            raise DatabaseException(
                message=f"Failed to load conversation: {str(e)}",
                operation="load_conversation"
            )

    async def rename_conversation(
        self,
        session_id: str,
        workspace_id: int,
        user_id: int,
        title: str,
    ) -> Dict[str, Any]:
        """
        Rename a conversation session.

        Args:
            session_id: Session identifier
            workspace_id: Workspace identifier
            user_id: User identifier
            title: New title

        Returns:
            Dict with updated session info
        """
        try:
            logger.info(
                "Renaming conversation",
                session_id=session_id,
                title=title,
            )

            if not title or not title.strip():
                raise ValidationException(
                    message="Title cannot be empty"
                )

            # Update in MongoDB
            success = await self.mongo_service.update_session_title(
                session_id=session_id,
                workspace_id=workspace_id,
                user_id=user_id,
                title=title.strip(),
            )

            if not success:
                raise NotFoundException(
                    message=f"Conversation not found: {session_id}",
                    resource_type="conversation",
                )

            logger.info(
                "Conversation renamed",
                session_id=session_id,
            )

            return {
                "session_id": session_id,
                "title": title.strip(),
                "status": "updated",
            }

        except APIException:
            raise
        except Exception as e:
            logger.error("Failed to rename conversation", error=e)
            raise DatabaseException(
                message=f"Failed to rename conversation: {str(e)}",
                operation="rename_conversation"
            )

    async def delete_conversation(
        self,
        session_id: str,
        workspace_id: int,
        user_id: int,
    ) -> Dict[str, Any]:
        """
        Delete a conversation session and all its messages.

        Args:
            session_id: Session identifier
            workspace_id: Workspace identifier
            user_id: User identifier

        Returns:
            Dict with deletion status

        Raises:
            NotFoundException: if the conversation does not exist for this
                user/workspace. Reporting "deleted" for a session that was never
                there hid both typo'd ids and cross-tenant probes.
        """
        try:
            logger.info(
                "Deleting conversation",
                session_id=session_id,
            )

            # Delete from MongoDB
            mongo_success = await self.mongo_service.delete_session(
                session_id=session_id,
                workspace_id=workspace_id,
                user_id=user_id,
            )

            if not mongo_success:
                logger.warning(
                    "Conversation not found for deletion",
                    session_id=session_id,
                    workspace_id=workspace_id,
                    user_id=user_id,
                )
                raise NotFoundException(
                    message=f"Conversation not found: {session_id}",
                    resource_type="conversation",
                )

            logger.info(
                "Conversation deleted",
                session_id=session_id,
            )

            return {
                "session_id": session_id,
                "status": "deleted",
                "message": "Conversation deleted successfully",
            }

        except APIException:
            raise
        except Exception as e:
            logger.error("Failed to delete conversation", error=e)
            raise DatabaseException(
                message=f"Failed to delete conversation: {str(e)}",
                operation="delete_conversation"
            )

    # ========================================================================
    # Message Processing
    # ========================================================================

    # async def message_gpt(
    #     self,
    #     user_message: str,
    #     session_id: str,
    #     workspace_id: int,
    #     user_id: int,
    #     role_id: int,
    #     industry: Optional[str] = None,
    #     sub_industry: Optional[str] = None,
    #     mode: str = "Search",
    #     agent_id: Optional[int] = None,
    #     knowledge_bases: Optional[List[str]] = None,
    #     file_names: Optional[List[str]] = None,
    #     file_contents: Optional[List[str]] = None,
    # ) -> Dict[str, Any]:
    #     """
    #     Process a user message and generate a response.

    #     Args:
    #         user_message: User's message
    #         session_id: Session identifier
    #         workspace_id: Workspace identifier
    #         user_id: User identifier
    #         role_id: Role identifier
    #         industry: Optional industry/domain
    #         sub_industry: Optional sub-industry
    #         mode: Conversation mode (Search, Query, Update)
    #         agent_id: Optional agent identifier
    #         knowledge_bases: Optional list of KBs to query
    #         file_names: Optional files to upload
    #         file_contents: Optional file contents

    #         Returns:
    #         Dict with response, sources, and task_ids
    #     """
    #     try:
    #         logger.info(
    #             "Processing message",
    #             session_id=session_id,
    #             workspace_id=workspace_id,
    #             message_length=len(user_message),
    #             mode=mode,
    #         )

    #         # Save user message
    #         await self.mongo_service.append_message(
    #             session_id=session_id,
    #             workspace_id=workspace_id,
    #             user_id=user_id,
    #             role="user",
    #             content=user_message,
    #         )

    #         # Handle file upload mode
    #         if file_names and file_contents:
    #             logger.info("Processing file upload")
    #             upload_result = await self.rag_service.upload_and_index_tool(
    #                 file_names=file_names,
    #                 file_contents=file_contents,
    #                 workspace_id=workspace_id,
    #                 user_id=user_id,
    #                 role_id=role_id,
    #                 domain=industry,
    #                 kb_name=sub_industry,
    #             )

    #             response_text = upload_result.get("response", "Files uploaded successfully")
    #             task_ids = upload_result.get("task_ids", [])

    #             # Save assistant response
    #             await self.mongo_service.append_message(
    #                 session_id=session_id,
    #                 workspace_id=workspace_id,
    #                 user_id=user_id,
    #                 role="assistant",
    #                 content=response_text,
    #                 task_ids=task_ids,
    #             )

    #             return {
    #                 "response": response_text,
    #                 "task_ids": task_ids,
    #                 "sources": [],
    #             }

    #         # Query RAG for knowledge-based response
    #         history, _ = await self.mongo_service.get_messages_page(
    #             session_id=session_id,
    #             workspace_id=workspace_id,
    #             user_id=user_id,
    #             page_size=settings.CHAT_HISTORY_TURNS_FOR_CONTEXT,  # Last 5 messages for context
    #         )

    #         # Get workspace storage paths for security (domain, kb_name from DB)
    #         storage_paths = await get_workspace_storage_paths(workspace_id)
    #         if not storage_paths:
    #             raise ValidationException(
    #                 message=f"Failed to retrieve workspace configuration for workspace {workspace_id}"
    #             )

    #         domain = storage_paths.get("domain", "")
    #         kb_name = storage_paths.get("kb_name", "")
    #         all_kb_titles = storage_paths.get("all_kb_titles", [])

    #         # For multi-KB workspaces, pass additional KBs
    #         additional_kbs = None
    #         if len(all_kb_titles) > 1:
    #             additional_kbs = all_kb_titles[1:]

    #         # Use optimized RAG query service
    #         rag_result = await self.rag_query_service.query(
    #             query=user_message,
    #             workspace_id=workspace_id,
    #             role_id=role_id,
    #             domain=domain,
    #             kb_name=kb_name,
    #             mode="hybrid",
    #             history=history,
    #             knowledge_bases=additional_kbs,
    #             agent_id=agent_id,
    #             is_kg=storage_paths.get("is_kg"),
    #         )

    #         response_text = rag_result.answer or "I don't have enough information to answer that question."
    #         # Convert EnrichedSource objects to dicts for MongoDB storage
    #         sources = [
    #             {
    #                 "file_name": src.file_name,
    #                 "download_url": src.download_url,
    #                 "citation": src.citation,
    #             }
    #             for src in rag_result.sources
    #         ]

    #         # Save assistant response
    #         await self.mongo_service.append_message(
    #             session_id=session_id,
    #             workspace_id=workspace_id,
    #             user_id=user_id,
    #             role="assistant",
    #             content=response_text,
    #             sources=sources,
    #         )

    #         # Keep the title in sync with the latest user message. This is a
    #         # no-op once the user has manually renamed the session (see
    #         # `update_session_title`'s `is_manual=False` guard on
    #         # `title_set_by_user`), so a rename always wins.
    #         title = user_message.strip()
    #         if len(title) > 50:
    #             title = title[:50] + "..."
    #         await self.mongo_service.update_session_title(
    #             session_id=session_id,
    #             workspace_id=workspace_id,
    #             user_id=user_id,
    #             title=title,
    #             is_manual=False,
    #         )

    #         logger.info(
    #             "Message processed",
    #             session_id=session_id,
    #             has_sources=bool(sources),
    #         )

    #         return {
    #             "response": response_text,
    #             "sources": sources,
    #             "task_ids": [],
    #         }

    #     except Exception as e:
    #         logger.error("Failed to process message", error=e)
    #         error_response = f"Sorry, something went wrong while processing your request: {str(e)}"

    #         # Try to save error response
    #         try:
    #             await self.mongo_service.append_message(
    #                 session_id=session_id,
    #                 workspace_id=workspace_id,
    #                 user_id=user_id,
    #                 role="assistant",
    #                 content=error_response,
    #             )
    #         except Exception:
    #             pass  # Don't fail if we can't save error

    #         return {
    #             "response": error_response,
    #             "error": str(e),
    #             "sources": [],
    #             "task_ids": [],
    #         }


# ============================================================================
# Singleton Instance
# ============================================================================

_chat_service_instance: Optional[ChatService] = None


def get_chat_service() -> ChatService:
    """Get or create singleton chat service instance"""
    global _chat_service_instance
    if _chat_service_instance is None:
        _chat_service_instance = ChatService()
    return _chat_service_instance
