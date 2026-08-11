"""
MongoDB Service for Conversation/Chat History Management

Handles conversation sessions, messages, and context storage.
"""
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from bson import ObjectId
from pymongo.errors import PyMongoError

from src.core.config import settings
from src.core.exceptions import DatabaseException
from src.core.logging import get_logger

logger = get_logger(__name__)


def _json_safe(value: Any) -> Any:
    """Convert MongoDB-returned values into JSON-serializable primitives."""
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


class MongoDBService:
    """
    MongoDB service for managing conversations, messages, and session context.
    Uses Motor for async MongoDB operations.
    """

    def __init__(self):
        self._client: Optional[AsyncIOMotorClient] = None
        self._db: Optional[AsyncIOMotorDatabase] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize MongoDB connection"""
        if self._initialized:
            logger.debug("MongoDB already initialized, skipping")
            return

        try:
            from src.core.config import settings

            mongodb_uri = settings.database.MONGODB_DATABASE_URI or settings.database.MONGODB_URI
            if not mongodb_uri:
                raise ValueError("MONGODB_DATABASE_URI environment variable not set")

            logger.info("Initializing MongoDB connection")

            self._client = AsyncIOMotorClient(
                mongodb_uri,
                serverSelectionTimeoutMS=5000,
                maxPoolSize=10,
                minPoolSize=1,
            )

            # Test connection
            await self._client.admin.command("ping")

            # Get database name from environment or use default
            db_name = settings.database.MONGODB_DATABASE_NAME or settings.database.MONGODB_DATABASE
            self._db = self._client[db_name]

            self._initialized = True
            logger.info("MongoDB connection initialized successfully", database=db_name)

        except Exception as e:
            logger.error("Failed to initialize MongoDB connection", error=e)
            self._client = None
            self._db = None
            self._initialized = False
            raise DatabaseException(
                message=f"Failed to connect to MongoDB: {str(e)}",
                operation="initialize"
            )

    async def close(self) -> None:
        """Close MongoDB connection"""
        if self._client:
            try:
                logger.info("Closing MongoDB connection")
                self._client.close()
                self._client = None
                self._db = None
                self._initialized = False
                logger.info("MongoDB connection closed successfully")
            except Exception as e:
                logger.error("Error closing MongoDB connection", error=e)
                raise

    @property
    def db(self) -> AsyncIOMotorDatabase:
        """Get MongoDB database instance"""
        if not self._initialized or self._db is None:
            raise DatabaseException(
                message="MongoDB not initialized. Call initialize() first.",
                operation="get_db"
            )
        return self._db

    # ========================================================================
    # Session Management
    # ========================================================================

    async def create_session(
        self,
        workspace_id: int,
        user_id: int,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Create a new conversation session.

        Args:
            workspace_id: Workspace identifier
            user_id: User identifier
            session_id: Optional custom session ID

        Returns:
            Session ID
        """
        try:
            if session_id is None:
                session_id = str(uuid.uuid4())

            session_doc = {
                "session_id": session_id,
                "workspace_id": workspace_id,
                "user_id": user_id,
                "title": None,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
                "message_count": 0,
                "is_active": True,
            }

            await self.db.sessions.insert_one(session_doc)

            logger.info(
                "Created conversation session",
                session_id=session_id,
                workspace_id=workspace_id,
                user_id=user_id,
            )

            return session_id

        except PyMongoError as e:
            logger.error("Failed to create session", error=e)
            raise DatabaseException(
                message=f"Failed to create session: {str(e)}",
                operation="create_session"
            )

    async def get_session(
        self,
        session_id: str,
        workspace_id: int,
        user_id: int,
    ) -> Optional[Dict[str, Any]]:
        """Get session metadata"""
        try:
            session = await self.db.sessions.find_one({
                "session_id": session_id,
                "workspace_id": workspace_id,
                "user_id": user_id,
            })
            return session
        except PyMongoError as e:
            logger.error("Failed to get session", error=e, session_id=session_id)
            raise DatabaseException(
                message=f"Failed to get session: {str(e)}",
                operation="get_session"
            )

    async def update_session_title(
        self,
        session_id: str,
        workspace_id: int,
        user_id: int,
        title: str,
    ) -> bool:
        """Update session title"""
        try:
            result = await self.db.sessions.update_one(
                {
                    "session_id": session_id,
                    "workspace_id": workspace_id,
                    "user_id": user_id,
                },
                {
                    "$set": {
                        "title": title,
                        "updated_at": datetime.now(timezone.utc),
                    }
                }
            )

            success = result.modified_count > 0
            if success:
                logger.info("Updated session title", session_id=session_id, title=title)
            else:
                logger.warning("Session not found for title update", session_id=session_id)

            return success

        except PyMongoError as e:
            logger.error("Failed to update session title", error=e)
            raise DatabaseException(
                message=f"Failed to update session title: {str(e)}",
                operation="update_session_title"
            )

    async def delete_session(
        self,
        session_id: str,
        workspace_id: int,
        user_id: int,
    ) -> bool:
        """Delete a session and all its messages"""
        try:
            # Delete all messages first
            await self.db.messages.delete_many({
                "session_id": session_id,
                "workspace_id": workspace_id,
                "user_id": user_id,
            })

            # Delete session
            result = await self.db.sessions.delete_one({
                "session_id": session_id,
                "workspace_id": workspace_id,
                "user_id": user_id,
            })

            success = result.deleted_count > 0
            if success:
                logger.info("Deleted session", session_id=session_id)
            else:
                logger.warning("Session not found for deletion", session_id=session_id)

            return success

        except PyMongoError as e:
            logger.error("Failed to delete session", error=e)
            raise DatabaseException(
                message=f"Failed to delete session: {str(e)}",
                operation="delete_session"
            )

    async def list_sessions(
        self,
        workspace_id: int,
        user_id: int,
        limit: int = 50,
        skip: int = 0,
    ) -> List[Dict[str, Any]]:
        """List user's conversation sessions"""
        try:
            cursor = self.db.sessions.find(
                {
                    "workspace_id": workspace_id,
                    "user_id": user_id,
                    "is_active": True,
                },
                {
                    "_id": 0,
                    "session_id": 1,
                    "title": 1,
                    "message_count": 1,
                    "created_at": 1,
                    "updated_at": 1,
                }
            ).sort("updated_at", -1).skip(skip).limit(limit)

            sessions = await cursor.to_list(length=limit)

            logger.info(
                "Listed sessions",
                workspace_id=workspace_id,
                user_id=user_id,
                count=len(sessions),
            )

            return [_json_safe(s) for s in sessions]

        except PyMongoError as e:
            logger.error("Failed to list sessions", error=e)
            raise DatabaseException(
                message=f"Failed to list sessions: {str(e)}",
                operation="list_sessions"
            )

    # ========================================================================
    # Message Management
    # ========================================================================

    async def append_message(
        self,
        session_id: str,
        workspace_id: int,
        user_id: int,
        role: str,  # "user" or "assistant"
        content: str,
        sources: Optional[List[Dict]] = None,
        task_ids: Optional[List[int]] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Append a message to a conversation session.

        Args:
            session_id: Session identifier
            workspace_id: Workspace identifier
            user_id: User identifier
            role: Message role ("user" or "assistant")
            content: Message content
            sources: Optional source documents
            task_ids: Optional task IDs
            metadata: Optional additional metadata

        Returns:
            Message ID
        """
        try:
            message_id = str(uuid.uuid4())
            message_doc = {
                "message_id": message_id,
                "session_id": session_id,
                "workspace_id": workspace_id,
                "user_id": user_id,
                "role": role,
                "content": content,
                "sources": sources or [],
                "task_ids": task_ids or [],
                "metadata": metadata or {},
                "timestamp": datetime.now(timezone.utc),
            }

            await self.db.messages.insert_one(message_doc)

            # Update session message count and timestamp
            await self.db.sessions.update_one(
                {
                    "session_id": session_id,
                    "workspace_id": workspace_id,
                    "user_id": user_id,
                },
                {
                    "$inc": {"message_count": 1},
                    "$set": {"updated_at": datetime.now(timezone.utc)},
                }
            )

            logger.debug(
                "Appended message",
                session_id=session_id,
                message_id=message_id,
                role=role,
            )

            return message_id

        except PyMongoError as e:
            logger.error("Failed to append message", error=e)
            raise DatabaseException(
                message=f"Failed to append message: {str(e)}",
                operation="append_message"
            )

    async def get_conversation_history(
        self,
        session_id: str,
        workspace_id: int,
        user_id: int,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session.

        Args:
            session_id: Session identifier
            workspace_id: Workspace identifier
            user_id: User identifier
            limit: Optional limit on number of messages

        Returns:
            List of messages
        """
        try:
            query = {
                "session_id": session_id,
                "workspace_id": workspace_id,
                "user_id": user_id,
            }

            cursor = self.db.messages.find(
                query,
                {"_id": 0}
            ).sort("timestamp", 1)

            if limit:
                # Get last N messages
                cursor = cursor.skip(max(0, await self.db.messages.count_documents(query) - limit))

            messages = await cursor.to_list(length=limit if limit else None)

            logger.debug(
                "Retrieved conversation history",
                session_id=session_id,
                message_count=len(messages),
            )

            return [_json_safe(m) for m in messages]

        except PyMongoError as e:
            logger.error("Failed to get conversation history", error=e)
            raise DatabaseException(
                message=f"Failed to get conversation history: {str(e)}",
                operation="get_conversation_history"
            )

    # ========================================================================
    # Context Management
    # ========================================================================

    async def save_context(
        self,
        session_id: str,
        workspace_id: int,
        user_id: int,
        context_data: Dict[str, Any],
    ) -> None:
        """Save session context data"""
        try:
            await self.db.context.update_one(
                {
                    "session_id": session_id,
                    "workspace_id": workspace_id,
                    "user_id": user_id,
                },
                {
                    "$set": {
                        "context_data": context_data,
                        "updated_at": datetime.now(timezone.utc),
                    }
                },
                upsert=True,
            )

            logger.debug("Saved session context", session_id=session_id)

        except PyMongoError as e:
            logger.error("Failed to save context", error=e)
            raise DatabaseException(
                message=f"Failed to save context: {str(e)}",
                operation="save_context"
            )

    async def load_context(
        self,
        session_id: str,
        workspace_id: int,
        user_id: int,
    ) -> Optional[Dict[str, Any]]:
        """Load session context data"""
        try:
            context_doc = await self.db.context.find_one({
                "session_id": session_id,
                "workspace_id": workspace_id,
                "user_id": user_id,
            })

            if context_doc:
                return context_doc.get("context_data")
            return None

        except PyMongoError as e:
            logger.error("Failed to load context", error=e)
            raise DatabaseException(
                message=f"Failed to load context: {str(e)}",
                operation="load_context"
            )


# ============================================================================
# Singleton Instance
# ============================================================================

_mongodb_service_instance: Optional[MongoDBService] = None


def get_mongodb_service() -> MongoDBService:
    """Get or create singleton MongoDB service instance"""
    global _mongodb_service_instance
    if _mongodb_service_instance is None:
        _mongodb_service_instance = MongoDBService()
    return _mongodb_service_instance
