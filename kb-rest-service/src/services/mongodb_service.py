"""
MongoDB Service for Conversation/Chat History Management

Handles conversation sessions, messages, and context storage.
"""
import asyncio
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
from src.core.redis import invalidate_conversation_cache

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


def _id_match(value: int) -> Dict[str, Any]:
    """
    Match an id stored either as an int (this service) or as a string
    (legacy KnowledgeCurator writes), so both generations of documents are
    readable without a migration.

    Writes always use the canonical int form; only queries widen.
    """
    return {"$in": [value, str(value)]}


def _scope(session_id: str, workspace_id: int, user_id: int) -> Dict[str, Any]:
    """Standard (session, workspace, user) filter, tolerant of legacy string ids."""
    return {
        "session_id": session_id,
        "workspace_id": _id_match(workspace_id),
        "user_id": _id_match(user_id),
    }


def _normalize_message(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Bring a stored message up to the current schema.

    Legacy documents use `tasks` instead of `task_ids`, carry no `message_id`,
    and store workspace_id/user_id as strings. Normalizing on read means
    callers see one shape regardless of which app wrote the document.
    """
    doc = _json_safe(doc)

    if "task_ids" not in doc:
        doc["task_ids"] = doc.pop("tasks", []) or []
    else:
        doc.pop("tasks", None)

    doc.setdefault("message_id", None)
    doc.setdefault("sources", [])
    doc.setdefault("metadata", {})

    for key in ("workspace_id", "user_id"):
        if isinstance(doc.get(key), str):
            try:
                doc[key] = int(doc[key])
            except ValueError:
                pass

    return doc


class MongoDBService:
    """
    MongoDB service for managing conversations, messages, and session context.
    Uses Motor for async MongoDB operations.

    Collections:
        kb_session       - one document per conversation (title, message_count, ...)
        kb_chat_history  - one document per message (role, content, sources, ...)
    """

    SESSION_COLLECTION = "kb_session"
    CHAT_HISTORY_COLLECTION = "kb_chat_history"

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

            mongodb_uri = settings.database.MONGODB_URI
            if not mongodb_uri:
                raise ValueError("MONGODB_URI environment variable not set")

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
            db_name = settings.database.MONGODB_DATABASE
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

    @property
    def sessions(self):
        """Session collection"""
        return self.db[self.SESSION_COLLECTION]

    @property
    def chat_history(self):
        """Chat history collection"""
        return self.db[self.CHAT_HISTORY_COLLECTION]

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
                "title_set_by_user": False,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
                "message_count": 0,
                "is_active": True,
            }

            await self.sessions.insert_one(session_doc)

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

    async def ensure_session(
        self,
        session_id: str,
        workspace_id: int,
        user_id: int,
    ) -> None:
        """
        Make sure a `kb_session` document exists for this (session, workspace, user).

        message_gpt accepts a client-supplied `session_id` and is often called
        without a prior `start_conversation`. Without this upsert the messages
        land in `kb_chat_history` but no session document exists, so
        `list_sessions` returns nothing and `load_conversation` raises
        "Session not found" — the conversation looks like it was never stored.
        """
        try:
            now = datetime.now(timezone.utc)
            await self.sessions.update_one(
                _scope(session_id, workspace_id, user_id),
                {
                    "$setOnInsert": {
                        "session_id": session_id,
                        "workspace_id": workspace_id,
                        "user_id": user_id,
                        "title": None,
                        "title_set_by_user": False,
                        "created_at": now,
                        "message_count": 0,
                        "is_active": True,
                    },
                    "$set": {"updated_at": now},
                },
                upsert=True,
            )
        except PyMongoError as e:
            logger.error("Failed to ensure session", error=e, session_id=session_id)
            raise DatabaseException(
                message=f"Failed to ensure session: {str(e)}",
                operation="ensure_session",
            )

    async def get_session(
        self,
        session_id: str,
        workspace_id: int,
        user_id: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Get session metadata.

        Falls back to deriving the metadata from `kb_chat_history` when no
        session document exists. Legacy conversations were written without one,
        so without this fallback they are unloadable even though their messages
        are present.
        """
        try:
            session = await self.sessions.find_one(
                _scope(session_id, workspace_id, user_id),
                {"_id": 0},
            )
            if session:
                return _json_safe(session)

            return await self._derive_session(session_id, workspace_id, user_id)
        except PyMongoError as e:
            logger.error("Failed to get session", error=e, session_id=session_id)
            raise DatabaseException(
                message=f"Failed to get session: {str(e)}",
                operation="get_session"
            )

    async def _derive_session(
        self,
        session_id: str,
        workspace_id: int,
        user_id: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Synthesize session metadata from a conversation's messages.

        Used for legacy conversations that have messages but no `kb_session`
        document. The title is the first user message, mirroring the
        auto-titling `_track_title` applies to new sessions.
        """
        docs = await self.chat_history.find(
            _scope(session_id, workspace_id, user_id),
            {"_id": 0, "role": 1, "content": 1, "timestamp": 1},
        ).sort("timestamp", 1).to_list(length=None)

        if not docs:
            return None

        title = next(
            (d.get("content", "") for d in docs if d.get("role") == "user"),
            docs[0].get("content", ""),
        ).strip()
        if len(title) > 50:
            title = title[:50] + "..."

        return _json_safe({
            "session_id": session_id,
            "workspace_id": workspace_id,
            "user_id": user_id,
            "title": title or None,
            "title_set_by_user": False,
            "created_at": docs[0].get("timestamp"),
            "updated_at": docs[-1].get("timestamp"),
            "message_count": len(docs),
            "is_active": True,
            "derived": True,
        })

    async def update_session_title(
        self,
        session_id: str,
        workspace_id: int,
        user_id: int,
        title: str,
        is_manual: bool = True,
    ) -> bool:
        """
        Update session title.

        Args:
            is_manual: True for an explicit user rename (sticky — marks the
                session so later auto-title updates leave it alone). False
                for the automatic "title tracks last user message" behavior,
                which is a no-op once the session has been manually renamed.
        """
        try:
            query: Dict[str, Any] = _scope(session_id, workspace_id, user_id)
            if not is_manual:
                # Atomic guard: only auto-update when the user hasn't renamed
                # this session themselves. Avoids a separate read-then-write.
                query["title_set_by_user"] = {"$ne": True}

            update: Dict[str, Any] = {
                "$set": {
                    "title": title,
                    "updated_at": datetime.now(timezone.utc),
                }
            }
            if is_manual:
                update["$set"]["title_set_by_user"] = True

            result = await self.sessions.update_one(query, update)

            success = result.modified_count > 0
            if success:
                logger.info(
                    "Updated session title",
                    session_id=session_id,
                    title=title,
                    is_manual=is_manual,
                )
            else:
                logger.debug(
                    "Session title not updated (not found, or manually renamed and this was an auto-update)",
                    session_id=session_id,
                    is_manual=is_manual,
                )

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
            deleted_messages = await self.chat_history.delete_many(
                _scope(session_id, workspace_id, user_id)
            )

            # Delete session
            result = await self.sessions.delete_one(
                _scope(session_id, workspace_id, user_id)
            )

            # A legacy conversation has no session document — deleting its
            # messages still counts as deleting the conversation.
            success = result.deleted_count > 0 or deleted_messages.deleted_count > 0
            if success:
                # Invalidate cache for this session
                invalidate_conversation_cache(session_id, workspace_id, user_id)
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
        """
        List a user's conversation sessions.

        Unions two sources so no conversation is invisible:
        1. `kb_session` documents (written by this service).
        2. Sessions derived from `kb_chat_history` for conversations that have
           no session document — legacy data written by the previous app.

        A real session document always wins over its derived counterpart.
        Both sources tolerate ids stored as ints or as strings.
        """
        try:
            window = skip + limit

            stored, derived = await asyncio.gather(
                self.sessions.find(
                    {
                        "workspace_id": _id_match(workspace_id),
                        "user_id": _id_match(user_id),
                        "is_active": {"$ne": False},
                    },
                    {
                        "_id": 0,
                        "session_id": 1,
                        "title": 1,
                        "message_count": 1,
                        "created_at": 1,
                        "updated_at": 1,
                    },
                ).sort("updated_at", -1).limit(window).to_list(length=window),  # noqa: E501
                self._derive_session_list(workspace_id, user_id, window),
            )

            # Normalize both sides before merging: datetimes become ISO strings
            # so the two sources are directly comparable when sorting.
            merged: Dict[str, Dict[str, Any]] = {
                s["session_id"]: s for s in derived if s.get("session_id")
            }
            merged.update({
                s["session_id"]: _json_safe(s) for s in stored if s.get("session_id")
            })

            sessions = sorted(
                merged.values(),
                key=lambda s: s.get("updated_at") or "",
                reverse=True,
            )[skip:window]

            logger.info(
                "Listed sessions",
                workspace_id=workspace_id,
                user_id=user_id,
                count=len(sessions),
            )

            return sessions

        except PyMongoError as e:
            logger.error("Failed to list sessions", error=e)
            raise DatabaseException(
                message=f"Failed to list sessions: {str(e)}",
                operation="list_sessions"
            )

    async def count_sessions(self, workspace_id: int, user_id: int) -> int:
        """
        Total conversations visible to `list_sessions`, for pagination metadata.

        Counts the union of session ids from both sources `list_sessions` merges,
        so a legacy conversation with no `kb_session` document is counted exactly
        once and the total always matches what paging through the list yields.
        """
        try:
            stored_ids, derived_ids = await asyncio.gather(
                self.sessions.distinct(
                    "session_id",
                    {
                        "workspace_id": _id_match(workspace_id),
                        "user_id": _id_match(user_id),
                        "is_active": {"$ne": False},
                    },
                ),
                self.chat_history.distinct(
                    "session_id",
                    {
                        "workspace_id": _id_match(workspace_id),
                        "user_id": _id_match(user_id),
                    },
                ),
            )

            session_ids = {sid for sid in (*stored_ids, *derived_ids) if sid}
            return len(session_ids)

        except PyMongoError as e:
            logger.error("Failed to count sessions", error=e)
            raise DatabaseException(
                message=f"Failed to count sessions: {str(e)}",
                operation="count_sessions",
            )

    async def _derive_session_list(
        self,
        workspace_id: int,
        user_id: int,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """
        Build session summaries straight from `kb_chat_history`.

        Covers legacy conversations that never got a `kb_session` document.
        The first user message becomes the title, matching how new sessions are
        auto-titled.
        """
        pipeline = [
            {
                "$match": {
                    "workspace_id": _id_match(workspace_id),
                    "user_id": _id_match(user_id),
                }
            },
            {"$sort": {"timestamp": 1}},
            {
                "$group": {
                    "_id": "$session_id",
                    "message_count": {"$sum": 1},
                    "created_at": {"$first": "$timestamp"},
                    "updated_at": {"$last": "$timestamp"},
                    "title": {
                        "$first": {
                            "$cond": [{"$eq": ["$role", "user"]}, "$content", None]
                        }
                    },
                }
            },
            {"$sort": {"updated_at": -1}},
            {"$limit": limit},
        ]

        rows = await self.chat_history.aggregate(pipeline).to_list(length=limit)

        sessions = []
        for row in rows:
            title = (row.get("title") or "").strip()
            if len(title) > 50:
                title = title[:50] + "..."
            sessions.append(_json_safe({
                "session_id": row["_id"],
                "title": title or None,
                "message_count": row.get("message_count", 0),
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at"),
            }))

        return sessions

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

            await self.chat_history.insert_one(message_doc)

            # Update session message count and timestamp
            await self.sessions.update_one(
                _scope(session_id, workspace_id, user_id),
                {
                    "$inc": {"message_count": 1},
                    "$set": {"updated_at": datetime.now(timezone.utc)},
                    # Self-healing: a message must never exist without its session
                    # document, otherwise the conversation is invisible to
                    # list_sessions / load_conversation.
                    "$setOnInsert": {
                        "session_id": session_id,
                        "workspace_id": workspace_id,
                        "user_id": user_id,
                        "title": None,
                        "title_set_by_user": False,
                        "created_at": datetime.now(timezone.utc),
                        "is_active": True,
                    },
                },
                upsert=True,
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

    async def get_messages_page(
        self,
        session_id: str,
        workspace_id: int,
        user_id: int,
        page: int = 1,
        page_size: int = 50,
        newest_first: bool = True,
    ) -> tuple[List[Dict[str, Any]], int]:
        """
        Get one page of a conversation's messages, plus the total message count.

        Args:
            newest_first: when True (the default), page 1 is the newest slice of
                the transcript and paging forward walks backwards in time — what
                a chat UI needs when opening a long conversation. The messages
                *within* each page are always returned oldest-first, so a page
                renders top-to-bottom as written either way.

        Returns:
            (messages, total_count)
        """
        try:
            query = _scope(session_id, workspace_id, user_id)
            page = max(int(page), 1)
            page_size = max(int(page_size), 1)
            skip = (page - 1) * page_size

            total_count = await self.chat_history.count_documents(query)

            messages = await (
                self.chat_history.find(query, {"_id": 0})
                .sort("timestamp", -1 if newest_first else 1)
                .skip(skip)
                .limit(page_size)
                .to_list(length=page_size)
            )

            if newest_first:
                messages.reverse()

            logger.debug(
                "Retrieved message page",
                session_id=session_id,
                page=page,
                page_size=page_size,
                message_count=len(messages),
                total_count=total_count,
            )

            return [_normalize_message(m) for m in messages], total_count

        except PyMongoError as e:
            logger.error("Failed to get message page", error=e)
            raise DatabaseException(
                message=f"Failed to get message page: {str(e)}",
                operation="get_messages_page",
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
