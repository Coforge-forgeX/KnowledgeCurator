"""Database connection and models for PostgreSQL - KB REST Service Only"""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional

from sqlalchemy import Boolean, DateTime, Integer, String, Text, JSON
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from .config import settings
from .logging import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """Base class for all database models"""
    pass


# ============================================================================
# KB REST Service Models - Only what this service needs
# ============================================================================


class Workspace(Base):
    """
    Workspace model - Basic workspace information for authorization.
    Note: Full workspace management is in user-management service.
    """
    __tablename__ = "workspace_master"

    workspace_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    workspace_name: Mapped[str] = mapped_column(String(100), nullable=False)
    workspace_desc: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    namespace: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    keywords: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    workspace_logo: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_date: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.utcnow())
    last_updated: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.utcnow())


class FileTask(Base):
    """
    File/Document indexing task tracking.
    Stores status of documents being uploaded and indexed.

    Status values:
    - uploading: File being uploaded (default)
    - pending: Queued for processing
    - processing: Currently being indexed
    - indexed: Successfully indexed
    - completed: Fully completed
    - failed: Indexing failed (see error_message)
    """
    __tablename__ = "file_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    container_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, comment="Azure blob container name")
    upload_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="Upload path prefix in blob storage")
    file_path: Mapped[str] = mapped_column(Text, nullable=False, comment="Full blob path")
    workspace_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True, comment="Workspace ID (INTEGER)")
    status: Mapped[str] = mapped_column(
        String(255),
        default="uploading",
        index=True,
        comment="Task status: uploading, pending, processing, indexed, completed, failed"
    )
    file_size: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, comment="Human-readable file size (e.g., '144.93 KB')")
    uploaded_by: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, comment="User full name who uploaded")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
        comment="Task creation timestamp"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        comment="Last update timestamp"
    )
    domain: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, comment="Domain name")
    kb_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, comment="Knowledge base name")
    # Linking columns (added via migration 001)
    # file_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True, comment="Original file name")
    # full_doc_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True, comment="Unique document ID for linking to lightrag")


class DocumentMetadata(Base):
    """
    Master document metadata record (created via migration 002).
    Links file_tasks to lightrag_vdb_chunks via full_doc_id.
    Provides normalized document tracking with proper relationships.
    """
    __tablename__ = "document_metadata"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    full_doc_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True, comment="Unique document ID linking to lightrag_vdb_chunks")
    file_task_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True, comment="Reference to file_tasks.id")
    workspace_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True, comment="Workspace ID")
    kb_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True, comment="Knowledge base ID - populated only for documents uploaded to KG workspaces for sharing across workspaces")
    file_name: Mapped[str] = mapped_column(String(255), nullable=False, comment="Original filename")
    file_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True, comment="Full blob path")
    file_size_bytes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, comment="File size in bytes")
    content_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True, comment="MD5/SHA256 hash for deduplication")
    total_chunks: Mapped[int] = mapped_column(Integer, default=0, comment="Number of chunks in lightrag_vdb_chunks")
    doc_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, comment="Document type: pdf, docx, txt, etc.")
    indexed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True, index=True, comment="When indexing completed")
    doc_metadata: Mapped[Optional[dict]] = mapped_column("metadata", JSON, nullable=True, comment="Additional flexible metadata")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), comment="Creation timestamp")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), comment="Last update timestamp")


class ConversationSession(Base):
    """
    Conversation/chat session metadata.
    Stores conversation sessions for chatbot functionality.
    Note: Actual messages stored in MongoDB for performance.
    """
    __tablename__ = "conversation_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    workspace_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    title: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class User(Base):
    """
    User model - Basic user information.
    Note: Full user management is in user-management service.
    """
    __tablename__ = "users"

    user_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    namespace: Mapped[str] = mapped_column(String(100), nullable=False)
    email_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    first_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    last_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    password: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    role_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    created_date: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_updated: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class Role(Base):
    """
    Role model - Role definitions.
    """
    __tablename__ = "role_master"

    role_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    role_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    role_desc: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    workflow_stage: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_date: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_updated: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class UserMap(Base):
    """
    User-Workspace mapping with role assignment.
    Maps users to workspaces with specific roles.
    """
    __tablename__ = "workspace_users_mapping"

    user_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    workspace_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    role_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    namespace: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    can_curate_kb: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_date: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_updated: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class Industry(Base):
    """
    Industry master table.
    """
    __tablename__ = "industry_master"

    industry_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    industry_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_date: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_updated: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class SubIndustry(Base):
    """
    Sub-industry master table.
    """
    __tablename__ = "subindustry_master"

    subindustry_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    subindustry_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    industry_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_date: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_updated: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class KnowledgeBase(Base):
    """
    Knowledge base master table.
    """
    __tablename__ = "knowledge_base_master"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    industry_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    sub_industry_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.utcnow())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.utcnow())


class WorkspaceIndustryIntentMap(Base):
    """
    Workspace-Industry-Intent-KB mapping.
    Links workspaces to knowledge bases via industry and intent.
    """
    __tablename__ = "workspace_industry_intent_mapping"

    workspace_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    industry_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    subindustry_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    intent_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    kb_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_date: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_updated: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


# ============================================================================
# Database Connection Manager
# ============================================================================


class Database:
    """Async database connection manager with connection pooling"""

    def __init__(self):
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None

    async def initialize(self) -> None:
        """Initialize database engine and session factory with connection pooling"""
        if self._engine is not None:
            logger.debug("Database already initialized, skipping")
            return

        try:
            logger.info(
                "Initializing database connection",
                pool_size=settings.database.DB_POOL_SIZE,
                max_overflow=settings.database.DB_MAX_OVERFLOW,
                pool_recycle=settings.database.DB_POOL_RECYCLE,
            )

            self._engine = create_async_engine(
                settings.database.postgresql_url,
                echo=settings.database.DB_ECHO,
                pool_size=settings.database.DB_POOL_SIZE,
                max_overflow=settings.database.DB_MAX_OVERFLOW,
                pool_pre_ping=True,
                pool_recycle=settings.database.DB_POOL_RECYCLE,
                pool_timeout=settings.database.DB_POOL_TIMEOUT,
            )

            self._session_factory = async_sessionmaker(
                self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autocommit=False,
                autoflush=False,
            )

            logger.info("Database connection initialized successfully")

        except Exception as e:
            logger.error(
                "Failed to initialize database connection",
                error=e,
                error_type=type(e).__name__,
            )
            self._engine = None
            self._session_factory = None
            raise

    async def close(self) -> None:
        """Close database engine and cleanup connections"""
        if self._engine is not None:
            try:
                logger.info("Closing database connection")
                await self._engine.dispose()
                self._engine = None
                self._session_factory = None
                logger.info("Database connection closed successfully")
            except Exception as e:
                logger.error(
                    "Error while closing database connection",
                    error=e,
                    error_type=type(e).__name__,
                )
                raise

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session with automatic cleanup"""
        if self._session_factory is None:
            logger.debug("Session factory not initialized, initializing now")
            await self.initialize()

        session = None
        try:
            async with self._session_factory() as session:
                logger.debug("Database session created")
                try:
                    yield session
                    await session.commit()
                    logger.debug("Database transaction committed successfully")
                except SQLAlchemyError as e:
                    logger.error(
                        "Database error during transaction",
                        error=e,
                        error_type=type(e).__name__,
                    )
                    await session.rollback()
                    logger.warning("Database transaction rolled back")
                    raise
                except Exception as e:
                    logger.error(
                        "Unexpected error during database transaction",
                        error=e,
                        error_type=type(e).__name__,
                    )
                    await session.rollback()
                    logger.warning(
                        "Database transaction rolled back due to unexpected error"
                    )
                    raise
        except Exception as e:
            logger.error(
                "Error creating database session",
                error=e,
                error_type=type(e).__name__,
            )
            raise
        finally:
            if session:
                logger.debug("Database session closed")

    async def create_tables(self) -> None:
        """Create all tables defined in Base"""
        try:
            if self._engine is None:
                await self.initialize()

            logger.info("Creating database tables")
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")

        except Exception as e:
            logger.error(
                "Failed to create database tables",
                error=e,
                error_type=type(e).__name__,
            )
            raise

    async def drop_tables(self) -> None:
        """Drop all tables defined in Base"""
        try:
            if self._engine is None:
                await self.initialize()

            logger.warning(
                "Dropping all database tables - this is a destructive operation"
            )
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            logger.info("Database tables dropped successfully")

        except Exception as e:
            logger.error(
                "Failed to drop database tables",
                error=e,
                error_type=type(e).__name__,
            )
            raise

    @property
    def engine(self) -> Optional[AsyncEngine]:
        """Get database engine"""
        return self._engine


# ============================================================================
# Global Database Instance & Helpers
# ============================================================================

# Global database instance
db = Database()

# Async session for async operations
get_async_session = db.get_session


def get_database() -> Database:
    """Get global database instance"""
    return db


async def get_postgres_pool():
    """Get PostgreSQL connection pool (engine)"""
    if db._engine is None:
        await db.initialize()
    return db.engine
