"""Database connection and models for PostgreSQL - KB REST Service Only"""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text, JSON
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

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

    workspace_id = Column(Integer, primary_key=True)
    workspace_name = Column(String(100), nullable=False)
    workspace_desc = Column(Text)
    namespace = Column(String(100))
    is_active = Column(Boolean, default=True)
    created_date = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_updated = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class FileTask(Base):
    """
    File/Document indexing task tracking.
    Stores status of documents being indexed.
    """
    __tablename__ = "file_tasks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    container_name = Column(String(255))
    upload_path = Column(String(500))
    domain = Column(String(255))
    kb_name = Column(String(255))
    file_path = Column(String(500))
    file_name = Column(String(255), nullable=False)
    workspace_id = Column(Integer, nullable=False, index=True)
    status = Column(String(50), default="pending", index=True)  # pending, processing, completed, failed
    file_size = Column(Integer)
    uploaded_by = Column(String(255))
    error_message = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class DocumentMetadata(Base):
    """
    Indexed document metadata.
    Stores information about documents that have been indexed into LightRAG.
    """
    __tablename__ = "document_metadata"

    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(String(255), unique=True, nullable=False, index=True)
    file_name = Column(String(255), nullable=False)
    workspace_id = Column(Integer, nullable=False, index=True)
    file_path = Column(String(500))
    file_size = Column(Integer)
    chunk_count = Column(Integer)
    metadata = Column(JSON)  # Additional metadata as JSON
    indexed_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class ConversationSession(Base):
    """
    Conversation/chat session metadata.
    Stores conversation sessions for chatbot functionality.
    Note: Actual messages stored in MongoDB for performance.
    """
    __tablename__ = "conversation_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), unique=True, nullable=False, index=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    title = Column(String(500))
    message_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


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
