"""Database connection and models for PostgreSQL and Neo4j - Indexer Service"""
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional

from neo4j import AsyncGraphDatabase, AsyncDriver
from sqlalchemy import Boolean, DateTime, Integer, String, Text, JSON
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from core.config import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base class for all database models"""
    pass


# ============================================================================
# Indexer Service Models
# ============================================================================


class IndexingJob(Base):
    """
    Indexing job state tracking for retry/resume functionality.
    Stores checkpoint data for resumable indexing operations.

    State values:
    - pending: Queued, not started
    - downloading: Downloading file from blob storage
    - downloaded: File downloaded successfully
    - extracting: Extracting text from document
    - extracted: Text extraction complete
    - indexing: Indexing into LightRAG
    - indexed: Indexing complete
    - updating_metadata: Updating metadata tables
    - completed: Job finished successfully
    - failed: Job failed (see last_error)
    - retrying: Job being retried after failure
    """
    __tablename__ = "indexing_jobs"

    job_id: Mapped[str] = mapped_column(String(255), primary_key=True, comment="Unique job identifier from queue message")
    workspace_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True, comment="Workspace ID")
    document_url: Mapped[str] = mapped_column(Text, nullable=False, comment="Document URL/path")
    kb_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, comment="Knowledge base ID")

    # State tracking
    state: Mapped[str] = mapped_column(String(50), nullable=False, default="pending", index=True, comment="Current state")
    checkpoint_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True, default={}, comment="JSON checkpoint data for resume")

    # Retry tracking
    retry_count: Mapped[int] = mapped_column(Integer, default=0, comment="Number of retry attempts")
    last_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="Last error message")

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), index=True)


class FileTask(Base):
    """
    File/Document indexing task tracking (shared with kb-rest-service).
    Stores status of documents being indexed.

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
        DateTime(timezone=False),
        default=datetime.utcnow,
        index=True,
        comment="Task creation timestamp"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        comment="Last update timestamp"
    )
    domain: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, comment="Domain name")
    kb_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, comment="Knowledge base name")
    # Linking columns (added via migration 001)
    # file_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True, comment="Original file name")
    # full_doc_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True, comment="Unique document ID for linking to lightrag")


class DocumentMetadata(Base):
    """
    Master document metadata record.
    Links file_tasks to lightrag_vdb_chunks via full_doc_id.
    """
    __tablename__ = "document_metadata"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    full_doc_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    file_task_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    workspace_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    kb_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True, comment="Knowledge base ID - populated only for documents uploaded to KG workspaces for sharing across workspaces")
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    file_size_bytes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    content_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    total_chunks: Mapped[int] = mapped_column(Integer, default=0)
    doc_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    doc_metadata: Mapped[Optional[dict]] = mapped_column("metadata", JSON, nullable=True)
    indexed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class Workspace(Base):
    """Workspace model used for workspace type checks."""
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


class WorkspaceIndustryIntentMap(Base):
    """Workspace-to-KB mapping model used for kb_id resolution."""
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


class DatabaseManager:
    """Manages database connections for PostgreSQL (SQLAlchemy 2.0) and Neo4j"""

    def __init__(self):
        self.pg_engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None
        self.neo4j_driver: Optional[AsyncDriver] = None
        self._initialized = False

    async def initialize(self):
        """Initialize database connections"""
        if self._initialized:
            return

        logger.info("Initializing database connections")

        # Initialize PostgreSQL
        if settings.database.postgresql_url:
            await self._initialize_postgresql()

        # Initialize Neo4j
        if settings.database.neo4j_uri:
            await self._initialize_neo4j()

        self._initialized = True
        logger.info("Database connections initialized successfully")

    async def _initialize_postgresql(self):
        """Initialize PostgreSQL async engine with SQLAlchemy 2.0"""
        try:
            # Use asyncpg driver for async operations
            url = settings.database.postgresql_url.replace(
                "postgresql://", "postgresql+asyncpg://"
            )
            self.pg_engine = create_async_engine(
                url,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_timeout=30,
                pool_recycle=3600,
                echo=settings.DEBUG,
            )

            # Create session factory
            self._session_factory = async_sessionmaker(
                self.pg_engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autocommit=False,
                autoflush=False,
            )

            logger.info("PostgreSQL connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL: {e}")
            raise

    async def _initialize_neo4j(self):
        """Initialize Neo4j async driver"""
        try:
            self.neo4j_driver = AsyncGraphDatabase.driver(
                settings.database.neo4j_uri,
                auth=(
                    settings.database.NEO4J_DATABASE_NEO4J_USER,
                    settings.database.NEO4J_DATABASE_NEO4J_PASSWORD,
                ),
                max_connection_pool_size=50,
                connection_timeout=30.0,
            )
            # Verify connectivity
            await self.neo4j_driver.verify_connectivity()
            logger.info("Neo4j driver initialized and connectivity verified")
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j: {e}")
            raise

    async def close(self):
        """Close all database connections"""
        logger.info("Closing database connections")

        if self.pg_engine:
            await self.pg_engine.dispose()
            logger.info("PostgreSQL connection pool closed")

        if self.neo4j_driver:
            await self.neo4j_driver.close()
            logger.info("Neo4j driver closed")

        self._initialized = False
        self._session_factory = None

    def get_pg_engine(self) -> AsyncEngine:
        """Get PostgreSQL async engine"""
        if not self.pg_engine:
            raise RuntimeError("PostgreSQL engine not initialized")
        return self.pg_engine

    def get_neo4j_driver(self) -> AsyncDriver:
        """Get Neo4j async driver"""
        if not self.neo4j_driver:
            raise RuntimeError("Neo4j driver not initialized")
        return self.neo4j_driver

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get async database session with automatic transaction management.

        Usage:
            async with db_manager.get_session() as session:
                result = await session.execute(select(User))
                # session.commit() called automatically on success
        """
        if self._session_factory is None:
            logger.debug("Session factory not initialized, initializing now")
            await self.initialize()

        if self._session_factory is None:
            raise RuntimeError("Failed to initialize session factory")

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
                        "Database error during transaction (%s): %s",
                        type(e).__name__,
                        str(e),
                    )
                    await session.rollback()
                    logger.warning("Database transaction rolled back")
                    raise
                except Exception as e:
                    logger.error(
                        "Unexpected error during database transaction (%s): %s",
                        type(e).__name__,
                        str(e),
                    )
                    await session.rollback()
                    logger.warning("Database transaction rolled back due to unexpected error")
                    raise
        except Exception as e:
            logger.error(
                "Error creating database session (%s): %s",
                type(e).__name__,
                str(e),
            )
            raise
        finally:
            if session:
                logger.debug("Database session closed")

    async def create_tables(self) -> None:
        """Create all tables defined in Base"""
        try:
            if self.pg_engine is None:
                await self.initialize()

            if self.pg_engine is None:
                raise RuntimeError("Failed to initialize database engine")

            logger.info("Creating database tables")
            async with self.pg_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")

        except Exception as e:
            logger.error(
                "Failed to create database tables (%s): %s",
                type(e).__name__,
                str(e),
            )
            raise


# ============================================================================
# Global Database Instance & Helpers
# ============================================================================

# Global database manager instance
db_manager = DatabaseManager()


async def get_pg_engine() -> AsyncEngine:
    """Helper function to get PostgreSQL engine"""
    if not db_manager._initialized:
        await db_manager.initialize()
    return db_manager.get_pg_engine()


async def get_neo4j_driver() -> AsyncDriver:
    """Helper function to get Neo4j driver"""
    if not db_manager._initialized:
        await db_manager.initialize()
    return db_manager.get_neo4j_driver()


def get_async_session():
    """Helper function to get async database session context manager"""
    return db_manager.get_session()
