"""Database connection management for Indexer Service"""
import logging
from typing import Optional

from neo4j import AsyncGraphDatabase, AsyncDriver
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from core.config import settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections for PostgreSQL and Neo4j"""

    def __init__(self):
        self.pg_engine: Optional[AsyncEngine] = None
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
        """Initialize PostgreSQL async engine"""
        try:
            # Use asyncpg driver for async operations
            url = settings.database.postgresql_url.replace(
                "postgresql://", "postgresql+asyncpg://"
            )
            self.pg_engine = create_async_engine(
                url,
                pool_size=10,
                max_overflow=20,
                pool_timeout=30,
                pool_recycle=3600,
                echo=settings.DEBUG,
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
