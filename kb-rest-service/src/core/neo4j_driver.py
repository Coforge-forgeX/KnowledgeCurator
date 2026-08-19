"""Neo4j database driver for KB REST service"""
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional

from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncSession

from .config import settings
from .logging import Logger

logger = Logger("neo4j_driver")


class Neo4jDriver:
    """
    Neo4j database driver class for managing connections and executing queries.

    This class provides a centralized way to manage Neo4j database connections
    with proper error handling and resource management.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None
    ):
        """
        Initialize Neo4j driver with connection parameters.

        Args:
            uri: Neo4j database URI. Defaults to settings
            username: Database username. Defaults to settings
            password: Database password. Defaults to settings
        """
        self.uri = uri or settings.database.NEO4J_URI
        self.username = username or settings.database.NEO4J_USER
        self.password = password or settings.database.NEO4J_PASSWORD

        self._driver: Optional[AsyncDriver] = None

        logger.info(f"Neo4j driver initialized with URI: {self.uri}")

    async def connect(self) -> None:
        """
        Establish connection to Neo4j database.

        Raises:
            Exception: If connection fails
        """
        try:
            self._driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                max_connection_pool_size=settings.database.DB_POOL_SIZE,
                max_connection_lifetime=settings.database.DB_POOL_RECYCLE,
            )

            # Test the connection
            await self.verify_connectivity()
            logger.info("Successfully connected to Neo4j database")

        except Exception as e:
            logger.error("Failed to connect to Neo4j database", error=e)
            raise

    async def close(self) -> None:
        """Close the Neo4j driver connection"""
        if self._driver:
            await self._driver.close()
            self._driver = None
            logger.info("Neo4j driver connection closed")

    async def verify_connectivity(self) -> bool:
        """
        Verify that the database connection is working.

        Returns:
            bool: True if connection is successful

        Raises:
            RuntimeError: If driver not initialized
            Exception: If connection verification fails
        """
        if not self._driver:
            raise RuntimeError("Driver not initialized. Call connect() first.")

        try:
            await self._driver.verify_connectivity()
            return True
        except Exception as e:
            logger.error("Neo4j connectivity verification failed", error=e)
            raise

    @asynccontextmanager
    async def session(self, database: Optional[str] = None) -> AsyncGenerator[AsyncSession, None]:
        """
        Create an async context manager for Neo4j sessions.

        Args:
            database: Optional database name

        Yields:
            AsyncSession: Neo4j async session

        Raises:
            RuntimeError: If driver is not connected
        """
        if not self._driver:
            raise RuntimeError("Driver not connected. Call connect() first.")

        session = self._driver.session(database=database)
        try:
            yield session
        finally:
            await session.close()

    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and return results.

        Args:
            query: Cypher query string
            parameters: Query parameters
            database: Optional database name

        Returns:
            List[Dict[str, Any]]: Query results

        Raises:
            RuntimeError: If driver is not connected
            Exception: If query execution fails
        """
        if not self._driver:
            await self.connect()
        else:
            try:
                # Health check for serverless environments
                await self.verify_connectivity()
            except Exception as e:
                logger.warning("Neo4j connection lost, reconnecting", error=e)
                await self.close()
                await self.connect()

        try:
            async with self.session(database=database) as session:
                result = await session.run(query, parameters or {})
                records = await result.data()
                return records

        except Exception as e:
            logger.error("Query execution failed", error=e, query=query, parameters=parameters)
            raise

    async def execute_write_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a write transaction query.

        Args:
            query: Cypher query string
            parameters: Query parameters
            database: Optional database name

        Returns:
            List[Dict[str, Any]]: Query results

        Raises:
            RuntimeError: If driver is not connected
            Exception: If query execution fails
        """
        if not self._driver:
            await self.connect()
        else:
            try:
                # Health check for serverless environments
                await self.verify_connectivity()
            except Exception as e:
                logger.warning("Neo4j connection lost, reconnecting", error=e)
                await self.close()
                await self.connect()

        try:
            async with self.session(database=database) as session:
                result = await session.execute_write(
                    self._execute_query_tx, query, parameters or {}
                )
                return result

        except Exception as e:
            logger.error("Write query execution failed", error=e, query=query, parameters=parameters)
            raise

    async def execute_read_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a read transaction query.

        Args:
            query: Cypher query string
            parameters: Query parameters
            database: Optional database name

        Returns:
            List[Dict[str, Any]]: Query results

        Raises:
            RuntimeError: If driver is not connected
            Exception: If query execution fails
        """
        if not self._driver:
            await self.connect()
        else:
            try:
                # Health check for serverless environments
                await self.verify_connectivity()
            except Exception as e:
                logger.warning("Neo4j connection lost, reconnecting", error=e)
                await self.close()
                await self.connect()

        try:
            async with self.session(database=database) as session:
                result = await session.execute_read(
                    self._execute_query_tx, query, parameters or {}
                )
                return result

        except Exception as e:
            logger.error("Read query execution failed", error=e, query=query, parameters=parameters)
            raise

    @staticmethod
    async def _execute_query_tx(tx, query: str, parameters: Dict[str, Any]):
        """
        Helper method for transaction execution.

        Args:
            tx: Transaction object
            query: Cypher query string
            parameters: Query parameters

        Returns:
            List[Dict[str, Any]]: Query results
        """
        result = await tx.run(query, parameters)
        return await result.data()

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()


# Singleton instance for global use
_neo4j_driver_instance: Optional[Neo4jDriver] = None


def get_neo4j_driver() -> Neo4jDriver:
    """
    Get or create a singleton Neo4j driver instance.

    Returns:
        Neo4j driver: Singleton driver instance
    """
    global _neo4j_driver_instance

    if _neo4j_driver_instance is None:
        _neo4j_driver_instance = Neo4jDriver()

    return _neo4j_driver_instance


async def initialize_neo4j_driver(
    uri: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None
) -> Neo4jDriver:
    """
    Initialize and connect the global Neo4j driver instance.

    Args:
        uri: Neo4j database URI
        username: Database username
        password: Database password

    Returns:
        Neo4jDriver: Connected driver instance
    """
    global _neo4j_driver_instance

    _neo4j_driver_instance = Neo4jDriver(uri, username, password)
    await _neo4j_driver_instance.connect()

    return _neo4j_driver_instance


async def close_neo4j_driver() -> None:
    """Close the global Neo4j driver instance"""
    global _neo4j_driver_instance

    if _neo4j_driver_instance:
        await _neo4j_driver_instance.close()
        _neo4j_driver_instance = None
