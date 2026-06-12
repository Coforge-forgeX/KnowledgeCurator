from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
import os
import logging
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


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
            uri: Neo4j database URI. Defaults to NEO4J_URI environment variable
            username: Database username. Defaults to NEO4J_USERNAME environment variable
            password: Database password. Defaults to NEO4J_PASSWORD environment variable
        """
        self.uri = uri or os.environ.get("NEO4J_URI", "neo4j://20.55.248.225:7687")
        self.username = username or os.environ.get("NEO4J_USERNAME", "neo4j")
        self.password = password or os.environ.get("NEO4J_PASSWORD", "")
        
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
                auth=(self.username, self.password)
            )
            
            # Test the connection
            await self.verify_connectivity()
            logger.info("Successfully connected to Neo4j database")
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j database: {str(e)}")
            raise
    
    async def close(self) -> None:
        """
        Close the Neo4j driver connection.
        """
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
            Exception: If connection verification fails
        """
        if not self._driver:
            raise RuntimeError("Driver not initialized. Call connect() first.")
        
        try:
            await self._driver.verify_connectivity()
            return True
        except Exception as e:
            logger.error(f"Neo4j connectivity verification failed: {str(e)}")
            raise
    
    @asynccontextmanager
    async def session(self, database: Optional[str] = None) -> AsyncSession:
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
            raise RuntimeError("Driver not connected. Call connect() first.")
        
        try:
            async with self.session(database=database) as session:
                result = await session.run(query, parameters or {})
                records = await result.data()
                return records
                
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            logger.error(f"Query: {query}")
            logger.error(f"Parameters: {parameters}")
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
            raise RuntimeError("Driver not connected. Call connect() first.")
        
        try:
            async with self.session(database=database) as session:
                result = await session.execute_write(
                    self._execute_query_tx, query, parameters or {}
                )
                return result
                
        except Exception as e:
            logger.error(f"Write query execution failed: {str(e)}")
            logger.error(f"Query: {query}")
            logger.error(f"Parameters: {parameters}")
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
            raise RuntimeError("Driver not connected. Call connect() first.")
        
        try:
            async with self.session(database=database) as session:
                result = await session.execute_read(
                    self._execute_query_tx, query, parameters or {}
                )
                return result
                
        except Exception as e:
            logger.error(f"Read query execution failed: {str(e)}")
            logger.error(f"Query: {query}")
            logger.error(f"Parameters: {parameters}")
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
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Singleton instance for global use
_neo4j_driver_instance: Optional[Neo4jDriver] = None


def get_neo4j_driver() -> Neo4jDriver:
    """
    Get or create a singleton Neo4j driver instance.
    
    Returns:
        Neo4jDriver: Singleton driver instance
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
    """
    Close the global Neo4j driver instance.
    """
    global _neo4j_driver_instance
    
    if _neo4j_driver_instance:
        await _neo4j_driver_instance.close()
        _neo4j_driver_instance = None