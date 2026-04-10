import logging
import threading
import certifi
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from typing import Optional, Any
import os
import re
from dotenv import load_dotenv
from common_adapters.database import DatabaseAdapterFactory, DatabaseAdapter

load_dotenv()


class MongoDBSingleton:
    """
    Thread-safe singleton implementation for database client connection.
    Supports both MongoDB and AWS DocumentDB through the adapter pattern.
    Ensures only one database client instance is created and reused across the application.
    
    The database type is determined by the DB_TYPE environment variable:
    - 'mongodb' (default): Uses standard MongoDB
    - 'documentdb': Uses AWS DocumentDB
    """
    
    _instance: Optional['MongoDBSingleton'] = None
    _lock = threading.Lock()
    _client: Optional[Any] = None  # Can be MongoDB or DocumentDB client
    _adapter: Optional[DatabaseAdapter] = None
    _db_chatbot = None
    _db_workflow = None
    _is_initialized = False
    
    def __new__(cls):
        """
        Thread-safe singleton pattern using double-checked locking.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(MongoDBSingleton, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """
        Initialize MongoDB client only once.
        """
        # Prevent re-initialization using class variable
        if MongoDBSingleton._is_initialized:
            return
            
        with self._lock:
            # Double-check with class variable
            if not MongoDBSingleton._is_initialized:
                self._initialize_connection()
    
    def _initialize_connection(self):
        """
        Initialize database connection using the appropriate adapter and centralized DBSettings config.
        Supports both MongoDB and AWS DocumentDB based on DB_TYPE environment variable.
        """
        from common_adapters.database.config import initialize_db_settings
        try:
            # Load DB config from environment using the new config pattern
            settings = initialize_db_settings()
            db_type = settings.db_type
            masked_uri = re.sub(r'://([^:]+):([^@]+)@', r'://\1:****@', settings.db_uri)
            logging.info(f"Connecting to {db_type.upper()} with URI: {masked_uri}")
            # Create adapter and client using the factory and settings
            self._adapter = DatabaseAdapterFactory.create_adapter(db_type)
            self._client = self._adapter.create_client(settings.db_uri, db_type=db_type)
            # Verify connection
            if not self._adapter.ping(self._client):
                raise ConnectionError(f"Failed to ping {db_type} server. Please check connection URI and network connectivity.")
            # Initialize database references
            chatbot_db_name = os.getenv("MONGODB_CHATBOT_DATABASE_NAME", "chatbot_db")
            self._db_chatbot = self._adapter.get_database(self._client, chatbot_db_name)
            workflow_db_name = os.getenv("MONGODB_WORKFLOW_DATABASE_NAME", "npd_workflow_db")
            self._db_workflow = self._adapter.get_database(self._client, workflow_db_name)
            MongoDBSingleton._is_initialized = True
            logging.info(f"✅ {db_type.upper()} connection initialized successfully (Singleton)")
            logging.info(f"   Database endpoints - Chatbot: {chatbot_db_name}, Workflow: {workflow_db_name}")
        except ValueError as e:
            logging.error(f"❌ Configuration error: {e}")
            self._client = None
            self._adapter = None
            MongoDBSingleton._is_initialized = False
            raise
        except ConnectionError as e:
            logging.error(f"❌ Connection error: {e}")
            self._client = None
            self._adapter = None
            MongoDBSingleton._is_initialized = False
            raise
        except Exception as e:
            logging.error(f"❌ Unexpected error initializing database connection: {e}")
            logging.error(f"   DB_TYPE: {os.getenv('DB_TYPE', 'mongodb')}")
            logging.error(f"   Check that your connection URI is valid and the database is accessible.")
            self._client = None
            self._adapter = None
            MongoDBSingleton._is_initialized = False
            raise
    
    @property
    def client(self) -> Any:
        """
        Get the database client instance.
        
        Returns:
            Database client instance (MongoClient for both MongoDB and DocumentDB)
        """
        if not MongoDBSingleton._is_initialized or self._client is None:
            raise RuntimeError("Database client not initialized")
        return self._client
    
    @property
    def chatbot_db(self):
        """
        Get the chatbot database instance.
        
        Returns:
            Database instance for chatbot operations
        """
        if not MongoDBSingleton._is_initialized or self._db_chatbot is None:
            raise RuntimeError("Database client not initialized")
        return self._db_chatbot
    
    @property
    def workflow_db(self):
        """
        Get the workflow database instance.
        
        Returns:
            Database instance for workflow operations
        """
        if not MongoDBSingleton._is_initialized or self._db_workflow is None:
            raise RuntimeError("Database client not initialized")
        return self._db_workflow
    
    def close(self):
        """
        Close the database connection gracefully.
        Uses the adapter's close method to ensure proper cleanup.
        Call this on application shutdown.
        """
        print("🔧 close() method called!")
        with self._lock:
            print(f"🔧 _client exists: {self._client is not None}")
            if self._client and self._adapter:
                print("🔧 Closing database client...")
                self._adapter.close(self._client)
                self._client = None
                self._adapter = None
                # Reset class variable
                MongoDBSingleton._is_initialized = False
                print("✅ Database connection closed gracefully")
                logging.info("✅ Database connection closed gracefully")
            else:
                print("⚠️ Database client was already None")
                
    @classmethod
    def reset_instance(cls):
        """
        Reset the singleton instance. Useful for testing.
        WARNING: Only use this in test environments.
        """
        with cls._lock:
            if cls._instance is not None:
                if cls._instance._client is not None and cls._instance._adapter is not None:
                    cls._instance._adapter.close(cls._instance._client)
                cls._instance._adapter = None
            cls._instance = None
            cls._is_initialized = False


def get_mongodb_client() -> MongoDBSingleton:
    """
    Get the database singleton instance.
    
    This function returns a singleton instance that provides access to either
    MongoDB or AWS DocumentDB based on the DB_TYPE environment variable.
    
    Returns:
        MongoDBSingleton: The singleton instance with database connections
        
    Example:
        >>> db = get_mongodb_client()
        >>> chatbot_collection = db.chatbot_db["my_collection"]
    """
    return MongoDBSingleton()