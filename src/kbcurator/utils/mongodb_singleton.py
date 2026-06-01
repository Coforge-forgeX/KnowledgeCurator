import os
from urllib.parse import unquote
import threading
import logging
import certifi
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class MongoDBSingleton:
    """
    Thread-safe singleton implementation for MongoDB client connection.
    Ensures only one MongoDB client instance is created and reused across the application.
    """
    
    _instance: Optional['MongoDBSingleton'] = None
    _lock = threading.Lock()
    _client: Optional[MongoClient] = None
    _db_chatbot = None
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
        """Initialize MongoDB connection with connection pooling."""
        try:
            mongodb_uri = os.getenv("MONGODB_DATABASE_URI", "").strip()
            if not mongodb_uri:
                raise ValueError("MONGODB_DATABASE_URI environment variable is required")

            # If an encoded URI is provided, decode it once so MongoClient gets a valid URI.
            if "%" in mongodb_uri and "://" not in mongodb_uri:
                mongodb_uri = unquote(mongodb_uri)
            
            self._client = MongoClient(
                mongodb_uri,
                server_api=ServerApi('1'),
                tlsCAFile=certifi.where(),
                maxPoolSize=50,
                minPoolSize=0,
                maxIdleTimeMS=45000,
                waitQueueTimeoutMS=10000,
                serverSelectionTimeoutMS=10000,  # 10 seconds
                connectTimeoutMS=10000,          # 10 seconds
                socketTimeoutMS=10000
            )
            
            # Initialize instance variables for database references
            self._db_chatbot = self._client["chatbot_db"]
            
            # Mark as initialized using class variable
            MongoDBSingleton._is_initialized = True
            logging.info("✅ MongoDB connection initialized successfully (Singleton)")
        except Exception as e:
            logging.error(f"❌ Error initializing MongoDB connection: {e}")
            raise
    
    @property
    def client(self) -> MongoClient:
        """Get the MongoDB client instance."""
        if not MongoDBSingleton._is_initialized or self._client is None:
            raise RuntimeError("MongoDB client not initialized")
        return self._client
    
    @property
    def chatbot_db(self):
        """Get the chatbot database instance."""
        if not MongoDBSingleton._is_initialized or self._db_chatbot is None:
            raise RuntimeError("MongoDB client not initialized")
        return self._db_chatbot
    
    def close(self):
        """Close the MongoDB connection. Call this on application shutdown."""
        print("🔧 close() method called!")
        with self._lock:
            print(f"🔧 _client exists: {self._client is not None}")
            if self._client:
                print("🔧 Closing MongoDB client...")
                self._client.close()
                self._client = None
                # Reset class variable
                MongoDBSingleton._is_initialized = False
                print("✅ MongoDB connection closed gracefully")
                logging.info("✅ MongoDB connection closed gracefully")
    
    @classmethod
    def reset_instance(cls):
        """
        Reset the singleton instance. Useful for testing.
        WARNING: Only use this in test environments.
        """
        with cls._lock:
            if cls._instance is not None and cls._instance._client is not None:
                cls._instance._client.close()
            cls._instance = None
            cls._is_initialized = False


def get_mongodb_client() -> MongoDBSingleton:
    """
    Factory function to get the MongoDB singleton instance.
    
    Returns:
        MongoDBSingleton: The singleton MongoDB client instance
    """
    return MongoDBSingleton()