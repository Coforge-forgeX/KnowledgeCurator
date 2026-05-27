from fastmcp import FastMCP
from contextlib import asynccontextmanager
from typing import AsyncIterator
from kbcurator.utils.mongodb_singleton import get_mongodb_client
import logging
from dotenv import load_dotenv
load_dotenv()
from common_adapters.langfuse_instrumentation import flush as langfuse_flush
from common_adapters.sharepoint import SharePointClientManagerAsync
from common_adapters.cache import CacheFactory
from ..utils.session_history_manager import (
    SessionHistoryManager, 
    UserConfigManager
)
# Global variables
sharepoint_client_manager = None
user_config_manager = None
r = None
session = None
session_context = None
@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[None]:
    # Initialize MongoDB singleton
    mongo_client = get_mongodb_client()
    global r, sharepoint_client_manager, user_config_manager, session
    try:
        # Initialize other services
        logging.info("🔧 Initializing services...")
        
        logging.debug("Initializing Redis cache...")
        CacheFactory.initialize()  # (optional, usually called automatically)
        r = CacheFactory.get_cache(prefix="kb-member-")
        # r = CacheFactory("kb-member-")
        logging.info("  ✅ Redis cache initialized")
        
        user_config_manager = UserConfigManager(mongo_client)
        # Initialize SharePoint client manager with config manager
        sharepoint_client_manager = SharePointClientManagerAsync(
            redis_client=r,  # Optional: for caching, replace with actual redis client if available
            config_manager=user_config_manager  # Required
        )
        logging.info("✅ SharePoint client manager initialized")

        logging.debug("Initializing session history manager...")
        session = SessionHistoryManager(mongo_client)
        logging.info("  ✅ Session history manager initialized")
        
    except Exception as e:
        logging.error(f"✗ Startup initialization failed: {e}")
    try:
        yield
    finally:
        # Close MongoDB connection on shutdown
        logging.info("🔧 Shutting down lifespan, closing MongoDB...")
        mongo_client.close()
        logging.info("✅ Lifespan cleanup complete")
        try:
            langfuse_flush()
        except Exception:
            pass

mcp = FastMCP("kbCuratorAdapter", lifespan=lifespan)