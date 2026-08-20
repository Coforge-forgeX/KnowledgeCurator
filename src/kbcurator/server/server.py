from fastmcp import FastMCP
from contextlib import asynccontextmanager
from typing import AsyncIterator
from kbcurator.utils.mongodb_singleton import get_mongodb_client
import logging
import os
from urllib.parse import quote_plus
from dotenv import load_dotenv
load_dotenv()
from common_adapters.langfuse_instrumentation import flush as langfuse_flush
from common_adapters.sharepoint import SharePointClientManagerAsync
from common_adapters.cache import CacheFactory
from common_adapters.cancel_convesation import cancel_conversation
from kbcurator.utils.session_history_manager import (
    SessionHistoryManager, 
    UserConfigManager
)
from common_adapters.trustai import  TrustAIDatabaseManager
from common_adapters.trustai.workspace_integration import TrustAIWorkspaceIntegration
from .scheduler import start_scheduler, scheduler

sharepoint_client_manager = None
user_config_manager = None
trustai_workspace_integration = None
trustai_db_manager = None
r = None
session = None
session_context = None

#trustai helper method
def get_postgres_connection_string(database_env: str = "POSTGRESQL_DATABASE_DATABASE",) -> str | None:
    """
    Builds a PostgreSQL SQLAlchemy connection string from environment variables.
    Returns:
    Connection string if all required values are present.
    None if any required value is missing or an error occurs.
    """
    try:
        host = os.getenv("POSTGRESQL_DATABASE_HOST")
        port = os.getenv("POSTGRESQL_DATABASE_PORT", "5432")
        database = os.getenv(database_env)
        user = os.getenv("POSTGRESQL_DATABASE_USER")
        password = os.getenv("POSTGRESQL_DATABASE_PASSWORD")
        if not all([host, port, database, user, password]):
            return None
        password = quote_plus(password)
        return (
            f"postgresql+psycopg2://"
            f"{user}:{password}"
            f"@{host}:{port}/{database}"
            f"?sslmode=require"
        )
    except Exception:
        return None
    
@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[None]:
    # Initialize MongoDB singleton
    mongo_client = get_mongodb_client()
    global r, sharepoint_client_manager, user_config_manager, session, trustai_workspace_integration, trustai_db_manager
    try:
        
        # Initialize other services
        logging.info("🔧 Initializing services...")
        
        # Start worker scheduler job for trustai aggregation tables.
        start_scheduler()
        logging.info("Start scheduler for trustai_analytics...")
        
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

        db_url = get_postgres_connection_string()
        trustai_db_manager = TrustAIDatabaseManager(db_url)
        trustai_workspace_integration = TrustAIWorkspaceIntegration(trustai_db_manager)
        logging.info("  ✅ Trust AI Database & integration initialized")
        
        # Sync existing workspaces with TrustAI (backward compatibility)
        # try:
        #     from kbcurator.tools.user_management_system import sync_trustai_workspaces
        #     await sync_trustai_workspaces()
        # except Exception as sync_error:
        #     logging.error(f"  ⚠️ TrustAI workspace sync failed: {sync_error}")
    except Exception as e:
        logging.error(f"✗ Startup initialization failed: {e}")
    try:
        yield
    finally:
        # shutdown scheduler job
        scheduler.shutdown()
        # Close MongoDB connection on shutdown
        logging.info("🔧 Shutting down lifespan, closing MongoDB...")
        mongo_client.close()
        logging.info("✅ Lifespan cleanup complete")
        try:
            langfuse_flush()
        except Exception:
            pass

mcp = FastMCP("kbCuratorAdapter", lifespan=lifespan)

# Expose tool name `cancel_conversation` so the UI can stop in-flight requests.
mcp.tool()(cancel_conversation)
