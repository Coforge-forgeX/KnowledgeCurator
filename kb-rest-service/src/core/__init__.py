"""Core modules for kb-rest-service"""
from .auth import (
    create_jwt_token,
    create_refresh_token,
    decode_and_verify_token,
    get_claims,
    get_email,
    get_user_id,
    get_workspace_ids,
    hash_password,
    require_auth,
    revoke_token,
    verify_password,
)
from .config import settings
from .database import (
    Agent,
    AgentMap,
    Base,
    Database,
    FavouriteMappingAgent,
    FavouriteMappingTool,
    Industry,
    KnowledgeBase,
    Role,
    SubIndustry,
    Tool,
    ToolMap,
    User,
    UserMap,
    Workspace,
    db,
    get_async_session,
)
from .exceptions import (
    APIException,
    AuthenticationException,
    AuthorizationException,
    BusinessLogicException,
    ConfigurationException,
    ConflictException,
    DatabaseException,
    ExternalServiceException,
    LightRAGException,
    NotFoundException,
    QueueException,
    ValidationException,
)
from .lightrag_service import (
    LightRAGService,
    get_lightrag_service,
    initialize_lightrag_service,
)
from .logging import Logger, get_logger, setup_logging
from .middleware import AzureFunctionMiddleware, azure_http_decorator
from .neo4j_driver import (
    Neo4jDriver,
    close_neo4j_driver,
    get_neo4j_driver,
    initialize_neo4j_driver,
)
from .redis import get_redis_client, is_redis_available

__all__ = [
    # Config
    "settings",
    # Database
    "Base",
    "Database",
    "User",
    "UserMap",
    "Workspace",
    "Role",
    "Agent",
    "AgentMap",
    "Tool",
    "ToolMap",
    "Industry",
    "SubIndustry",
    "KnowledgeBase",
    "FavouriteMappingAgent",
    "FavouriteMappingTool",
    "db",
    "get_async_session",
    # Auth
    "create_jwt_token",
    "create_refresh_token",
    "decode_and_verify_token",
    "get_claims",
    "get_user_id",
    "get_email",
    "get_workspace_ids",
    "require_auth",
    "hash_password",
    "verify_password",
    "revoke_token",
    # Exceptions
    "APIException",
    "ValidationException",
    "AuthenticationException",
    "AuthorizationException",
    "NotFoundException",
    "ConflictException",
    "DatabaseException",
    "ConfigurationException",
    "BusinessLogicException",
    "ExternalServiceException",
    "LightRAGException",
    "QueueException",
    # Logging
    "Logger",
    "get_logger",
    "setup_logging",
    # Middleware
    "AzureFunctionMiddleware",
    "azure_http_decorator",
    # LightRAG
    "LightRAGService",
    "get_lightrag_service",
    "initialize_lightrag_service",
    # Neo4j
    "Neo4jDriver",
    "get_neo4j_driver",
    "initialize_neo4j_driver",
    "close_neo4j_driver",
    # Redis
    "get_redis_client",
    "is_redis_available",
]
