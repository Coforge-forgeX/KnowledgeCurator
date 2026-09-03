"""
Clean Configuration Management for KB REST API

Multi-cloud support with canonical environment variable names.
Aligned with common_adapters package requirements.
"""
from typing import List, Optional, Union
from urllib.parse import quote_plus

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# Database Settings
# =============================================================================

class DatabaseSettings(BaseSettings):
    """Database configuration settings"""

    # -------------------------------------------------------------------------
    # PostgreSQL (Main Application Database)
    # -------------------------------------------------------------------------
    POSTGRESQL_HOST: str = Field(default="localhost")
    POSTGRESQL_PORT: int = Field(default=5432)
    POSTGRESQL_USER: str = Field(default="postgres")
    POSTGRESQL_PASSWORD: str = Field(default="")
    POSTGRESQL_DATABASE: str = Field(default="kbcurator")

    # -------------------------------------------------------------------------
    # MongoDB (for common_adapters llm_router_config_store)
    # -------------------------------------------------------------------------
    MONGODB_URI: str = Field(default="")
    MONGODB_DATABASE: str = Field(default="chatbot_db")

    # -------------------------------------------------------------------------
    # Redis
    # -------------------------------------------------------------------------
    REDIS_HOST: Optional[str] = Field(default=None)
    REDIS_PORT: int = Field(default=6379)
    REDIS_PASSWORD: Optional[str] = Field(default=None)
    REDIS_DB: int = Field(default=0)

    # -------------------------------------------------------------------------
    # Neo4j (for LightRAG Graph Storage)
    # -------------------------------------------------------------------------
    NEO4J_URI: Optional[str] = Field(default=None)
    NEO4J_USER: str = Field(default="neo4j")
    NEO4J_PASSWORD: Optional[str] = Field(default=None)

    # -------------------------------------------------------------------------
    # Connection Pool Settings
    # -------------------------------------------------------------------------
    DB_POOL_SIZE: int = Field(default=5)
    DB_MAX_OVERFLOW: int = Field(default=5)
    DB_POOL_TIMEOUT: int = Field(default=30)
    DB_POOL_RECYCLE: int = Field(default=3600)
    DB_ECHO: bool = Field(default=False)

    # Deployment mode (serverless optimization)
    SERVERLESS: bool = Field(default=True)

    @property
    def get_pool_size(self) -> int:
        """Get pool size optimized for deployment mode"""
        return min(self.DB_POOL_SIZE, 5) if self.SERVERLESS else self.DB_POOL_SIZE

    @property
    def get_max_overflow(self) -> int:
        """Get max overflow optimized for deployment mode"""
        return 0 if self.SERVERLESS else self.DB_MAX_OVERFLOW

    @property
    def get_pool_recycle(self) -> int:
        """Get pool recycle time optimized for deployment mode"""
        return 300 if self.SERVERLESS else self.DB_POOL_RECYCLE

    @property
    def postgresql_url(self) -> str:
        """Build PostgreSQL async connection URL"""
        encoded_user = quote_plus(self.POSTGRESQL_USER)
        encoded_password = quote_plus(self.POSTGRESQL_PASSWORD)
        return (
            f"postgresql+asyncpg://{encoded_user}:{encoded_password}"
            f"@{self.POSTGRESQL_HOST}:{self.POSTGRESQL_PORT}"
            f"/{self.POSTGRESQL_DATABASE}?ssl=require"
        )

    @property
    def redis_url(self) -> Optional[str]:
        """Build Redis connection URL"""
        if not self.REDIS_HOST:
            return None
        if self.REDIS_PASSWORD:
            encoded_password = quote_plus(self.REDIS_PASSWORD)
            return f"redis://:{encoded_password}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


# =============================================================================
# Cache Settings
# =============================================================================

class CacheSettings(BaseSettings):
    """Redis caching configuration settings"""

    # Redis connection
    REDIS_URL: str = Field(default="redis://localhost:6379/0")
    REDIS_ENABLED: bool = Field(default=True)

    # Cache TTL settings (in seconds)
    QUERY_CACHE_TTL: int = Field(default=3600)  # 1 hour
    CONVERSATION_HISTORY_CACHE_TTL: int = Field(default=3600)  # 1 hour

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


# =============================================================================
# Multi-Cloud Storage Settings
# =============================================================================

class StorageSettings(BaseSettings):
    """Storage configuration (multi-cloud support)"""

    # Provider selection
    STORAGE_PROVIDER: str = Field(default="azure")  # azure, aws, gcp, local

    # Container/Bucket names
    STORAGE_CONTAINER_NAME: str = Field(default="aksknowledgecurator")
    WORKSPACE_CONTAINER_NAME: str = Field(default="workspace")

    # -------------------------------------------------------------------------
    # Azure Blob Storage
    # -------------------------------------------------------------------------
    AZURE_STORAGE_CONNECTION_STRING: Optional[str] = Field(default=None)

    # -------------------------------------------------------------------------
    # AWS S3
    # -------------------------------------------------------------------------
    AWS_ACCESS_KEY_ID: Optional[str] = Field(default=None)
    AWS_SECRET_ACCESS_KEY: Optional[str] = Field(default=None)
    AWS_REGION: str = Field(default="us-east-1")
    S3_BUCKET_NAME: Optional[str] = Field(default=None)
    S3_PATH_PREFIX: str = Field(default="")
    S3_URL_EXPIRY_MINUTES: int = Field(default=60)

    # -------------------------------------------------------------------------
    # GCP Cloud Storage
    # -------------------------------------------------------------------------
    GCP_PROJECT_ID: Optional[str] = Field(default=None)
    GCP_CREDENTIALS_PATH: Optional[str] = Field(default=None)
    GCS_BUCKET_NAME: Optional[str] = Field(default=None)
    GCS_PATH_PREFIX: str = Field(default="")
    GCS_URL_EXPIRY_MINUTES: int = Field(default=60)

    # -------------------------------------------------------------------------
    # Local Storage (for development)
    # -------------------------------------------------------------------------
    LOCAL_STORAGE_PATH: str = Field(default="./local_storage")
    LOCAL_STORAGE_PATH_PREFIX: str = Field(default="")
    LOCAL_STORAGE_BASE_URL: str = Field(default="")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


# =============================================================================
# Multi-Cloud Queue Settings
# =============================================================================

class QueueSettings(BaseSettings):
    """Queue configuration (multi-cloud support)"""

    # Provider selection
    QUEUE_PROVIDER: str = Field(default="azure")  # azure, aws, redis

    # -------------------------------------------------------------------------
    # Azure Queue/Service Bus
    # -------------------------------------------------------------------------
    AZURE_STORAGE_CONNECTION_STRING: Optional[str] = Field(default=None)
    AZURE_INDEXING_QUEUE_NAME: str = Field(default="indexing-queue")

    # Service Bus (recommended for production)
    SERVICE_BUS_CONNECTION_STRING: Optional[str] = Field(default=None)
    SERVICE_BUS_TOPIC_NAME: Optional[str] = Field(default=None)
    SERVICE_BUS_SUBSCRIPTION_NAME: Optional[str] = Field(default=None)

    # -------------------------------------------------------------------------
    # AWS SQS
    # -------------------------------------------------------------------------
    SQS_QUEUE_URL: Optional[str] = Field(default=None)
    SQS_QUEUE_NAME: str = Field(default="indexing-jobs")
    AWS_REGION: str = Field(default="us-east-1")

    # -------------------------------------------------------------------------
    # Redis Queue
    # -------------------------------------------------------------------------
    REDIS_QUEUE_URL: Optional[str] = Field(default=None)
    REDIS_QUEUE_NAME: str = Field(default="indexing-jobs")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


# =============================================================================
# Multi-Cloud OCR Settings
# =============================================================================

class OCRSettings(BaseSettings):
    """
    OCR configuration (multi-cloud support)

    Aligned with shared/text_extraction package:
    - Uses AZURE_DOC_INTELLIGENCE_* (shorter form)
    - shared/text_extraction/config.py supports both long and short forms
    """

    # Provider selection
    OCR_PROVIDER: str = Field(default="azure")  # azure, aws, gcp, noop
    PDF_MIN_TEXT_CHARS: int = Field(default=200, ge=1)

    # -------------------------------------------------------------------------
    # Azure Document Intelligence
    # Canonical keys: AZURE_DOC_INTELLIGENCE_* (shorter)
    # shared/text_extraction also accepts: AZURE_DOCUMENT_INTELLIGENCE_* (longer)
    # -------------------------------------------------------------------------
    AZURE_DOC_INTELLIGENCE_ENDPOINT: Optional[str] = Field(default=None)
    AZURE_DOC_INTELLIGENCE_KEY: Optional[str] = Field(default=None)

    # -------------------------------------------------------------------------
    # AWS Textract
    # -------------------------------------------------------------------------
    AWS_TEXTRACT_REGION: Optional[str] = Field(default=None)
    AWS_TEXTRACT_ACCESS_KEY_ID: Optional[str] = Field(default=None)
    AWS_TEXTRACT_SECRET_ACCESS_KEY: Optional[str] = Field(default=None)

    # -------------------------------------------------------------------------
    # GCP Document AI
    # -------------------------------------------------------------------------
    GCP_DOCUMENT_AI_PROJECT_ID: Optional[str] = Field(default=None)
    GCP_DOCUMENT_AI_LOCATION: str = Field(default="us")
    GCP_DOCUMENT_AI_PROCESSOR_ID: Optional[str] = Field(default=None)
    GCP_DOCUMENT_AI_CREDENTIALS_PATH: Optional[str] = Field(default=None)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


# =============================================================================
# Security Settings
# =============================================================================

class SecuritySettings(BaseSettings):
    """Security and authentication settings"""

    # JWT settings
    JWT_SECRET_KEY: str = Field(default="your-secret-key-change-in-production")
    JWT_ALGORITHM: str = Field(default="HS256")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30)
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7)

    # Backend service authentication
    BACKEND_SERVICE_SECRET_KEY: Optional[str] = Field(default=None)

    # CORS settings
    CORS_ORIGINS: Union[List[str], str] = Field(default=["*"])
    CORS_ALLOW_CREDENTIALS: bool = Field(default=True)
    CORS_ALLOW_METHODS: Union[List[str], str] = Field(default=["*"])
    CORS_ALLOW_HEADERS: Union[List[str], str] = Field(default=["*"])

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            if not v or v.strip() == "":
                return ["*"]
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @field_validator("JWT_SECRET_KEY")
    @classmethod
    def validate_jwt_secret(cls, v):
        if len(v) < 32:
            raise ValueError("JWT_SECRET_KEY must be at least 32 characters long")
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


# =============================================================================
# LightRAG Settings
# =============================================================================

class LightRAGSettings(BaseSettings):
    """
    LightRAG configuration settings

    Note: LightRAG PostgreSQL settings are SEPARATE from main database.
    They can point to a different database instance.
    """

    # Working directory
    LIGHTRAG_WORKING_DIR: str = Field(default="/tmp/lightrag_data")

    # Storage backend types
    VECTOR_STORAGE_TYPE: str = Field(default="PGVectorStorage")
    GRAPH_STORAGE_TYPE: str = Field(default="Neo4JStorage")
    COSINE_THRESHOLD: float = Field(default=0.2, ge=0.0, le=1.0)

    # -------------------------------------------------------------------------
    # PostgreSQL for PGVectorStorage (SEPARATE from main database)
    # -------------------------------------------------------------------------
    LIGHTRAG_POSTGRESQL_HOST: Optional[str] = Field(default=None)
    LIGHTRAG_POSTGRESQL_USER: Optional[str] = Field(default=None)
    LIGHTRAG_POSTGRESQL_PASSWORD: Optional[str] = Field(default=None)
    LIGHTRAG_POSTGRESQL_DATABASE: Optional[str] = Field(default=None)

    # Chunk settings
    CHUNK_TOKEN_SIZE: int = Field(default=600)
    CHUNK_OVERLAP_TOKEN_SIZE: int = Field(default=150)

    # Embedding settings
    EMBEDDING_DIM: int = Field(default=1024)
    MAX_TOKEN_SIZE: int = Field(default=8192)
    EMBEDDING_TIMEOUT_SECONDS: int = Field(default=120)
    EMBEDDING_FUNC_MAX_ASYNC: int = Field(default=4)
    EMBEDDING_BATCH_NUM: int = Field(default=4)

    # -------------------------------------------------------------------------
    # Azure OpenAI LLM Settings (for common_adapters compatibility)
    # -------------------------------------------------------------------------
    AZURE_OPENAI_LLM_MODEL_API_KEY: Optional[str] = Field(default=None)
    AZURE_OPENAI_LLM_MODEL_API_BASE: Optional[str] = Field(default=None)
    AZURE_OPENAI_LLM_MODEL_API_VERSION: str = Field(default="2024-02-15-preview")
    AZURE_OPENAI_LLM_MODEL_LLM_MODEL: Optional[str] = Field(default=None)

    # -------------------------------------------------------------------------
    # Azure OpenAI Embedding Settings (for common_adapters compatibility)
    # -------------------------------------------------------------------------
    AZURE_OPENAI_EMBEDDING_MODEL_API_KEY: Optional[str] = Field(default=None)
    AZURE_OPENAI_EMBEDDING_MODEL_API_BASE: Optional[str] = Field(default=None)
    AZURE_OPENAI_EMBEDDING_MODEL_API_VERSION: str = Field(default="2024-02-01")
    AZURE_OPENAI_EMBEDDING_MODEL_EMBEDDING_MODEL: Optional[str] = Field(default=None)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


# =============================================================================
# Progress/Event Bus Settings
# =============================================================================

class ProgressSettings(BaseSettings):
    """Progress and event bus configuration settings"""

    # Progress backend configuration
    PROGRESS_BACKEND: str = Field(default="auto")
    EVENT_BUS_PROVIDER: Optional[str] = Field(default=None)

    # Azure Service Bus settings
    EVENT_BUS_CONNECTION_STRING: Optional[str] = Field(default=None)
    SERVICE_BUS_CONNECTION_STRING: Optional[str] = Field(default=None)
    PROGRESS_QUEUE: Optional[str] = Field(default=None)
    PROGRESS_TOPIC: str = Field(default="agent-progress")

    # AWS EventBridge settings
    PROGRESS_EVENT_BUS: str = Field(default="default")

    # Local relay settings
    PROGRESS_LOCAL_RELAY_URL: str = Field(default="http://127.0.0.1:8090/publish")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


# =============================================================================
# Main Settings
# =============================================================================

class Settings(BaseSettings):
    """Main application settings"""

    # -------------------------------------------------------------------------
    # Application Settings
    # -------------------------------------------------------------------------
    APP_NAME: str = Field(default="KB REST Service")
    VERSION: str = Field(default="2.0.0")
    ENVIRONMENT: str = Field(default="development")
    DEBUG: bool = Field(default=False)

    # -------------------------------------------------------------------------
    # Multi-Cloud Deployment
    # -------------------------------------------------------------------------
    CLOUD_PROVIDER: str = Field(default="azure")  # azure, aws, gcp

    # -------------------------------------------------------------------------
    # Server Settings
    # -------------------------------------------------------------------------
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)

    # -------------------------------------------------------------------------
    # Logging Settings
    # -------------------------------------------------------------------------
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FORMAT: str = Field(default="json")

    # -------------------------------------------------------------------------
    # Request Validation
    # -------------------------------------------------------------------------
    MAX_REQUEST_SIZE: int = Field(default=10485760)  # 10MB default

    # -------------------------------------------------------------------------
    # Chat Settings (for common_adapters context_compaction)
    # -------------------------------------------------------------------------
    CHAT_HISTORY_TURNS_FOR_CONTEXT: int = Field(default=5)
    CHAT_CONTEXT_TOKEN_THRESHOLD: int = Field(default=200_000)

    # Intent Detection
    INTENT_DETECTOR_TYPE: str = Field(default="rule")
    INTENT_CONFIDENCE_THRESHOLD: float = Field(default=0.8)
    INTENT_CACHE_ENABLED: bool = Field(default=True)
    INTENT_CACHE_TTL: int = Field(default=600)

    # -------------------------------------------------------------------------
    # Debug Toggles
    # -------------------------------------------------------------------------
    SKIP_DUPLICATE_CHECK: bool = Field(default=False)

    # -------------------------------------------------------------------------
    # Legacy Settings (for backward compatibility with some code)
    # -------------------------------------------------------------------------
    AWS_REGION: str = Field(default="us-east-1")
    SQS_QUEUE_NAME: str = Field(default="indexing-jobs")
    SQS_QUEUE_URL: Optional[str] = Field(default=None)
    REDIS_QUEUE_NAME: str = Field(default="indexing-jobs")
    REDIS_QUEUE_URL: Optional[str] = Field(default=None)

    # -------------------------------------------------------------------------
    # Nested Settings
    # -------------------------------------------------------------------------
    database: DatabaseSettings = DatabaseSettings()
    cache: CacheSettings = CacheSettings()
    storage: StorageSettings = StorageSettings()
    queue: QueueSettings = QueueSettings()
    ocr: OCRSettings = OCRSettings()
    security: SecuritySettings = SecuritySettings()
    lightrag: LightRAGSettings = LightRAGSettings()
    progress: ProgressSettings = ProgressSettings()

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()

    @field_validator("LOG_FORMAT")
    @classmethod
    def validate_log_format(cls, v):
        valid_formats = ["console", "json"]
        if v.lower() not in valid_formats:
            raise ValueError(f"LOG_FORMAT must be one of {valid_formats}")
        return v.lower()

    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, v):
        valid_envs = ["development", "staging", "production"]
        if v.lower() not in valid_envs:
            raise ValueError(f"ENVIRONMENT must be one of {valid_envs}")
        return v.lower()

    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.ENVIRONMENT == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.ENVIRONMENT == "development"

    @property
    def app_name(self) -> str:
        """Get application name"""
        return self.APP_NAME

    @property
    def active_queue_provider(self) -> str:
        """Get effective queue provider with cloud provider fallback."""
        return (self.queue.QUEUE_PROVIDER or self.CLOUD_PROVIDER).lower()

    @property
    def active_queue_name(self) -> str:
        """Resolve queue name for active provider."""
        provider = self.active_queue_provider
        if provider == "aws":
            return self.SQS_QUEUE_NAME
        if provider == "redis":
            return self.REDIS_QUEUE_NAME
        return self.queue.AZURE_INDEXING_QUEUE_NAME

    @property
    def active_queue_connection(self) -> Optional[str]:
        """Resolve queue connection string/URL for active provider."""
        provider = self.active_queue_provider
        if provider == "aws":
            return self.SQS_QUEUE_URL
        if provider == "redis":
            return self.REDIS_QUEUE_URL or self.database.redis_url
        if provider == "azure_service_bus":
            return self.queue.SERVICE_BUS_CONNECTION_STRING
        return self.queue.AZURE_STORAGE_CONNECTION_STRING

    # Backward compatibility properties
    @property
    def STORAGE_PROVIDER(self) -> str:
        return self.storage.STORAGE_PROVIDER

    @property
    def QUEUE_PROVIDER(self) -> str:
        return self.queue.QUEUE_PROVIDER

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow"  # Allow extra fields from nested settings
    )


# =============================================================================
# Global Settings Instance
# =============================================================================

settings = Settings()


def get_settings() -> Settings:
    """
    Get global settings instance.

    This function provides a singleton pattern for settings access,
    allowing for easier dependency injection in FastAPI routes.

    Returns:
        Settings: Global settings instance
    """
    return settings
