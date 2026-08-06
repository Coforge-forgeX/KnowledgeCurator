"""Application configuration management for Indexer Service."""
from typing import Optional
from urllib.parse import quote_plus

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""

    # PostgreSQL settings
    POSTGRESQL_DATABASE_HOST: str = Field(default="localhost")
    POSTGRESQL_DATABASE_PORT: int = Field(default=5432)
    POSTGRESQL_DATABASE_USER: str = Field(default="postgres")
    POSTGRESQL_DATABASE_PASSWORD: str = Field(default="")
    POSTGRESQL_DATABASE_DATABASE: str = Field(default="kbcurator")

    # Neo4j settings
    NEO4J_DATABASE_NEO4J_BOLT_URI: str = Field(default="bolt://localhost:7687")
    NEO4J_DATABASE_NEO4J_USER: str = Field(default="neo4j")
    NEO4J_DATABASE_NEO4J_PASSWORD: str = Field(default="")

    @property
    def postgresql_url(self) -> str:
        """Build PostgreSQL connection URL with encoded credentials."""
        encoded_user = quote_plus(self.POSTGRESQL_DATABASE_USER)
        encoded_password = quote_plus(self.POSTGRESQL_DATABASE_PASSWORD)
        return (
            f"postgresql://{encoded_user}:"
            f"{encoded_password}@{self.POSTGRESQL_DATABASE_HOST}:"
            f"{self.POSTGRESQL_DATABASE_PORT}/{self.POSTGRESQL_DATABASE_DATABASE}"
        )

    @property
    def neo4j_uri(self) -> str:
        """Return Neo4j URI."""
        return self.NEO4J_DATABASE_NEO4J_BOLT_URI

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


class StorageSettings(BaseSettings):
    """Storage configuration (multi-cloud support)."""

    # Storage provider
    STORAGE_PROVIDER: str = Field(default="azure")
    STORAGE_CONTAINER_NAME: str = Field(default="aksknowledgecurator")

    # Azure Blob Storage
    AZURE_BLOB_STORAGE_CONNECTION_STRING: Optional[str] = Field(default=None)

    # AWS S3
    AWS_ACCESS_KEY_ID: Optional[str] = Field(default=None)
    AWS_SECRET_ACCESS_KEY: Optional[str] = Field(default=None)
    AWS_REGION: str = Field(default="us-east-1")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


class QueueSettings(BaseSettings):
    """Queue configuration (multi-cloud support)."""

    # Queue provider
    QUEUE_PROVIDER: str = Field(default="azure")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


class AzureSettings(BaseSettings):
    """Azure-specific configuration settings."""

    # Azure Storage Queue
    AZURE_STORAGE_CONNECTION_STRING: str = Field(default="")
    INDEXING_QUEUE_NAME: str = Field(
        default="kb-indexing-jobs",
        validation_alias="AZURE_INDEXING_QUEUE_NAME",
    )
    QUEUE_POLL_INTERVAL: int = Field(default=5)
    WORKSPACE_CONTAINER_NAME: str = Field(default="workspace")

    # Azure Service Bus (recommended for production)
    SERVICE_BUS_CONNECTION_STRING: Optional[str] = Field(
        default=None,
        env="SERVICE_BUS_CONNECTION_STRING",
    )
    SERVICE_BUS_TOPIC_NAME: Optional[str] = Field(
        default=None,
        env="SERVICE_BUS_TOPIC_NAME",
    )
    SERVICE_BUS_SUBSCRIPTION_NAME: Optional[str] = Field(
        default=None,
        env="SERVICE_BUS_SUBSCRIPTION_NAME",
    )

    # Azure Document Intelligence
    AZURE_DOC_INTELLIGENCE_ENDPOINT: Optional[str] = Field(
        default=None,
        validation_alias="AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT",
    )
    AZURE_DOC_INTELLIGENCE_KEY: Optional[str] = Field(
        default=None,
        validation_alias="AZURE_DOCUMENT_INTELLIGENCE_KEY",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )


class LLMSettings(BaseSettings):
    """LLM and embedding configuration."""

    # Azure OpenAI LLM
    AZURE_OPENAI_LLM_MODEL_API_KEY: str = Field(default="")
    AZURE_OPENAI_LLM_MODEL_API_BASE: str = Field(default="")
    AZURE_OPENAI_LLM_MODEL_API_VERSION: str = Field(default="2024-02-15-preview")
    AZURE_OPENAI_LLM_MODEL_LLM_MODEL: str = Field(default="gpt-4")

    # Azure OpenAI embeddings
    AZURE_OPENAI_EMBEDDING_MODEL_API_KEY: str = Field(default="")
    AZURE_OPENAI_EMBEDDING_MODEL_API_BASE: str = Field(default="")
    AZURE_OPENAI_EMBEDDING_MODEL_API_VERSION: str = Field(default="2024-02-15-preview")
    AZURE_OPENAI_EMBEDDING_MODEL_EMBEDDING_MODEL: str = Field(default="text-embedding-3-large")

    # Ollama (for embeddings)
    OLLAMA_MODEL_BASE_URL: str = Field(default="http://localhost:11434")
    OLLAMA_MODEL_EMBEDDING_MODEL: str = Field(default="mxbai-embed-large")
    OLLAMA_MODEL_EMBEDDING_MODEL_DIMS: int = Field(default=1024)
    OLLAMA_MODEL_EMBEDDING_MODEL_MAX_TOKENS: int = Field(default=8192)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


class ProcessingSettings(BaseSettings):
    """Document processing configuration."""

    # PDF processing
    PDF_MIN_TEXT_CHARS: int = Field(default=200)
    PDF_MIN_TEXT_PER_PAGE: int = Field(default=100)
    PDF_PER_PAGE_OCR: bool = Field(default=True)

    # Cache directories
    INDEXER_CACHE_DIR: str = Field(default="./indexer_cache")
    INDEXER_STATE_DIR: str = Field(default="./indexer_state")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


class LightRAGSettings(BaseSettings):
    """LightRAG configuration settings."""

    WORKING_DIR: str = Field(default="./lightrag_data", validation_alias="LIGHTRAG_WORKING_DIR")
    EMBEDDING_DIM: int = Field(default=3072, validation_alias="LIGHTRAG_EMBEDDING_DIM")
    MAX_TOKEN_SIZE: int = Field(default=8192, validation_alias="LIGHTRAG_MAX_TOKEN_SIZE")
    CHUNK_TOKEN_SIZE: int = Field(default=1000, validation_alias="LIGHTRAG_CHUNK_TOKEN_SIZE")
    CHUNK_OVERLAP_TOKEN_SIZE: int = Field(default=200, validation_alias="LIGHTRAG_CHUNK_OVERLAP_TOKEN_SIZE")
    GRAPH_STORAGE_TYPE: str = Field(default="Neo4JStorage", validation_alias="LIGHTRAG_GRAPH_STORAGE_TYPE")
    VECTOR_STORAGE_TYPE: str = Field(default="PGVectorStorage", validation_alias="LIGHTRAG_VECTOR_STORAGE_TYPE")
    USE_EMBEDDING_MODEL_SUFFIX: bool = Field(default=False, validation_alias="LIGHTRAG_USE_EMBEDDING_MODEL_SUFFIX")
    EMBEDDING_TIMEOUT_SECONDS: int = Field(default=120, validation_alias="LIGHTRAG_EMBEDDING_TIMEOUT_SECONDS")
    EMBEDDING_FUNC_MAX_ASYNC: int = Field(default=4, validation_alias="LIGHTRAG_EMBEDDING_FUNC_MAX_ASYNC")
    EMBEDDING_BATCH_NUM: int = Field(default=4, validation_alias="LIGHTRAG_EMBEDDING_BATCH_NUM")
    INSERT_MODE: str = Field(default="custom_chunks", validation_alias="LIGHTRAG_INSERT_MODE")

    # Azure OpenAI LLM settings
    AZURE_OPENAI_LLM_API_KEY: Optional[str] = Field(default=None, validation_alias="AZURE_OPENAI_LLM_MODEL_API_KEY")
    AZURE_OPENAI_LLM_API_BASE: Optional[str] = Field(default=None, validation_alias="AZURE_OPENAI_LLM_MODEL_API_BASE")
    AZURE_OPENAI_LLM_API_VERSION: str = Field(default="2024-12-01-preview", validation_alias="AZURE_OPENAI_LLM_MODEL_API_VERSION")
    AZURE_OPENAI_LLM_DEPLOYMENT: Optional[str] = Field(default=None, validation_alias="AZURE_OPENAI_LLM_MODEL_LLM_MODEL")

    # Azure OpenAI embedding settings
    AZURE_OPENAI_EMBEDDING_API_KEY: Optional[str] = Field(default=None, validation_alias="AZURE_OPENAI_EMBEDDING_MODEL_API_KEY")
    AZURE_OPENAI_EMBEDDING_API_BASE: Optional[str] = Field(default=None, validation_alias="AZURE_OPENAI_EMBEDDING_MODEL_API_BASE")
    AZURE_OPENAI_EMBEDDING_API_VERSION: str = Field(default="2024-02-01", validation_alias="AZURE_OPENAI_EMBEDDING_MODEL_API_VERSION")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: Optional[str] = Field(default=None, validation_alias="AZURE_OPENAI_EMBEDDING_MODEL_EMBEDDING_MODEL")

    # Ollama settings
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434", validation_alias="OLLAMA_MODEL_BASE_URL")
    OLLAMA_EMBEDDING_MODEL: str = Field(default="mxbai-embed-large", validation_alias="OLLAMA_MODEL_EMBEDDING_MODEL")
    OLLAMA_EMBEDDING_DIMS: int = Field(default=1024, validation_alias="OLLAMA_MODEL_EMBEDDING_MODEL_DIMS")
    OLLAMA_MAX_TOKENS: int = Field(default=8192, validation_alias="OLLAMA_MODEL_EMBEDDING_MODEL_MAX_TOKENS")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )


class Settings(BaseSettings):
    """Main application settings."""

    # Application settings
    APP_NAME: str = Field(default="KB Indexer Service")
    VERSION: str = Field(default="1.0.0", validation_alias="APP_VERSION")
    ENVIRONMENT: str = Field(default="development")
    DEBUG: bool = Field(default=False)

    # Deployment settings
    CLOUD_PROVIDER: str = Field(default="azure")
    STORAGE_PROVIDER: Optional[str] = Field(default=None)

    # Server settings
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8081)

    # Logging settings
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FORMAT: str = Field(default="console")

    # Worker settings
    MAX_CONCURRENT_JOBS: int = Field(default=10)
    MESSAGE_VISIBILITY_TIMEOUT: int = Field(default=300)
    MAX_RETRIES: int = Field(default=3)
    # Maximum lock renewal duration for Service Bus messages (in seconds)
    # Set this to the maximum expected job duration to prevent lock expiration
    # Default: 1800 seconds (30 minutes)
    MAX_LOCK_RENEWAL_DURATION: int = Field(default=1800)

    # Nested settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    queue: QueueSettings = Field(default_factory=QueueSettings)
    azure: AzureSettings = Field(default_factory=AzureSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    lightrag: LightRAGSettings = Field(default_factory=LightRAGSettings)

    @property
    def QUEUE_PROVIDER(self) -> str:
        """Get effective queue provider with cloud provider fallback."""
        return self.active_queue_provider

    @property
    def active_queue_provider(self) -> str:
        """Get effective queue provider with cloud provider fallback."""
        return (self.queue.QUEUE_PROVIDER or self.CLOUD_PROVIDER or "azure").lower()

    @property
    def active_storage_provider(self) -> str:
        """Get effective storage provider with cloud provider fallback."""
        return (
            self.storage.STORAGE_PROVIDER
            or self.STORAGE_PROVIDER
            or self.CLOUD_PROVIDER
            or "azure"
        ).lower()

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()

    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        valid_envs = ["development", "dev", "staging", "stage", "production", "prod"]
        normalized = v.lower()
        if normalized not in valid_envs:
            raise ValueError(f"ENVIRONMENT must be one of {valid_envs}")
        if normalized == "dev":
            return "development"
        if normalized in {"stage", "staging"}:
            return "staging"
        if normalized == "prod":
            return "production"
        return normalized

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT == "development"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
    )


settings = Settings()
