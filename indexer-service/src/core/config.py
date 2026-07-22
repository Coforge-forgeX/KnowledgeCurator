"""Application configuration management for Indexer Service"""
from typing import Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseSettings):
    """Database configuration settings"""

    # PostgreSQL settings
    POSTGRESQL_DATABASE_HOST: str = Field(default="localhost", env="POSTGRESQL_DATABASE_HOST")
    POSTGRESQL_DATABASE_PORT: int = Field(default=5432, env="POSTGRESQL_DATABASE_PORT")
    POSTGRESQL_DATABASE_USER: str = Field(default="postgres", env="POSTGRESQL_DATABASE_USER")
    POSTGRESQL_DATABASE_PASSWORD: str = Field(default="", env="POSTGRESQL_DATABASE_PASSWORD")
    POSTGRESQL_DATABASE_DATABASE: str = Field(default="kbcurator", env="POSTGRESQL_DATABASE_DATABASE")

    # Neo4j settings
    NEO4J_DATABASE_NEO4J_BOLT_URI: str = Field(
        default="bolt://localhost:7687", env="NEO4J_DATABASE_NEO4J_BOLT_URI"
    )
    NEO4J_DATABASE_NEO4J_USER: str = Field(default="neo4j", env="NEO4J_DATABASE_NEO4J_USER")
    NEO4J_DATABASE_NEO4J_PASSWORD: str = Field(default="", env="NEO4J_DATABASE_NEO4J_PASSWORD")

    @property
    def postgresql_url(self) -> str:
        """Build PostgreSQL connection URL"""
        return (
            f"postgresql://{self.POSTGRESQL_DATABASE_USER}:"
            f"{self.POSTGRESQL_DATABASE_PASSWORD}@{self.POSTGRESQL_DATABASE_HOST}:"
            f"{self.POSTGRESQL_DATABASE_PORT}/{self.POSTGRESQL_DATABASE_DATABASE}"
        )

    @property
    def neo4j_uri(self) -> str:
        """Return Neo4j URI"""
        return self.NEO4J_DATABASE_NEO4J_BOLT_URI


class StorageSettings(BaseSettings):
    """Storage configuration (multi-cloud support)"""

    # Storage provider
    STORAGE_PROVIDER: str = Field(default="azure", env="STORAGE_PROVIDER")
    STORAGE_CONTAINER_NAME: str = Field(default="aksknowledgecurator", env="STORAGE_CONTAINER_NAME")

    # Azure Blob Storage
    AZURE_BLOB_STORAGE_CONNECTION_STRING: Optional[str] = Field(
        default=None, env="AZURE_BLOB_STORAGE_CONNECTION_STRING"
    )

    # AWS S3
    AWS_ACCESS_KEY_ID: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    AWS_REGION: str = Field(default="us-east-1", env="AWS_REGION")


class AzureSettings(BaseSettings):
    """Azure-specific configuration settings"""

    # Azure Storage Queue
    AZURE_STORAGE_CONNECTION_STRING: str = Field(
        default="UseDevelopmentStorage=true", env="AZURE_STORAGE_CONNECTION_STRING"
    )
    INDEXING_QUEUE_NAME: str = Field(default="kb-indexing-jobs", env="INDEXING_QUEUE_NAME")
    QUEUE_POLL_INTERVAL: int = Field(default=5, env="QUEUE_POLL_INTERVAL")

    # Azure Document Intelligence
    AZURE_DOC_INTELLIGENCE_ENDPOINT: Optional[str] = Field(
        default=None, env="AZURE_DOC_INTELLIGENCE_ENDPOINT"
    )
    AZURE_DOC_INTELLIGENCE_KEY: Optional[str] = Field(
        default=None, env="AZURE_DOC_INTELLIGENCE_KEY"
    )


class LLMSettings(BaseSettings):
    """LLM and Embedding configuration"""

    # Azure OpenAI LLM
    AZURE_OPENAI_LLM_MODEL_API_KEY: str = Field(default="", env="AZURE_OPENAI_LLM_MODEL_API_KEY")
    AZURE_OPENAI_LLM_MODEL_API_BASE: str = Field(
        default="", env="AZURE_OPENAI_LLM_MODEL_API_BASE"
    )
    AZURE_OPENAI_LLM_MODEL_API_VERSION: str = Field(
        default="2024-02-15-preview", env="AZURE_OPENAI_LLM_MODEL_API_VERSION"
    )
    AZURE_OPENAI_LLM_MODEL_LLM_MODEL: str = Field(
        default="gpt-4", env="AZURE_OPENAI_LLM_MODEL_LLM_MODEL"
    )

    # Azure OpenAI Embeddings
    AZURE_OPENAI_EMBEDDING_MODEL_API_KEY: str = Field(
        default="", env="AZURE_OPENAI_EMBEDDING_MODEL_API_KEY"
    )
    AZURE_OPENAI_EMBEDDING_MODEL_API_BASE: str = Field(
        default="", env="AZURE_OPENAI_EMBEDDING_MODEL_API_BASE"
    )
    AZURE_OPENAI_EMBEDDING_MODEL_API_VERSION: str = Field(
        default="2024-02-15-preview", env="AZURE_OPENAI_EMBEDDING_MODEL_API_VERSION"
    )
    AZURE_OPENAI_EMBEDDING_MODEL_EMBEDDING_MODEL: str = Field(
        default="text-embedding-3-large", env="AZURE_OPENAI_EMBEDDING_MODEL_EMBEDDING_MODEL"
    )

    # Ollama (for embeddings)
    OLLAMA_MODEL_BASE_URL: str = Field(default="http://localhost:11434", env="OLLAMA_MODEL_BASE_URL")
    OLLAMA_MODEL_EMBEDDING_MODEL: str = Field(
        default="mxbai-embed-large", env="OLLAMA_MODEL_EMBEDDING_MODEL"
    )
    OLLAMA_MODEL_EMBEDDING_MODEL_DIMS: int = Field(
        default=1024, env="OLLAMA_MODEL_EMBEDDING_MODEL_DIMS"
    )
    OLLAMA_MODEL_EMBEDDING_MODEL_MAX_TOKENS: int = Field(
        default=8192, env="OLLAMA_MODEL_EMBEDDING_MODEL_MAX_TOKENS"
    )


class ProcessingSettings(BaseSettings):
    """Document processing configuration"""

    # PDF processing
    PDF_MIN_TEXT_CHARS: int = Field(default=200, env="PDF_MIN_TEXT_CHARS")
    PDF_MIN_TEXT_PER_PAGE: int = Field(default=100, env="PDF_MIN_TEXT_PER_PAGE")
    PDF_PER_PAGE_OCR: bool = Field(default=True, env="PDF_PER_PAGE_OCR")

    # Cache directories
    INDEXER_CACHE_DIR: str = Field(default="./indexer_cache", env="INDEXER_CACHE_DIR")
    INDEXER_STATE_DIR: str = Field(default="./indexer_state", env="INDEXER_STATE_DIR")


class Settings(BaseSettings):
    """Main application settings"""

    # Application settings
    APP_NAME: str = Field(default="KB Indexer Service", env="APP_NAME")
    VERSION: str = Field(default="1.0.0", env="APP_VERSION")
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=False, env="DEBUG")

    # Logging settings
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(default="json", env="LOG_FORMAT")

    # Worker settings
    MAX_CONCURRENT_JOBS: int = Field(default=10, env="MAX_CONCURRENT_JOBS")
    MESSAGE_VISIBILITY_TIMEOUT: int = Field(default=300, env="MESSAGE_VISIBILITY_TIMEOUT")
    MAX_RETRIES: int = Field(default=3, env="MAX_RETRIES")

    # Nested settings
    database: DatabaseSettings = DatabaseSettings()
    azure: AzureSettings = AzureSettings()
    llm: LLMSettings = LLMSettings()
    storage: StorageSettings = StorageSettings()
    processing: ProcessingSettings = ProcessingSettings()

    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()

    @validator("ENVIRONMENT")
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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
