"""Application configuration management for Indexer Service"""
import os
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

# Get the absolute path to the .env file (in the project root)
# Config file is in: services/indexer-service/src/core/config.py
# .env file is in: services/indexer-service/.env
_config_dir = Path(__file__).parent  # src/core/
_project_root = _config_dir.parent.parent  # services/indexer-service/
_env_file_path = _project_root / ".env"

# Explicitly load .env file into os.environ before Pydantic initializes
# This ensures environment variables are available when Settings is instantiated
if _env_file_path.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_file_path, override=False)


class DatabaseSettings(BaseModel):
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


class StorageSettings(BaseModel):
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



class AzureSettings(BaseModel):
    """Azure-specific configuration settings"""

    # Azure Storage Queue
    AZURE_STORAGE_CONNECTION_STRING: str = Field(
        default="", env="AZURE_STORAGE_CONNECTION_STRING"
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


class LLMSettings(BaseModel):
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



class ProcessingSettings(BaseModel):
    """Document processing configuration"""

    # PDF processing
    PDF_MIN_TEXT_CHARS: int = Field(default=200, env="PDF_MIN_TEXT_CHARS")
    PDF_MIN_TEXT_PER_PAGE: int = Field(default=100, env="PDF_MIN_TEXT_PER_PAGE")
    PDF_PER_PAGE_OCR: bool = Field(default=True, env="PDF_PER_PAGE_OCR")

    # Cache directories
    INDEXER_CACHE_DIR: str = Field(default="./indexer_cache", env="INDEXER_CACHE_DIR")
    INDEXER_STATE_DIR: str = Field(default="./indexer_state", env="INDEXER_STATE_DIR")



class LightRAGSettings(BaseModel):
    """LightRAG configuration settings"""

    WORKING_DIR: str = Field(default="./lightrag_data")
    EMBEDDING_DIM: int = Field(default=3072)
    MAX_TOKEN_SIZE: int = Field(default=8192)
    CHUNK_TOKEN_SIZE: int = Field(default=1000)
    CHUNK_OVERLAP_TOKEN_SIZE: int = Field(default=200)
    GRAPH_STORAGE_TYPE: str = Field(default="Neo4JStorage")
    VECTOR_STORAGE_TYPE: str = Field(default="PGVectorStorage")

    # Azure OpenAI LLM settings (copy from LLMSettings for easier access)
    AZURE_OPENAI_LLM_API_KEY: Optional[str] = Field(default=None)
    AZURE_OPENAI_LLM_API_BASE: Optional[str] = Field(default=None)
    AZURE_OPENAI_LLM_API_VERSION: str = Field(default="2024-12-01-preview")
    AZURE_OPENAI_LLM_DEPLOYMENT: Optional[str] = Field(default=None)

    # Azure OpenAI Embedding settings
    AZURE_OPENAI_EMBEDDING_API_KEY: Optional[str] = Field(default=None)
    AZURE_OPENAI_EMBEDDING_API_BASE: Optional[str] = Field(default=None)
    AZURE_OPENAI_EMBEDDING_API_VERSION: str = Field(default="2024-02-01")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: Optional[str] = Field(default=None)

    @model_validator(mode='before')
    @classmethod
    def load_from_env(cls, values):
        """Load values from environment variables if not provided"""
        import os

        # Map of field names to environment variable names
        env_map = {
            'WORKING_DIR': 'LIGHTRAG_WORKING_DIR',
            'EMBEDDING_DIM': 'LIGHTRAG_EMBEDDING_DIM',
            'MAX_TOKEN_SIZE': 'LIGHTRAG_MAX_TOKEN_SIZE',
            'CHUNK_TOKEN_SIZE': 'LIGHTRAG_CHUNK_TOKEN_SIZE',
            'CHUNK_OVERLAP_TOKEN_SIZE': 'LIGHTRAG_CHUNK_OVERLAP_TOKEN_SIZE',
            'GRAPH_STORAGE_TYPE': 'LIGHTRAG_GRAPH_STORAGE_TYPE',
            'VECTOR_STORAGE_TYPE': 'LIGHTRAG_VECTOR_STORAGE_TYPE',
            'AZURE_OPENAI_LLM_API_KEY': 'AZURE_OPENAI_LLM_MODEL_API_KEY',
            'AZURE_OPENAI_LLM_API_BASE': 'AZURE_OPENAI_LLM_MODEL_API_BASE',
            'AZURE_OPENAI_LLM_API_VERSION': 'AZURE_OPENAI_LLM_MODEL_API_VERSION',
            'AZURE_OPENAI_LLM_DEPLOYMENT': 'AZURE_OPENAI_LLM_MODEL_LLM_MODEL',
            'AZURE_OPENAI_EMBEDDING_API_KEY': 'AZURE_OPENAI_EMBEDDING_MODEL_API_KEY',
            'AZURE_OPENAI_EMBEDDING_API_BASE': 'AZURE_OPENAI_EMBEDDING_MODEL_API_BASE',
            'AZURE_OPENAI_EMBEDDING_API_VERSION': 'AZURE_OPENAI_EMBEDDING_MODEL_API_VERSION',
            'AZURE_OPENAI_EMBEDDING_DEPLOYMENT': 'AZURE_OPENAI_EMBEDDING_MODEL_EMBEDDING_MODEL',
        }

        # Load from environment if not in values
        for field_name, env_name in env_map.items():
            if field_name not in values or values.get(field_name) is None:
                env_value = os.getenv(env_name)
                if env_value is not None:
                    # Convert to int for numeric fields
                    if field_name in ['EMBEDDING_DIM', 'MAX_TOKEN_SIZE', 'CHUNK_TOKEN_SIZE', 'CHUNK_OVERLAP_TOKEN_SIZE']:
                        values[field_name] = int(env_value)
                    else:
                        values[field_name] = env_value

        return values

    # Ollama settings
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434", env="OLLAMA_MODEL_BASE_URL")
    OLLAMA_EMBEDDING_MODEL: str = Field(default="mxbai-embed-large", env="OLLAMA_MODEL_EMBEDDING_MODEL")
    OLLAMA_EMBEDDING_DIMS: int = Field(default=1024, env="OLLAMA_MODEL_EMBEDDING_MODEL_DIMS")
    OLLAMA_MAX_TOKENS: int = Field(default=8192, env="OLLAMA_MODEL_EMBEDDING_MODEL_MAX_TOKENS")



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
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    azure: AzureSettings = Field(default_factory=AzureSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    lightrag: LightRAGSettings = Field(default_factory=LightRAGSettings)

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()

    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, v):
        valid_envs = ["dev", "stage", "prod"]
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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow"  # Allow extra fields from nested settings
    )


# Global settings instance
settings = Settings()
