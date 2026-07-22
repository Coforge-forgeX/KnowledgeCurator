"""Configuration management for KB REST API"""
from typing import List, Optional, Union
from urllib.parse import quote_plus

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseSettings):
    """Database configuration settings"""

    # PostgreSQL settings
    POSTGRESQL_DATABASE_HOST: str = Field(
        default="localhost", env="POSTGRESQL_DATABASE_HOST"
    )
    POSTGRESQL_DATABASE_PORT: int = Field(default=5432, env="POSTGRESQL_DATABASE_PORT")
    POSTGRESQL_DATABASE_USER: str = Field(
        default="postgres", env="POSTGRESQL_DATABASE_USER"
    )
    POSTGRESQL_DATABASE_PASSWORD: str = Field(
        default="", env="POSTGRESQL_DATABASE_PASSWORD"
    )
    POSTGRESQL_DATABASE_DATABASE: str = Field(
        default="kbcurator", env="POSTGRESQL_DATABASE_DATABASE"
    )

    # MongoDB settings
    MONGODB_URI: str = Field(default="", env="MONGODB_URI")
    MONGODB_DATABASE_URI: str = Field(default="", env="MONGODB_DATABASE_URI")  # Alias
    MONGODB_DATABASE: str = Field(default="kb_sessions", env="MONGODB_DATABASE")
    MONGODB_DATABASE_NAME: str = Field(default="kb_conversations", env="MONGODB_DATABASE_NAME")  # Alias

    # Redis settings
    REDIS_HOST: Optional[str] = Field(default=None, env="REDIS_HOST")
    REDIS_PORT: int = Field(default=6379, env="REDIS_PORT")
    REDIS_PASSWORD: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    REDIS_DB: int = Field(default=0, env="REDIS_DB")

    # Neo4j settings
    NEO4J_URI: Optional[str] = Field(default=None, env="NEO4J_DATABASE_NEO4J_BOLT_URI")
    NEO4J_USER: str = Field(default="neo4j", env="NEO4J_DATABASE_NEO4J_USER")
    NEO4J_PASSWORD: Optional[str] = Field(default=None, env="NEO4J_DATABASE_NEO4J_PASSWORD")

    # Connection pool settings
    DB_POOL_SIZE: int = Field(default=5, env="DB_POOL_SIZE")
    DB_MAX_OVERFLOW: int = Field(default=5, env="DB_MAX_OVERFLOW")
    DB_POOL_TIMEOUT: int = Field(default=30, env="DB_POOL_TIMEOUT")
    DB_POOL_RECYCLE: int = Field(default=3600, env="DB_POOL_RECYCLE")
    DB_ECHO: bool = Field(default=False, env="DB_ECHO")

    @property
    def postgresql_url(self) -> str:
        """Build PostgreSQL async connection URL with properly encoded credentials"""
        # URL-encode username and password to handle special characters like @
        encoded_user = quote_plus(self.POSTGRESQL_DATABASE_USER)
        encoded_password = quote_plus(self.POSTGRESQL_DATABASE_PASSWORD)

        return (
            f"postgresql+asyncpg://{encoded_user}:"
            f"{encoded_password}@{self.POSTGRESQL_DATABASE_HOST}:"
            f"{self.POSTGRESQL_DATABASE_PORT}/{self.POSTGRESQL_DATABASE_DATABASE}"
            f"?ssl=require"
        )

    @property
    def redis_url(self) -> Optional[str]:
        """Build Redis connection URL if Redis is configured"""
        if not self.REDIS_HOST:
            return None
        if self.REDIS_PASSWORD:
            # URL-encode password to handle special characters
            encoded_password = quote_plus(self.REDIS_PASSWORD)
            return f"redis://:{encoded_password}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"


class StorageSettings(BaseSettings):
    """Storage configuration (multi-cloud support)"""

    # Storage provider
    STORAGE_PROVIDER: str = Field(default="azure", env="STORAGE_PROVIDER")
    STORAGE_CONTAINER_NAME: str = Field(default="aksknowledgecurator", env="STORAGE_CONTAINER_NAME")

    # Azure Blob Storage
    AZURE_BLOB_STORAGE_CONNECTION_STRING: Optional[str] = Field(
        default=None, env="AZURE_BLOB_STORAGE_CONNECTION_STRING"
    )
    AZURE_BLOB_STORAGE_CONTAINER_NAME: str = Field(
        default="aksknowledgecurator", env="AZURE_BLOB_STORAGE_CONTAINER_NAME"
    )

    # AWS S3
    AWS_ACCESS_KEY_ID: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    AWS_REGION: str = Field(default="us-east-1", env="AWS_REGION")


class BlobSettings(BaseSettings):
    """Azure Blob specific settings (for blob_client.py)"""

    AZURE_STORAGE_CONNECTION_STRING: str = Field(default="", env="AZURE_STORAGE_CONNECTION_STRING")
    BLOB_CONTAINER_NAME: str = Field(default="documents", env="BLOB_CONTAINER_NAME")
    BLOB_PATH_PREFIX: str = Field(default="", env="BLOB_PATH_PREFIX")
    BLOB_URL_EXPIRY_MINUTES: int = Field(default=60, env="BLOB_URL_EXPIRY_MINUTES")


class AzureSettings(BaseSettings):
    """Azure-specific configuration settings"""

    # Azure Functions settings
    AZURE_STORAGE_CONNECTION_STRING: str = Field(default="", env="AzureWebJobsStorage")

    # Azure Blob Storage (legacy)
    BLOB_STORAGE_CONNECTION_STRING: str = Field(
        default="", env="BLOB_STORAGE_CONNECTION_STRING"
    )
    BLOB_CONTAINER_NAME: str = Field(default="kb-documents", env="AZURE_BLOB_STORAGE_CONTAINER_NAME")
    WORKSPACE_CONTAINER_NAME: str = Field(default="workspace", env="AZURE_BLOB_STORAGE_WORKSPACE_CONTAINER_NAME")

    # Azure Queue Storage
    QUEUE_STORAGE_CONNECTION_STRING: str = Field(
        default="", env="AZURE_QUEUE_STORAGE_CONNECTION_STRING"
    )
    INDEXING_QUEUE_NAME: str = Field(default="indexing-queue", env="AZURE_INDEXING_QUEUE_NAME")


class SecuritySettings(BaseSettings):
    """Security and authentication settings"""

    # JWT settings
    JWT_SECRET_KEY: str = Field(
        default="your-secret-key-change-in-production", env="JWT_SECRET_KEY"
    )
    JWT_ALGORITHM: str = Field(default="HS256", env="JWT_ALGORITHM")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30, env="JWT_ACCESS_TOKEN_EXPIRE_MINUTES"
    )
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = Field(
        default=7, env="JWT_REFRESH_TOKEN_EXPIRE_DAYS"
    )

    # CORS settings
    CORS_ORIGINS: Union[List[str], str] = Field(default=["*"], env="CORS_ORIGINS")
    CORS_ALLOW_CREDENTIALS: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    CORS_ALLOW_METHODS: List[str] = Field(default=["*"], env="CORS_ALLOW_METHODS")
    CORS_ALLOW_HEADERS: List[str] = Field(default=["*"], env="CORS_ALLOW_HEADERS")

    @validator("CORS_ORIGINS", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            # Handle empty string
            if not v or v.strip() == "":
                return ["*"]
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @validator("JWT_SECRET_KEY")
    def validate_jwt_secret(cls, v):
        """Validate JWT secret key is not using default insecure value"""
        if v in [
            "your-secret-key",
            "secret",
            "changeme",
            "your-secret-key-change-in-production",
        ]:
            import warnings

            warnings.warn(
                "JWT_SECRET_KEY is using a default insecure value. "
                "Please change it in production!",
                UserWarning,
            )
        if len(v) < 32:
            raise ValueError("JWT_SECRET_KEY must be at least 32 characters long")
        return v


class LightRAGSettings(BaseSettings):
    """LightRAG configuration settings"""

    # LightRAG working directory
    LIGHTRAG_WORKING_DIR: str = Field(
        default="./lightrag_data", env="LIGHTRAG_WORKING_DIR"
    )

    # Vector storage settings
    VECTOR_STORAGE_TYPE: str = Field(
        default="PGVectorStorage", env="LIGHTRAG_VECTOR_STORAGE_TYPE"
    )
    GRAPH_STORAGE_TYPE: str = Field(
        default="Neo4JStorage", env="LIGHTRAG_GRAPH_STORAGE_TYPE"
    )

    # PostgreSQL settings for PGVectorStorage
    LIGHTRAG_POSTGRESQL_HOST: Optional[str] = Field(
        default=None, env="LIGHTRAG_POSTGRESQL_DATABASE_HOST"
    )
    LIGHTRAG_POSTGRESQL_USER: Optional[str] = Field(
        default=None, env="LIGHTRAG_POSTGRESQL_DATABASE_USER"
    )
    LIGHTRAG_POSTGRESQL_PASSWORD: Optional[str] = Field(
        default=None, env="LIGHTRAG_POSTGRESQL_DATABASE_PASSWORD"
    )
    LIGHTRAG_POSTGRESQL_DATABASE: Optional[str] = Field(
        default=None, env="LIGHTRAG_POSTGRESQL_DATABASE_DATABASE"
    )

    # Chunk settings
    CHUNK_TOKEN_SIZE: int = Field(default=1000, env="LIGHTRAG_CHUNK_TOKEN_SIZE")
    CHUNK_OVERLAP_TOKEN_SIZE: int = Field(
        default=200, env="LIGHTRAG_CHUNK_OVERLAP_TOKEN_SIZE"
    )

    # Ollama embedding settings
    OLLAMA_MODEL_BASE_URL: Optional[str] = Field(
        default=None, env="OLLAMA_MODEL_BASE_URL"
    )
    OLLAMA_MODEL_EMBEDDING_MODEL: Optional[str] = Field(
        default=None, env="OLLAMA_MODEL_EMBEDDING_MODEL"
    )
    OLLAMA_MODEL_EMBEDDING_MODEL_DIMS: int = Field(
        default=3072, env="OLLAMA_MODEL_EMBEDDING_MODEL_DIMS"
    )
    OLLAMA_MODEL_EMBEDDING_MODEL_MAX_TOKENS: int = Field(
        default=8192, env="OLLAMA_MODEL_EMBEDDING_MODEL_MAX_TOKENS"
    )

    # Azure OpenAI LLM settings
    AZURE_OPENAI_LLM_MODEL_API_KEY: Optional[str] = Field(
        default=None, env="AZURE_OPENAI_LLM_MODEL_API_KEY"
    )
    AZURE_OPENAI_LLM_MODEL_API_BASE: Optional[str] = Field(
        default=None, env="AZURE_OPENAI_LLM_MODEL_API_BASE"
    )
    AZURE_OPENAI_LLM_MODEL_API_VERSION: str = Field(
        default="2024-02-15-preview", env="AZURE_OPENAI_LLM_MODEL_API_VERSION"
    )
    AZURE_OPENAI_LLM_MODEL_LLM_MODEL: Optional[str] = Field(
        default=None, env="AZURE_OPENAI_LLM_MODEL_LLM_MODEL"
    )


class Settings(BaseSettings):
    """Main application settings"""

    # Application settings
    APP_NAME: str = Field(default="KB REST Service", env="APP_NAME")
    VERSION: str = Field(default="1.0.0", env="APP_VERSION")
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=False, env="DEBUG")

    # Server settings
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")

    # Logging settings
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(default="json", env="LOG_FORMAT")

    # Rate limiting
    RATE_LIMIT_ENABLED: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    RATE_LIMIT_REQUESTS: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    RATE_LIMIT_WINDOW: int = Field(default=60, env="RATE_LIMIT_WINDOW")

    # Request validation
    MAX_REQUEST_SIZE: int = Field(
        default=10485760, env="MAX_REQUEST_SIZE"
    )  # 10MB default

    # Nested settings
    database: DatabaseSettings = DatabaseSettings()
    azure: AzureSettings = AzureSettings()
    security: SecuritySettings = SecuritySettings()
    lightrag: LightRAGSettings = LightRAGSettings()
    storage: StorageSettings = StorageSettings()
    blob: BlobSettings = BlobSettings()

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

    @property
    def app_name(self) -> str:
        """Get application name"""
        return self.APP_NAME

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
