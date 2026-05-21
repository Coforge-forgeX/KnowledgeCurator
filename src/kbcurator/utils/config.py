
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # =====================
    # Database Configuration
    # =====================
    POSTGRES_HOST: str = Field(..., validation_alias="POSTGRESQL_DATABASE_HOST")
    POSTGRES_PORT: int = Field(5432, validation_alias="POSTGRESQL_DATABASE_PORT")
    POSTGRES_DB: str = Field(..., validation_alias="POSTGRESQL_DATABASE_DATABASE")
    POSTGRES_USER: str = Field(..., validation_alias="POSTGRESQL_DATABASE_USER")
    POSTGRES_PASSWORD: str = Field(..., validation_alias="POSTGRESQL_DATABASE_PASSWORD")
    POSTGRES_TABLE_WORKSPACE: str = Field("workspace_master", validation_alias="POSTGRESQL_DATABASE_WORKSPACE_TABLE")
    POSTGRES_TABLE_USER: str = Field("user_details", validation_alias="POSTGRESQL_DATABASE_USER_TABLE")

    # MongoDB
    MONGODB_DATABASE_URI: Optional[str] = Field(None, env="MONGODB_DATABASE_URI")
    MONGODB_DATABASE_TTL: int = Field(180, env="MONGODB_DATABASE_TTL")

    # Neo4j
    NEO4J_DATABASE_NEO4J_BOLT_URI: Optional[str] = Field(None, env="NEO4J_DATABASE_NEO4J_BOLT_URI")
    NEO4J_DATABASE_NEO4J_USER: Optional[str] = Field(None, env="NEO4J_DATABASE_NEO4J_USER")
    NEO4J_DATABASE_NEO4J_PASSWORD: Optional[str] = Field(None, env="NEO4J_DATABASE_NEO4J_PASSWORD")

    # =====================
    # JWT & Auth
    # =====================
    JWT_SECRET: str = Field("change-this-in-prod", env="JWT_SECRET")
    JWT_ALGORITHM: str = Field("HS256", env="JWT_ALGORITHM")
    JWT_ACCESS_TOKEN_EXPIRY_SECONDS: int = Field(86400, env="JWT_ACCESS_TOKEN_EXPIRY_SECONDS")
    JWT_REFRESH_TOKEN_EXPIRY_SECONDS: int = Field(86400, env="JWT_REFRESH_TOKEN_EXPIRY_SECONDS")
    JWT_TRANSPORT_ENCODE: bool = Field(True, env="JWT_TRANSPORT_ENCODE")
    JWT_SET_ACCESS_COOKIE: bool = Field(True, env="JWT_SET_ACCESS_COOKIE")
    JWT_RETURN_RAW_ACCESS: bool = Field(False, env="JWT_RETURN_RAW_ACCESS")

    # =====================
    # Redis
    # =====================
    REDIS_HOST: str = Field("localhost", env="REDIS_HOST")
    REDIS_PORT: int = Field(6379, env="REDIS_PORT")
    REDIS_PASSWORD: Optional[str] = Field(None, env="REDIS_PASSWORD")
    REDIS_SSL: bool = Field(True, env="REDIS_SSL")

    # =====================
    # Azure Blob Storage
    # =====================
    AZURE_BLOB_STORAGE_CONNECTION_STRING: Optional[str] = Field(None, env="AZURE_BLOB_STORAGE_CONNECTION_STRING")
    AZURE_BLOB_STORAGE_CONTAINER_NAME: Optional[str] = Field(None, env="AZURE_BLOB_STORAGE_CONTAINER_NAME")

    # =====================
    # Azure Document Intelligence
    # =====================
    AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT: Optional[str] = Field(None, env="AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    AZURE_DOCUMENT_INTELLIGENCE_KEY: Optional[str] = Field(None, env="AZURE_DOCUMENT_INTELLIGENCE_KEY")

    # =====================
    # Azure OpenAI
    # =====================
    AZURE_OPENAI_EMBEDDING_MODEL_API_BASE: Optional[str] = Field(None, env="AZURE_OPENAI_EMBEDDING_MODEL_API_BASE")
    AZURE_OPENAI_EMBEDDING_MODEL_API_KEY: Optional[str] = Field(None, env="AZURE_OPENAI_EMBEDDING_MODEL_API_KEY")
    AZURE_OPENAI_EMBEDDING_MODEL_API_VERSION: Optional[str] = Field(None, env="AZURE_OPENAI_EMBEDDING_MODEL_API_VERSION")
    AZURE_OPENAI_EMBEDDING_MODEL_EMBEDDING_MODEL: Optional[str] = Field(None, env="AZURE_OPENAI_EMBEDDING_MODEL_EMBEDDING_MODEL")
    AZURE_OPENAI_LLM_MODEL_API_BASE: Optional[str] = Field(None, env="AZURE_OPENAI_LLM_MODEL_API_BASE")
    AZURE_OPENAI_LLM_MODEL_API_KEY: Optional[str] = Field(None, env="AZURE_OPENAI_LLM_MODEL_API_KEY")
    AZURE_OPENAI_LLM_MODEL_API_VERSION: Optional[str] = Field(None, env="AZURE_OPENAI_LLM_MODEL_API_VERSION")
    AZURE_OPENAI_LLM_MODEL_LLM_MODEL: Optional[str] = Field(None, env="AZURE_OPENAI_LLM_MODEL_LLM_MODEL")

    # =====================
    # Quasar & External APIs
    # =====================
    QUASAR_ENDPOINT_URL: Optional[str] = Field(None, env="QUASAR_ENDPOINT_URL")
    QUASAR_MODEL: Optional[str] = Field(None, env="QUASAR_MODEL")
    QUSAR_API_KEY: Optional[str] = Field(None, env="QUSAR_API_KEY")

    # =====================
    # Langfuse
    # =====================
    LANGFUSE_HOST: Optional[str] = Field(None, env="LANGFUSE_HOST")
    LANGFUSE_PUBLIC_KEY: Optional[str] = Field(None, env="LANGFUSE_PUBLIC_KEY")
    LANGFUSE_SECRET_KEY: Optional[str] = Field(None, env="LANGFUSE_SECRET_KEY")

    # =====================
    # OLLAMA Model
    # =====================
    OLLAMA_MODEL_BASE_URL: Optional[str] = Field(None, env="OLLAMA_MODEL_BASE_URL")
    OLLAMA_MODEL_EMBEDDING_MODEL: Optional[str] = Field(None, env="OLLAMA_MODEL_EMBEDDING_MODEL")
    OLLAMA_MODEL_EMBEDDING_MODEL_DIMS: Optional[int] = Field(None, env="OLLAMA_MODEL_EMBEDDING_MODEL_DIMS")
    OLLAMA_MODEL_EMBEDDING_MODEL_MAX_TOKENS: Optional[int] = Field(None, env="OLLAMA_MODEL_EMBEDDING_MODEL_MAX_TOKENS")
    OLLAMA_MODEL_LLM_MODEL: Optional[str] = Field(None, env="OLLAMA_MODEL_LLM_MODEL")
    OLLAMA_MODEL_TAG_API: Optional[str] = Field(None, env="OLLAMA_MODEL_TAG_API")

    # =====================
    # SharePoint Integration
    # =====================
    SHAREPOINT_INTEGRATION_CLIENT_ID: Optional[str] = Field(None, env="SHAREPOINT_INTEGRATION_CLIENT_ID")
    SHAREPOINT_INTEGRATION_CLIENT_SECRET: Optional[str] = Field(None, env="SHAREPOINT_INTEGRATION_CLIENT_SECRET")
    SHAREPOINT_INTEGRATION_TENANT_ID: Optional[str] = Field(None, env="SHAREPOINT_INTEGRATION_TENANT_ID")

    # =====================
    # Miscellaneous
    # =====================
    KC_SERVICE_URL: Optional[str] = Field(None, env="KC_SERVICE_URL")
    N8N_ORCHESTRATOR_INTEGRATION_WEBHOOK_URL: Optional[str] = Field(None, env="N8N_ORCHESTRATOR_INTEGRATION_WEBHOOK_URL")
    HTTP_FORWARDED_COUNT: Optional[int] = Field(1, env="HTTP_FORWARDED_COUNT")
    WEBSITE_PRESERVE_HOST_HEADER: Optional[bool] = Field(None, env="WEBSITE_PRESERVE_HOST_HEADER")
    WEBSITES_ENABLE_APP_SERVICE_STORAGE: Optional[bool] = Field(None, env="WEBSITES_ENABLE_APP_SERVICE_STORAGE")
    WEBSITES_PORT: Optional[int] = Field(9000, env="WEBSITES_PORT")
    PO_HOST: Optional[str] = Field(None, env="PO_HOST")
    PO_PORT: Optional[int] = Field(None, env="PO_PORT")
    REQUIRED_SCOPE: Optional[str] = Field(None, env="REQUIRED_SCOPE")
    TENANT_ID: Optional[str] = Field(None, env="TENANT_ID")
    AUDIENCE: Optional[str] = Field(None, env="AUDIENCE")
    CLIENT_ID: Optional[str] = Field(None, env="CLIENT_ID")
    VECTOR_STORE_VECTOR_CHUNK_SIZE: Optional[int] = Field(None, env="VECTOR_STORE_VECTOR_CHUNK_SIZE")
    VECTOR_STORE_VECTOR_OVERLAP_SIZE: Optional[int] = Field(None, env="VECTOR_STORE_VECTOR_OVERLAP_SIZE")
    VECTOR_STORE_MAX_CHUNKS: Optional[int] = Field(None, env="VECTOR_STORE_MAX_CHUNKS")
    VECTOR_STORE_BATCH_SIZE: Optional[int] = Field(None, env="VECTOR_STORE_BATCH_SIZE")
    SIMILARITY_SEARCH_SIM_SEARCH_K: Optional[int] = Field(None, env="SIMILARITY_SEARCH_SIM_SEARCH_K")
    RETRIEVER_RETRIEVER_SEARCH_TYPE: Optional[str] = Field(None, env="RETRIEVER_RETRIEVER_SEARCH_TYPE")
    RETRIEVER_RETRIEVER_K: Optional[int] = Field(None, env="RETRIEVER_RETRIEVER_K")

settings = Settings()
