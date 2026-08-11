"""Test configuration management"""
import pytest

from src.core.config import DatabaseSettings, AzureSettings, LLMSettings, Settings


def test_database_settings_defaults():
    """Test DatabaseSettings default values"""
    db_settings = DatabaseSettings()
    assert db_settings.POSTGRESQL_DATABASE_HOST == "localhost"
    assert db_settings.POSTGRESQL_DATABASE_PORT == 5432
    assert db_settings.NEO4J_DATABASE_NEO4J_USER == "neo4j"


def test_database_settings_postgresql_url():
    """Test PostgreSQL URL construction"""
    db_settings = DatabaseSettings(
        POSTGRESQL_DATABASE_USER="testuser",
        POSTGRESQL_DATABASE_PASSWORD="testpass",
        POSTGRESQL_DATABASE_HOST="testhost",
        POSTGRESQL_DATABASE_PORT=5432,
        POSTGRESQL_DATABASE_DATABASE="testdb",
    )
    expected_url = "postgresql://testuser:testpass@testhost:5432/testdb"
    assert db_settings.postgresql_url == expected_url


def test_database_settings_neo4j_uri():
    """Test Neo4j URI property"""
    db_settings = DatabaseSettings(
        NEO4J_DATABASE_NEO4J_BOLT_URI="bolt://testhost:7687"
    )
    assert db_settings.neo4j_uri == "bolt://testhost:7687"


def test_azure_settings_defaults():
    """Test AzureSettings default values"""
    azure_settings = AzureSettings()
    assert azure_settings.INDEXING_QUEUE_NAME == "kb-indexing-jobs"
    assert azure_settings.QUEUE_POLL_INTERVAL == 5


def test_llm_settings_defaults():
    """Test LLMSettings default values"""
    llm_settings = LLMSettings()
    assert llm_settings.AZURE_OPENAI_LLM_MODEL_LLM_MODEL == "gpt-4"
    assert (
        llm_settings.AZURE_OPENAI_EMBEDDING_MODEL_EMBEDDING_MODEL
        == "text-embedding-3-large"
    )


def test_settings_nested_structure():
    """Test Settings with nested configurations"""
    settings = Settings()
    assert isinstance(settings.database, DatabaseSettings)
    assert isinstance(settings.azure, AzureSettings)
    assert isinstance(settings.llm, LLMSettings)


def test_settings_log_level_validation():
    """Test LOG_LEVEL validation"""
    settings = Settings(LOG_LEVEL="DEBUG")
    assert settings.LOG_LEVEL == "DEBUG"

    with pytest.raises(ValueError):
        Settings(LOG_LEVEL="INVALID")


def test_settings_environment_validation():
    """Test ENVIRONMENT validation"""
    settings = Settings(ENVIRONMENT="production")
    assert settings.ENVIRONMENT == "production"
    assert settings.is_production is True
    assert settings.is_development is False

    with pytest.raises(ValueError):
        Settings(ENVIRONMENT="invalid")
