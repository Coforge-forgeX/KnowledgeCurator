"""MongoDB-backed store for LLM router credentials and provider selection.

This module now delegates to the common_adapters package so all agents
share the same implementation without code duplication.
"""

from common_adapters.configurableAI.llm_router_config_store import (  # noqa: F401
    LLMRouterConfigStore,
    llm_router_config_store,
    SUPPORTED_PROVIDERS,
    DEFAULT_CONFIG_KEY,
    LLM_CONFIG_DB_NAME,
    LLM_CONFIG_COLLECTION_NAME,
)
