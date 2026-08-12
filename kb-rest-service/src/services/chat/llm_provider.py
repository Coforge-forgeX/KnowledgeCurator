"""
LLMProvider backed by `common_adapters` (ConfigurableAIManager), the same
package the rest of the platform uses for provider-agnostic LLM access,
instead of a bespoke Azure-only client.

Provider/model selection is workspace- and agent-specific (mirrors
`LightRAGService._resolve_llm_router_config`): the effective config is
resolved from `common_adapters.configurableAI.llm_router_config_store`
first, falling back to the AZURE_OPENAI_LLM_MODEL_* environment settings
already used elsewhere in this service when no per-workspace override
exists. Built managers are cached per (workspace_id, agent_id) since
resolving the router config hits MongoDB.
"""
import asyncio
from typing import Dict, Optional, Tuple

from common_adapters.configurableAI.manager import ConfigurableAIManager

try:
    from common_adapters.configurableAI.llm_router_config_store import (
        llm_router_config_store,
    )
except ImportError:  # pragma: no cover - defensive, mirrors lightrag_service
    llm_router_config_store = None

from src.core.config import settings
from src.core.exceptions import ConfigurationException
from src.core.logging import get_logger
from src.services.intent_detection.protocols import LLMProvider

logger = get_logger(__name__)


def _env_fallback_config() -> Optional[Dict[str, object]]:
    api_key = settings.lightrag.AZURE_OPENAI_LLM_MODEL_API_KEY
    endpoint = settings.lightrag.AZURE_OPENAI_LLM_MODEL_API_BASE
    model = settings.lightrag.AZURE_OPENAI_LLM_MODEL_LLM_MODEL

    if not all([api_key, endpoint, model]):
        return None

    return {
        "provider_name": "azure",
        "api_key": api_key,
        "endpoint": endpoint,
        "model": model,
        "deployment_name": model,
        "api_version": settings.lightrag.AZURE_OPENAI_LLM_MODEL_API_VERSION,
    }


def _resolve_provider_config(workspace_id: int, agent_id: Optional[int]) -> Optional[Dict[str, object]]:
    """Resolve the effective provider config for a workspace/agent, common_adapters-first."""
    if llm_router_config_store is not None and agent_id is not None:
        try:
            effective = llm_router_config_store.get_effective_configuration(workspace_id, agent_id)
            provider = ((effective or {}).get("current_provider") or "").strip().lower()
            if provider:
                config = llm_router_config_store.build_config_dict(
                    workspace_id,
                    provider,
                    model_override=effective.get("current_model"),
                )
                if config:
                    return config
        except Exception as e:
            logger.warning(
                "Failed to resolve LLM router config, falling back to env settings",
                workspace_id=workspace_id,
                agent_id=agent_id,
                error=str(e),
            )

    return _env_fallback_config()


class CommonAdaptersLLMProvider(LLMProvider):
    """Adapts `ConfigurableAIManager` to the `LLMProvider` interface."""

    def __init__(self, workspace_id: int, agent_id: Optional[int]) -> None:
        self._workspace_id = workspace_id
        self._agent_id = agent_id
        self._manager: Optional[ConfigurableAIManager] = None
        self._provider_name: Optional[str] = None
        self._lock = asyncio.Lock()

    async def _ensure_manager(self) -> ConfigurableAIManager:
        if self._manager is not None:
            return self._manager

        async with self._lock:
            if self._manager is not None:
                return self._manager

            config = await asyncio.to_thread(
                _resolve_provider_config, self._workspace_id, self._agent_id
            )
            if not config:
                raise ConfigurationException(
                    message="No LLM provider is configured for this workspace",
                    config_key="AZURE_OPENAI_LLM_MODEL",
                )

            manager = ConfigurableAIManager(default_provider=config["provider_name"])
            manager.configure_provider(config["provider_name"], config)
            manager.set_current_provider(config["provider_name"])

            self._manager = manager
            self._provider_name = config["provider_name"]
            return manager

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 100,
        temperature: float = 0.0,
    ) -> str:
        manager = await self._ensure_manager()
        # The underlying providers only accept a single user prompt (no
        # dedicated system-message slot), so fold the system prompt in.
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        return await manager.generate_text_async(
            full_prompt,
            provider=self._provider_name,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    async def invoke_async(self, sys_prompt: str, input: str) -> str:
        """Adapter for `common_adapters.context_compaction`, which expects this shape."""
        return await self.complete(prompt=input, system_prompt=sys_prompt)


_provider_cache: Dict[Tuple[int, Optional[int]], LLMProvider] = {}


def get_llm_provider(workspace_id: int, agent_id: Optional[int] = None) -> LLMProvider:
    """Get (or build) the cached LLM provider for a workspace/agent pair."""
    key = (workspace_id, agent_id)
    if key not in _provider_cache:
        _provider_cache[key] = CommonAdaptersLLMProvider(workspace_id=workspace_id, agent_id=agent_id)
    return _provider_cache[key]
