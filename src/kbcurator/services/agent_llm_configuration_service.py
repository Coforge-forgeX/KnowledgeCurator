"""Agent LLM configuration service backed by MongoDB config documents."""

from __future__ import annotations

from typing import Dict, List, Optional

from kbcurator.services.llm_router_config_store import llm_router_config_store


class AgentLLMConfigurationService:
    """Compatibility service delegating agent configuration operations to MongoDB config store."""

    def get_configuration(self, workspace_id: int, agent_id: Optional[int] = None) -> Optional[Dict]:
        return llm_router_config_store.get_configuration(workspace_id, agent_id)

    def get_effective_configuration(self, workspace_id: int, agent_id: int) -> Optional[Dict]:
        return llm_router_config_store.get_effective_configuration(workspace_id, agent_id)

    def create_or_update_configuration(
        self,
        workspace_id: int,
        agent_id: Optional[int] = None,
        configured_providers: Optional[List[str]] = None,
        current_provider: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> Dict:
        return llm_router_config_store.create_or_update_configuration(
            workspace_id=workspace_id,
            agent_id=agent_id,
            configured_providers=configured_providers,
            current_provider=current_provider,
            user_id=user_id,
        )

    def switch_provider(
        self,
        workspace_id: int,
        provider: str,
        agent_id: Optional[int] = None,
        user_id: Optional[int] = None,
    ) -> Dict:
        return llm_router_config_store.switch_provider(
            workspace_id=workspace_id,
            provider=provider,
            agent_id=agent_id,
            user_id=user_id,
        )

    def add_provider(
        self,
        workspace_id: int,
        provider: str,
        agent_id: Optional[int] = None,
        set_as_current: bool = False,
        user_id: Optional[int] = None,
    ) -> Dict:
        return llm_router_config_store.add_provider(
            workspace_id=workspace_id,
            provider=provider,
            agent_id=agent_id,
            set_as_current=set_as_current,
            user_id=user_id,
        )

    def get_current_provider(
        self,
        workspace_id: int,
        agent_id: Optional[int] = None,
    ) -> Optional[str]:
        return llm_router_config_store.get_current_provider(
            workspace_id=workspace_id,
            agent_id=agent_id,
        )

    def list_configured_providers(
        self,
        workspace_id: int,
        agent_id: Optional[int] = None,
    ) -> List[str]:
        return llm_router_config_store.list_configured_providers(
            workspace_id=workspace_id,
            agent_id=agent_id,
        )

    def delete_configuration(
        self,
        workspace_id: int,
        agent_id: Optional[int] = None,
        user_id: Optional[int] = None,
    ) -> bool:
        return llm_router_config_store.delete_configuration(
            workspace_id=workspace_id,
            agent_id=agent_id,
            user_id=user_id,
        )

    def bulk_create_agent_configurations(
        self,
        workspace_id: int,
        agent_ids: List[int],
        configured_providers: Optional[List[str]] = None,
        current_provider: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> List[Dict]:
        return llm_router_config_store.bulk_create_agent_configurations(
            workspace_id=workspace_id,
            agent_ids=agent_ids,
            configured_providers=configured_providers,
            current_provider=current_provider,
            user_id=user_id,
        )

    def get_workspace_configurations(self, workspace_id: int) -> List[Dict]:
        return llm_router_config_store.get_workspace_configurations(workspace_id)

    def delete_workspace_configurations(self, workspace_id: int, user_id: Optional[int] = None) -> int:
        return llm_router_config_store.delete_workspace_configurations(
            workspace_id=workspace_id,
            user_id=user_id,
        )


agent_llm_config_service = AgentLLMConfigurationService()
