"""Workspace provider credentials service backed by MongoDB config documents."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from kbcurator.services.llm_router_config_store import (
    SUPPORTED_PROVIDERS,
    llm_router_config_store,
)


class WorkspaceProviderCredentialsService:
    """Compatibility service delegating credential operations to MongoDB config store."""

    def get_provider_credentials(
        self, workspace_id: int, provider_name: str
    ) -> Optional[Dict[str, Any]]:
        return llm_router_config_store.get_provider_credentials(workspace_id, provider_name)

    def list_workspace_providers(self, workspace_id: int) -> List[Dict[str, Any]]:
        return llm_router_config_store.list_workspace_providers(workspace_id)

    def build_config_dict(
        self, workspace_id: int, provider_name: str
    ) -> Optional[Dict[str, Any]]:
        return llm_router_config_store.build_config_dict(workspace_id, provider_name)

    def upsert_provider_credentials(
        self,
        workspace_id: int,
        provider_name: str,
        api_key: str,
        endpoint: str,
        model: str,
        api_version: Optional[str] = None,
        deployment_name: Optional[str] = None,
        extra_config: Optional[Dict[str, Any]] = None,
        user_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        return llm_router_config_store.upsert_provider_credentials(
            workspace_id=workspace_id,
            provider_name=provider_name,
            api_key=api_key,
            endpoint=endpoint,
            model=model,
            api_version=api_version,
            deployment_name=deployment_name,
            extra_config=extra_config,
            user_id=user_id,
        )

    def deactivate_provider_credentials(
        self,
        workspace_id: int,
        provider_name: str,
        user_id: Optional[int] = None,
    ) -> bool:
        return llm_router_config_store.deactivate_provider_credentials(
            workspace_id=workspace_id,
            provider_name=provider_name,
            user_id=user_id,
        )


workspace_provider_credentials_service = WorkspaceProviderCredentialsService()
