"""MongoDB-backed store for LLM router credentials and provider selection."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import logging

from kbcurator.utils.mongodb_singleton import get_mongodb_client

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = ["azure", "quasar"]
DEFAULT_CONFIG_KEY = "__workspace_default__"
LLM_CONFIG_DB_NAME = "llm_configs"
LLM_CONFIG_COLLECTION_NAME = "workspace_configs"


class LLMRouterConfigStore:
    """CRUD abstraction over a single MongoDB config document per workspace."""

    def __init__(self):
        self._mongo = get_mongodb_client()
        self._collection = self._mongo.client[LLM_CONFIG_DB_NAME][LLM_CONFIG_COLLECTION_NAME]
        self._indexes_ready = False
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        if self._indexes_ready:
            return
        self._collection.create_index("workspace_id", unique=True, name="uq_workspace_id")
        self._collection.create_index("updated_at", name="idx_updated_at")
        self._indexes_ready = True

    @staticmethod
    def _utcnow() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _agent_key(agent_id: Optional[int]) -> str:
        return str(agent_id) if agent_id is not None else DEFAULT_CONFIG_KEY

    def _ensure_workspace_document(self, workspace_id: int) -> None:
        now = self._utcnow()
        self._collection.update_one(
            {"workspace_id": workspace_id},
            {
                "$setOnInsert": {
                    "workspace_id": workspace_id,
                    "provider_credentials": {},
                    "agent_configs": {},
                    "created_at": now,
                    "updated_at": now,
                }
            },
            upsert=True,
        )

    def _get_workspace_document(self, workspace_id: int) -> Optional[Dict[str, Any]]:
        return self._collection.find_one({"workspace_id": workspace_id})

    # ------------------------------------------------------------------
    # Provider credentials
    # ------------------------------------------------------------------

    def get_provider_credentials(self, workspace_id: int, provider_name: str) -> Optional[Dict[str, Any]]:
        provider = provider_name.lower().strip()
        doc = self._get_workspace_document(workspace_id)
        if not doc:
            return None

        entry = (doc.get("provider_credentials") or {}).get(provider)
        if not entry or not entry.get("is_active", True):
            return None

        return {
            "workspace_id": workspace_id,
            "provider_name": provider,
            "api_key": entry.get("api_key"),
            "endpoint": entry.get("endpoint"),
            "model": entry.get("model"),
            "api_version": entry.get("api_version"),
            "deployment_name": entry.get("deployment_name"),
            "extra_config": entry.get("extra_config") or {},
            "is_active": True,
            "created_at": entry.get("created_at"),
            "updated_at": entry.get("updated_at"),
            "created_by": entry.get("created_by"),
            "updated_by": entry.get("updated_by"),
        }

    def list_workspace_providers(self, workspace_id: int) -> List[Dict[str, Any]]:
        doc = self._get_workspace_document(workspace_id)
        if not doc:
            return []

        providers = []
        for provider_name in sorted((doc.get("provider_credentials") or {}).keys()):
            creds = self.get_provider_credentials(workspace_id, provider_name)
            if creds:
                providers.append(creds)
        return providers

    def build_config_dict(self, workspace_id: int, provider_name: str) -> Optional[Dict[str, Any]]:
        creds = self.get_provider_credentials(workspace_id, provider_name)
        if not creds:
            return None

        provider = provider_name.lower().strip()
        config = {
            "provider_name": provider,
            "api_key": creds["api_key"],
            "endpoint": creds["endpoint"],
            "model": creds["model"],
            "extra_params": creds.get("extra_config") or {},
        }
        if provider == "azure":
            config["deployment_name"] = creds.get("deployment_name") or creds.get("model")
            config["api_version"] = creds.get("api_version")
        return config

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
        provider = provider_name.lower().strip()
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Provider '{provider}' is not supported. Supported: {SUPPORTED_PROVIDERS}"
            )

        self._ensure_workspace_document(workspace_id)

        existing = self.get_provider_credentials(workspace_id, provider)
        now = self._utcnow()
        payload = {
            "api_key": api_key,
            "endpoint": endpoint,
            "model": model,
            "api_version": api_version,
            "deployment_name": deployment_name or model,
            "extra_config": extra_config or {},
            "is_active": True,
            "created_at": existing.get("created_at") if existing else now,
            "updated_at": now,
            "created_by": existing.get("created_by") if existing else user_id,
            "updated_by": user_id,
        }

        self._collection.update_one(
            {"workspace_id": workspace_id},
            {
                "$set": {
                    f"provider_credentials.{provider}": payload,
                    "updated_at": now,
                }
            },
        )

        return self.get_provider_credentials(workspace_id, provider)

    def deactivate_provider_credentials(
        self,
        workspace_id: int,
        provider_name: str,
        user_id: Optional[int] = None,
    ) -> bool:
        provider = provider_name.lower().strip()
        doc = self._get_workspace_document(workspace_id)
        entry = ((doc or {}).get("provider_credentials") or {}).get(provider)
        if not entry or not entry.get("is_active", True):
            return False

        now = self._utcnow()
        self._collection.update_one(
            {"workspace_id": workspace_id},
            {
                "$set": {
                    f"provider_credentials.{provider}.is_active": False,
                    f"provider_credentials.{provider}.updated_at": now,
                    f"provider_credentials.{provider}.updated_by": user_id,
                    "updated_at": now,
                }
            },
        )
        return True

    # ------------------------------------------------------------------
    # Agent configuration
    # ------------------------------------------------------------------

    def _normalize_providers(self, providers: Optional[List[str]]) -> List[str]:
        values = []
        for provider in providers or []:
            p = provider.lower().strip()
            if p not in SUPPORTED_PROVIDERS:
                raise ValueError(f"Invalid provider: {p}")
            if p not in values:
                values.append(p)
        return values

    def get_configuration(self, workspace_id: int, agent_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        doc = self._get_workspace_document(workspace_id)
        if not doc:
            return None

        key = self._agent_key(agent_id)
        cfg = (doc.get("agent_configs") or {}).get(key)
        if not cfg or not cfg.get("is_active", True):
            return None

        return {
            "id": f"{workspace_id}:{key}",
            "workspace_id": workspace_id,
            "agent_id": agent_id,
            "configured_providers": cfg.get("configured_providers") or [],
            "current_provider": cfg.get("current_provider"),
            "created_at": cfg.get("created_at"),
            "updated_at": cfg.get("updated_at"),
            "created_by": cfg.get("created_by"),
            "updated_by": cfg.get("updated_by"),
        }

    def get_effective_configuration(self, workspace_id: int, agent_id: int) -> Optional[Dict[str, Any]]:
        return self.get_configuration(workspace_id, agent_id) or self.get_configuration(workspace_id, None)

    def create_or_update_configuration(
        self,
        workspace_id: int,
        agent_id: Optional[int] = None,
        configured_providers: Optional[List[str]] = None,
        current_provider: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        self._ensure_workspace_document(workspace_id)
        existing = self.get_configuration(workspace_id, agent_id)

        providers = (
            self._normalize_providers(configured_providers)
            if configured_providers is not None
            else list(existing.get("configured_providers") or [])
            if existing
            else []
        )

        selected_provider = current_provider.lower().strip() if current_provider else None
        if selected_provider and selected_provider not in SUPPORTED_PROVIDERS:
            raise ValueError(f"Invalid provider: {selected_provider}")
        if selected_provider and selected_provider not in providers:
            providers.append(selected_provider)

        now = self._utcnow()
        key = self._agent_key(agent_id)
        payload = {
            "configured_providers": providers,
            "current_provider": selected_provider if current_provider is not None else (existing or {}).get("current_provider"),
            "is_active": True,
            "created_at": (existing or {}).get("created_at", now),
            "updated_at": now,
            "created_by": (existing or {}).get("created_by", user_id),
            "updated_by": user_id,
        }

        self._collection.update_one(
            {"workspace_id": workspace_id},
            {
                "$set": {
                    f"agent_configs.{key}": payload,
                    "updated_at": now,
                }
            },
        )
        return self.get_configuration(workspace_id, agent_id)

    def switch_provider(
        self,
        workspace_id: int,
        provider: str,
        agent_id: Optional[int] = None,
        user_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        selected = provider.lower().strip()
        if selected not in SUPPORTED_PROVIDERS:
            raise ValueError(f"Invalid provider: {selected}")

        cfg = self.get_configuration(workspace_id, agent_id)
        providers = list(cfg.get("configured_providers") or []) if cfg else []
        if selected not in providers:
            providers.append(selected)

        return self.create_or_update_configuration(
            workspace_id=workspace_id,
            agent_id=agent_id,
            configured_providers=providers,
            current_provider=selected,
            user_id=user_id,
        )

    def add_provider(
        self,
        workspace_id: int,
        provider: str,
        agent_id: Optional[int] = None,
        set_as_current: bool = False,
        user_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        selected = provider.lower().strip()
        if selected not in SUPPORTED_PROVIDERS:
            raise ValueError(f"Invalid provider: {selected}")

        cfg = self.get_configuration(workspace_id, agent_id)
        providers = list(cfg.get("configured_providers") or []) if cfg else []
        if selected not in providers:
            providers.append(selected)

        return self.create_or_update_configuration(
            workspace_id=workspace_id,
            agent_id=agent_id,
            configured_providers=providers,
            current_provider=selected if set_as_current else (cfg or {}).get("current_provider"),
            user_id=user_id,
        )

    def get_workspace_configurations(self, workspace_id: int) -> List[Dict[str, Any]]:
        doc = self._get_workspace_document(workspace_id)
        if not doc:
            return []

        configs: List[Dict[str, Any]] = []
        for key, cfg in (doc.get("agent_configs") or {}).items():
            if not cfg.get("is_active", True):
                continue
            agent_id = None if key == DEFAULT_CONFIG_KEY else int(key)
            configs.append(
                {
                    "id": f"{workspace_id}:{key}",
                    "workspace_id": workspace_id,
                    "agent_id": agent_id,
                    "configured_providers": cfg.get("configured_providers") or [],
                    "current_provider": cfg.get("current_provider"),
                    "created_at": cfg.get("created_at"),
                    "updated_at": cfg.get("updated_at"),
                    "created_by": cfg.get("created_by"),
                    "updated_by": cfg.get("updated_by"),
                }
            )

        configs.sort(key=lambda x: (-1 if x["agent_id"] is None else x["agent_id"]))
        return configs

    def get_current_provider(self, workspace_id: int, agent_id: Optional[int] = None) -> Optional[str]:
        config = self.get_effective_configuration(workspace_id, agent_id) if agent_id is not None else self.get_configuration(workspace_id, agent_id)
        return config.get("current_provider") if config else None

    def list_configured_providers(self, workspace_id: int, agent_id: Optional[int] = None) -> List[str]:
        config = self.get_effective_configuration(workspace_id, agent_id) if agent_id is not None else self.get_configuration(workspace_id, agent_id)
        return list(config.get("configured_providers") or []) if config else []

    def delete_configuration(
        self,
        workspace_id: int,
        agent_id: Optional[int] = None,
        user_id: Optional[int] = None,
    ) -> bool:
        key = self._agent_key(agent_id)
        doc = self._get_workspace_document(workspace_id)
        cfg = ((doc or {}).get("agent_configs") or {}).get(key)
        if not cfg or not cfg.get("is_active", True):
            return False

        now = self._utcnow()
        self._collection.update_one(
            {"workspace_id": workspace_id},
            {
                "$set": {
                    f"agent_configs.{key}.is_active": False,
                    f"agent_configs.{key}.updated_at": now,
                    f"agent_configs.{key}.updated_by": user_id,
                    "updated_at": now,
                }
            },
        )
        return True

    def bulk_create_agent_configurations(
        self,
        workspace_id: int,
        agent_ids: List[int],
        configured_providers: Optional[List[str]] = None,
        current_provider: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        created = []
        if not agent_ids:
            return created

        defaults = configured_providers or ["azure"]
        active_provider = current_provider or "azure"

        for agent_id in agent_ids:
            existing = self.get_configuration(workspace_id, agent_id)
            if existing:
                continue
            created.append(
                self.create_or_update_configuration(
                    workspace_id=workspace_id,
                    agent_id=agent_id,
                    configured_providers=defaults,
                    current_provider=active_provider,
                    user_id=user_id,
                )
            )
        return created

    def delete_workspace_configurations(self, workspace_id: int, user_id: Optional[int] = None) -> int:
        doc = self._get_workspace_document(workspace_id)
        if not doc:
            return 0

        count = 0
        updates: Dict[str, Any] = {}
        now = self._utcnow()
        for key, cfg in (doc.get("agent_configs") or {}).items():
            if cfg.get("is_active", True):
                updates[f"agent_configs.{key}.is_active"] = False
                updates[f"agent_configs.{key}.updated_at"] = now
                updates[f"agent_configs.{key}.updated_by"] = user_id
                count += 1

        if count == 0:
            return 0

        updates["updated_at"] = now
        self._collection.update_one({"workspace_id": workspace_id}, {"$set": updates})
        return count


llm_router_config_store = LLMRouterConfigStore()
