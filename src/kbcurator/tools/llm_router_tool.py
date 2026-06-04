"""
LLM Router Tool — Admin-managed provider configuration.

Admin-only tools:
    admin_configure_llm_provider   — store credentials + enable for chosen agents
    admin_list_llm_providers       — list what is configured in this workspace
    admin_remove_llm_provider      — deactivate a provider from the workspace

Authenticated-user tools:
    switch_llm_provider            — toggle between already-configured providers
    test_llm_generation            — smoke-test the active provider

Credential source: MongoDB config documents (`chatbot_db.llm_router_config`).
Environment variables are used only to bootstrap MongoDB connectivity.
"""

import logging
from typing import Any, Dict, List, Optional

from kbcurator.server.server import mcp
from kbcurator.services.agent_llm_configuration_service import agent_llm_config_service
from kbcurator.services.workspace_provider_credentials_service import (
    workspace_provider_credentials_service,
    SUPPORTED_PROVIDERS,
)
from kbcurator.utils.auth import require_auth_async, get_current_user
from kbcurator.utils.constants import Role
from common_adapters.configurableAI import ConfigurableAIManager, clear_ai_manager_cache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_admin_role(role_id: int) -> bool:
    return role_id in (Role.ADMIN.id, Role.WS_ADMIN.id)


def _build_manager_from_db(workspace_id: int, agent_id: Optional[int]) -> ConfigurableAIManager:
    """
    Construct a ConfigurableAIManager whose providers are loaded from
    MongoDB workspace config, not environment variables.
    """
    manager = ConfigurableAIManager()
    config = agent_llm_config_service.get_effective_configuration(workspace_id, agent_id)
    if not config:
        return manager

    configured_providers: List[str] = config.get("configured_providers") or []
    current_provider: Optional[str] = config.get("current_provider")

    for provider in configured_providers:
        creds_dict = workspace_provider_credentials_service.build_config_dict(workspace_id, provider)
        if creds_dict:
            try:
                manager.configure_provider(provider, creds_dict)
            except Exception as e:
                logger.warning(f"Could not configure provider '{provider}': {e}")
        else:
            logger.warning(f"No credentials found for provider '{provider}' in workspace {workspace_id}")

    if current_provider and current_provider in manager.list_configured_providers():
        try:
            manager.set_current_provider(current_provider)
        except Exception as e:
            logger.warning(f"Could not set current provider '{current_provider}': {e}")

    return manager


# ---------------------------------------------------------------------------
# ADMIN TOOLS
# ---------------------------------------------------------------------------

@mcp.tool()
@require_auth_async
async def admin_configure_llm_provider(
    provider: str,
    api_key: str,
    endpoint: str,
    model: str,
    agent_ids: List[int],
    workspace_id: int,
    api_version: Optional[str] = None,
    deployment_name: Optional[str] = None,
    set_as_current: bool = False,
) -> Dict[str, Any]:
    """
    [ADMIN ONLY] Store LLM provider credentials for a workspace and enable the
    provider for the specified agents.

    Args:
        provider:         Provider name — 'azure' or 'quasar'.
        api_key:          API key for the provider.
        endpoint:         API endpoint URL.
        model:            Model / deployment name.
        agent_ids:        List of agent IDs (from agents_details table) that
                          should be able to use this provider.
        workspace_id:     Workspace to configure.
        api_version:      (Azure only) API version, e.g. '2024-12-01-preview'.
        deployment_name:  (Azure only) Deployment name if different from model.
        set_as_current:   If True, immediately set this provider as active for
                          all listed agents.

    Returns:
        Summary dict with success flag and details.
    """
    claims, user_id = get_current_user()
    caller_role = int(claims.get("role_id", -1))

    if not _is_admin_role(caller_role):
        return {
            "success": False,
            "error": "Forbidden: only Workspace Admins or Platform Admins can configure LLM providers.",
        }

    provider = provider.lower().strip()
    if provider not in SUPPORTED_PROVIDERS:
        return {
            "success": False,
            "error": f"Unsupported provider '{provider}'. Supported: {SUPPORTED_PROVIDERS}",
        }

    if not api_key or not endpoint or not model:
        return {
            "success": False,
            "error": "api_key, endpoint and model are all required.",
        }

    try:
        workspace_provider_credentials_service.upsert_provider_credentials(
            workspace_id=workspace_id,
            provider_name=provider,
            api_key=api_key,
            endpoint=endpoint,
            model=model,
            api_version=api_version,
            deployment_name=deployment_name,
            user_id=user_id,
        )
        logger.info(f"Admin {user_id} saved credentials for provider '{provider}' in workspace {workspace_id}")

        enabled_agents = []
        for agent_id in agent_ids:
            try:
                agent_llm_config_service.add_provider(
                    workspace_id=workspace_id,
                    agent_id=agent_id,
                    provider=provider,
                    set_as_current=set_as_current,
                    user_id=user_id,
                )
                enabled_agents.append(agent_id)
            except Exception as e:
                logger.error(f"Failed to enable provider '{provider}' for agent {agent_id}: {e}")

        clear_ai_manager_cache(workspace_id=workspace_id)

        return {
            "success": True,
            "message": (
                f"Provider '{provider}' configured for workspace {workspace_id} "
                f"and enabled for {len(enabled_agents)}/{len(agent_ids)} agent(s)."
            ),
            "provider": provider,
            "workspace_id": workspace_id,
            "enabled_agent_ids": enabled_agents,
            "set_as_current": set_as_current,
        }

    except Exception as e:
        logger.error(f"admin_configure_llm_provider error: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
@require_auth_async
async def admin_list_llm_providers(
    workspace_id: int,
) -> Dict[str, Any]:
    """
    [ADMIN ONLY] List all LLM providers that have been configured for a workspace,
    along with which agents each provider is enabled for.

    Args:
        workspace_id: Workspace to inspect.

    Returns:
        Dict with configured providers and per-agent activation status.
    """
    claims, user_id = get_current_user()
    caller_role = int(claims.get("role_id", -1))

    if not _is_admin_role(caller_role):
        return {
            "success": False,
            "error": "Forbidden: only Workspace Admins or Platform Admins can view LLM provider configuration.",
        }

    try:
        credential_records = workspace_provider_credentials_service.list_workspace_providers(workspace_id)
        agent_configs = agent_llm_config_service.get_workspace_configurations(workspace_id)

        providers_summary = []
        for cred in credential_records:
            pname = cred["provider_name"]
            agents_enabled = [
                {
                    "agent_id": ac["agent_id"],
                    "is_current": ac["current_provider"] == pname,
                }
                for ac in agent_configs
                if ac["agent_id"] is not None and pname in (ac["configured_providers"] or [])
            ]
            providers_summary.append({
                "provider": pname,
                "endpoint": cred["endpoint"],
                "model": cred["model"],
                "api_version": cred.get("api_version"),
                "configured_at": str(cred["created_at"]),
                "configured_by": cred["created_by"],
                "agents_enabled": agents_enabled,
            })

        return {
            "success": True,
            "workspace_id": workspace_id,
            "configured_providers": providers_summary,
            "supported_providers": SUPPORTED_PROVIDERS,
        }

    except Exception as e:
        logger.error(f"admin_list_llm_providers error: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
@require_auth_async
async def admin_remove_llm_provider(
    workspace_id: int,
    provider: str,
) -> Dict[str, Any]:
    """
    [ADMIN ONLY] Deactivate an LLM provider from a workspace.

    Args:
        workspace_id: Workspace to modify.
        provider:     Provider name to remove ('azure' or 'quasar').

    Returns:
        Success/failure dict.
    """
    claims, user_id = get_current_user()
    caller_role = int(claims.get("role_id", -1))

    if not _is_admin_role(caller_role):
        return {
            "success": False,
            "error": "Forbidden: only Workspace Admins or Platform Admins can remove LLM providers.",
        }

    provider = provider.lower().strip()
    try:
        removed = workspace_provider_credentials_service.deactivate_provider_credentials(
            workspace_id=workspace_id,
            provider_name=provider,
            user_id=user_id,
        )
        if not removed:
            return {
                "success": False,
                "error": f"Provider '{provider}' was not found or is already inactive.",
            }

        clear_ai_manager_cache(workspace_id=workspace_id)

        return {
            "success": True,
            "message": f"Provider '{provider}' has been deactivated for workspace {workspace_id}.",
            "provider": provider,
            "workspace_id": workspace_id,
        }
    except Exception as e:
        logger.error(f"admin_remove_llm_provider error: {e}")
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# AUTHENTICATED-USER TOOLS
# ---------------------------------------------------------------------------

@mcp.tool()
@require_auth_async
async def switch_llm_provider(
    provider: str,
    workspace_id: int,
    agent_id: int,
) -> Dict[str, Any]:
    """
    Switch the active LLM provider for an agent in a workspace.

    The provider must already be admin-configured for this workspace AND
    enabled for this agent. This call only toggles the active provider —
    it does not accept or store credentials.

    Args:
        provider:     Provider name to switch to ('azure' or 'quasar').
        workspace_id: Workspace ID.
        agent_id:     Agent ID.

    Returns:
        Updated state dict.
    """
    claims, user_id = get_current_user()
    provider = provider.lower().strip()

    creds = workspace_provider_credentials_service.get_provider_credentials(workspace_id, provider)
    if not creds:
        return {
            "success": False,
            "error": (
                f"Provider '{provider}' has not been configured for workspace {workspace_id}. "
                "An admin must configure it first via admin_configure_llm_provider."
            ),
        }

    config = agent_llm_config_service.get_effective_configuration(workspace_id, agent_id)
    if not config or provider not in (config.get("configured_providers") or []):
        return {
            "success": False,
            "error": (
                f"Provider '{provider}' is not enabled for agent {agent_id} "
                f"in workspace {workspace_id}. An admin must enable it first."
            ),
        }

    updated = agent_llm_config_service.switch_provider(
        workspace_id=workspace_id,
        provider=provider,
        agent_id=agent_id,
        user_id=user_id,
    )

    clear_ai_manager_cache(workspace_id=workspace_id, agent_id=agent_id)

    return {
        "success": True,
        "message": f"Switched to provider '{provider}'.",
        "provider": provider,
        "workspace_id": workspace_id,
        "agent_id": agent_id,
        "configured_providers": updated.get("configured_providers", []),
    }


@mcp.tool()
async def test_llm_generation(
    prompt: str = "Hello, how are you?",
    workspace_id: int = 0,
    agent_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Smoke-test the currently active LLM provider for an agent.

    Args:
        prompt:       Text prompt.
        workspace_id: Workspace ID.
        agent_id:     Agent ID.

    Returns:
        Generated text and metadata.
    """
    try:
        manager = _build_manager_from_db(workspace_id, agent_id)
        current = manager.get_current_provider()

        if not current:
            return {
                "success": False,
                "error": "No provider is currently configured. Ask an admin to run admin_configure_llm_provider.",
            }

        response = await manager.generate_text_async(prompt)
        return {
            "success": True,
            "provider_used": current,
            "workspace_id": workspace_id,
            "agent_id": agent_id,
            "prompt": prompt,
            "response": response,
            "response_length": len(response),
        }
    except Exception as e:
        logger.error(f"test_llm_generation error: {e}")
        return {"success": False, "error": str(e)}
