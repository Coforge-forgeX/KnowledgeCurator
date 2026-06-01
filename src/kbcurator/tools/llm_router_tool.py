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
from fastmcp.exceptions import ToolError, ValidationError

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

async def _validate_llm_credentials(
    provider: str,
    api_key: str,
    endpoint: str,
    model: str,
    api_version: Optional[str] = None,
    deployment_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Test LLM credentials by attempting a simple generation request.
    
    Returns:
        Dict with success flag and validation details.
    """
    try:
        # Create a temporary ConfigurableAI manager to test the credentials
        manager = ConfigurableAIManager()
        
        # Build config dict for the provider
        config_dict = {
            "api_key": api_key,
            "endpoint": endpoint,
            "model": model,
        }
        
        if provider == "azure":
            config_dict["api_version"] = api_version or "2024-12-01-preview"
            config_dict["deployment_name"] = deployment_name or model
        
        # Configure the provider temporarily
        manager.configure_provider(provider, config_dict)
        manager.set_current_provider(provider)
        
        # Test with a simple prompt
        test_prompt = "Hello, respond with just 'OK' to confirm connection."
        response = await manager.generate_text_async(test_prompt)
        
        return {
            "success": True,
            "message": "LLM credentials validated successfully.",
            "test_response": response[:100] if response else "No response",
        }
        
    except Exception as e:
        logger.error(f"LLM credential validation failed for provider '{provider}': {e}")
        return {
            "success": False,
            "error": f"LLM credential validation failed: {str(e)}",
            "details": "Please check your API key, endpoint, model name, and ensure the service is accessible.",
        }


@mcp.tool()
@require_auth_async
async def admin_configure_llm_provider(
    provider: str,
    workspace_id: int,
    agent_ids: Optional[List[int]] = None,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    model: Optional[str] = None,
    api_version: Optional[str] = None,
    deployment_name: Optional[str] = None,
    set_as_current: bool = False,
    skip_validation: bool = False,
) -> Dict[str, Any]:
    """
    [ADMIN ONLY] Configure or update LLM provider for a workspace. This unified tool can:
    1. Create new provider configuration (when provider doesn't exist)
    2. Update existing provider configuration (when provider exists)
    3. Add/manage agents for the provider
    
    For NEW providers: api_key, endpoint, and model are required.
    For EXISTING providers: only provide the fields you want to update.

    Args:
        provider:         Provider name — 'azure' or 'quasar'.
        workspace_id:     Workspace to configure.
        agent_ids:        List of agent IDs to enable/add for this provider (optional).
        api_key:          API key for the provider (required for new, optional for updates).
        endpoint:         API endpoint URL (required for new, optional for updates).
        model:            Model / deployment name (required for new, optional for updates).
        api_version:      (Azure only) API version, e.g. '2024-12-01-preview'.
        deployment_name:  (Azure only) Deployment name if different from model.
        set_as_current:   If True, set this provider as active for all listed agents.
        skip_validation:  If True, skip credential validation (use with caution).
        
    Note: When agent_ids is provided, it replaces the entire agent list for this provider.
    Agents not in the list will have this provider removed from their configuration.

    Returns:
        Summary dict with success flag and details.
    """
    claims, user_id = get_current_user()
    caller_role = int(claims.get("role_id", -1))

    if not _is_admin_role(caller_role):
        raise ToolError("Forbidden: only Workspace Admins or Platform Admins can configure LLM providers.")

    provider = provider.lower().strip()
    if provider not in SUPPORTED_PROVIDERS:
        raise ValidationError(f"Unsupported provider '{provider}'. Supported: {SUPPORTED_PROVIDERS}")

    # Check if provider already exists
    existing_creds = workspace_provider_credentials_service.get_provider_credentials(workspace_id, provider)
    is_new_provider = existing_creds is None
    
    # Determine operation mode and validate inputs
    if is_new_provider:
        # Creating new provider - require essential credentials
        if not api_key or not endpoint or not model:
            raise ValidationError(
                f"For new provider configuration, api_key, endpoint, and model are all required. "
                f"Provider '{provider}' does not exist in workspace {workspace_id}."
            )
        operation_mode = "create"
        config_to_use = {
            "api_key": api_key,
            "endpoint": endpoint,
            "model": model,
            "api_version": api_version,
            "deployment_name": deployment_name,
        }
    else:
        # Updating existing provider - merge with existing config
        operation_mode = "update"
        config_to_use = {
            "api_key": api_key if api_key is not None else existing_creds["api_key"],
            "endpoint": endpoint if endpoint is not None else existing_creds["endpoint"],
            "model": model if model is not None else existing_creds["model"],
            "api_version": api_version if api_version is not None else existing_creds.get("api_version"),
            "deployment_name": deployment_name if deployment_name is not None else existing_creds.get("deployment_name"),
        }

    # Track what configuration fields are being changed
    config_changes = []
    if is_new_provider:
        config_changes = ["api_key", "endpoint", "model"]
        if api_version:
            config_changes.append("api_version")
        if deployment_name:
            config_changes.append("deployment_name")
    else:
        if api_key is not None:
            config_changes.append("api_key")
        if endpoint is not None:
            config_changes.append("endpoint")
        if model is not None:
            config_changes.append("model")
        if api_version is not None:
            config_changes.append("api_version")
        if deployment_name is not None:
            config_changes.append("deployment_name")

    # Validate credentials if configuration is being changed
    credentials_validated = False
    if not skip_validation and (is_new_provider or any([api_key, endpoint, model])):
        logger.info(f"Validating LLM credentials for provider '{provider}'...")
        validation_result = await _validate_llm_credentials(
            provider=provider,
            api_key=config_to_use["api_key"],
            endpoint=config_to_use["endpoint"],
            model=config_to_use["model"],
            api_version=config_to_use["api_version"],
            deployment_name=config_to_use["deployment_name"],
        )
        
        if not validation_result["success"]:
            raise ValidationError(
                f"LLM credential validation failed: {validation_result['error']}. "
                f"{validation_result.get('details', '')}"
            )
        
        logger.info(f"LLM credentials validated successfully for provider '{provider}'")
        credentials_validated = True

    try:
        operations_performed = []
        enabled_agents = []
        skipped_agents = []

        # Update/create provider credentials if any config changes
        if config_changes:
            workspace_provider_credentials_service.upsert_provider_credentials(
                workspace_id=workspace_id,
                provider_name=provider,
                api_key=config_to_use["api_key"],
                endpoint=config_to_use["endpoint"],
                model=config_to_use["model"],
                api_version=config_to_use["api_version"],
                deployment_name=config_to_use["deployment_name"],
                user_id=user_id,
            )
            operations_performed.append(f"credentials_{operation_mode}d")
            logger.info(f"Admin {user_id} {operation_mode}d provider '{provider}' credentials in workspace {workspace_id}")

        # Handle agent configuration if agent_ids provided
        if agent_ids is not None:  # Allow empty list to remove all agents
            removed_agents = []
            
            # Always replace agents when agent_ids is provided for existing providers
            if not is_new_provider:
                # Get all agents currently configured with this provider
                all_workspace_configs = agent_llm_config_service.get_workspace_configurations(workspace_id)
                current_agents_with_provider = []
                
                for config in all_workspace_configs:
                    if config.get("agent_id") is not None:  # Skip workspace default
                        configured_providers = config.get("configured_providers", [])
                        if provider in configured_providers:
                            current_agents_with_provider.append(config["agent_id"])
                
                # Remove provider from agents not in the new agent_ids list
                agents_to_remove = [aid for aid in current_agents_with_provider if aid not in agent_ids]
                for agent_id in agents_to_remove:
                    try:
                        # Get current config for this agent
                        config = agent_llm_config_service.get_effective_configuration(workspace_id, agent_id)
                        if config:
                            current_providers = config.get("configured_providers", [])
                            if provider in current_providers:
                                # Remove this provider from the agent's configured providers
                                updated_providers = [p for p in current_providers if p != provider]
                                current_provider = config.get("current_provider")
                                
                                # If this was the current provider, switch to another or clear
                                new_current = None
                                if current_provider == provider:
                                    if updated_providers:
                                        new_current = updated_providers[0]  # Switch to first available
                                else:
                                    new_current = current_provider  # Keep existing current
                                
                                agent_llm_config_service.create_or_update_configuration(
                                    workspace_id=workspace_id,
                                    agent_id=agent_id,
                                    configured_providers=updated_providers,
                                    current_provider=new_current,
                                    user_id=user_id,
                                )
                                removed_agents.append(agent_id)
                                logger.info(f"Removed provider '{provider}' from agent {agent_id}")
                    except Exception as e:
                        logger.error(f"Failed to remove provider '{provider}' from agent {agent_id}: {e}")
            
            # Now handle the agents in the provided agent_ids list
            for agent_id in agent_ids:
                try:
                    # Check if agent already has this provider
                    config = agent_llm_config_service.get_effective_configuration(workspace_id, agent_id)
                    existing_providers = config.get("configured_providers", []) if config else []
                    
                    if provider in existing_providers:
                        # Agent already has provider, but check if we need to set as current
                        if set_as_current:
                            current_provider = config.get("current_provider") if config else None
                            if current_provider != provider:
                                # Switch to this provider as current
                                agent_llm_config_service.switch_provider(
                                    workspace_id=workspace_id,
                                    provider=provider,
                                    agent_id=agent_id,
                                    user_id=user_id,
                                )
                                enabled_agents.append(agent_id)
                                logger.info(f"Agent {agent_id} switched to provider '{provider}' as current")
                            else:
                                skipped_agents.append(agent_id)
                                logger.info(f"Agent {agent_id} already has provider '{provider}' as current")
                        else:
                            skipped_agents.append(agent_id)
                            logger.info(f"Agent {agent_id} already has provider '{provider}' configured")
                        continue
                    
                    # Agent doesn't have this provider, add it
                    agent_llm_config_service.add_provider(
                        workspace_id=workspace_id,
                        agent_id=agent_id,
                        provider=provider,
                        set_as_current=set_as_current,
                        user_id=user_id,
                    )
                    enabled_agents.append(agent_id)
                    logger.info(f"Agent {agent_id} added provider '{provider}'" + (f" as current" if set_as_current else ""))
                    
                except Exception as e:
                    logger.error(f"Failed to enable provider '{provider}' for agent {agent_id}: {e}")
                    skipped_agents.append(agent_id)
            
            operations_performed.append("agents_managed")
            
            # Add removed agents info to the response if any were removed
            if removed_agents:
                operations_performed.append("agents_removed")

        # Clear cache to ensure new configuration is used
        clear_ai_manager_cache(workspace_id=workspace_id)

        # Build success message
        messages = []
        if config_changes:
            if is_new_provider:
                messages.append(f"Provider '{provider}' created for workspace {workspace_id}")
            else:
                messages.append(f"Provider '{provider}' updated (fields: {', '.join(config_changes)})")
            
            if credentials_validated:
                messages.append("credentials validated successfully")
        
        if agent_ids is not None:
            agent_messages = []
            if enabled_agents:
                agent_messages.append(f"processed {len(enabled_agents)} agent(s)")
            if skipped_agents:
                agent_messages.append(f"{len(skipped_agents)} already configured/current")
            if removed_agents:
                agent_messages.append(f"removed {len(removed_agents)} agent(s)")
            
            if agent_messages:
                messages.append(", ".join(agent_messages))

        success_message = ". ".join(messages).capitalize() + "."

        return {
            "success": True,
            "message": success_message,
            "provider": provider,
            "workspace_id": workspace_id,
            "operation_mode": operation_mode,
            "operations_performed": operations_performed,
            "config_changes": config_changes,
            "enabled_agent_ids": enabled_agents,
            "skipped_agent_ids": skipped_agents,
            "removed_agent_ids": removed_agents,
            "set_as_current": set_as_current,
            "credentials_validated": credentials_validated,
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
        raise ToolError("Forbidden: only Workspace Admins or Platform Admins can view LLM provider configuration.")

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
        raise ToolError("Forbidden: only Workspace Admins or Platform Admins can remove LLM providers.")

    provider = provider.lower().strip()

    # Protect the default azure provider — it must never be removed
    if provider == "azure":
        raise ToolError(
            "Cannot remove the default Azure (gpt-4.1) provider. "
            "This is the system default and must always remain configured."
        )

    try:
        # Prevent removing the last configured provider — at least one must remain
        active_providers = workspace_provider_credentials_service.list_workspace_providers(workspace_id)
        active_provider_names = [p["provider_name"] for p in active_providers if p.get("is_active", True)]

        if provider not in active_provider_names:
            raise ToolError(f"Provider '{provider}' was not found or is already inactive.")

        if len(active_provider_names) <= 1:
            raise ToolError(
                f"Cannot remove provider '{provider}' — it is the only configured provider for this workspace. "
                "At least one LLM provider must always remain configured."
            )

        removed = workspace_provider_credentials_service.deactivate_provider_credentials(
            workspace_id=workspace_id,
            provider_name=provider,
            user_id=user_id,
        )
        if not removed:
            raise ToolError(f"Provider '{provider}' was not found or could not be deactivated.")

        clear_ai_manager_cache(workspace_id=workspace_id)

        return {
            "success": True,
            "message": f"Provider '{provider}' has been deactivated for workspace {workspace_id}.",
            "provider": provider,
            "workspace_id": workspace_id,
        }
    except ToolError:
        raise
    except Exception as e:
        logger.error(f"admin_remove_llm_provider error: {e}")
        raise ToolError(f"Failed to remove provider '{provider}': {str(e)}")


# ---------------------------------------------------------------------------
# AUTHENTICATED-USER TOOLS
# ---------------------------------------------------------------------------

@mcp.tool()
@require_auth_async
async def list_available_llm_providers(
    workspace_id: int,
    agent_id: int,
) -> Dict[str, Any]:
    """
    List all LLM providers that have been configured by an admin for a specific
    workspace-agent, along with which provider is currently active.

    Regular users call this to discover available LLMs before calling
    switch_llm_provider.

    Args:
        workspace_id: Workspace ID.
        agent_id:     Agent ID.

    Returns:
        Dict with configured providers list, current provider, and switch hint.
    """
    try:
        config = agent_llm_config_service.get_effective_configuration(workspace_id, agent_id)
        if not config:
            return {
                "success": True,
                "workspace_id": workspace_id,
                "agent_id": agent_id,
                "configured_providers": [],
                "current_provider": None,
                "message": "No LLM providers have been configured for this workspace-agent yet. Contact an admin.",
            }

        configured_providers: List[str] = config.get("configured_providers") or []
        current_provider: Optional[str] = config.get("current_provider")

        # Enrich with lightweight public metadata (model name, endpoint host) from credentials
        provider_details = []
        for provider in configured_providers:
            creds = workspace_provider_credentials_service.get_provider_credentials(workspace_id, provider)
            
            # Safely extract endpoint host
            endpoint_host = None
            if creds and creds.get("endpoint"):
                try:
                    endpoint_parts = creds["endpoint"].split("/")
                    if len(endpoint_parts) >= 3:
                        endpoint_host = endpoint_parts[2]
                    else:
                        endpoint_host = creds["endpoint"]  # Use full endpoint if splitting fails
                except Exception as e:
                    logger.warning(f"Failed to parse endpoint for provider '{provider}': {e}")
                    endpoint_host = creds["endpoint"]
            
            provider_details.append({
                "provider": provider,
                "model": creds["model"] if creds else None,
                "endpoint_host": endpoint_host,
                "is_current": provider == current_provider,
            })

        return {
            "success": True,
            "workspace_id": workspace_id,
            "agent_id": agent_id,
            "configured_providers": provider_details,
            "current_provider": current_provider,
            "can_switch": len(configured_providers) > 1,
            "switch_hint": (
                "Use switch_llm_provider with provider=<name> to change the active LLM."
                if len(configured_providers) > 1 else None
            ),
        }
    except Exception as e:
        logger.error(f"list_available_llm_providers error: {e}")
        return {"success": False, "error": str(e)}


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
        raise ValidationError(
            f"Provider '{provider}' has not been configured for workspace {workspace_id}. "
            "An admin must configure it first via admin_configure_llm_provider."
        )

    config = agent_llm_config_service.get_effective_configuration(workspace_id, agent_id)
    if not config or provider not in (config.get("configured_providers") or []):
        raise ValidationError(
            f"Provider '{provider}' is not enabled for agent {agent_id} "
            f"in workspace {workspace_id}. An admin must enable it first."
        )

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
            raise ValidationError("No provider is currently configured. Ask an admin to run admin_configure_llm_provider.")

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
