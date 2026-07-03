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
from kbcurator.utils.permission import is_admin
from kbcurator.utils.constants import Role
from common_adapters.configurableAI import (
    ConfigurableAIManager,
    get_configured_llm_manager,
    invalidate_llm_cache,
)
from fastmcp.exceptions import ToolError, ValidationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _invalidate_manager_cache(workspace_id: int, agent_id: Optional[int] = None) -> None:
    """Clear LLM manager cache (delegates to common_adapters)."""
    invalidate_llm_cache(workspace_id=workspace_id, agent_id=agent_id)


def _build_manager_from_db(workspace_id: int, agent_id: Optional[int]) -> ConfigurableAIManager:
    """
    Get a ConfigurableAIManager pre-loaded from MongoDB config.
    Delegates to common_adapters.configurableAI.get_configured_llm_manager.
    """
    try:
        manager = get_configured_llm_manager(workspace_id, agent_id)
        current_provider = manager.get_current_provider()
        logger.info(
            f"LLM provider resolved: '{current_provider}' "
            f"(workspace_id={workspace_id}, agent_id={agent_id})"
        )
        return manager
    except ValueError:
        # Return empty manager if nothing is configured (for backward compat)
        logger.warning(
            f"No LLM provider configured — using empty manager "
            f"(workspace_id={workspace_id}, agent_id={agent_id})"
        )
        return ConfigurableAIManager()


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
    [ADMIN ONLY] Configure or update LLM provider for a workspace.
    
    This tool handles the key scenarios:
    1. Configure agent_ids [1,2,3] for claude-sonnet-4 (quasar) - adds model to those agents
    2. Later configure agent_ids [1,3] for claude-sonnet-4 (quasar) - removes from agent 2
    3. Configure agent_ids [1,3,4] for gpt-5-2-chat (quasar) - agent 1,3 now have multiple models
    
    For NEW providers: api_key, endpoint, and model are required.
    For EXISTING providers: only provide the fields you want to update.

    Args:
        provider:         Provider name — 'azure' or 'quasar'.
        workspace_id:     Workspace to configure.
        agent_ids:        List of agent IDs to configure this model for. When provided,
                         this model will be configured ONLY for these agents (others removed).
        api_key:          API key for the provider (required for new, optional for updates).
        endpoint:         API endpoint URL (required for new, optional for updates).
        model:            Model name (required for new, optional for updates).
        api_version:      (Azure only) API version, e.g. '2024-12-01-preview'.
        deployment_name:  (Azure only) Deployment name if different from model.
        set_as_current:   If True, set this provider+model as active for all listed agents.
        skip_validation:  If True, skip credential validation (use with caution).

    Returns:
        Summary dict with success flag and details.
    """
    # For testing purposes, bypass auth check
    # TODO: Re-enable auth in production
    # claims, user_id = get_current_user()
    # if not is_admin(user_id, workspace_id):
    #     raise ToolError("Forbidden: only Workspace Admins can configure LLM providers.")
    user_id = 247  # Default user for testing

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
            "deployment_name": deployment_name or model,
        }
    else:
        # Updating existing provider - merge with existing config
        operation_mode = "update"
        config_to_use = {
            "api_key": api_key if api_key is not None else existing_creds["api_key"],
            "endpoint": endpoint if endpoint is not None else existing_creds["endpoint"],
            "model": model if model is not None else existing_creds["model"],
            "api_version": api_version if api_version is not None else existing_creds.get("api_version"),
            "deployment_name": deployment_name if deployment_name is not None else existing_creds.get("deployment_name", model or existing_creds["model"]),
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
        removed_agents = []
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
            # The specific model being configured for these agents
            target_model = model if model is not None else config_to_use["model"]

            # Special handling for Azure: cannot remove all agents from default model
            if provider == "azure" and target_model == "gpt-4.1":
                # Get all agents in workspace
                all_workspace_configs = agent_llm_config_service.get_workspace_configurations(workspace_id)
                all_agent_ids = [
                    ac["agent_id"] for ac in all_workspace_configs if ac["agent_id"] is not None
                ]
                # Check if user is trying to assign fewer agents than total for default Azure model
                if set(agent_ids) != set(all_agent_ids):
                    raise ToolError(
                        f"Cannot modify agent assignments for the default Azure model 'gpt-4.1'. "
                        f"All agents must have the default Azure LLM configured. "
                        f"You can add additional models to Azure or other providers, but gpt-4.1 must remain on all agents."
                    )

            # Get current agent configurations to see who actually has this model
            all_agent_configs = agent_llm_config_service.get_workspace_configurations(workspace_id)
            
            # Find agents that currently have this model configured
            agents_with_model = []
            for ac in all_agent_configs:
                if ac["agent_id"] is not None:
                    configured_models = ac.get("configured_models", {})
                    provider_models = configured_models.get(provider, [])
                    if target_model in provider_models:
                        agents_with_model.append(ac["agent_id"])
            
            # Find agents that need to be removed (currently have model but not in new list)
            agents_to_remove = [aid for aid in agents_with_model if aid not in agent_ids]
            
            print(f"DEBUG: agents_with_model for {target_model}: {agents_with_model}")
            print(f"DEBUG: new agent_ids: {agent_ids}")
            print(f"DEBUG: agents_to_remove: {agents_to_remove}")
            logger.info(f"DEBUG: agents_with_model for {target_model}: {agents_with_model}")
            logger.info(f"DEBUG: new agent_ids: {agent_ids}")
            logger.info(f"DEBUG: agents_to_remove: {agents_to_remove}")
            
            # Remove the model from agents that are no longer assigned
            removal_failures = []
            for agent_id in agents_to_remove:
                try:
                    print(f"DEBUG: Attempting to remove model '{target_model}' from agent {agent_id}")
                    logger.info(f"DEBUG: Attempting to remove model '{target_model}' from agent {agent_id}")
                    result = agent_llm_config_service.remove_model_from_agent(
                        workspace_id=workspace_id,
                        provider=provider,
                        model=target_model,
                        agent_id=agent_id,
                        user_id=user_id,
                    )
                    print(f"DEBUG: Remove result for agent {agent_id}: {result}")
                    logger.info(f"DEBUG: Remove result for agent {agent_id}: {result}")
                    removed_agents.append(agent_id)
                    print(f"Successfully removed model '{target_model}' from agent {agent_id}")
                    logger.info(f"Successfully removed model '{target_model}' from agent {agent_id}")
                except Exception as e:
                    print(f"Failed to remove model '{target_model}' from agent {agent_id}: {e}")
                    logger.error(f"Failed to remove model '{target_model}' from agent {agent_id}: {e}")
                    removal_failures.append({"agent_id": agent_id, "error": str(e)})
            
            # If there were removal failures, don't claim success
            if removal_failures:
                logger.error(f"Model removal failures: {removal_failures}")
                # Continue but track the failures

            # Store model→agents assignment in provider_credentials (for UI display)
            workspace_provider_credentials_service.set_model_assignments(
                workspace_id=workspace_id,
                provider_name=provider,
                model_name=target_model,
                agent_ids=agent_ids,
                user_id=user_id,
            )

            # Add this model to each agent's configured_models for this provider
            for agent_id in agent_ids:
                try:
                    agent_llm_config_service.add_model_to_agent(
                        workspace_id=workspace_id,
                        provider=provider,
                        model=target_model,
                        agent_id=agent_id,
                        set_as_current=set_as_current,
                        user_id=user_id,
                    )
                    enabled_agents.append(agent_id)
                    logger.info(f"Added model '{target_model}' to agent {agent_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to add model '{target_model}' to agent {agent_id}: {e}")
                    skipped_agents.append(agent_id)
            
            operations_performed.append("agents_managed")

        # Clear cache to ensure new configuration is used
        _invalidate_manager_cache(workspace_id=workspace_id)

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
                agent_messages.append(f"configured {len(enabled_agents)} agent(s)")
            if removed_agents:
                agent_messages.append(f"removed from {len(removed_agents)} agent(s)")
            if skipped_agents:
                agent_messages.append(f"{len(skipped_agents)} failed")
            
            if agent_messages:
                messages.append(", ".join(agent_messages))

        success_message = ". ".join(messages).capitalize() + "."
        
        # Check if there were any removal failures
        has_removal_failures = 'removal_failures' in locals() and removal_failures
        if has_removal_failures:
            success_message += f" Warning: {len(removal_failures)} agent(s) failed to be removed."

        return {
            "success": True,
            "message": success_message,
            "provider": provider,
            "workspace_id": workspace_id,
            "operation_mode": operation_mode,
            "operations_performed": operations_performed,
            "config_changes": config_changes,
            "enabled_agent_ids": enabled_agents,
            "removed_agent_ids": removed_agents,
            "skipped_agent_ids": skipped_agents,
            "set_as_current": set_as_current,
            "credentials_validated": credentials_validated,
            "removal_failures": removal_failures if has_removal_failures else [],
        }

    except Exception as e:
        logger.error(f"admin_configure_llm_provider error: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
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
    # For testing purposes, bypass auth check
    # TODO: Re-enable auth in production
    # claims, user_id = get_current_user()
    # if not is_admin(user_id, workspace_id):
    #     raise ToolError("Forbidden: only Workspace Admins can view LLM provider configuration.")
    user_id = 247  # Default user for testing

    try:
        credential_records = workspace_provider_credentials_service.list_workspace_providers(workspace_id)
        agent_configs = agent_llm_config_service.get_workspace_configurations(workspace_id)

        # Return one row per MODEL (not per provider) for clear model-agent mapping
        providers_summary = []
        
        for cred in credential_records:
            pname = cred["provider_name"]
            available_models = cred.get("available_models") or []
            model_assignments = cred.get("model_assignments") or {}

            # If no available_models tracked, use the top-level model as a single entry
            if not available_models:
                available_models = [{"model_name": cred["model"], "deployment_name": cred.get("deployment_name") or cred["model"]}]

            # Create one entry per model
            for model_entry in available_models:
                model_name = model_entry["model_name"]
                deployment_name = model_entry.get("deployment_name", model_name)
                
                # Build agents list for this specific model based on actual agent configurations
                agents_for_model = []
                
                # Check each agent to see if they have this model configured
                for ac in agent_configs:
                    if ac["agent_id"] is None:
                        continue
                    
                    configured_models = ac.get("configured_models") or {}
                    provider_models = configured_models.get(pname) or []
                    
                    # Only include agents that actually have this model configured
                    if model_name in provider_models:
                        is_current = (ac.get("current_provider") == pname and ac.get("current_model") == model_name)
                        agents_for_model.append({
                            "agent_id": ac["agent_id"],
                            "is_current": is_current
                        })

                # Create one row for this model
                providers_summary.append({
                    "provider": pname,
                    "model": model_name,
                    "deployment_name": deployment_name,
                    "endpoint": cred["endpoint"],
                    "api_version": cred.get("api_version"),
                    "configured_at": str(cred["created_at"]),
                    "configured_by": cred["created_by"],
                    "agents_enabled": agents_for_model,
                    "agent_count": len(agents_for_model),
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
async def admin_remove_llm_provider(
    workspace_id: int,
    provider: str,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    [ADMIN ONLY] Remove a specific model from a provider, or deactivate an entire provider.

    If 'model' is specified, only that model is removed from the provider's available_models.
    Agents using that model will have the provider removed from their configuration.
    If 'model' is not specified, the entire provider is deactivated.

    Args:
        workspace_id: Workspace to modify.
        provider:     Provider name ('azure' or 'quasar').
        model:        (Optional) Specific model to remove from the provider.

    Returns:
        Success/failure dict.
    """
    # For testing purposes, bypass auth check
    # TODO: Re-enable auth in production
    # claims, user_id = get_current_user()
    # if not is_admin(user_id, workspace_id):
    #     raise ToolError("Forbidden: only Workspace Admins can remove LLM providers.")
    user_id = 247  # Default user for testing

    provider = provider.lower().strip()

    # Protect the default azure provider — it must never be fully removed
    if provider == "azure" and model is None:
        raise ToolError(
            "Cannot remove the Azure provider. "
            "Azure (gpt-4.1) is the system default LLM configured at workspace creation and must always remain active. "
            "You may update its model/credentials via admin_configure_llm_provider, but it cannot be deleted."
        )

    try:
        if model is not None:
            # Remove a specific model from the provider
            creds = workspace_provider_credentials_service.get_provider_credentials(workspace_id, provider)
            if not creds:
                raise ToolError(f"Provider '{provider}' was not found or is already inactive.")

            available_models = creds.get("available_models") or []
            updated_models = [m for m in available_models if m["model_name"] != model]

            if len(updated_models) == len(available_models):
                raise ToolError(f"Model '{model}' not found in provider '{provider}'.")

            # Get agents assigned to this model from model_assignments
            model_assignments = creds.get("model_assignments") or {}
            removed_from_agents = model_assignments.get(model) or []

            if not updated_models:
                # No models left — deactivate the entire provider
                workspace_provider_credentials_service.deactivate_provider_credentials(
                    workspace_id=workspace_id,
                    provider_name=provider,
                    user_id=user_id,
                )
            else:
                # Update available_models and set top-level model to the first remaining
                workspace_provider_credentials_service.remove_model_from_provider(
                    workspace_id=workspace_id,
                    provider_name=provider,
                    model_name=model,
                    user_id=user_id,
                )

            _invalidate_manager_cache(workspace_id=workspace_id)

            return {
                "success": True,
                "message": f"Model '{model}' removed from provider '{provider}' in workspace {workspace_id}.",
                "provider": provider,
                "model": model,
                "workspace_id": workspace_id,
                "removed_from_agents": removed_from_agents,
            }

        # Full provider removal (original behavior)
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

        _invalidate_manager_cache(workspace_id=workspace_id)

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
async def list_available_llm_providers(
    workspace_id: int,
    agent_id: int,
) -> Dict[str, Any]:
    """
    List all LLM providers and models that have been configured for a specific agent.
    Only shows models that are actually configured for this agent.

    Regular users call this to discover available LLMs before calling
    switch_llm_provider.

    Args:
        workspace_id: Workspace ID.
        agent_id:     Agent ID.

    Returns:
        Dict with configured providers list, current provider, and switch hint.
    """
    try:
        # Get agent-specific config first, then workspace default for proper hierarchy
        agent_config = agent_llm_config_service.get_configuration(workspace_id, agent_id)
        workspace_config = agent_llm_config_service.get_configuration(workspace_id, None)
        
        # Use agent config if exists, otherwise workspace default
        effective_config = agent_config or workspace_config
        
        if not effective_config:
            return {
                "success": True,
                "workspace_id": workspace_id,
                "agent_id": agent_id,
                "configured_providers": [],
                "current_provider": None,
                "message": "No LLM providers have been configured for this workspace-agent yet. Contact an admin.",
            }

        configured_providers: List[str] = effective_config.get("configured_providers") or []
        configured_models_dict: Dict[str, List[str]] = effective_config.get("configured_models") or {}
        current_provider: Optional[str] = effective_config.get("current_provider")
        current_model: Optional[str] = effective_config.get("current_model")

        # Build provider details showing only configured models for this agent
        provider_details = []
        
        for provider in configured_providers:
            # Validate that provider has credentials in this specific workspace
            creds = workspace_provider_credentials_service.get_provider_credentials(workspace_id, provider)
            
            # Skip providers that have no active credentials in this workspace
            if not creds or not creds.get("is_active", True):
                logger.debug(f"Provider '{provider}' listed in config but has no active credentials in workspace {workspace_id} — hiding from user.")
                continue
            
            # Get configured models for this agent and provider
            agent_provider_models = configured_models_dict.get(provider, [])
            
            # Skip providers with no configured models for this agent
            if not agent_provider_models:
                logger.debug(f"Provider '{provider}' has no configured models for agent {agent_id} — skipping.")
                continue
            
            # Safely extract endpoint host for display
            endpoint_host = None
            if creds.get("endpoint"):
                try:
                    endpoint_parts = creds["endpoint"].split("/")
                    if len(endpoint_parts) >= 3:
                        endpoint_host = endpoint_parts[2]
                    else:
                        endpoint_host = creds["endpoint"]
                except Exception as e:
                    logger.warning(f"Failed to parse endpoint for provider '{provider}': {e}")
                    endpoint_host = creds["endpoint"]
            
            # Determine current model for this provider
            provider_current_model = current_model if provider == current_provider else None
            
            # Build available models list for this agent (only show configured models)
            available_models = []
            for model_name in agent_provider_models:
                # Find deployment info from provider credentials
                deployment_name = model_name  # default
                for model_entry in (creds.get("available_models") or []):
                    if isinstance(model_entry, dict) and model_entry.get("model_name") == model_name:
                        deployment_name = model_entry.get("deployment_name", model_name)
                        break
                
                available_models.append({
                    "model_name": model_name,
                    "deployment_name": deployment_name,
                    "is_current": (provider == current_provider and model_name == current_model)
                })
            
            provider_details.append({
                "provider": provider,
                "endpoint_host": endpoint_host,
                "is_current": provider == current_provider,
                "configured_models": agent_provider_models,
                "available_models": available_models,
                "current_model": provider_current_model,
                "model_count": len(agent_provider_models),
            })

        # Count total models available to this agent
        total_models = sum(len(p["configured_models"]) for p in provider_details)

        return {
            "success": True,
            "workspace_id": workspace_id,
            "agent_id": agent_id,
            "configured_providers": provider_details,
            "current_provider": current_provider,
            "current_model": current_model,
            "total_models": total_models,
            "can_switch": total_models > 1,
            "config_source": "agent" if agent_config else "workspace_default",
            "switch_hint": (
                "Use switch_llm_provider with provider=<name> to change the active provider. "
                "Use switch_llm_provider with provider=<name> and model=<name> to change to a specific model."
                if total_models > 1 else "Only one model configured. Contact admin to add more models."
            ),
        }
    except Exception as e:
        logger.error(f"list_available_llm_providers error: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def switch_llm_provider(
    provider: str,
    workspace_id: int,
    agent_id: int,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Switch the active LLM provider and/or model for an agent in a workspace.

    The provider must already be admin-configured for this workspace AND
    the model must be configured for this agent. This allows agents to toggle
    between their configured models.

    Args:
        provider:     Provider name to switch to ('azure' or 'quasar').
        workspace_id: Workspace ID.
        agent_id:     Agent ID.
        model:        (Optional) Model name to use within the provider (e.g. 'gpt-4.1').
                      If not specified, uses the first configured model for this provider.

    Returns:
        Updated state dict.
    """
    # For testing purposes, bypass auth check
    # TODO: Re-enable auth in production
    # claims, user_id = get_current_user()
    user_id = 247  # Default user for testing
    provider = provider.lower().strip()

    # Check if provider exists in workspace
    creds = workspace_provider_credentials_service.get_provider_credentials(workspace_id, provider)
    if not creds:
        raise ValidationError(
            f"Provider '{provider}' has not been configured for workspace {workspace_id}. "
            "An admin must configure it first via admin_configure_llm_provider."
        )

    # Check if agent has this provider configured
    config = agent_llm_config_service.get_effective_configuration(workspace_id, agent_id)
    if not config or provider not in (config.get("configured_providers") or []):
        raise ValidationError(
            f"Provider '{provider}' is not enabled for agent {agent_id} "
            f"in workspace {workspace_id}. An admin must enable it first."
        )

    # Get configured models for this agent and provider
    configured_models = config.get("configured_models", {})
    agent_provider_models = configured_models.get(provider, [])
    
    if not agent_provider_models:
        raise ValidationError(
            f"No models configured for provider '{provider}' on agent {agent_id}. "
            "An admin must configure models first."
        )

    # Validate and resolve model
    target_model = model
    if target_model:
        if target_model not in agent_provider_models:
            raise ValidationError(
                f"Model '{target_model}' is not configured for agent {agent_id} with provider '{provider}'. "
                f"Configured models: {agent_provider_models}"
            )
    else:
        # Use first configured model if no model specified
        target_model = agent_provider_models[0]

    updated = agent_llm_config_service.switch_provider(
        workspace_id=workspace_id,
        provider=provider,
        agent_id=agent_id,
        model=target_model,
        user_id=user_id,
    )

    _invalidate_manager_cache(workspace_id=workspace_id, agent_id=agent_id)

    return {
        "success": True,
        "message": f"Switched to provider '{provider}' with model '{target_model}'.",
        "provider": provider,
        "model": target_model,
        "workspace_id": workspace_id,
        "agent_id": agent_id,
        "configured_providers": updated.get("configured_providers", []),
        "current_provider": updated.get("current_provider"),
        "current_model": updated.get("current_model"),
        "configured_models": updated.get("configured_models", {}),
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
