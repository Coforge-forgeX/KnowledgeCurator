"""
LLM Router Tool with Database Persistence.

This tool provides MCP endpoints for managing LLM provider configurations
with persistent database storage. It stores only provider selection and current provider,
with all credentials coming from environment variables.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

from kbcurator.server.server import mcp
from kbcurator.utils.request_context import request_var
from kbcurator.services.agent_llm_configuration_service import agent_llm_config_service
from kbcurator.utils.auth import require_auth_async
from common_adapters.configurableAI import ConfigurableAIManager, get_ai_manager, clear_ai_manager_cache, get_cached_manager_count
from common_adapters.configurableAI.config import (
    OpenAIConfig, 
    AzureOpenAIConfig, 
    GCPConfig,
    QuasarConfig
)

logger = logging.getLogger(__name__)

def _save_configuration_to_db(workspace_id: int, agent_id: Optional[int], provider: str, set_as_current: bool = False, user_id: Optional[int] = None):
    """Save provider configuration to database."""
    try:
        if set_as_current:
            agent_llm_config_service.switch_provider(
                workspace_id=workspace_id,
                provider=provider,
                agent_id=agent_id,
                user_id=user_id
            )
        else:
            agent_llm_config_service.add_provider(
                workspace_id=workspace_id,
                provider=provider,
                agent_id=agent_id,
                set_as_current=False,
                user_id=user_id
            )
        logger.info(f"Saved provider {provider} configuration to database")
    except Exception as e:
        logger.error(f"Failed to save configuration to database: {e}")


def _extract_workspace_and_agent_ids() -> tuple[int, Optional[int]]:
    """Extract workspace_id and agent_id from request context."""
    try:
        request = request_var.get()
        
        # For public tools, try to get from JWT claims if available
        if hasattr(request, 'state') and hasattr(request.state, 'jwt_claims'):
            claims = request.state.jwt_claims
            workspace_id = claims.get("workspace_id")
            agent_id = claims.get("agent_id")
            
            if workspace_id is not None:
                return int(workspace_id), int(agent_id) if agent_id is not None else None
        
        # If no JWT claims or no workspace_id in claims, check if it's a dict context
        if hasattr(request, 'get'):
            workspace_id = request.get("workspace_id")
            agent_id = request.get("agent_id")
            
            if workspace_id is not None:
                return int(workspace_id), int(agent_id) if agent_id is not None else None
        
        # Fallback: return default values for testing/public tools
        logger.warning("No workspace_id found in request context, using default values")
        return 1, None
        
    except Exception as e:
        logger.error(f"Failed to extract workspace/agent IDs from context: {e}")
        # Fallback to default values for testing
        return 1, None


@mcp.tool()
async def use_llm_provider(
    provider: str,
    workspace_id: Optional[int] = None,
    agent_id: Optional[int] = None,
    set_as_current: bool = True
) -> Dict[str, Any]:
    """
    Configure and optionally switch to an LLM provider with persistent storage.
    
    This tool configures an LLM provider using environment variables and saves
    the provider selection to the database for persistence across service restarts.
    
    Args:
        provider: Provider name ('azure', 'openai', 'gcp', 'quasar')
        workspace_id: Workspace ID (extracted from context if not provided)
        agent_id: Agent ID (None for workspace default, extracted from context if not provided)
        set_as_current: Whether to set as the current active provider
    
    Returns:
        Dictionary with configuration status and result
    """
    try:
        # Handle empty string parameters by converting them to None
        if agent_id == '' or agent_id == 'None':
            agent_id = None
        if workspace_id == '' or workspace_id == 'None':
            workspace_id = None
            
        # Convert string numbers to integers if needed
        if isinstance(agent_id, str) and agent_id.isdigit():
            agent_id = int(agent_id)
        if isinstance(workspace_id, str) and workspace_id.isdigit():
            workspace_id = int(workspace_id)
            
        # Extract workspace and agent IDs from context if not provided
        if workspace_id is None or agent_id is None:
            context_workspace_id, context_agent_id = _extract_workspace_and_agent_ids()
            workspace_id = workspace_id or context_workspace_id
            agent_id = agent_id if agent_id is not None else context_agent_id
        
        # Get AI manager with persistence
        ai_manager = get_ai_manager(workspace_id=workspace_id, agent_id=agent_id)
        
        # Try to configure provider from environment variables
        # Load environment variables from KnowledgeCurator side and pass to common package
        import os
        from dotenv import load_dotenv
        load_dotenv()  # Ensure .env is loaded from KnowledgeCurator
        
        # Create config dict with environment variables from this service
        if provider.lower() == "azure":
            config_dict = {
                "provider_name": "azure",
                "api_key": os.getenv("AZURE_OPENAI_LLM_MODEL_API_KEY"),
                "endpoint": os.getenv("AZURE_OPENAI_LLM_MODEL_API_BASE"),
                "model": os.getenv("AZURE_OPENAI_LLM_MODEL_LLM_MODEL"),
                "deployment_name": os.getenv("AZURE_OPENAI_LLM_MODEL_LLM_MODEL"),
                "api_version": os.getenv("AZURE_OPENAI_LLM_MODEL_API_VERSION", "2023-12-01-preview"),
                "extra_params": {}
            }
        elif provider.lower() == "quasar":
            config_dict = {
                "provider_name": "quasar",
                "api_key": os.getenv("QUASAR_API_KEY"),
                "endpoint": os.getenv("QUASAR_ENDPOINT_URL"),
                "model": os.getenv("QUASAR_MODEL", "claude-sonnet-4"),
                "extra_params": {}
            }
        elif provider.lower() == "openai":
            config_dict = {
                "provider_name": "openai",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "endpoint": os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1"),
                "model": os.getenv("OPENAI_MODEL", "gpt-4"),
                "organization": os.getenv("OPENAI_ORGANIZATION"),
                "extra_params": {}
            }
        elif provider.lower() == "gcp":
            config_dict = {
                "provider_name": "gcp",
                "api_key": os.getenv("GCP_API_KEY"),
                "endpoint": os.getenv("GCP_ENDPOINT"),
                "model": os.getenv("GCP_MODEL"),
                "project_id": os.getenv("GCP_PROJECT_ID"),
                "extra_params": {}
            }
        else:
            # Fallback for unknown providers
            prefix = provider.upper()
            config_dict = {
                "provider_name": provider,
                "api_key": os.getenv(f"{prefix}_API_KEY"),
                "endpoint": os.getenv(f"{prefix}_ENDPOINT"),
                "model": os.getenv(f"{prefix}_MODEL"),
                "extra_params": {}
            }
        
        # Validate required fields
        if not config_dict.get("api_key"):
            logger.error(f"Missing API key for {provider} provider")
            success = False
        elif not config_dict.get("endpoint"):
            logger.error(f"Missing endpoint for {provider} provider")
            success = False
        elif not config_dict.get("model"):
            logger.error(f"Missing model for {provider} provider")
            success = False
        else:
            # Configure provider with the config dict
            try:
                ai_manager.configure_provider(provider, config_dict)
                success = True
                logger.info(f"Successfully configured {provider} provider from environment variables")
            except Exception as e:
                logger.error(f"Failed to configure {provider} provider: {e}")
                success = False
        
        if not success:
            return {
                "success": False,
                "error": f"Failed to configure {provider} from environment variables. Please check your environment configuration."
            }
        
        # Save to database
        _save_configuration_to_db(workspace_id, agent_id, provider, set_as_current)
        
        # Set as current provider if requested
        if set_as_current:
            ai_manager.set_current_provider(provider)
        
        # Get current status
        try:
            status = ai_manager.get_configuration_status()
        except AttributeError:
            # Fallback if method doesn't exist
            status = {
                "current_provider": ai_manager.get_current_provider(),
                "configured_providers": ai_manager.list_configured_providers(),
                "available_providers": ["azure", "openai", "gcp", "quasar"]
            }
        
        return {
            "success": True,
            "message": f"Successfully configured {provider} from environment variables" + 
                      (f" and set as current provider" if set_as_current else ""),
            "provider": provider,
            "workspace_id": workspace_id,
            "agent_id": agent_id,
            "config_source": "environment",
            "current_provider": status.get("current_provider"),
            "configured_providers": status.get("configured_providers"),
            "persistence_enabled": True
        }
        
    except Exception as e:
        logger.error(f"Failed to configure provider {provider}: {e}")
        return {
            "success": False,
            "error": f"Configuration failed: {str(e)}"
        }


@mcp.tool()
async def query_llm_router_status(
    workspace_id: Optional[int] = None,
    agent_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Query the current status of the LLM router.
    
    Args:
        workspace_id: Workspace ID (extracted from context if not provided)
        agent_id: Agent ID (extracted from context if not provided)
    
    Returns:
        Dictionary with current router status and configuration
    """
    try:
        # Handle empty string parameters by converting them to None
        if agent_id == '' or agent_id == 'None':
            agent_id = None
        if workspace_id == '' or workspace_id == 'None':
            workspace_id = None
            
        # Convert string numbers to integers if needed
        if isinstance(agent_id, str) and agent_id.isdigit():
            agent_id = int(agent_id)
        if isinstance(workspace_id, str) and workspace_id.isdigit():
            workspace_id = int(workspace_id)
            
        # Extract workspace and agent IDs from context if not provided
        if workspace_id is None or agent_id is None:
            context_workspace_id, context_agent_id = _extract_workspace_and_agent_ids()
            workspace_id = workspace_id or context_workspace_id
            agent_id = agent_id if agent_id is not None else context_agent_id
        
        # Get AI manager with persistence
        ai_manager = get_ai_manager(workspace_id=workspace_id, agent_id=agent_id)
        
        # Get configuration status
        try:
            status = ai_manager.get_configuration_status()
        except AttributeError:
            # Fallback if method doesn't exist
            status = {
                "current_provider": ai_manager.get_current_provider(),
                "configured_providers": ai_manager.list_configured_providers(),
                "available_providers": ["azure", "openai", "gcp", "quasar"]
            }
        
        # Get database configuration
        db_config = agent_llm_config_service.get_effective_configuration(workspace_id, agent_id)
        
        return {
            "success": True,
            "workspace_id": workspace_id,
            "agent_id": agent_id,
            "current_provider": status.get("current_provider"),
            "configured_providers": status.get("configured_providers", []),
            "available_providers": ["azure", "openai", "gcp", "quasar"]
        }
        
    except Exception as e:
        logger.error(f"Failed to query router status: {e}")
        return {
            "success": False,
            "error": f"Status query failed: {str(e)}"
        }


@mcp.tool()
@require_auth_async
async def switch_llm_provider(
    provider: str,
    workspace_id: Optional[int] = None,
    agent_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Switch to a different LLM provider (must be already configured).
    
    Args:
        provider: Provider name to switch to
        workspace_id: Workspace ID (extracted from context if not provided)
        agent_id: Agent ID (extracted from context if not provided)
    
    Returns:
        Dictionary with switch result
    """
    try:
        # Handle empty string parameters by converting them to None
        if agent_id == '' or agent_id == 'None':
            agent_id = None
        if workspace_id == '' or workspace_id == 'None':
            workspace_id = None
            
        # Convert string numbers to integers if needed
        if isinstance(agent_id, str) and agent_id.isdigit():
            agent_id = int(agent_id)
        if isinstance(workspace_id, str) and workspace_id.isdigit():
            workspace_id = int(workspace_id)
            
        # Extract workspace and agent IDs from context if not provided
        if workspace_id is None or agent_id is None:
            context_workspace_id, context_agent_id = _extract_workspace_and_agent_ids()
            workspace_id = workspace_id or context_workspace_id
            agent_id = agent_id if agent_id is not None else context_agent_id
        
        # Get AI manager with persistence
        ai_manager = get_ai_manager(workspace_id=workspace_id, agent_id=agent_id)
        
        # Check if provider is configured
        configured_providers = ai_manager.list_configured_providers()
        if provider not in configured_providers:
            return {
                "success": False,
                "error": f"Provider {provider} is not configured. Configured providers: {configured_providers}"
            }
        
        # Switch provider
        ai_manager.set_current_provider(provider)
        
        # Save to database
        agent_llm_config_service.switch_provider(
            workspace_id=workspace_id,
            provider=provider,
            agent_id=agent_id
        )
        
        return {
            "success": True,
            "message": f"Successfully switched to provider: {provider}",
            "current_provider": provider,
            "workspace_id": workspace_id,
            "agent_id": agent_id,
            "configured_providers": configured_providers
        }
        
    except Exception as e:
        logger.error(f"Failed to switch provider to {provider}: {e}")
        return {
            "success": False,
            "error": f"Provider switch failed: {str(e)}"
        }


@mcp.tool()
async def test_llm_generation(
    prompt: str = "Hello, how are you?",
    workspace_id: Optional[int] = None,
    agent_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Test text generation with the LLM router using the current provider.
    
    Args:
        prompt: Text prompt to generate response for
        workspace_id: Workspace ID (extracted from context if not provided)
        agent_id: Agent ID (extracted from context if not provided)
    
    Returns:
        Dictionary with generation result
    """
    try:
        # Handle empty string parameters by converting them to None
        if agent_id == '' or agent_id == 'None':
            agent_id = None
        if workspace_id == '' or workspace_id == 'None':
            workspace_id = None
            
        # Convert string numbers to integers if needed
        if isinstance(agent_id, str) and agent_id.isdigit():
            agent_id = int(agent_id)
        if isinstance(workspace_id, str) and workspace_id.isdigit():
            workspace_id = int(workspace_id)
            
        # Extract workspace and agent IDs from context if not provided
        if workspace_id is None or agent_id is None:
            context_workspace_id, context_agent_id = _extract_workspace_and_agent_ids()
            workspace_id = workspace_id or context_workspace_id
            agent_id = agent_id if agent_id is not None else context_agent_id
        
        # Get AI manager with persistence
        ai_manager = get_ai_manager(workspace_id=workspace_id, agent_id=agent_id)
        
        # Use current provider only
        current_provider = ai_manager.get_current_provider()
        if not current_provider:
            return {
                "success": False,
                "error": "No provider is currently set. Please configure and set a provider first."
            }
        
        # Generate text using current provider
        response = ai_manager.generate_text(prompt)
        
        return {
            "success": True,
            "prompt": prompt,
            "response": response,
            "provider_used": current_provider,
            "workspace_id": workspace_id,
            "agent_id": agent_id,
            "response_length": len(response) if response else 0
        }
        
    except Exception as e:
        logger.error(f"Failed to generate text: {e}")
        return {
            "success": False,
            "error": f"Text generation failed: {str(e)}"
        }


@mcp.tool()
@require_auth_async
async def list_llm_providers(
    workspace_id: Optional[int] = None,
    agent_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    List all available and configured LLM providers.
    
    Args:
        workspace_id: Workspace ID (extracted from context if not provided)
        agent_id: Agent ID (extracted from context if not provided)
    
    Returns:
        Dictionary with provider information
    """
    try:
        # Handle empty string parameters by converting them to None
        if agent_id == '' or agent_id == 'None':
            agent_id = None
        if workspace_id == '' or workspace_id == 'None':
            workspace_id = None
            
        # Convert string numbers to integers if needed
        if isinstance(agent_id, str) and agent_id.isdigit():
            agent_id = int(agent_id)
        if isinstance(workspace_id, str) and workspace_id.isdigit():
            workspace_id = int(workspace_id)
        
        # Extract workspace and agent IDs from context if not provided
        if workspace_id is None or agent_id is None:
            context_workspace_id, context_agent_id = _extract_workspace_and_agent_ids()
            workspace_id = workspace_id or context_workspace_id
            agent_id = agent_id if agent_id is not None else context_agent_id
        
        # Get AI manager with persistence
        ai_manager = get_ai_manager(workspace_id=workspace_id, agent_id=agent_id)
        
        configured_providers = ai_manager.list_configured_providers()
        current_provider = ai_manager.get_current_provider()
        
        # Get database configuration info
        db_configs = agent_llm_config_service.get_workspace_configurations(workspace_id)
        
        return {
            "success": True,
            "workspace_id": workspace_id,
            "agent_id": agent_id,
            "available_providers": ["azure", "openai", "gcp", "quasar"],
            "configured_providers": configured_providers,
            "current_provider": current_provider,
            "database_configurations": [
                {
                    "id": config['id'],
                    "agent_id": config['agent_id'],
                    "current_provider": config['current_provider'],
                    "configured_providers": config['configured_providers'],
                    "is_workspace_default": config['agent_id'] is None
                }
                for config in db_configs
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to list providers: {e}")
        return {
            "success": False,
            "error": f"Provider listing failed: {str(e)}"
        }


@mcp.tool()
@require_auth_async
async def reset_llm_router(
    workspace_id: Optional[int] = None,
    agent_id: Optional[int] = None,
    delete_from_database: bool = False
) -> Dict[str, Any]:
    """
    Reset the LLM router configuration.
    
    Args:
        workspace_id: Workspace ID (extracted from context if not provided)
        agent_id: Agent ID (extracted from context if not provided)
        delete_from_database: Whether to delete configuration from database
    
    Returns:
        Dictionary with reset result
    """
    try:
        # Handle empty string parameters by converting them to None
        if agent_id == '' or agent_id == 'None':
            agent_id = None
        if workspace_id == '' or workspace_id == 'None':
            workspace_id = None
            
        # Convert string numbers to integers if needed
        if isinstance(agent_id, str) and agent_id.isdigit():
            agent_id = int(agent_id)
        if isinstance(workspace_id, str) and workspace_id.isdigit():
            workspace_id = int(workspace_id)
            
        # Extract workspace and agent IDs from context if not provided
        if workspace_id is None or agent_id is None:
            context_workspace_id, context_agent_id = _extract_workspace_and_agent_ids()
            workspace_id = workspace_id or context_workspace_id
            agent_id = agent_id if agent_id is not None else context_agent_id
        
        # Clear cache using common package function
        from common_adapters.configurableAI import clear_ai_manager_cache
        clear_ai_manager_cache(workspace_id, agent_id)
        # Clear cache functionality not available in current version
        logger.info(f"Cache clearing requested for workspace_id={workspace_id}, agent_id={agent_id} (function not available)")
        
        # Delete from database if requested
        if delete_from_database:
            success = agent_llm_config_service.delete_configuration(workspace_id, agent_id)
            message = "Reset router and deleted configuration from database" if success else "Reset router (no database configuration found)"
        else:
            message = "Reset router configuration (database configuration preserved)"
        
        return {
            "success": True,
            "message": message,
            "workspace_id": workspace_id,
            "agent_id": agent_id,
            "deleted_from_database": delete_from_database
        }
        
    except Exception as e:
        logger.error(f"Failed to reset router: {e}")
        return {
            "success": False,
            "error": f"Router reset failed: {str(e)}"
        }