"""
LLM Router Tool for testing ConfigurableAI common adapter.

This tool provides a single MCP endpoint to test the LLM routing functionality
similar to test_azure_production.py, allowing users to:
- Configure and switch LLM providers (auto-fallback from env to manual)
- Query current provider status
- Test text generation with different providers
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

from kbcurator.server.server import mcp
from kbcurator.utils.request_context import request_var
from common_adapters.configurableAI import ConfigurableAIManager
from common_adapters.configurableAI.config import (
    OpenAIConfig, 
    AzureOpenAIConfig, 
    GCPConfig,
    QuasarConfig
)

logger = logging.getLogger(__name__)

# Global AI manager instance for testing
_ai_manager: Optional[ConfigurableAIManager] = None


def get_ai_manager() -> ConfigurableAIManager:
    """Get or create the global AI manager instance with Azure as default provider."""
    global _ai_manager
    if _ai_manager is None:
        _ai_manager = ConfigurableAIManager()
        
        # Auto-configure Azure as default provider if environment variables are available
        try:
            _ai_manager.configure_from_env("azure")
            logger.info("Successfully auto-configured Azure as default provider")
        except Exception as e:
            logger.warning(f"Failed to auto-configure Azure from environment: {e}")
            # Try manual configuration as fallback
            try:
                manual_config = _get_manual_config("azure")
                if manual_config:
                    _ai_manager.configure_provider("azure", manual_config)
                    logger.info("Successfully configured Azure with manual fallback as default")
                else:
                    logger.warning("Azure environment variables not found, no default provider configured")
            except Exception as manual_error:
                logger.error(f"Failed to configure Azure manually: {manual_error}")
    
    return _ai_manager


@mcp.tool()
def use_llm_provider(provider_name: str) -> Dict[str, Any]:
    """
    Configure and switch to an LLM provider. Auto-fallback from env to manual configuration.
    
    Args:
        provider_name: Name of the provider ('openai', 'azure', 'gcp', 'quasar')
    
    Returns:
        Configuration and switch result with status and details
    """
    try:
        manager = get_ai_manager()
        provider_name = provider_name.lower()
        
        # Check if provider is already configured
        configured_providers = manager.list_configured_providers()
        
        if provider_name in configured_providers:
            # Provider already configured, just switch to it
            previous_provider = manager.get_current_provider()
            manager.set_current_provider(provider_name)
            
            return {
                "status": "success",
                "action": "switched",
                "message": f"Switched to existing provider '{provider_name}'",
                "provider": provider_name,
                "previous_provider": previous_provider,
                "current_provider": manager.get_current_provider(),
                "configured_providers": manager.list_configured_providers()
            }
        
        # Provider not configured, try to configure it
        configuration_method = None
        configuration_error = None
        
        # First try: configure from environment variables (like test_configure_from_env)
        try:
            manager.configure_from_env(provider_name)
            configuration_method = "environment"
            logger.info(f"Successfully configured {provider_name} from environment")
            
        except Exception as env_error:
            configuration_error = str(env_error)
            logger.warning(f"Failed to configure {provider_name} from environment: {env_error}")
            
            # Second try: manual configuration with default values (like test_manual_configuration)
            try:
                manual_config = _get_manual_config(provider_name)
                if manual_config:
                    manager.configure_provider(provider_name, manual_config)
                    configuration_method = "manual_fallback"
                    logger.info(f"Successfully configured {provider_name} with manual fallback")
                else:
                    raise ValueError(f"No manual configuration available for {provider_name}")
                    
            except Exception as manual_error:
                error_result = {
                    "status": "error",
                    "action": "configuration_failed",
                    "message": f"Failed to configure {provider_name}",
                    "provider": provider_name,
                    "errors": {
                        "environment_config": configuration_error,
                        "manual_config": str(manual_error)
                    },
                    "required_env_vars": _get_required_env_vars(provider_name)
                }
                # Raise exception to set isError: true in MCP response
                raise ValueError(f"Configuration failed for {provider_name}: {error_result['message']}")
        
        # Configuration successful, provider is now active
        result = {
            "status": "success",
            "action": "configured_and_switched",
            "message": f"Successfully configured and switched to {provider_name}",
            "provider": provider_name,
            "configuration_method": configuration_method,
            "current_provider": manager.get_current_provider(),
            "configured_providers": manager.list_configured_providers()
        }
        
        logger.info(f"LLM provider use result: {result}")
        return result
        
    except Exception as e:
        error_result = {
            "status": "error",
            "action": "unexpected_error",
            "message": f"Unexpected error using {provider_name}: {str(e)}",
            "provider": provider_name,
            "error_type": type(e).__name__
        }
        logger.error(f"LLM provider use error: {error_result}")
        # Re-raise to ensure isError: true in MCP response
        raise


def _get_manual_config(provider_name: str) -> Optional[Dict[str, Any]]:
    """Get manual configuration for a provider using environment variables."""
    if provider_name == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        return {
            "api_key": api_key,
            "model": "gpt-3.5-turbo",
            "organization": os.getenv("OPENAI_ORGANIZATION")
        }
    
    elif provider_name == "azure":
        api_key = os.getenv("AZURE_OPENAI_LLM_MODEL_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_LLM_MODEL_API_BASE")
        deployment = os.getenv("AZURE_OPENAI_LLM_MODEL_LLM_MODEL")
        api_version = os.getenv("AZURE_OPENAI_LLM_MODEL_API_VERSION")
        
        if not all([api_key, endpoint, deployment, api_version]):
            return None
        
        return {
            "api_key": api_key,
            "endpoint": endpoint,
            "deployment_name": deployment,
            "api_version": api_version,
            "model": deployment
        }
    
    elif provider_name == "gcp":
        project_id = os.getenv("GCP_PROJECT_ID")
        credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        
        if not all([project_id, credentials]):
            return None
            
        return {
            "project_id": project_id,
            "credentials_path": credentials
        }
    
    elif provider_name == "quasar":
        endpoint_url = os.getenv("QUASAR_ENDPOINT_URL")
        api_key = os.getenv("QUASAR_API_KEY")
        model = os.getenv("QUASAR_MODEL", "claude-sonnet-4")
        
        if not all([endpoint_url, api_key]):
            return None
            
        return {
            "endpoint_url": endpoint_url,
            "api_key": api_key,
            "model": model
        }
    
    return None


@mcp.tool()
def query_llm_router_status() -> Dict[str, Any]:
    """
    Query the current status of the LLM router.
    
    Returns:
        Current router status including providers and configuration
    """
    try:
        manager = get_ai_manager()
        
        # Get environment variable status for debugging
        env_vars = {}
        for provider in ["openai", "azure", "gcp", "quasar"]:
            if provider == "openai":
                env_vars["openai"] = {
                    "api_key": "found" if os.getenv("OPENAI_API_KEY") else "missing",
                    "organization": "found" if os.getenv("OPENAI_ORGANIZATION") else "missing"
                }
            elif provider == "azure":
                env_vars["azure"] = {
                    "api_key": "found" if os.getenv("AZURE_OPENAI_LLM_MODEL_API_KEY") else "missing",
                    "endpoint": "found" if os.getenv("AZURE_OPENAI_LLM_MODEL_API_BASE") else "missing",
                    "deployment": "found" if os.getenv("AZURE_OPENAI_LLM_MODEL_LLM_MODEL") else "missing",
                    "api_version": "found" if os.getenv("AZURE_OPENAI_LLM_MODEL_API_VERSION") else "missing"
                }
            elif provider == "gcp":
                env_vars["gcp"] = {
                    "project_id": "found" if os.getenv("GCP_PROJECT_ID") else "missing",
                    "credentials": "found" if os.getenv("GOOGLE_APPLICATION_CREDENTIALS") else "missing"
                }
            elif provider == "quasar":
                env_vars["quasar"] = {
                    "endpoint_url": "found" if os.getenv("QUASAR_ENDPOINT_URL") else "missing",
                    "api_key": "found" if os.getenv("QUASAR_API_KEY") else "missing",
                    "model": "found" if os.getenv("QUASAR_MODEL") else "missing"
                }
        
        result = {
            "status": "success",
            "current_provider": manager.get_current_provider(),
            "configured_providers": manager.list_configured_providers(),
            "available_providers": manager.list_available_providers(),
            "environment_variables": env_vars,
            "total_configured": len(manager.list_configured_providers()),
            "has_active_provider": manager.get_current_provider() is not None
        }
        
        logger.info(f"LLM router status query result: {result}")
        return result
        
    except Exception as e:
        error_result = {
            "status": "error",
            "message": f"Failed to query router status: {str(e)}",
            "error_type": type(e).__name__
        }
        logger.error(f"LLM router status query error: {error_result}")
        return error_result


@mcp.tool()
def list_llm_providers() -> Dict[str, Any]:
    """
    List all available and configured LLM providers.
    
    Returns:
        Detailed information about providers
    """
    try:
        manager = get_ai_manager()
        
        available_providers = manager.list_available_providers()
        configured_providers = manager.list_configured_providers()
        current_provider = manager.get_current_provider()
        
        # Create detailed provider info
        provider_details = {}
        for provider in available_providers:
            provider_details[provider] = {
                "available": True,
                "configured": provider in configured_providers,
                "is_current": provider == current_provider,
                "required_env_vars": _get_required_env_vars(provider)
            }
        
        result = {
            "status": "success",
            "available_providers": available_providers,
            "configured_providers": configured_providers,
            "current_provider": current_provider,
            "provider_details": provider_details,
            "summary": {
                "total_available": len(available_providers),
                "total_configured": len(configured_providers),
                "has_current": current_provider is not None
            }
        }
        
        logger.info(f"LLM providers list result: {result}")
        return result
        
    except Exception as e:
        error_result = {
            "status": "error",
            "message": f"Failed to list providers: {str(e)}",
            "error_type": type(e).__name__
        }
        logger.error(f"LLM providers list error: {error_result}")
        return error_result


@mcp.tool()
async def test_llm_generation(
    prompt: str = "Explain what artificial intelligence is in one sentence.",
    provider: Optional[str] = None,
    max_tokens: int = 100,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """
    Test text generation with the current or specified LLM provider.
    
    Args:
        prompt: Text prompt for generation
        provider: Specific provider to use (optional, uses current if not specified)
        max_tokens: Maximum tokens to generate
        temperature: Generation temperature
        
    Returns:
        Generation result with response and metadata
    """
    try:
        manager = get_ai_manager()
        
        # Determine which provider to use
        test_provider = provider or manager.get_current_provider()
        
        if not test_provider:
            return {
                "status": "error",
                "message": "No provider specified and no current provider set",
                "configured_providers": manager.list_configured_providers()
            }
        
        if test_provider not in manager.list_configured_providers():
            return {
                "status": "error",
                "message": f"Provider '{test_provider}' is not configured",
                "configured_providers": manager.list_configured_providers()
            }
        
        # Generate text
        start_time = asyncio.get_event_loop().time()
        response = await manager.generate_text_async(
            prompt,
            provider=test_provider,
            max_tokens=max_tokens,
            temperature=temperature
        )
        end_time = asyncio.get_event_loop().time()
        
        result = {
            "status": "success",
            "provider": test_provider,
            "prompt": prompt,
            "response": response,
            "parameters": {
                "max_tokens": max_tokens,
                "temperature": temperature
            },
            "metadata": {
                "response_time_seconds": round(end_time - start_time, 3),
                "response_length": len(response) if response else 0
            }
        }
        
        logger.info(f"LLM generation test result: {result}")
        return result
        
    except Exception as e:
        error_result = {
            "status": "error",
            "message": f"Failed to generate text: {str(e)}",
            "provider": provider,
            "prompt": prompt,
            "error_type": type(e).__name__
        }
        logger.error(f"LLM generation test error: {error_result}")
        return error_result


@mcp.tool()
def reset_llm_router() -> Dict[str, Any]:
    """
    Reset the LLM router, clearing all configurations.
    
    Returns:
        Reset operation result
    """
    try:
        global _ai_manager
        
        # Store previous state for logging
        previous_state = {}
        if _ai_manager:
            previous_state = {
                "current_provider": _ai_manager.get_current_provider(),
                "configured_providers": _ai_manager.list_configured_providers()
            }
        
        # Reset by creating new instance
        _ai_manager = ConfigurableAIManager()
        
        result = {
            "status": "success",
            "message": "LLM router reset successfully",
            "previous_state": previous_state,
            "current_state": {
                "current_provider": _ai_manager.get_current_provider(),
                "configured_providers": _ai_manager.list_configured_providers()
            }
        }
        
        logger.info(f"LLM router reset result: {result}")
        return result
        
    except Exception as e:
        error_result = {
            "status": "error",
            "message": f"Failed to reset router: {str(e)}",
            "error_type": type(e).__name__
        }
        logger.error(f"LLM router reset error: {error_result}")
        return error_result


@mcp.tool()
def check_default_azure_config() -> Dict[str, Any]:
    """
    Check if Azure is properly configured as the default provider.
    
    Returns:
        Status of Azure default configuration
    """
    try:
        manager = get_ai_manager()
        
        # Check if Azure is configured
        configured_providers = manager.list_configured_providers()
        current_provider = manager.get_current_provider()
        
        # Check Azure environment variables
        azure_env_vars = {
            "AZURE_OPENAI_LLM_MODEL_API_KEY": "found" if os.getenv("AZURE_OPENAI_LLM_MODEL_API_KEY") else "missing",
            "AZURE_OPENAI_LLM_MODEL_API_BASE": "found" if os.getenv("AZURE_OPENAI_LLM_MODEL_API_BASE") else "missing",
            "AZURE_OPENAI_LLM_MODEL_LLM_MODEL": "found" if os.getenv("AZURE_OPENAI_LLM_MODEL_LLM_MODEL") else "missing",
            "AZURE_OPENAI_LLM_MODEL_API_VERSION": "found" if os.getenv("AZURE_OPENAI_LLM_MODEL_API_VERSION") else "missing"
        }
        
        azure_configured = "azure" in configured_providers
        azure_is_current = current_provider == "azure"
        all_env_vars_present = all(status == "found" for status in azure_env_vars.values())
        
        result = {
            "status": "success",
            "azure_configured": azure_configured,
            "azure_is_current_provider": azure_is_current,
            "azure_environment_variables": azure_env_vars,
            "all_azure_env_vars_present": all_env_vars_present,
            "current_provider": current_provider,
            "configured_providers": configured_providers,
            "default_setup_successful": azure_configured and azure_is_current
        }
        
        logger.info(f"Azure default config check result: {result}")
        return result
        
    except Exception as e:
        error_result = {
            "status": "error",
            "message": f"Failed to check Azure default config: {str(e)}",
            "error_type": type(e).__name__
        }
        logger.error(f"Azure default config check error: {error_result}")
        return error_result


def _get_required_env_vars(provider: str) -> List[str]:
    """Get required environment variables for a provider."""
    env_vars = {
        "openai": ["OPENAI_API_KEY"],
        "azure": [
            "AZURE_OPENAI_LLM_MODEL_API_KEY",
            "AZURE_OPENAI_LLM_MODEL_API_BASE", 
            "AZURE_OPENAI_LLM_MODEL_LLM_MODEL",
            "AZURE_OPENAI_LLM_MODEL_API_VERSION"
        ],
        "gcp": ["GCP_PROJECT_ID", "GOOGLE_APPLICATION_CREDENTIALS"],
        "quasar": ["QUASAR_ENDPOINT_URL", "QUASAR_API_KEY", "QUASAR_MODEL"]
    }
    return env_vars.get(provider, [])