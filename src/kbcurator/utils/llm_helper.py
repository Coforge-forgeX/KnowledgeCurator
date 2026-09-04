"""
LLM Helper Module - Unified interface for LLM calls.

Supports two backends controlled by USE_TRUSTAI env var:
- USE_TRUSTAI=true  → Routes through TrustAI (guardrails, logging, quota)
- USE_TRUSTAI=false → Routes through ConfigurableAI (legacy)
"""

import os
import logging
import asyncio
from typing import Optional, List, Dict
from urllib.parse import quote_plus

from langchain.messages import HumanMessage, SystemMessage, AIMessage

from kbcurator.utils.request_context import request_var
from kbcurator.utils.auth import get_useremail_userid

logger = logging.getLogger(__name__)

# Environment flag to toggle TrustAI
USE_TRUSTAI = os.getenv("USE_TRUSTAI", "false").lower() == "true"


# =============================================================================
# TrustAI Helpers
# =============================================================================

def _get_postgres_connection_string(database_env: str = "POSTGRESQL_DATABASE_DATABASE") -> str | None:
    """Build PostgreSQL connection string from environment variables."""
    try:
        host = os.getenv("POSTGRESQL_DATABASE_HOST")
        port = os.getenv("POSTGRESQL_DATABASE_PORT", "5432")
        database = os.getenv(database_env)
        user = os.getenv("POSTGRESQL_DATABASE_USER")
        password = os.getenv("POSTGRESQL_DATABASE_PASSWORD")
        if not all([host, port, database, user, password]):
            return None
        password = quote_plus(password)
        return (
            f"postgresql+psycopg2://"
            f"{user}:{password}"
            f"@{host}:{port}/{database}"
            f"?sslmode=require"
        )
    except Exception:
        return None


def _get_user_context() -> Dict[str, str]:
    """Extract user_email and user_id from request headers."""
    request = request_var.get(None)
    if not request:
        return {"user_email": "", "user_id": None}
    try:
        return get_useremail_userid(request.headers)
    except Exception as e:
        logger.warning(f"Failed to get user context: {e}")
        return {"user_email": "", "user_id": None}


def _get_trustai_llm(workspace_id: int, agent_id: Optional[int], temperature: float = 0.7):
    """Get TrustAI router LLM instance."""
    from common_adapters.trustai import get_llm_helper
    
    db_url = os.getenv("POSTGRESQL_DATABASE_URL") or _get_postgres_connection_string()
    user_data = _get_user_context()
    
    helper = get_llm_helper(db_url)
    return helper.get_router_llm(
        workspace_id=str(workspace_id),
        agent_id=str(agent_id or 0),
        user_email=user_data.get("user_email", ""),
        user_id=user_data.get("user_id"),
        temperature=temperature,
    )


def _build_messages(
    prompt: str,
    sys_prompt: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> List:
    """Build LangChain message list from prompt, system prompt, and history."""
    messages = []
    
    if sys_prompt:
        messages.append(SystemMessage(content=sys_prompt))
    
    if history:
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
    
    messages.append(HumanMessage(content=prompt))
    return messages


# =============================================================================
# Legacy ConfigurableAI Helpers
# =============================================================================

def _get_legacy_manager(workspace_id: int, agent_id: Optional[int]):
    """Get ConfigurableAI manager (legacy path)."""
    from kbcurator.tools.llm_router_tool import _build_manager_from_db
    return _build_manager_from_db(workspace_id, agent_id)


def _build_legacy_prompt(
    prompt: str,
    sys_prompt: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Build combined prompt string for legacy ConfigurableAI."""
    full_prompt = ""
    
    if sys_prompt:
        full_prompt = f"System: {sys_prompt}\n\n"
    
    if history:
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            full_prompt += f"{role.capitalize()}: {content}\n"
    
    full_prompt += f"User: {prompt}"
    return full_prompt


# =============================================================================
# Public API
# =============================================================================

def get_llm_response(
    workspace_id: int,
    prompt: str,
    agent_id: Optional[int] = None
) -> str:
    """
    Generate text using the configured LLM for the given workspace/agent.
    
    Args:
        workspace_id: The workspace ID
        prompt: The input prompt text
        agent_id: Optional agent ID for agent-specific configuration
        
    Returns:
        The generated text response
    """
    if USE_TRUSTAI:
        llm = _get_trustai_llm(workspace_id, agent_id)
        messages = _build_messages(prompt)
        response = llm.invoke(messages)
        return response.content
    else:
        manager = _get_legacy_manager(workspace_id, agent_id)
        return manager.generate_text(prompt)


def get_llm_response_with_context(
    workspace_id: int,
    user_input: str,
    sys_prompt: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    agent_id: Optional[int] = None
) -> str:
    """
    Generate text with system prompt and conversation history.
    
    Args:
        workspace_id: The workspace ID
        user_input: The user's current input
        sys_prompt: Optional system prompt
        history: Optional list of conversation history dicts with 'role' and 'content' keys
        agent_id: Optional agent ID for agent-specific configuration
        
    Returns:
        The generated text response
    """
    if USE_TRUSTAI:
        llm = _get_trustai_llm(workspace_id, agent_id)
        messages = _build_messages(user_input, sys_prompt, history)
        response = llm.invoke(messages)
        return response.content
    else:
        manager = _get_legacy_manager(workspace_id, agent_id)
        full_prompt = _build_legacy_prompt(user_input, sys_prompt, history)
        return manager.generate_text(full_prompt)


async def get_llm_response_async(
    workspace_id: int,
    prompt: str,
    agent_id: Optional[int] = None
) -> str:
    """
    Async version of get_llm_response.
    
    Args:
        workspace_id: The workspace ID
        prompt: The input prompt text
        agent_id: Optional agent ID for agent-specific configuration
        
    Returns:
        The generated text response
    """
    if USE_TRUSTAI:
        llm = _get_trustai_llm(workspace_id, agent_id)
        messages = _build_messages(prompt)
        response = await llm.ainvoke(messages)
        return response.content
    else:
        manager = _get_legacy_manager(workspace_id, agent_id)
        return await manager.generate_text_async(prompt)


async def get_llm_response_with_context_async(
    workspace_id: int,
    user_input: str,
    sys_prompt: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    agent_id: Optional[int] = None
) -> str:
    """
    Async version of get_llm_response_with_context.
    
    Args:
        workspace_id: The workspace ID
        user_input: The user's current input
        sys_prompt: Optional system prompt
        history: Optional list of conversation history dicts with 'role' and 'content' keys
        agent_id: Optional agent ID for agent-specific configuration
        
    Returns:
        The generated text response
    """
    print(f"[LLM_HELPER] get_llm_response_with_context_async called | USE_TRUSTAI={USE_TRUSTAI} | workspace={workspace_id} | agent={agent_id}")
    if USE_TRUSTAI:
        print("[LLM_HELPER] Using TrustAI path...")
        llm = _get_trustai_llm(workspace_id, agent_id)
        messages = _build_messages(user_input, sys_prompt, history)
        print(f"[LLM_HELPER] Calling llm.ainvoke with {len(messages)} messages...")
        response = await llm.ainvoke(messages)
        print(f"[LLM_HELPER] TrustAI response received: {len(response.content)} chars")
        return response.content
    else:
        print("[LLM_HELPER] Using legacy ConfigurableAI path...")
        manager = _get_legacy_manager(workspace_id, agent_id)
        print(f"[LLM_HELPER] Manager obtained, provider={manager.get_current_provider() if hasattr(manager, 'get_current_provider') else 'unknown'}")
        full_prompt = _build_legacy_prompt(user_input, sys_prompt, history)
        print(f"[LLM_HELPER] Calling manager.generate_text_async (prompt len={len(full_prompt)})...")
        result = await manager.generate_text_async(full_prompt)
        print(f"[LLM_HELPER] Legacy response received: {len(result) if result else 0} chars")
        return result


# =============================================================================
# Utility: Get Router LLM directly (for callers who need LangChain-compatible LLM)
# =============================================================================

def get_router_llm(workspace_id: int, agent_id: Optional[int] = None, temperature: float = 0.7):
    """
    Get a LangChain-compatible LLM instance for direct use.
    
    Args:
        workspace_id: The workspace ID
        agent_id: Optional agent ID
        temperature: Sampling temperature (default 0.7)
        
    Returns:
        TrustAI router LLM if USE_TRUSTAI=true, else raises NotImplementedError
    """
    if USE_TRUSTAI:
        return _get_trustai_llm(workspace_id, agent_id, temperature)
    else:
        raise NotImplementedError(
            "get_router_llm requires USE_TRUSTAI=true. "
            "Use get_llm_response* functions for legacy path."
        )
