"""
LLM Helper Module - Unified interface for LLM calls using LLM Router.

This module provides common functions to replace direct AzureCustomLLM usage
with the new LLM Router approach that supports multi-provider configuration.
"""

from typing import Optional, List, Dict
from kbcurator.tools.llm_router_tool import _build_manager_from_db
import asyncio


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
        
    Example:
        response = get_llm_response(
            workspace_id=782,
            prompt="What is Python?"
        )
    """
    manager = _build_manager_from_db(workspace_id, agent_id)
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
    
    The ConfigurableAIManager sends everything as a single user message,
    so we build a combined prompt that includes system prompt and history.
    
    Args:
        workspace_id: The workspace ID
        user_input: The user's current input
        sys_prompt: Optional system prompt
        history: Optional list of conversation history dicts with 'role' and 'content' keys
        agent_id: Optional agent ID for agent-specific configuration
        
    Returns:
        The generated text response
        
    Example:
        response = get_llm_response_with_context(
            workspace_id=782,
            user_input="What is Python?",
            sys_prompt="You are a helpful assistant.",
            history=[
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello! How can I help?"}
            ]
        )
    """
    manager = _build_manager_from_db(workspace_id, agent_id)
    
    # Build a combined prompt (ConfigurableAIManager sends as single user message)
    full_prompt = ""
    
    if sys_prompt:
        full_prompt = f"System: {sys_prompt}\n\n"
    
    if history:
        for msg in history:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            full_prompt += f"{role.capitalize()}: {content}\n"
    
    full_prompt += f"User: {user_input}"
    
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
    manager = _build_manager_from_db(workspace_id, agent_id)
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
    manager = _build_manager_from_db(workspace_id, agent_id)
    
    # Build a combined prompt
    full_prompt = ""
    
    if sys_prompt:
        full_prompt = f"System: {sys_prompt}\n\n"
    
    if history:
        for msg in history:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            full_prompt += f"{role.capitalize()}: {content}\n"
    
    full_prompt += f"User: {user_input}"
    
    return await manager.generate_text_async(full_prompt)
