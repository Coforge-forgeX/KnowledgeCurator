from typing import Optional
from kbcurator.utils.llm_helper import get_llm_response_with_context_async
from kbcurator.utils.prompt_builder import PromptBuilder

async def classifier(user_prompt: str, sys_prompt: str, workspace_id: int, history: Optional[list|None] = None, agent_id: Optional[int] = None) -> str:
    """
    Classifies the user prompt based on the system prompt.
    
    Args:
        user_prompt (str): The user prompt to classify.
        sys_prompt (str): The system prompt for classification.
        workspace_id (int): The workspace ID.
        history (list|None = None): history of conversations.
        agent_id (int|None = None): Optional agent ID for agent-specific configuration.
        
    Returns:
        str: The classification result.
    """
    
    # Use the LLM Router to classify the user prompt
    classification = await get_llm_response_with_context_async(
        workspace_id=workspace_id,
        user_input=user_prompt,
        sys_prompt=sys_prompt,
        history=history,
        agent_id=agent_id
    )
    
    # if isinstance(classification, str):
    #     classification = classification.strip()
    #     # Remove any prefix like "Intent: "
    #     if classification.lower().startswith("intent:"):
    #         classification = classification.split(":", 1)[1].strip()
    return classification
