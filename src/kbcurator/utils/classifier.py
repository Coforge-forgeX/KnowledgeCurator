from typing import Optional
from .azurecustomllm import AzureCustomLLM
from .prompt_builder import PromptBuilder

llm_classifier = AzureCustomLLM()
 
async def classifier(user_prompt: str , sys_prompt: str, history : Optional[list|None] = None )-> str:
    """
    Classifies the user prompt based on the system prompt.
    
    Args:
        user_prompt (str): The user prompt to classify.
        history (list|None = None): history of conversations.
        
    Returns:
        str: The classification result.
    """
    
    # sys_prompt = PromptBuilder.get_intent_prompt(user_prompt)
    # print(f"System_prompt", sys_prompt)
    # Use the LLM to classify the user prompt
    classification = llm_classifier.invoke(input = user_prompt , sys_prompt = sys_prompt , history = history)
    
    # if isinstance(classification, str):
    #     classification = classification.strip()
    #     # Remove any prefix like "Intent: "
    #     if classification.lower().startswith("intent:"):
    #         classification = classification.split(":", 1)[1].strip()
    return classification
