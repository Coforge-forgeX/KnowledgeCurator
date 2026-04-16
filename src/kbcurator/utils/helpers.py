from kbcurator.utils.azurecustomllm import AzureCustomLLM
import re
from kbcurator.utils.prompt_builder import PromptBuilder

# Function to evaluate quality of prompt
def evaluate_user_input(user_input,industry,sub_industry):
    llm_classifier = AzureCustomLLM()
    quality_prompt = PromptBuilder.get_intent_prompt(user_input, industry, sub_industry) 

    response = llm_classifier._call(
            input=user_input,
            sys_prompt=quality_prompt
        )
    
    return response


def preprocessing_for_edits(parsed_Data: dict, industry:str, sub_industry:str):
    print(f"Preprocessing assistant message for edits: {parsed_Data}")
    
    entity = parsed_Data.get("entity")
    entity_new_value = parsed_Data.get("entity_new_value")
    descr_new_value = parsed_Data.get("descr_new_value")
    properties = parsed_Data.get("property", [])
    properties = [p.lower() for p in parsed_Data.get("property", [])]

    updated_data = {}
    edit_arguments = {}

    # 1. Both label and description are being updated
    if "label" in properties and entity_new_value and entity_new_value != entity and \
    "description" in properties and descr_new_value:
        updated_data["entity_id"] = entity_new_value
        updated_data["labels"] = [entity_new_value]
        updated_data["description"] = descr_new_value

    # 2. Only label is being updated
    elif "label" in properties and entity_new_value and entity_new_value != entity:
        print('inside label updation')
        updated_data["entity_id"] = entity_new_value
        updated_data["labels"] = [entity_new_value]
        # updated_data["description"] = descr_new_value
        
    # 3. Only description is being updated
    elif "description" in properties and descr_new_value:
        print('inside description updation')
        # updated_data["entity_id"] = entity
        # updated_data["labels"] = [entity]
        updated_data["description"] = descr_new_value

    # # 4. If entity_new_value is None or empty, keep entity_id and labels as entity name
    # elif not entity_new_value:
    #     updated_data["entity_id"] = entity
    #     updated_data["labels"] = [entity]
    #     if "description" in properties and descr_new_value:
    #         updated_data["description"] = descr_new_value

    edit_arguments = {
    "domain": industry,
    "kb_name": sub_industry,
    "entity_name": entity,  # original value
    "updated_data": updated_data}
    
    return edit_arguments
