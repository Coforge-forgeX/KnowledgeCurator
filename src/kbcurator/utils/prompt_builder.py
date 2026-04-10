from typing import List, Dict, Any

class PromptBuilder:
    """
    Generic prompt builder for various chatbot use cases.
    All methods are static for easy access.
    """

    @staticmethod
    def get_intent_prompt(user_input: str, industry: str, sub_industry:str) -> str:
        """
        Build a prompt for intent classification.
        """
        training_data = [
            # QUERY examples
            "What is the weather today?",
            "How do I reset my password?",
            "Can you tell me about Python programming?",
            "Where is the nearest restaurant?",
            "What time is it in New York?",
            "Help me find information about machine learning",
            "Generate mock data for testing",             
            "Create a user manual",            
            "Generate a status report",            
            "Generate API documentation",
            "Create a new report",
            "Could you please provide a details?"
            # UPLOAD examples
            "Upload this CSV file for analysis",
            "Process the attached document",
            "Can you analyze this spreadsheet?",
            "I need to submit this file",
            "Please import this data",
            "Load this JSON file",
            "Parse the uploaded PDF",
            "Extract data from this Excel file",
            "Process the image I'm sending",
            "upload the file",
            # UPDATE examples            
            "Build a dashboard for metrics",
            "Make a presentation outline",           
            "Build a configuration file",
            "Update the node",
            "Modify the details"
        ]

        example_intents = []
        for ex in training_data:
            ex_lower = ex.lower()
            if ex_lower.startswith(("what", "how", "can you", "where", "who", "why", "which", "show", "tell", "find", "search", "help", "explain", "generate", "build", "provide", "give", "list", "summarize","could you","could you please","can you please")):
                example_intents.append({"user": ex, "intent": "QUERY"})
            elif any(word in ex_lower for word in ["upload", "import", "load", "process", "submit", "file", "document", "data", "attachment", "csv", "pdf", "excel", "json", "image", "analyze"]):
                example_intents.append({"user": ex, "intent": "UPLOAD"})
            elif any(word in ex_lower for word in ["create", "modify", "update", "delete", "make", "produce", "new", "design", "develop", "construct", "template", "report", "document", "dashboard", "guide"]):
                example_intents.append({"user": ex, "intent": "CREATE"})
            else:
                example_intents.append({"user": ex, "intent": "OTHER"})

        validation_examples = [
            {"user": "What is the status of my order?", "intent": "QUERY"},
            {"user": "Could you please provide me the list of files?", "intent": "QUERY"},
            {"user": "Please process this CSV file", "intent": "UPLOAD"},
            {"user": "Modify the details", "intent": "CREATE"},
            {"user": "risk", "intent": "OTHER"},
            {"user": "hello", "intent": "OTHER"},
            {"user": "update", "intent": "OTHER"},
            {"user": "graph", "intent": "OTHER"},
            {"user": "file", "intent": "OTHER"},
            {"user": "data", "intent": "OTHER"},
            {"user": "good morning", "intent": "OTHER"},
            {"user": "information", "intent": "OTHER"},
            {"user": "Can you help?", "intent": "OTHER"},
            {"user": "Please assist", "intent": "OTHER"}
        ]

        example_intents.extend(validation_examples)

        prompt_examples = "\n".join(
            [f'User: "{ex["user"]}"\nIntent: {ex["intent"]}' for ex in example_intents]
        )
        
        final_prompt = (
            "You are an expert intent classifier for a multi-functional chatbot.\n"
            "The chatbot is designed to handle queries within the following domain:\n"
            # f"Industry: {industry}\n"
            # f"Sub-Industry: {sub_industry}\n\n"
            # "If the user's query is unrelated to this domain, classify the intent as OTHER.\n"
            "If the user is greeting, or provides a single word, vague, or contextless input, classify the intent as OTHER.\n"
            "Classify the user's intent as one word: CREATE, UPLOAD, QUERY, UPDATE, or OTHER.\n"
            "Return only the intent word (CREATE, UPLOAD, QUERY, UPDATE, OTHER). Do not include any prefix or explanation.\n"
            "Use the following examples:\n"
            f"{prompt_examples}\n"
            f'User: "{user_input}"\nIntent:'
        )

        return final_prompt

    @staticmethod
    def get_system_prompt() -> List[Dict[str, Any]]:
        """
        Return the system prompt for the chatbot.
        """
        sys_prompt = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant for graph database operations. Respond to every user request with clarity and friendliness:\n"
                    "1. Confirm the requested operation.\n"
                    "2. Indicate when processing is in progress.\n"
                    "3. Share results with relevant metrics and use emojis (✅ success, 🔄 processing, ❌ error).\n"
                    "Always reference the input source (file, data, etc.) and keep your tone professional and supportive.\n"
                    "\n"
                    "Special Cases:\n"
                    "- Greeting: If the user greets, reply: 'Hello! How can I help you with graph operations today?'\n"
                    "- Thanks: If the user expresses gratitude (e.g., 'thanks', 'thank you'), reply: 'Glad I could help! Let me know if you want to do more graph operations.'\n"
                    "- Insufficient Context: If the input is a single word, vague, or contextless, reply: 'Could you please provide more details or context about your graph request?'\n"
                    "\n"
                    "Operation Types:\n"
                    "- CREATE: Add new nodes/relationships.\n"
                    "- SEARCH: Query existing data.\n"
                    "- UPDATE: Modify existing elements.\n"
                    "- UPLOAD: Add new nodes/relationships via file.\n"
                    "\n"
                    "Sample Prompts:\n"
                    "*CREATE:*\n"
                    "- 'Sure, I'll create {element_type} in the graph as per your input.'\n"
                    "- '🔄 Creating new {element_type}...'\n"
                    "- '✅ Created {count} new {element_type}. The file has been processed.'\n"
                    "*SEARCH:*\n"
                    "- 'I'll search the graph for your request.'\n"
                    "- '🔄 Running your query...'\n"
                    "- '✅ Found {count} results. Here are the details.'\n"
                    "*UPLOAD:*\n"
                    "- 'I'll process the CSV file for graph updates.'\n"
                    "- '📊 Processing rows and updating the graph...'\n"
                    "- '✅ File processed! {rows_processed} rows, {nodes_affected} nodes, {relationships_affected} relationships updated.'\n"
                    "\n"
                    "Variables:\n"
                    "- {element_type}: nodes, relationships, entities, connections, data points\n"
                    "- {count}: number of items created or found\n"
                    "- Always reference the input source\n"
                    "\n"
                    "Do NOT include technical details or file processing info if no context/file is provided."
                )
            }
        ]
        return sys_prompt

    @staticmethod
    def get_parser_prompt(user_input: str) -> str:
        """
        Build a prompt for extracting structured data from user input.
        """
        training_data = [
            {
                "user": 'Update node KG:Product:12345 — set title to "Acme Wireless Headphones Pro".',
                "action": "UPDATE",
                "entity": "KG:Product:12345",
                "property": ["label"],
                "entity_new_value": "Acme Wireless Headphones Pro",
                "descr_new_value": None
            },
            {
                "user": 'Change description of person:Q778 to "British chemist and Nobel laureate (1954)".',
                "action": "UPDATE",
                "entity": "person:Q778",
                "property": ["description"],
                "entity_new_value": None,
                "descr_new_value": "British chemist and Nobel laureate (1954)"
            },
            {
                "user": 'Update label "Family" to "Family_2" and description to "Modified the details of family class".',
                "action": "UPDATE",
                "entity": "Family",
                "property": ["label", "description"],
                "entity_new_value": "Family_2",
                "descr_new_value": "Modified the details of family class"
            },
            {
                "user": "Please update the entity 'British'",
                "action": "UPDATE",
                "entity": "British",
                "property": ["label"],
                "entity_new_value": "British",
                "descr_new_value": None
            },
            {
                "user": "Please delete the node 'British'",
                "action": "DELETE",
                "entity": "British",
                "property": ["label"],
                "entity_new_value": None,
                "descr_new_value": None
            },
            {
                "user": "delete the entity 'British'",
                "action": "DELETE",
                "entity": "British",
                "property": ["label"],
                "entity_new_value": None,
                "descr_new_value": None
            }
        ]
        
        prompt_examples = "\n".join([
            f'User: "{ex["user"]}"\nAction: {ex["action"]}\nEntity: {ex["entity"]}\nProperty: {ex["property"]}\nEntity_New_Value: {ex["entity_new_value"]}\nDescr_New_Value: {ex["descr_new_value"]}'
            for ex in training_data
        ])
        final_prompt = (
            "You are a structured data extractor for knowledge graph operations. Your task is to analyze user instructions and extract the following fields:\n"
            "1. action: The type of operation (ADD, DELETE, UPDATE). For action if there is insufficient data or not able to categorise the action, put the action as 'OTHER'\n"
            "2. entity: The identifier of the node being acted upon (e.g., KG:Product:12345, person:Q778).\n"
            "3. property: A list of properties being updated (e.g., ['label'], ['description'], ['label', 'description']).\n"
            "4. entity_new_value: The new value for the entity name, label, or title (if applicable).\n"
            "5. descr_new_value: The new value for the description, summary, or shortDescription (if applicable).\n"
            "Return the result in valid JSON format with keys: action, entity, property, entity_new_value, descr_new_value.\n"
            "If a field is not applicable, set its value to null.\n"
            "If the user is greeting, or provides a single word, vague, or contextless input, classify the user's action as one word: OTHER.\n"
            f"Use the following examples:\n{prompt_examples}\n\n"
            f'Now extract the fields from this input:\n"{user_input}"'
        )
        return final_prompt

    @staticmethod
    def evaluate_prompt_quality(user_input: str) -> str:
        """
        Build a prompt for evaluating the quality of user input.
        """
        llm_prompt = (
            "You are a helpful and friendly AI assistant focused on graph database operations.\n"
            "A user has submitted an input. Your task is to:\n"
            "1. Evaluate the quality of the input.\n"
            "2. If the input is a greeting (e.g., 'hi', 'hello'), a single word, contains vague or incomplete information, garbage text, or only special characters, classify the input as NO.\n"
            "3. If the input contains meaningful context, clear intent, and sufficient detail to proceed with graph database operations, classify it as YES.\n"
            "4. Respond with a polite and encouraging message if the input is classified as NO, asking the user to provide more details.\n\n"
            "Examples:\n"
            "- Input: 'hello' → Classification: NO → Response: \"Hi there! Could you please share a bit more about what you're looking for so I can help you better?\"\n"
            "- Input: 'risk' → Classification: NO → Response: \"Could you please provide more context or details about what you'd like to do with the graph database?\"\n"
            "- Input: 'recommend' → Classification: NO → Response: \"Sure! Could you let me know what you'd like recommendations for?\"\n"
            "- Input: 'I want to analyze the risk relationships between entities in the insurance graph.' → Classification: YES\n\n"
            "Always keep the tone friendly, supportive, and non-judgmental.\n"
            f"Now using this context, classify the following input as either YES or NO:\n\"{user_input}\""
        )
        return llm_prompt
    
    @staticmethod
    def evaluate_rag_response(user_input: str) -> str:
        """
        Build a prompt for evaluating the RAG response.
        """
        prompt = f"""You are a classification assistant.

                    Your task is to analyze the **first paragraph** of a response retrieved from a knowledge base and classify it into one of the following categories:

                    - **NOSHOW**: If the **first sentence** clearly indicates that the requested information is unavailable, missing, or not covered in the knowledge base, classify the entire paragraph as NOSHOW—even if the rest of the paragraph contains related or contextual information.
                    - **SHOW**: If the **first sentence** provides relevant or specific information, or does not indicate unavailability, then classify the paragraph as SHOW.

                    ### Examples:
                    1. "The provided knowledge base does not specify exact probabilities or statistical likelihoods for entity." → NOSHOW  
                    2. "Based on the provided Knowledge Base, there is no information available regarding entity." → NOSHOW  
                    3. "The entity is a software component used in distributed systems to manage stateful interactions." → SHOW  
                    4. "The knowledge base does not contain real-time information about entity." → NOSHOW  
                    5. "Entity X is typically used in scenarios involving asynchronous messaging and event-driven architecture." → SHOW  
                    6. "The knowledge base does not contain real-time weather information. However, it discusses weather conditions in the context of insurance and risk management..." → NOSHOW

                    - Give only single word as output - either SHOW or NOSHOW.
                    ### Response to classify:
                    \"\"\"{user_input}\"\"\"
                    """
        return prompt
    

    def get_parser_prompt_for_delete(user_message: str, file_names: list ) -> str:
        
        """
        Build a prompt for extracting structured data from user input and classify whether it's update or delete action.
        """
        
        prompt = f'''
            You are a file operation assistant.
            Classify the user's intent as one of: UPDATE, DELETE, or OTHER.

            Rules:
            - If the user wants to UPDATE, file_names (attached files) will be provided and must be used as filename2 (list of files to update). Return always in list format.
            - If the user wants to DELETE, the file name will be mentioned in the user prompt only (filename1), and filename2 should be null. Return always in list format.

            Return a JSON object with:
            - action: UPDATE, DELETE, or OTHER
            - filename1: (for UPDATE: source file mentioned in prompt, for DELETE: file to delete from prompt)
            - filename2: (for UPDATE: list of attached files, for DELETE: null)
            - raw: the original user message

            Attached files: {file_names}
            User prompt: "{user_message}"
            Respond ONLY with the JSON object.
            '''
        return prompt