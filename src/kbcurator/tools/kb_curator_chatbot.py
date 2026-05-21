import os
import sys
import logging
import ast
from dotenv import load_dotenv
import json
import psycopg2
from typing import List, Optional, Dict, Any
from datetime import datetime
# Third-party and internal imports
sys.path.append("../utils")
from ..utils.prompt_builder import PromptBuilder
from ..utils.azurecustomllm import AzureCustomLLM
from ..utils.classifier import classifier
from ..utils.mcp_service_client import MCPServiceClient
from ..server.server import mcp
from ..server.main import session
from ..utils.helpers import evaluate_user_input
import difflib
from ..utils.chatbot_context import ChatbotContext
import re
from urllib.parse import urlparse
from ..utils.access_validation import validate_user_workspace_access
from ..utils.request_context import request_var
# from tools.userManagementSystem import Session, UserMap
from ..utils.db import db
from fastmcp.server.dependencies import get_http_headers

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

llm_classifier = AzureCustomLLM()

class IntentDetector:
    """
    LLM-based intent classifier using OpenAI API, with example phrases for each intent.
    """
    def __init__(self):
        
        self.intents = [
            "search_kb",
            "upload_file",
            "add_entity",
            "delete_entity",
            "index_url",
            "update_entity",
            "delete_file",
            "greeting",
            "help"
        ]
        self.examples = {
            "search_kb": ['search', 'find', 'lookup','look for','what is','tell me about','what are','information on','information about','describe'],
            "upload_file": ['upload', 'add file', 'add document', 'attach file', 'attach document','import file','import document','index file','index document'],
            "add_entity": ['add entity', 'create entity', 'new entity','define entity','add new entity'],
            "delete_entity": ['delete entity', 'remove entity', 'discard entity','erase entity','delete the entity','remove the entity'],
            "index_url":['index url','index this url','index data from url','index the url'],
            "update_entity": ['update entity', 'modify entity', 'change entity','edit entity','update the entity','modify the entity'],
            "delete_file": ['delete file', 'remove file', 'delete a file','discard file','erase file','delete the file','remove the file'],
            "greeting": ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening'],
            "help": ['help', 'assist me', 'support', 'i need help', 'can you help me','what can you do','what are your capabilities']
        }
        self.llm_classifier = AzureCustomLLM()

    def detect_intent(self, user_message: str) -> str:
        # Build a prompt with examples for each intent
        prompt = """
            You are an intent classifier for a chatbot. Classify the user message into one of these intents:
            - search_kb
            - upload_file
            - add_entity
            - delete_entity
            - update_entity
            - delete_file
            - greeting
            - help

            Respond with only the intent label, nothing else.
            Here are example phrases for each intent:
            """
        
        for intent, phrases in self.examples.items():
            prompt += f"\n{intent}: {', '.join(phrases)}"
        
        prompt += f"\n\nUser message: \"{user_message}\"\nIntent:"

        response = self.llm_classifier._call(
            input=user_message,
            sys_prompt=prompt
        )

        # print(f"Response from classifier is {response}")
        
        intent = response.strip().split()[0].lower()
        if intent not in self.intents:
            return "search_kb"
        return intent

def extract_filename(user_message):
    # Extract a filename from quoted or unquoted user text.
    # Supports common characters used in real file names (spaces, underscores, parentheses, hyphens).
    import re

    # Try explicit command-style phrasing first (unquoted)
    verb_match = re.search(
        r"(?:delete|remove|erase|confirm)\s+(?:file\s+)?(.+?\.[A-Za-z0-9]+)\s*$",
        user_message,
        re.IGNORECASE,
    )
    if verb_match:
        return verb_match.group(1).strip().strip('"\'')

    # Try quoted filename next
    match = re.search(r'"([^"]+\.[A-Za-z0-9]+)"|\'([^\']+\.[A-Za-z0-9]+)\'', user_message)
    if match:
        return (match.group(1) or match.group(2)).strip()

    # Generic unquoted filename/path pattern (use last match to avoid leading verbs)
    matches = re.findall(r'([A-Za-z0-9_\-()\s/\\]+\.[A-Za-z0-9]+)', user_message)
    if matches:
        return matches[-1].strip().strip('"\'')

    return None

def find_similar_files(filename, indexed_files):
    return difflib.get_close_matches(filename, indexed_files.keys(), n=3, cutoff=0.5)

def _is_confirm_message(user_message: str) -> bool:
    return "confirm" in (user_message or "").lower()

def _normalize_filename_for_match(value: str) -> str:
    if not value:
        return ""
    base_name = os.path.basename(value)
    base_name = base_name.replace("\\", "/").split("/")[-1]
    return re.sub(r"[^a-z0-9]", "", base_name.lower())

def resolve_indexed_filename(requested_filename: str, indexed_files: Dict[str, list]) -> Optional[str]:
    if not requested_filename or not indexed_files:
        return None

    # 1) Exact key match first
    if requested_filename in indexed_files:
        return requested_filename

    requested_norm = _normalize_filename_for_match(requested_filename)

    # 2) Exact normalized basename match
    for key in indexed_files.keys():
        if _normalize_filename_for_match(key) == requested_norm:
            return key

    # 3) Containment normalized match (handles shortened names)
    for key in indexed_files.keys():
        key_norm = _normalize_filename_for_match(key)
        if requested_norm and key_norm and (
            requested_norm in key_norm or key_norm in requested_norm
        ):
            return key

    # 4) Best fuzzy match as final fallback
    candidates = find_similar_files(requested_filename, indexed_files)
    if candidates:
        return candidates[0]

    return None

async def get_parsed_data(message: str) -> json:
    parser_prompt = PromptBuilder.get_parser_prompt(message)
    parsed_data = await classifier(message, parser_prompt)
    print(f"Parsed data from classifier: {parsed_data[:10]}")
    parsed_data = json.loads(parsed_data)
    return parsed_data

def extract_url(user_message: str) -> Optional[str]:
    """Extract URL from user message with various formats."""
    # Try to match URL in double or single quotes
    match = re.search(r'"(https?://[^"]+)"|\'(https?://[^\']+)\'', user_message)
    if match:
        return match.group(1) or match.group(2)
   
    # Try to match URL without quotes
    match = re.search(r'(https?://[^\s]+)', user_message)
    if match:
        return match.group(1)
   
    return None
 
def validate_url(url: str) -> bool:
    """Validate if the URL is properly formatted."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

class Chatbot:
    """ Interactive chatbot for knowledge base management. """
    def __init__(
            self, 
            industry: str, 
            sub_industry: str, 
            workspace_id: int, 
            user_id: int, 
            role_id: int, 
            session_id: str, 
            token: str | None,
            knowledge_bases: list = None, 
            file_names: list = None, 
            file_contents: list = None, 
            mode: str = 'Search'
            ):
        self.intent_detector = IntentDetector()
        self.session = session  # Use the module-level session manager from main
        
        load_dotenv(os.path.abspath(os.path.join(os.getcwd(), '.env')))
        server_url = os.environ.get("KC_SERVICE_URL")
        self.industry = industry
        self.sub_industry = sub_industry
        self.knowledge_bases = knowledge_bases
        self.workspace_id = workspace_id
        self.user_id = user_id
        self.role_id = role_id
        self.session_id = session_id
        self.server_url = server_url
        self.file_names = file_names
        self.file_contents = file_contents
        self.mode = mode
        self.task_id = None
        self.token = token
        self.mcp_tool_obj = MCPServiceClient(
            server_url = self.server_url,
            industry= self.industry,
            sub_industry = self.sub_industry,
            knowledge_bases = self.knowledge_bases,
            token = self.token
            )
        
    def get_or_create_context(self, session_id: str) -> ChatbotContext:
        context = self.session.load_context(session_id)
        if context:
            return context
        context = ChatbotContext(session_id=session_id)
        self.session.save_context(context)
        return context

    def save_context(self, context: ChatbotContext):
        self.session.save_context(context)

    def _parse_indexed_files_response(self, indexed_files: Any) -> Dict[str, list]:
        try:
            if not indexed_files:
                return {}
            if isinstance(indexed_files, dict):
                return indexed_files
            content = getattr(indexed_files, "content", None)
            if not content:
                return {}
            text_json = getattr(content[0], "text", "") if len(content) > 0 else ""
            parsed = json.loads(text_json) if text_json else {}
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    def _get_latest_task_ids_from_history(self) -> List[int]:
        """
        Return the most recent non-empty task_ids list from session history.
        This lets query responses carry a stable task reference even when
        the query tool itself does not emit task_ids.
        """
        try:
            history = self.session.load_history(self.workspace_id, self.user_id, self.session_id)
            for msg in reversed(history or []):
                task_ids = msg.get("task_ids") if isinstance(msg, dict) else None
                if isinstance(task_ids, list):
                    cleaned = [t for t in task_ids if t is not None]
                    if cleaned:
                        return cleaned
            return []
        except Exception:
            return []

    async def process_message(self, message: str):
        try:
            print(f"Inside Process message: {message}")
            context = self.get_or_create_context(self.session_id)
            insert_id = self.session.append_message(self.workspace_id, self.user_id, self.session_id, "user", message, [])
            context.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "user": self.user_id,
                "intent": None,
                "assistant": None})

            if context.pending_confirmation:
                response = await self.handle_confirmation(message, context)
                context.conversation_history[-1]["assistant"] = response
                insert_id = self.session.append_message(self.workspace_id, self.user_id, self.session_id, "assistant", response, [])
                self.save_context(context)
                return response

            if self.mode.upper() in ['SEARCH','QUERY']:
                if self.file_names:
                    intent = 'upload_file'
                else:
                    intent = self.intent_detector.detect_intent(message)
                    if intent in ['help']:
                        intent = 'help'
                    elif intent in ['greeting']:
                        intent = 'greeting'
                    else:
                        intent = 'search_kb'
            elif self.mode.upper() == 'UPDATE':
                intent = self.intent_detector.detect_intent(message)
            else:
                intent = 'search_kb'

            context.last_intent = intent
            context.conversation_history[-1]["intent"] = intent

        #     print(f"Detected intent: {intent} for message: {message[:50]}")

        #     intent_response = await self.route_intent(intent, message, context)
        #     print("Query RAG response: ", intent_response[:50])
        #     if type(intent_response) == dict:
        #         response = intent_response["response"]
        #         task_ids = intent_response["task_ids"]
        #         print(f"Response: {response[:50]}, tasks: {task_ids}")
        #     else:
        #         response = intent_response
        #         task_ids = []
        #         print(f"Response: {response[:50]}, no tasks for this tool")

        #     context.conversation_history[-1]["assistant"] = response
        #     insert_id = self.session.append_message(self.workspace_id, self.user_id, self.session_id, "assistant", response, task_ids)
        #     self.save_context(context)

        #     # print(f"Updated context history length: {context.conversation_history}")
        #     return response
        # except Exception as e:
        #     print(f"Error processing message: {e}")
        #     return "Sorry, something went wrong while processing your request. Please try again"
            print(f"Detected intent: {intent} for message: {message[:50]}")

            intent_response = await self.route_intent(intent, message, context)
            print("Query RAG response: ", str(intent_response)[:50])
            
            # Handle different response types
            if type(intent_response) == dict:
                response = intent_response.get("response", "")
                task_ids = intent_response.get("task_ids", [])
                sources = intent_response.get("sources", [])

                if not task_ids:
                    task_ids = self._get_latest_task_ids_from_history()
                print(f"Response: {response[:50]}, tasks: {task_ids}, sources: {len(sources)}")
            else:
                response = intent_response
                task_ids = self._get_latest_task_ids_from_history()
                sources = []
                print(f"Response: {str(response)[:50]}, no tasks for this tool")

            context.conversation_history[-1]["assistant"] = response
            
            # Save message with sources if available
            if sources:
                insert_id = self.session.append_message(
                    self.workspace_id, 
                    self.user_id, 
                    self.session_id, 
                    "assistant", 
                    response, 
                    sources  # Pass sources instead of task_ids for search responses
                )
            else:
                insert_id = self.session.append_message(
                    self.workspace_id, 
                    self.user_id, 
                    self.session_id, 
                    "assistant", 
                    response, 
                    task_ids
                )
            
            self.save_context(context)

            # Return structured response with sources if available
            if sources:
                return {
                    "response": response,
                    "sources": sources,
                    "task_ids": task_ids
                }
            
            if task_ids:
                return {
                    "response": response,
                    "task_ids": task_ids
                }
            
            # print(f"Updated context history length: {context.conversation_history}")
            return response
        except Exception as e:
            print(f"Error processing message: {e}")
            return "Sorry, something went wrong while processing your request. Please try again"
        
    async def route_intent(self, intent: str, message: str, context: ChatbotContext):
        """Route to the appropriate handler based on detected intent."""
        if intent == "search_kb":
            return await self.handle_search(message, context)
        elif intent == "upload_file":
            return await self.handle_upload(message, context, intent)
        elif intent == "add_entity":
            return await self.handle_add_entity(message, context, intent)
        elif intent == "delete_entity":
            return await self.handle_delete_entity(message, context, intent)
        elif intent == "index_url":
            return await self.handle_index_url(message, context, intent)
        elif intent == "update_entity":
            return await self.handle_add_entity(message, context, intent)
        elif intent == "delete_file":
            return await self.handle_delete_file(message, context)
        elif intent == "greeting":
            return "Hello! How can I assist you with your knowledge base today?"
        elif intent == "help":
            return ("I can help you manage your knowledge base. You can ask me to search for information, "
                    "upload files, add or delete entities, and more. What would you like to do?")
        else:
            return "I'm not sure how to help with that. Could you please rephrase?"
    
    # async def handle_search(self, message: str, context: ChatbotContext) -> str:
    #     # Extract search query
    #     try:
    #         print(f"Inside Search kb {message}")
    #         history = self.session.load_history(self.workspace_id, self.user_id, self.session_id)
    #         history = history[-5:]
    #         # print(f"History: {history}, type: {type(history)}")
    #         assistant_message = await self.mcp_tool_obj.query_rag('Search',message, history, self.workspace_id, self.role_id)
    #         print(assistant_message[:50])
    #         return assistant_message
    #     except Exception as e:
    #         return (f"Error occurred while handling search: {e}")

    async def handle_search(self, message: str, context: ChatbotContext) -> dict:
        # Extract search query
        try:
            print(f"Inside Search kb {message}")
            history = self.session.load_history(self.workspace_id, self.user_id, self.session_id)
            history = history[-5:]
            # print(f"History: {history}, type: {type(history)}")
            assistant_message = await self.mcp_tool_obj.query_rag('Search',message, history, self.workspace_id, self.role_id)
            print(f"Query RAG response type: {type(assistant_message)}")
            
            # Check if response is structured (dict with sources) or plain text
            if isinstance(assistant_message, dict) and "sources" in assistant_message:
                print(f"Structured response with {len(assistant_message.get('sources', []))} sources")
                return {
                    "response": assistant_message.get("response", ""),
                    "sources": assistant_message.get("sources", []),
                    "task_ids": assistant_message.get("task_ids", [])
                }
            elif isinstance(assistant_message, dict):
                return {
                    "response": assistant_message.get("response", ""),
                    "task_ids": assistant_message.get("task_ids", [])
                }
            else:
                # Backward compatibility: plain text response
                print(f"Plain text response: {str(assistant_message)[:50]}")
                return str(assistant_message)
        except Exception as e:
            return (f"Error occurred while handling search: {e}")

    async def handle_delete_entity(self, message: str, context: ChatbotContext, intent: str) -> str:
        try:
            parsed_data = await get_parsed_data(message)
            print(f"Parsed data for deletion: {parsed_data}")

            delet_args = { 
                    "domain": self.industry,
                    "kb_name": self.sub_industry,
                    "entity_name": parsed_data.get("entity")
                    }
            
            print(f"List of Arguments has sent for deletion: {delet_args}")
            delete_response = await self.mcp_tool_obj.delete_node(intent ,delet_args)
            assistant_message = f"{delete_response.structuredContent['message']}"
            print(f"Assistant message after deletion: {assistant_message}")
            return f"Entity {parsed_data.get('entity')} has been deleted successfully."
        except Exception as e:
            return (f"Error occurred while handling delete entity: {e}")
    
    async def handle_add_entity(self, message: str, context: ChatbotContext, intent: str) -> str:
        # Add entity logic here
        try:
            parsed_data = await get_parsed_data(message)
            print(f"Parsed data for addition: {parsed_data}")

            add_args = { 
                "domain": self.industry,
                "kb_name": self.sub_industry,
                "user_query": message
                }
            
            print(f"List of Arguments has sent for adding: {add_args}")
            add_response = await self.mcp_tool_obj.add_node(intent,add_args)
            assistant_message = f"The data has been successfully processed and updated."
            # print(f"Assistant message after adding new node:", add_response)
            return assistant_message
        except Exception as e:
            return (f"Error occurred while handling add entity: {e}")
    
    async def handle_upload(self, message: str, context: ChatbotContext, intent: str):
        try:
            # Upload file logic here
            assistant_message = "Files has been uploaded successfully and added to the knowledge base."
            print(f"In handle_upload with message: {assistant_message}")
            result = await self.mcp_tool_obj.upload_rag(intent, self.file_names, self.workspace_id, self.user_id, self.role_id, self.file_contents)
            # Extract 'response' if result is a dict and has 'response' key
            if isinstance(result, dict) and "response" in result:
                assistant_message = result["response"]
                assistant_tasks = result["task_ids"]
            else:
                assistant_message = str(result)
                assistant_tasks = []
            return {
                    "response": assistant_message,
                    "task_ids": assistant_tasks
                }
        except Exception as e:
            return f"Error occurred while handling upload: {e}"
    
    async def handle_delete_file(self, message: str, context: ChatbotContext) -> str:
        # Extract file name from message
        try:
            print(f"Handling delete file for message: {message}")
           # indexed_files = await self.mcp_tool_obj.get_indexed_files()
            indexed_files = await self.mcp_tool_obj.get_indexed_files(self.workspace_id, self.role_id)
            print(f"Indexed files response: {indexed_files}")
            result_dict = self._parse_indexed_files_response(indexed_files)
            print(f"Indexed files: {result_dict}")

            filename = extract_filename(message)
            print(f"Extracted filename: {filename}")

            if not filename:
                return (
                    "I couldn't detect a file name in your request. "
                    "Please provide the exact file name including extension, "
                    "for example: Delete \"example.docx\"."
                )

            # If index metadata is temporarily empty/unavailable, avoid misleading "file not found"
            # and allow a confirm path for direct blob deletion.
            if not result_dict:
                context.pending_confirmation = {
                    "action": "delete_file",
                    "requested_filename": filename,
                    "options": [],
                    "indexed_match_key": None,
                    "indexed_candidate": None,
                }
                return (
                    f"Indexed file metadata is currently unavailable for '{filename}' in this workspace. "
                    "This can happen if indexing is still syncing. "
                    f"Reply with 'confirm {filename}' to proceed with blob deletion now, "
                    "or retry delete after a short wait to remove indexed records as well."
                )
            
            similar_files = find_similar_files(filename, result_dict)
            indexed_match_key = resolve_indexed_filename(filename, result_dict)
            context.pending_confirmation = {
                "action": "delete_file",
                "requested_filename": filename,
                "options": similar_files,
                "indexed_match_key": indexed_match_key,
                "indexed_candidate": similar_files[0] if similar_files else None,
            }

            if indexed_match_key:
                return (
                    f"Found indexed mapping for '{filename}' as '{indexed_match_key}'. "
                    f"Reply with 'confirm {filename}' to delete it."
                )

            if similar_files:
                return (
                    f"Exact file '{filename}' not found in index. "
                    f"Similar files: {', '.join(similar_files)}. "
                    f"Reply with 'confirm {filename}' to delete exact blob path, "
                    "or confirm one of the listed file names to delete indexed records."
                )

            return (
                f"Exact file '{filename}' not found in index and no similar files were found. "
                f"Reply with 'confirm {filename}' to attempt exact blob deletion."
            )
        except Exception as e:
            return (f"Error occurred while handling delete file: {e}")
    
    async def handle_index_url(self, message: str, context: ChatbotContext, intent: str) -> str:
        """Handle indexing content from a URL."""
        try:
            # Extract URL from message
            url = extract_url(message)
            if not url:
                return "I couldn't find a valid URL in your message. Please provide a URL like: index this url: \"https://example.com/demo/"
            if not validate_url(url):
                return f"The URL '{url}' doesn't appear to be valid. Please provide a complete URL starting with http:// or https://"
           
            print(f"Indexing URL: {url}")
            # Call MCP tool to index the URL
            assistant_message = await self.mcp_tool_obj.index_url(intent, url, self.industry, self.sub_industry)
            return assistant_message
           
        except Exception as e:
            return f"Error occurred while indexing URL: {str(e)}"
        

    async def delete_file_task_from_db(self, file_path: str):
        conn = psycopg2.connect(
            host=os.environ["POSTGRES_HOST"],
            user=os.environ["POSTGRES_USER"],
            password=os.environ["POSTGRES_PASSWORD"],
            dbname=os.environ.get("POSTGRES_DATABASE") or os.environ.get("POSTGRESQL_DATABASE_DATABASE_2"),
        )
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM public.file_tasks WHERE file_path = %s", (file_path,))
        finally:
            conn.close()
    

    async def handle_confirmation(self, user_message, context):
        try:
            print(f"Handling confirmation for action: {context.pending_confirmation}")
            
            if context.pending_confirmation and context.pending_confirmation["action"] == "delete_file":
                pending = context.pending_confirmation

                if not _is_confirm_message(user_message):
                    context.pending_confirmation = None
                    return "File deletion cancelled. No action taken."

                # Extract file name from confirmation, fallback to originally requested filename
                filename = extract_filename(user_message) or pending.get("requested_filename")
                print(f"Provided filename for deletion: {filename}")

                if not filename:
                    context.pending_confirmation = None
                    return "File deletion cancelled. No valid file name was provided."

                print(f"Proceed with file deletion: {filename}")
                indexed_files = await self.mcp_tool_obj.get_indexed_files(self.workspace_id, self.role_id)
                result_dict = self._parse_indexed_files_response(indexed_files)

                indexed_lookup_name = pending.get("indexed_match_key") or filename
                list_doc_ids_to_delete = result_dict.get(indexed_lookup_name) or []

                # Fallback: if exact filename key is not present, use best indexed candidate captured earlier.
                if not list_doc_ids_to_delete:
                    candidate_name = pending.get("indexed_candidate")
                    if candidate_name and candidate_name in result_dict:
                        list_doc_ids_to_delete = result_dict.get(candidate_name) or []
                        indexed_lookup_name = candidate_name

                print(f"Document IDs to delete: {list_doc_ids_to_delete}")

                deleted_doc_count = 0
                failed_doc_count = 0
                deletion_in_progress = False
                deletion_error_text = ""
                if list_doc_ids_to_delete:
                    bulk_result = await self.mcp_tool_obj.delete_files_by_doc_ids(
                        list_doc_ids_to_delete,
                        self.workspace_id,
                        self.role_id,
                    )
                    print(f"Bulk delete result: {bulk_result}")

                    summary = {}
                    try:
                        if isinstance(bulk_result, dict):
                            # Client-side timeout/transport errors can happen while server-side
                            # deletion continues; treat this as in-progress, not hard failure.
                            if bulk_result.get("status") == "client_exception":
                                deletion_in_progress = True
                                deletion_error_text = bulk_result.get("error") or ""
                        else:
                            structured = getattr(bulk_result, "structured_content", None)
                            if isinstance(structured, dict):
                                summary = structured.get("summary", {}) or {}
                    except Exception:
                        summary = {}

                    deleted_doc_count = int(summary.get("success", 0) or 0)
                    # Treat both explicit failures and not_found as not deleted for response purposes.
                    failed_doc_count = int(summary.get("failed", 0) or 0) + int(summary.get("not_found", 0) or 0)

                # Always attempt blob deletion for the exact confirmed filename.
                delet_from_blob = await self.mcp_tool_obj.delete_files_from_blob([filename], self.workspace_id, self.role_id)
                print(f"Deleted file from blob storage: {delet_from_blob}")

                blob_deleted_files = []
                blob_result_text = ""
                blob_error = False

                try:
                    blob_error = bool(getattr(delet_from_blob, "is_error", False))
                    content = getattr(delet_from_blob, "content", None)
                    if content:
                        first_item = content[0]
                        blob_result_text = getattr(first_item, "text", "") or ""

                    # Expected shape from tool text: "Deleted files: ['path1', ...]"
                    if blob_result_text and "Deleted files:" in blob_result_text:
                        raw_list = blob_result_text.split("Deleted files:", 1)[1].strip()
                        parsed_list = ast.literal_eval(raw_list)
                        if isinstance(parsed_list, list):
                            blob_deleted_files = parsed_list
                except Exception:
                    # Keep response resilient even if tool payload format changes.
                    pass

                context.pending_confirmation = None

                if deleted_doc_count > 0:
                    if blob_error:
                        return (
                            f"Removed {deleted_doc_count} indexed document record(s) for '{filename}'. "
                            "Blob deletion reported an error."
                        )

                    if blob_deleted_files:
                        for blob_path in blob_deleted_files:
                            await self.delete_file_task_from_db(blob_path)
                        return (
                            f"Removed {deleted_doc_count} indexed document record(s) for '{indexed_lookup_name}'. "
                            f"Deleted blob file(s): {blob_deleted_files}."
                        )

                    if blob_result_text:
                        return (
                            f"Removed {deleted_doc_count} indexed document record(s) for '{indexed_lookup_name}'. "
                            f"Blob deletion result: {blob_result_text}"
                        )

                    return (
                        f"File '{filename}' deleted. "
                        f"Removed {deleted_doc_count} indexed document record(s) and requested blob deletion."
                    )

                if deletion_in_progress:
                    if blob_deleted_files:
                        return (
                            f"Deletion started for {len(list_doc_ids_to_delete)} indexed document record(s) for '{indexed_lookup_name}'. "
                            f"Deleted blob file(s): {blob_deleted_files}. "
                            "Index cleanup is still running in background; please recheck indexed files shortly."
                        )

                    if blob_result_text:
                        return (
                            f"Deletion started for {len(list_doc_ids_to_delete)} indexed document record(s) for '{indexed_lookup_name}'. "
                            f"Blob deletion result: {blob_result_text}. "
                            "Index cleanup is still running in background; please recheck indexed files shortly."
                        )

                    if deletion_error_text:
                        return (
                            f"Deletion started for {len(list_doc_ids_to_delete)} indexed document record(s) for '{indexed_lookup_name}', "
                            f"but the client connection ended early ({deletion_error_text}). "
                            "The backend may still be completing index cleanup."
                        )

                    return (
                        f"Deletion started for {len(list_doc_ids_to_delete)} indexed document record(s) for '{indexed_lookup_name}'. "
                        "Index cleanup is still running in background; please recheck indexed files shortly."
                    )

                if failed_doc_count > 0:
                    return (
                        f"Unable to delete {failed_doc_count} indexed record(s) for '{indexed_lookup_name}' "
                        "in the current workspace context. Blob deletion was still attempted."
                    )

                if blob_error:
                    return (
                        f"No indexed document records were found for '{filename}'. "
                        "Blob deletion reported an error."
                    )

                if blob_deleted_files:
                    for blob_path in blob_deleted_files:
                        await self.delete_file_task_from_db(blob_path)
                    return (
                        f"No indexed document records were found for '{filename}'. "
                        f"Deleted blob file(s): {blob_deleted_files}."
                    )

                if blob_result_text:
                    return (
                        f"No indexed document records were found for '{filename}'. "
                        f"Blob deletion result: {blob_result_text}"
                    )

                return (
                    f"Requested deletion for '{filename}' from blob storage. "
                    "No indexed document records were found for this exact file name."
                )
        except Exception as e:
            return (f"Error occurred while handling confirmation: {e}")


@mcp.tool()
def start_conversation() -> Dict[str, Any]:
    """Start a new conversation session."""
    session_id = session.create_session()
    logger.info(f"Session started with id: {session_id}")
    return {"response": f"Session started with id: {session_id}", "session_id": session_id}


@mcp.tool()
async def message_gpt(
    workspace_id: str,
    user_id: str,
    role_id: str,
    user_message: str,
    session_id: str,
    industry: str,
    sub_industry: str,
    mode: Optional[str],
    knowledge_bases: Optional[list[str]] = None,
    file_names: Optional[List[str]] = None,
    file_contents: Optional[List[str]] = None
) -> dict:
    # --- JWT-based authentication and workspace-user mapping check (copied from ingestion_new.py tools) ---

    # Validate workspace_id presence
    if user_id is None:
        return {"status": "error", "error": "user_id cannot be null"}
    if not workspace_id:
        return {"error": "workspace_id is required for authentication."}

    # Validate user access to workspace
    valid, err = validate_user_workspace_access(
        user_id=user_id,
        workspace_id=workspace_id
        )
    if not valid:
        return {"error": err}


    # Enforce JWT-based access: only allow if user is mapped to the workspace and user_id matches JWT
    # request = request_var.get()
    # if not request or not hasattr(request.state, "jwt_claims"):
    #     return {"error": "Unauthorized: JWT claims not found in request context"}
    # claims = request.state.jwt_claims
    # jwt_user_id = claims.get("user_id") or claims.get("sub")
    # if not jwt_user_id:
    #     return {"error": "Unauthorized: user_id not found in token claims"}
    # if str(user_id) != str(jwt_user_id):
    #     return {"error": "Unauthorized: user_id in request does not match user in token"}

    # Check if user is mapped to this workspace
    session = db.Session()
    try:
        user_map = session.query(db.UserMap).filter_by(workspace_id=workspace_id, user_id=user_id, is_active=True).first()
        if not user_map:
            return {"error": "You are not authorized to access this workspace."}
    except Exception as e:
        return {"error": str(e)}
    finally:
        session.close()

    try:
        token = get_http_headers(include_all=True).get('authorization',"") or get_http_headers().get('Authorization',"")
        if token:
            print("'message_gpt' authorized call", token[:10]) 
        else: 
            print("'message_gpt' unauthorized call", token)
        bot = Chatbot(
            industry=industry, 
            sub_industry=sub_industry, 
            knowledge_bases=knowledge_bases, 
            workspace_id=workspace_id, 
            user_id=user_id,
            role_id=role_id,
            session_id=session_id, 
            file_names=file_names, 
            file_contents=file_contents, 
            mode=mode,
            token=token
            )
        response = await bot.process_message(user_message)
        #return {"response": response}
        # Check if response is structured with sources
        if isinstance(response, dict) and ("sources" in response or "task_ids" in response):
            return {
                "response": response.get("response", ""),
                "sources": response.get("sources", []),
             #   "sources": response.get("task_ids", [])
                "task_ids":response.get("task_ids",[])
            }
        else:
            return {"response": response}
    except Exception as e:
        print(f"Error in message_gpt: {e}")
        return {"error":f"Sorry, something went wrong while processing your request. Please try again.{e}"}

@mcp.tool()
def get_conversation_history(workspace_id: str = None, user_id: str = None, limit: Optional[int] = None) -> Dict[str, Any]:
    """Get recent conversation history for a user."""
    if user_id is None:
        return {"status": "error", "error": "user_id cannot be null"}
    # --- JWT-based authentication and workspace-user mapping check (copied from message_gpt and ingestion_new.py tools) ---
    if not workspace_id:
        return {"error": "workspace_id is required for authentication."}

    # Validate user access to workspace
    valid, err = validate_user_workspace_access(workspace_id=workspace_id)
    if not valid:
        return {"error": err}


    # Enforce JWT-based access: only allow if user is mapped to the workspace and user_id matches JWT
    request = request_var.get(None)
    if not request or not hasattr(request.state, "jwt_claims"):
        return {"error": "Unauthorized: JWT claims not found in request context"}
    claims = request.state.jwt_claims
    jwt_user_id = claims.get("user_id") or claims.get("sub")
    if not jwt_user_id:
        return {"error": "Unauthorized: user_id not found in token claims"}
    if user_id is not None and str(user_id) != str(jwt_user_id):
        return {"error": "Unauthorized: user_id in request does not match user in token"}

    # Check if user is mapped to this workspace
    session_db = db.Session()
    try:
        user_map = session_db.query(db.UserMap).filter_by(workspace_id=workspace_id, user_id=jwt_user_id, is_active=True).first()
        if not user_map:
            session_db.close()
            return {"error": "You are not authorized to access this workspace."}
    except Exception as e:
        session_db.close()
        return {"error": str(e)}
    finally:
        pass

    try:
        if limit == 5:
            con_hist = session.get_recent_sessions(workspace_id, user_id, limit=limit)
            last_messages = []
            for ses in con_hist:
                logger.info(f"Session ID: {ses}")
                history = session.load_history(workspace_id, user_id, ses)
                # Fetch the conversation title from context collection
                title = session.get_conversation_title(workspace_id, user_id, ses)
                if history and len(history) >= 2:
                    last_msg = history[-2]
                    last_messages.append({
                        "role": last_msg.get("role"),
                        "content": last_msg.get("content"),
                        "task_ids": last_msg.get("task_ids") if last_msg.get("task_ids") else None,
                        "session_id": ses,
                        "title": title,
                        "time_modified": last_msg.get("timestamp")
                    })
            return {"response": last_messages}
        else:
            # No threshold: return last user query AND response for each file
            res = session.get_recent_sessions(workspace_id, user_id, limit=0)
            conversations = []
            for ses in res:
                data = session.load_history(workspace_id, user_id, ses)
                # Fetch the conversation title from context collection
                title = session.get_conversation_title(workspace_id, user_id, ses)
                logger.info(f"Data for session {ses}: {data}")
                if isinstance(data, list) and len(data) >= 2: 
                    user_msg = next((msg for msg in reversed(data) if msg.get("role") == "user"), None)
                    assistant_msg = next((msg for msg in reversed(data) if msg.get("role") == "assistant"), None)
                    last_msg = data[-1]
                    time_modified_str = last_msg.get("timestamp", "N/A")
                    conversations.append({
                        "session_id": ses,
                        "time_modified": time_modified_str,
                        "title": title,
                        "user": user_msg["content"] if user_msg else None,
                        "assistant": assistant_msg["content"] if assistant_msg else None,
                        "task_ids": assistant_msg["task_ids"] if assistant_msg else None
                    })
                else:
                    conversations.append({
                        "session_id": ses,
                        "time_modified": "N/A",
                        "title": title,
                        "user": None,
                        "assistant": None,
                        "task_ids": assistant_msg.get("task_ids") if assistant_msg else None
                    })
            return {"response": conversations}
    except Exception as e:
        return {"error":f"Error occurred while retrieving conversation history: {e}"}

@mcp.tool()
def load_conversation(workspace_id: str, user_id: str, session_id: str) -> Dict[str, Any]:
    """Load the full conversation for a given user and session."""
    if user_id is None:
        return {"status": "error", "error": "user_id cannot be null"}
    # --- JWT-based authentication and workspace-user mapping check (copied from message_gpt and get_conversation_history) ---
    if not workspace_id:
        return {"error": "workspace_id is required for authentication."}

    # Validate user access to workspace
    valid, err = validate_user_workspace_access(workspace_id=workspace_id)
    if not valid:
        return {"error": err}


    # Enforce JWT-based access: only allow if user is mapped to the workspace and user_id matches JWT
    request = request_var.get(None)
    if not request or not hasattr(request.state, "jwt_claims"):
        return {"error": "Unauthorized: JWT claims not found in request context"}
    claims = request.state.jwt_claims
    jwt_user_id = claims.get("user_id") or claims.get("sub")
    if not jwt_user_id:
        return {"error": "Unauthorized: user_id not found in token claims"}
    if str(user_id) != str(jwt_user_id):
        return {"error": "Unauthorized: user_id in request does not match user in token"}

    # Check if user is mapped to this workspace
    session_db = db.Session()
    try:
        user_map = session_db.query(db.UserMap).filter_by(workspace_id=workspace_id, user_id=jwt_user_id, is_active=True).first()
        if not user_map:
            session_db.close()
            return {"error": "You are not authorized to access this workspace."}
    except Exception as e:
        session_db.close()
        return {"error": str(e)}
    finally:
        pass

    try:
        response = session.load_history(workspace_id, user_id, session_id)
        return {"response": response}
    except Exception as e:
        return {"error": f"Error occurred while loading conversation: {e}"}

@mcp.tool()
def delete_conversation(workspace_id: str, user_id: str, session_id: str) -> Dict[str, Any]:
    """Load the full conversation for a given user and session."""
    if user_id is None:
        return {"status": "error", "error": "user_id cannot be null"}
    # --- JWT-based authentication and workspace-user mapping check (copied from message_gpt and load_conversation) ---
    if not workspace_id:
        return {"error": "workspace_id is required for authentication."}

    # Validate user access to workspace
    valid, err = validate_user_workspace_access(workspace_id=workspace_id)
    if not valid:
        return {"error": err}


    # Enforce JWT-based access: only allow if user is mapped to the workspace and user_id matches JWT
    request = request_var.get(None)
    if not request or not hasattr(request.state, "jwt_claims"):
        return {"error": "Unauthorized: JWT claims not found in request context"}
    claims = request.state.jwt_claims
    jwt_user_id = claims.get("user_id") or claims.get("sub")
    if not jwt_user_id:
        return {"error": "Unauthorized: user_id not found in token claims"}
    if str(user_id) != str(jwt_user_id):
        return {"error": "Unauthorized: user_id in request does not match user in token"}

    # Check if user is mapped to this workspace
    session_db = db.Session()
    try:
        user_map = session_db.query(db.UserMap).filter_by(workspace_id=workspace_id, user_id=jwt_user_id, is_active=True).first()
        if not user_map:
            session_db.close()
            return {"error": "You are not authorized to access this workspace."}
    except Exception as e:
        session_db.close()
        return {"error": str(e)}
    finally:
        pass

    try:
        response = session.delete_session(workspace_id, user_id, session_id)
        return {"response": response}
    except Exception as e:
        return {"error": f"Error occurred while deleting conversation: {e}"}