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
from kbcurator.utils.prompt_builder import PromptBuilder
from kbcurator.utils.llm_helper import get_llm_response_with_context_async
from kbcurator.tools.llm_router_tool import _build_manager_from_db
from kbcurator.utils.classifier import classifier
from kbcurator.utils.mcp_service_client import MCPServiceClient
from kbcurator.server.server import mcp
from kbcurator.server.main import session
from kbcurator.utils.helpers import evaluate_user_input, workspace_id_to_alpha
import difflib
from kbcurator.utils.chatbot_context import ChatbotContext
import re
from urllib.parse import urlparse
from kbcurator.utils.access_validation import (
    validate_user_workspace_access,
    validate_chatbot_request_scope,
)
from kbcurator.utils.request_context import request_var
# from tools.userManagementSystem import Session, UserMap
from kbcurator.utils.db import db
from fastmcp.server.dependencies import get_http_headers

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

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

    async def detect_intent(self, user_message: str, workspace_id: int, agent_id: Optional[int] = None) -> str:
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

        print(f"MY agent Id: {agent_id}")
        response = await get_llm_response_with_context_async(
            workspace_id=workspace_id,
            user_input=user_message,
            sys_prompt=prompt,
            agent_id=agent_id
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

async def get_parsed_data(message: str, workspace_id: int, agent_id: Optional[int] = None) -> json:
    parser_prompt = PromptBuilder.get_parser_prompt(message)
    parsed_data = await classifier(message, parser_prompt, workspace_id=workspace_id, agent_id=agent_id)
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
            agent_id: str | int,
            session_id: str, 
            token: str | None,
            can_curate_kb: bool,
            knowledge_bases: list = None, 
            file_names: list = None, 
            file_contents: list = None, 
            mode: str = 'Search'
            ):
        self.intent_detector = IntentDetector()
        self.session = session  # Use the module-level session manager from main
        
        # Load .env file if it exists (for local development)
        env_path = os.path.abspath(os.path.join(os.getcwd(), '.env'))
        if os.path.exists(env_path):
            load_dotenv(env_path)
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
        self.agent_id = agent_id
        self.task_id = None
        self.token = token
        self.can_curate_kb = bool(can_curate_kb)
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
            if isinstance(context, ChatbotContext):
                return context
            if isinstance(context, dict):
                try:
                    return ChatbotContext.from_dict(context)
                except Exception:
                    return ChatbotContext(
                        session_id=session_id,
                        conversation_history=context.get("conversation_history", []) if isinstance(context, dict) else [],
                        pending_confirmation=context.get("pending_confirmation") if isinstance(context, dict) else None,
                        last_intent=context.get("last_intent") if isinstance(context, dict) else None,
                    )
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
                    intent = await self.intent_detector.detect_intent(message, workspace_id=self.workspace_id, agent_id=self.agent_id)
                    if intent in ['help']:
                        intent = 'help'
                    elif intent in ['greeting']:
                        intent = 'greeting'
                    else:
                        intent = 'search_kb'
            elif self.mode.upper() == 'UPDATE':
                intent = await self.intent_detector.detect_intent(message, workspace_id=self.workspace_id, agent_id=self.agent_id)
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

            intent_response = await self.route_intent(intent, message, context, self.agent_id)
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
        
    async def route_intent(self, intent: str, message: str, context: ChatbotContext, agent_id: str | int):
        """Route to the appropriate handler based on detected intent."""
        restricted_intents = {
            "upload_file",
            "add_entity",
            "delete_entity",
            "index_url",
            "update_entity",
            "delete_file",
        }

        # Non-curation users can only use search/help/greeting/default assistant replies.
        if not self.can_curate_kb and intent in restricted_intents:
            return (
                "You have search-only access in this workspace. "
                "Indexing or editing the knowledge base is not allowed for your account."
            )

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
            try:
                _provider = _build_manager_from_db(self.workspace_id, self.agent_id).get_current_provider()
                print(f"LLM provider for search (workspace_id={self.workspace_id}): {_provider}")
            except Exception as _e:
                print(f"Could not resolve LLM provider for search: {_e}")
            history = self.session.load_history(self.workspace_id, self.user_id, self.session_id)
            history = history[-5:]
            # print(f"History: {history}, type: {type(history)}")
            assistant_message = await self.mcp_tool_obj.query_rag('Search',message, history, self.workspace_id, self.role_id, agent_id=self.agent_id)
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
            parsed_data = await get_parsed_data(message, workspace_id=self.workspace_id, agent_id=self.agent_id)
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
            parsed_data = await get_parsed_data(message, workspace_id=self.workspace_id, agent_id=self.agent_id)
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
    
    async def _find_orphaned_neo4j_docs(self, workspace_id, role_id, doc_ids=None):
        """
        Query Neo4j directly to find orphaned documents that exist in the graph
        but have no entries in doc_status or vector storage.
        
        Args:
            workspace_id: Workspace identifier (e.g., "Nucor", "843", None for base)
            role_id: User role ID
            doc_ids: Optional list of doc_ids to search for (from other workspaces)
                    If None, returns ALL documents in Neo4j for this workspace
            
        Returns:
            List of doc_ids found in Neo4j for this workspace
        """
        print(f"DEBUG: _find_orphaned_neo4j_docs called for workspace_id={workspace_id}, doc_ids={'None' if doc_ids is None else f'{len(doc_ids)} items'}")
        try:
            # Construct workspace name same way as LightRAG does
            from kbcurator.utils.mcp_service_client import MCPServiceClient
            if  workspace_id and MCPServiceClient()._is_kg_workspace(workspace_id) :
                kb_name = self.sub_industry
            else:
                workspace_id_alpha = workspace_id_to_alpha(workspace_id)
                kb_name = f"{self.sub_industry}/{workspace_id_alpha}"
            
            workspace_name = ''.join(char for char in f"{self.industry}{kb_name}" if char.isalpha())
            
            print(f"DEBUG: Constructed workspace_name={workspace_name} from workspace_id={workspace_id}")

            
            # Access Neo4j using the Neo4jDriver class
            from kbcurator.services.neo4j_driver import Neo4jDriver
            import os
            import psycopg2
            
            print(f"DEBUG: Connecting to Neo4j using Neo4jDriver")
            
            found_doc_ids = []
            
            driver = Neo4jDriver()
            try:
                await driver.connect()
                print(f"DEBUG: Neo4j driver connected, executing queries...")
                
                # If doc_ids provided, search for specific matches
                if doc_ids:
                    print(f"DEBUG: Searching for specific doc_ids in workspace {workspace_name}")
                    query = f"""
                    MATCH (n:`{workspace_name}`)
                    WHERE n.source_id IN $doc_ids
                    RETURN DISTINCT n.source_id as source_id
                    """
                    records = await driver.execute_read_query(query, {"doc_ids": doc_ids})
                else:
                    # No doc_ids: get ALL documents in Neo4j for orphan cleanup
                    print(f"DEBUG: Searching for ALL documents in workspace {workspace_name}")
                    query = f"""
                    MATCH (n:`{workspace_name}`)
                    WHERE n.source_id IS NOT NULL AND n.source_id <> ''
                    RETURN DISTINCT n.source_id as source_id
                    LIMIT 1000
                    """
                    print(f"DEBUG: Executing Neo4j query: {query[:200]}...")
                    records = await driver.execute_read_query(query)
                
                print(f"DEBUG: Neo4j query executed, processing data...")
                
                print(f"DEBUG: Neo4j query returned {len(records)} records for workspace {workspace_name}")
                
                for record in records:
                    source_id = record.get('source_id')
                    if source_id and source_id.startswith('doc-'):
                        found_doc_ids.append(source_id)
                
                print(f"DEBUG: Found {len(found_doc_ids)} doc_ids with 'doc-' prefix in Neo4j workspace {workspace_name}")
                
                # CRITICAL: If no source_ids found but we're in orphan cleanup mode (doc_ids=None),
                # check if there are ANY nodes at all in this workspace
                if not found_doc_ids and doc_ids is None:
                    print(f"DEBUG: No source_ids found, checking for ANY nodes in workspace {workspace_name}...")
                    count_query = f"""
                    MATCH (n:`{workspace_name}`)
                    RETURN count(n) as node_count
                    """
                    count_records = await driver.execute_read_query(count_query)
                    node_count = count_records[0].get('node_count', 0) if count_records else 0
                    print(f"DEBUG: Found {node_count} total nodes without source_id in workspace {workspace_name}")
                    
                    if node_count > 0:
                        # These are orphaned nodes without source_id - mark for complete workspace cleanup
                        print(f"WARNING: Found {node_count} orphaned nodes in Neo4j workspace {workspace_name} without source_id property!")
                        print(f"WARNING: These nodes cannot be tracked to specific documents. Recommend manual cleanup.")
                        # Return a special marker to indicate workspace-level cleanup needed
                        found_doc_ids_to_return = ['__WORKSPACE_CLEANUP_NEEDED__']
                    else:
                        found_doc_ids_to_return = []
                    
                    # Now filter to only truly orphaned docs (not in doc_status or VDB)
                    if found_doc_ids:
                        try:
                            conn = psycopg2.connect(
                                host=os.environ["POSTGRES_HOST"],
                                user=os.environ["POSTGRES_USER"],
                                password=os.environ["POSTGRES_PASSWORD"],
                                dbname=os.environ.get("POSTGRES_DATABASE") or os.environ.get("POSTGRESQL_DATABASE_DATABASE_2")
                            )
                            cur = conn.cursor()
                            
                            # Check which doc_ids exist in VDB or doc_status
                            placeholders = ','.join(['%s'] * len(found_doc_ids))
                            check_query = f"""
                            SELECT DISTINCT full_doc_id FROM LIGHTRAG_VDB_CHUNKS 
                            WHERE workspace = %s AND full_doc_id IN ({placeholders})
                            """
                            cur.execute(check_query, [workspace_name] + found_doc_ids)
                            tracked_docs = {row[0] for row in cur.fetchall()}
                            
                            print(f"DEBUG: PostgreSQL VDB check found {len(tracked_docs)} tracked docs for workspace {workspace_name}")
                            
                            cur.close()
                            conn.close()
                            
                            # Keep only orphaned docs (not tracked in VDB)
                            orphaned = [doc_id for doc_id in found_doc_ids if doc_id not in tracked_docs]
                            
                            print(f"DEBUG: After filtering, {len(orphaned)} orphaned docs remain")
                            
                            if orphaned:
                                print(f"Found {len(orphaned)} orphaned doc(s) in Neo4j workspace {workspace_name} (out of {len(found_doc_ids)} total)")
                                found_doc_ids_to_return = orphaned
                            else:
                                print(f"No orphaned docs found - all {len(found_doc_ids)} docs are tracked in VDB")
                                found_doc_ids_to_return = []
                        except Exception as pg_err:
                            print(f"PostgreSQL check failed, returning all Neo4j docs: {pg_err}")
                            import traceback
                            traceback.print_exc()
                            found_doc_ids_to_return = found_doc_ids
                    else:
                        print(f"DEBUG: No doc_ids found in Neo4j workspace {workspace_name}")
                        found_doc_ids_to_return = []
            finally:
                print(f"DEBUG: Closing Neo4j driver")
                await driver.close()
            
            # Return after driver is closed
            if 'found_doc_ids_to_return' in locals():
                print(f"DEBUG: _find_orphaned_neo4j_docs returning {len(found_doc_ids_to_return)} doc_ids")
                return found_doc_ids_to_return
            
            print(f"DEBUG: _find_orphaned_neo4j_docs returning empty list (no path matched)")
            return []
        except Exception as e:
            print(f"Error finding orphaned Neo4j docs in workspace {workspace_id}: {e}")
            import traceback
            traceback.print_exc()
            return []

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
                
                # Delete from ALL workspaces where this file exists
                # This prevents orphaned data in Neo4j when querying across multiple workspaces
                indexed_lookup_name = pending.get("indexed_match_key") or filename
                all_workspaces_to_check = []
                
                # IMPORTANT: Always check the current user's workspace first
                # This is where files are actually uploaded and indexed
                if self.workspace_id:
                    all_workspaces_to_check.append(self.workspace_id)
                
                # Build list of additional workspaces to check (cross-workspace search pattern)
                # knowledge_bases contains suffixes like ['Nucor', '', 'eightthreenine']
                # Each represents a separate workspace: OtherDemoInstancesNucor, OtherDemoInstances, OtherDemoInstanceseightthreenine
                if self.knowledge_bases:
                    for kb in self.knowledge_bases:
                        # Each kb value is a workspace suffix
                        # kb="Nucor" → pass as workspace_id to get "OtherDemoInstancesNucor"
                        # kb="" → pass None as workspace_id to get "OtherDemoInstances"
                        # kb="eightthreenine" → pass as workspace_id to get "OtherDemoInstanceseightthreenine"
                        workspace_to_query = kb if kb else None
                        # Avoid duplicates - don't add if already in list
                        if workspace_to_query not in all_workspaces_to_check:
                            all_workspaces_to_check.append(workspace_to_query)
                
                # Collect doc_ids from ALL workspaces
                all_doc_ids_by_workspace = {}
                known_doc_ids = []  # Track doc_ids found in any workspace
                
                for ws_identifier in all_workspaces_to_check:
                    try:
                        # Get indexed files for this specific workspace
                        indexed_files = await self.mcp_tool_obj.get_indexed_files(ws_identifier, self.role_id)
                        result_dict = self._parse_indexed_files_response(indexed_files)
                        
                        doc_ids = result_dict.get(indexed_lookup_name) or []
                        
                        # Fallback: if exact filename key is not present, use best indexed candidate
                        if not doc_ids:
                            candidate_name = pending.get("indexed_candidate")
                            if candidate_name and candidate_name in result_dict:
                                doc_ids = result_dict.get(candidate_name) or []
                        
                        if doc_ids:
                            workspace_key = str(ws_identifier) if ws_identifier else "base"
                            all_doc_ids_by_workspace[workspace_key] = {
                                "ws_identifier": ws_identifier,
                                "doc_ids": doc_ids
                            }
                            # Track these doc_ids for orphan detection in other workspaces
                            known_doc_ids.extend(doc_ids)
                            print(f"Found {len(doc_ids)} doc IDs in workspace {workspace_key}")
                    except Exception as e:
                        print(f"Error checking workspace {ws_identifier}: {e}")
                
                # IMPORTANT: Check for orphaned Neo4j data using known doc_ids
                # If we found doc_ids in ANY workspace, check if those same doc_ids exist in OTHER workspaces' Neo4j graphs
                # This catches cases where files were uploaded to multiple workspaces but only properly tracked in one
                if known_doc_ids:
                    for ws_identifier in all_workspaces_to_check:
                        workspace_key = str(ws_identifier) if ws_identifier else "base"
                        # Skip if we already found doc_ids for this workspace
                        if workspace_key in all_doc_ids_by_workspace:
                            continue
                        
                        try:
                            # First attempt: Query Neo4j directly for orphaned data with these specific doc_ids
                            orphaned_doc_ids = await self._find_orphaned_neo4j_docs(
                                ws_identifier, 
                                self.role_id, 
                                known_doc_ids
                            )
                            
                            # Second attempt: If no matches found with specific doc_ids, search for ANY orphaned data
                            # This handles cases where the same file was uploaded to multiple workspaces at different times
                            # generating different doc_ids for the same file
                            if not orphaned_doc_ids:
                                print(f"No doc_id matches in workspace {workspace_key}, searching for ANY orphaned Neo4j data...")
                                orphaned_doc_ids = await self._find_orphaned_neo4j_docs(
                                    ws_identifier,
                                    self.role_id,
                                    None  # None = get ALL orphaned docs
                                )
                            
                            if orphaned_doc_ids:
                                all_doc_ids_by_workspace[workspace_key] = {
                                    "ws_identifier": ws_identifier,
                                    "doc_ids": orphaned_doc_ids
                                }
                                print(f"Found {len(orphaned_doc_ids)} orphaned Neo4j doc IDs in workspace {workspace_key}")
                        except Exception as e:
                            print(f"Error checking Neo4j orphans in workspace {ws_identifier}: {e}")
                
                print(f"Total workspaces with this file: {len(all_doc_ids_by_workspace)}")

                deleted_doc_count = 0
                failed_doc_count = 0
                deletion_in_progress = False
                deletion_error_text = ""
                
                # Delete from each workspace where the file exists
                for workspace_key, ws_data in all_doc_ids_by_workspace.items():
                    try:
                        bulk_result = await self.mcp_tool_obj.delete_files_by_doc_ids(
                            ws_data["doc_ids"],
                            ws_data["ws_identifier"],
                            self.role_id
                        )
                        print(f"Bulk delete result for {workspace_key}: {bulk_result}")
                        
                        summary = {}
                        try:
                            if isinstance(bulk_result, dict):
                                if bulk_result.get("status") == "client_exception":
                                    deletion_in_progress = True
                                    deletion_error_text = bulk_result.get("error") or ""
                            else:
                                structured = getattr(bulk_result, "structured_content", None)
                                if isinstance(structured, dict):
                                    summary = structured.get("summary", {}) or {}
                        except Exception:
                            summary = {}

                        deleted_doc_count += int(summary.get("success", 0) or 0)
                        failed_doc_count += int(summary.get("failed", 0) or 0) + int(summary.get("not_found", 0) or 0)
                    except Exception as e:
                        print(f"Error deleting from workspace {workspace_key}: {e}")
                        failed_doc_count += len(ws_data["doc_ids"])

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
                    total_doc_ids = sum(len(ws_data["doc_ids"]) for ws_data in all_doc_ids_by_workspace.values())
                    if blob_deleted_files:
                        return (
                            f"Deletion started for {total_doc_ids} indexed document record(s) for '{indexed_lookup_name}'. "
                            f"Deleted blob file(s): {blob_deleted_files}. "
                            "Index cleanup is still running in background; please recheck indexed files shortly."
                        )

                    if blob_result_text:
                        return (
                            f"Deletion started for {total_doc_ids} indexed document record(s) for '{indexed_lookup_name}'. "
                            f"Blob deletion result: {blob_result_text}. "
                            "Index cleanup is still running in background; please recheck indexed files shortly."
                        )

                    if deletion_error_text:
                        return (
                            f"Deletion started for {total_doc_ids} indexed document record(s) for '{indexed_lookup_name}', "
                            f"but the client connection ended early ({deletion_error_text}). "
                            "The backend may still be completing index cleanup."
                        )

                    return (
                        f"Deletion started for {total_doc_ids} indexed document record(s) for '{indexed_lookup_name}'. "
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
    agent_id: str | int,
    knowledge_bases: Optional[list[str]] = None,
    file_names: Optional[List[str]] = None,
    file_contents: Optional[List[str]] = None
) -> dict:
    # --- JWT-based authentication and workspace-user mapping check (copied from ingestion_new.py tools) ---

    # Validate user access to workspace
    valid, err = validate_user_workspace_access(
        user_id=user_id,
        workspace_id=workspace_id
        )
    if not valid:
        return {"error": err}

    # Reusable payload integrity check to reject tampered/corrupted inputs.
    valid_scope, scope_err, can_curate_kb = validate_chatbot_request_scope(
        user_id=user_id,
        workspace_id=workspace_id,
        role_id=role_id,
        industry=industry,
        sub_industry=sub_industry,
        knowledge_bases=knowledge_bases,
    )
    if not valid_scope:
        return {"error": scope_err}

    # Convert string IDs to integers for internal use
    workspace_id = int(workspace_id)
    user_id = int(user_id)
    role_id = int(role_id)
    agent_id = int(agent_id)

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
            can_curate_kb=can_curate_kb,
            file_names=file_names, 
            file_contents=file_contents, 
            mode=mode,
            agent_id=agent_id,
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
                assistant_msg = next((msg for msg in reversed(data) if msg.get("role") == "assistant"), None) if isinstance(data, list) else None
                if isinstance(data, list) and len(data) >= 2: 
                    user_msg = next((msg for msg in reversed(data) if msg.get("role") == "user"), None)
                    last_msg = data[-1]
                    time_modified_str = last_msg.get("timestamp", "N/A")
                    conversations.append({
                        "session_id": ses,
                        "time_modified": time_modified_str,
                        "title": title,
                        "user": user_msg["content"] if user_msg else None,
                        "assistant": assistant_msg["content"] if assistant_msg else None,
                        "task_ids": assistant_msg.get("task_ids") if assistant_msg else None
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