import socket
from kbcurator.client.mcp_client import MCPClient
import json
import asyncio
from threading import RLock
from time import monotonic
from dotenv import load_dotenv
import os
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from kbcurator.utils.db import db
from kbcurator.utils.constants import WorkspaceType

# Load .env file if it exists (for local development)
env_path = os.path.abspath(os.path.join(os.getcwd(), '.env'))
if os.path.exists(env_path):
    load_dotenv(env_path)

class MCPServiceClient:
    _WORKSPACE_TYPE_CACHE_TTL_SECONDS = 600
    _workspace_type_cache = {}
    _workspace_type_cache_lock = RLock()

    # def __init__(self, host, port, industry, sub_industry):
    def __init__(self, server_url, industry, sub_industry, knowledge_bases, token: str | None = None):
        # self.host = host #host
        # self.port = port #port
        self.server_url = server_url #server_url
        self.industry = industry #industry
        self.sub_industry = sub_industry #sub_industry
        self.knowledge_bases = knowledge_bases 
        if knowledge_bases is None:
            self.knowledge_bases = [""] #knowledge_bases
        else:
            self.knowledge_bases = list(knowledge_bases)
            self.knowledge_bases.append("") #knowledge_bases
            
        self.obj = MCPClient(server_url=self.server_url, token=token)
        self._token = token

    def _workspace_id_to_alpha(self, workspace_id) -> str:
        digit_map = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
        }
        result = []
        for c in str(workspace_id or ""):
            if c.isalpha():
                result.append(c)
            elif c.isdigit():
                result.append(digit_map[c])
        return ''.join(result)

    def _fetch_workspace_type(self, workspace_id) -> str | None:
        if not workspace_id:
            return None

        session = db.Session()
        try:
            ws = session.query(db.Workspace).filter(
                db.Workspace.workspace_id == workspace_id,
                db.Workspace.is_active == True
            ).first()
            if not ws:
                return None

            keyword = (getattr(ws, "keywords", None) or "").strip()
            if not keyword:
                return None

            # Keywords are persisted as codes like KG/DM/TR/PR.
            return keyword.split(",")[0].strip().upper() or None
        finally:
            session.close()

    def _get_workspace_type(self, workspace_id) -> str | None:
        if not workspace_id:
            return None

        cache_key = str(workspace_id)
        now = monotonic()
        with self._workspace_type_cache_lock:
            cached = self._workspace_type_cache.get(cache_key)
            if cached and cached.get("expires_at", 0) > now:
                return cached.get("value")

        workspace_type = self._fetch_workspace_type(workspace_id)
        with self._workspace_type_cache_lock:
            self._workspace_type_cache[cache_key] = {
                "value": workspace_type,
                "expires_at": now + self._WORKSPACE_TYPE_CACHE_TTL_SECONDS,
            }
        return workspace_type

    def _is_kg_workspace(self, workspace_id) -> bool:
        return self._get_workspace_type(workspace_id) == WorkspaceType.KG.name

    def _get_primary_knowledge_base(self) -> str | None:
        """Return first non-empty knowledge base token supplied in request context."""
        if not isinstance(self.knowledge_bases, list):
            return None
        for kb in self.knowledge_bases:
            text = (str(kb).strip() if kb is not None else "")
            if text:
                return text
        return None

    # def set_token(self, token: str | None):
    #     """Update bearer token to be used for subsequent connections."""
    #     self._token = token
    #     if self.obj:
    #         self.obj.set_token(token)

    # async def query_rag(self, intent, user_message, history, workspace_id, role_id):
    #     # Placeholder for MCP socket communication
    #     print(f"Calling MCP tool for intent: {intent} with message: {user_message}")
    #     # print(f"Host: {self.host}, Port: {self.port}, Industry: {self.industry}, Sub-industry: {self.sub_industry}")
    #     if str(role_id) == "34":
    #         print(f"SME query")
    #         pass
    #     else:
    #         digit_map = {
    #             '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
    #             '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
    #         }
    #         result = []
    #         for c in workspace_id:
    #             if c.isalpha():
    #                 result.append(c)
    #             elif c.isdigit():
    #                 result.append(digit_map[c])
    #             # skip non-alphanumeric characters
    #         workspace_id_alpha = ''.join(result)
    #         knowledge_bases = self.knowledge_bases
    #         knowledge_bases.append(workspace_id_alpha)
    #     try: 
    #         # arguments = {
    #         # "domain": "Banking & Financial Services",
    #         # "kb_name": "Asset & Wealth Management",
    #         # "question": "what is wealth management",
    #         # "history": [],
    #         # "user_prompt": "",
    #         # "mode": "mix"
    #         # }
    #         arguments = {
    #         "domain": self.industry,
    #         "kb_name": self.sub_industry,
    #         "question": user_message,
    #         "history": history,
    #         "knowledge_bases": knowledge_bases,
    #         "user_prompt": "",
    #         "mode": "mix"
    #         }

    #         tool_name = "query_rag"
    #         # print("Arguements for RAG query tool:", arguments)
    #         # await self.obj.connect_to_server()
    #         print("Connected to MCP server for RAG query.")
    #         # print(self._token)
    #         # response = await self.obj.session.call_tool(
    #         #     name=tool_name,
    #         #     arguments=arguments
    #         # )
    #         token = self._token if self._token else ""
    #         async with Client(
    #             transport=StreamableHttpTransport(
    #                 url=self.server_url,
    #                 headers={"Authorization": token},
    #             ),
    #         ) as client:
    #             response = await client.call_tool(
    #                 name = tool_name,
    #                 arguments = arguments
    #             )
    #         print("Received response from MCP server for RAG query.")
    #         # print("Cleaned up MCP client session after RAG query.")
                
    #         text_value = response.structured_content
    #         # await self.obj.cleanup()
    #         lightrag_text = text_value["LightRAG"] if text_value else ""
    #         print("Response from LightRAG:", lightrag_text[:10])
    #         return lightrag_text
        
    #         '''arguments = {
    #         "domain": self.industry,
    #         "kb_name": self.sub_industry,
    #         "question": user_message,
    #         "history": history
    #         }'''

    #         # call the aitl_reviewer tool with the response, state, and uuid
    #         # tool_name = "extract_text_from_query"
    #         # self.obj.connect_to_server()
    #         # response = await self.obj.session.call_tool(
    #         #    name=tool_name, 
    #         #    arguments=arguments
    #         # )
    #         # text_value = response.content[0].text
    #         # lightrag_text = json.loads(text_value)["LightRAG"]
    #         # self.obj.cleanup()

    #     except Exception:
    #         print("An error occurred while processing your request.")
    #         return "Unable to process your request at this time. Please try again later."

    async def query_rag(self, intent, user_message, history, workspace_id, role_id):
        print(f"Calling MCP tool for intent: {intent} with message: {user_message}")
        
        # Initialize kb_name
        kb_name = self.sub_industry
        
        #knowledge_bases = []

        # if str(role_id) == "34":
        #     print(f"SME query")
        #     pass
        # else:
        
        # digit_map = {
        #     '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        #     '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
        # }
        # result = []
        # for c in workspace_id:
        #     if c.isalpha():
        #         result.append(c)
        #     elif c.isdigit():
        #         result.append(digit_map[c])
        # workspace_id_alpha = ''.join(result)
        # role_id_str = str(role_id) if role_id is not None else ""
        # if role_id_str == "34":
        #     # SME: keep original cross-KB behavior
        #     knowledge_bases = self.knowledge_bases.copy()
        # else:
        #     # Workspace users: query only their workspace KB to avoid cross-scope contamination.
        #     if workspace_id:
        #         digit_map = {
        #             '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        #             '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
        #         }
        #         result = []
        #         for c in str(workspace_id):
        #             if c.isalpha():
        #                 result.append(c)
        #             elif c.isdigit():
        #                 result.append(digit_map[c])
        #         workspace_id_alpha = ''.join(result)
        #         if workspace_id_alpha:
        #             knowledge_bases = [workspace_id_alpha]
       # knowledge_bases.append(workspace_id_alpha)
            
        # Modified kb_name for workspace isolation
        # kb_name = f"{self.sub_industry}/{workspace_id_alpha}"
        
        try: 
            arguments = {
                "domain": self.industry,
                "kb_name": kb_name,
                "question": user_message,
                "history": history,
                "knowledge_bases": self.knowledge_bases,
                "user_prompt": "",
                "mode": "mix",
                "workspace_id": workspace_id,  # ADD THIS
                "role_id": role_id              # ADD THIS
            }

            tool_name = "query_rag"
            print("Connected to MCP server for RAG query.")
            token = self._token if self._token else ""
            async with Client(
                transport=StreamableHttpTransport(
                    url=self.server_url,
                    headers={"Authorization": token},
                ),
            ) as client:
                response = await client.call_tool(
                    name=tool_name,
                    arguments=arguments
                )
            print("Received response from MCP server for RAG query.")
                
            text_value = response.structured_content
            
            # Check if response has structured format with sources
            if text_value and isinstance(text_value, dict):
                if "sources" in text_value:
                    # Return full structured response with sources
                    return {
                        "response": text_value.get("LightRAG", text_value.get("response", "")),
                        "sources": text_value.get("sources", []),
                        "task_ids": text_value.get("task_ids", [])
                    }
                else:
                    # Backward compatibility: return text and preserve optional task_ids.
                    lightrag_text = text_value.get("LightRAG", text_value.get("response", ""))
                    print("Response from LightRAG:", lightrag_text[:10])
                    return {
                        "response": lightrag_text,
                        "task_ids": text_value.get("task_ids", [])
                    }
            else:
                return ""
        
        except Exception as e:
            print(f"Error calling MCP tool: {e}")
            return f"Error: {e}"
    
    async def upload_rag(self, intent, file_names, workspace_id, user_id, role_id, file_contents):
            """
            Calls the unified MCP tool `upload_and_index_tool`
            which handles upload + indexing asynchronously in background.
            """
            print(f"Calling MCP upload_and_index_tool for intent: {intent}")

            if self._is_kg_workspace(workspace_id):
                primary_kb = self._get_primary_knowledge_base()
                container_name = os.getenv('AZURE_BLOB_STORAGE_CONTAINER_NAME')
                if primary_kb:
                    upload_path = f"{self.industry}/{self.sub_industry}/{primary_kb}"
                    kb_name = f"{self.sub_industry}/{primary_kb}"

            else:
                container_name = "workspace"
                upload_path = f"{self.industry}/{self.sub_industry}/{workspace_id}"
                workspace_id_alpha = self._workspace_id_to_alpha(workspace_id)
                kb_name = f"{self.sub_industry}/{workspace_id_alpha}"

            try:
                upload_arguments = {
                    "container_name": container_name,
                    "upload_path": upload_path,
                    "file_names": file_names,
                    "file_contents": file_contents,
                    "domain": self.industry,
                    "kb_name": kb_name,
                    "workspace_id": workspace_id,
                    "user_id": user_id,
                    "expiry_years": 10
                }

                print("Upload + Index process started with arguments:", upload_arguments)
                tool_name = "upload_and_index_tool"
                token = self._token if self._token else ""
                async with Client(
                    transport=StreamableHttpTransport(
                        self.server_url,
                        headers={"Authorization": token},
                    ),
                ) as client:
                    response = await client.call_tool(
                        name=tool_name,
                        arguments=upload_arguments
                    )

                structured = response.structured_content if hasattr(response, "structured_content") else response.structuredContent or {}
                status = structured.get("status")
                tasks = structured.get("tasks")
                if isinstance(tasks, list) and tasks:
                    print("Upload + Index tasks submitted successfully.")
                    task_ids = [t.get("task_id") for t in tasks if t.get("task_id") is not None]
                    if task_ids:
                        print(f"Status: {status}, Tasks: {task_ids}")
                        return {
                            "response": "Upload and indexing started in background. Please check the status through the file storage section.",
                            "task_ids": task_ids
                        }

                task_id = structured.get("task_id")
                print("Upload + Index task submitted successfully.")
                print(f"Task ID: {task_id}, Status: {status}")
                return task_id

            except asyncio.CancelledError:
                print("Operation was cancelled.")
                return "Operation was cancelled. Please try again."
            except Exception:
                print("An error occurred while processing your request.")
                return "Unable to process your request at this time. Please try again later."
        
    async def index_url(self, intent, url, industry, sub_industry) -> str:
        print(f"Calling MCP URL indexing tool for intent: {intent}")
        try:
            index_arguments = {
                "url": url,
                "domain": industry,
                "kb_name": sub_industry
            }

            print("URL Index process started...")
            tool_name = "conversation_indexing_tool"
            token = self._token if self._token else ""
            async with Client(
                transport=StreamableHttpTransport(
                    self.server_url,
                    headers={"Authorization": token},
                ),
            ) as client:
                index_response = await client.call_tool(
                    name=tool_name,
                    arguments=index_arguments
                )
            print(f"URL Indexing process completed successfully. Find stat's below:", getattr(index_response, "structured_content", getattr(index_response, "structuredContent", None)))
            return "URL has been indexed successfully."
        except Exception:
            print("An error occurred while processing your request.")
            return "Unable to process your request at this time. Please try again later."
               
    async def edit_node(self, intent, args):
        print(f"Calling MCP editing tool for intent: {intent}")
        try:
            print("Edit process started...")
            tool_name = "edit_entity_in_kg"
            token = self._token if self._token else ""
            async with Client(
                transport=StreamableHttpTransport(
                    self.server_url,
                    headers={"Authorization": token},
                ),
            ) as client:
                edit_response = await client.call_tool(
                    name=tool_name,
                    arguments=args
                )
            print(f"Edit process completed successfully. Find stat's below:", edit_response)
            return edit_response
        except asyncio.CancelledError:
            print("Operation was cancelled.")
        except Exception:
            print("An error occurred while processing your request.")
        return None

    async def delete_node(self, intent, delete_arguments):
        print(f"Calling MCP deleting tool for intent: {intent}")
        try:
            print("Delete process started...")
            tool_name = "delete_entity_from_kg"
            token = self._token if self._token else ""
            async with Client(
                transport=StreamableHttpTransport(
                    self.server_url,
                    headers={"Authorization": token},
                ),
            ) as client:
                delete_response = await client.call_tool(
                    name=tool_name,
                    arguments=delete_arguments
                )
            print(f"Delete process completed successfully. Find stat's below:", delete_response)
            return delete_response
        except asyncio.CancelledError:
            print("Operation was cancelled.")
        except Exception:
            print("An error occurred while processing your request.")
        return None

    async def add_node(self, intent, index_arguments):
        print(f"Calling MCP Adding tool for intent: {intent}")
        try:
            print("Index process started...")
            tool_name = "conversation_indexing_tool"
            token = self._token if self._token else ""
            async with Client(
                transport=StreamableHttpTransport(
                    self.server_url,
                    headers={"Authorization": token},
                ),
            ) as client:
                index_response = await client.call_tool(
                    name=tool_name,
                    arguments=index_arguments
                )
            print(f"Adding process completed successfully. Find stat's below:", index_response)
            return index_response
        except asyncio.CancelledError:
            print("Operation was cancelled.")
        except Exception:
            print("An error occurred while processing your request.")
        return None

    # async def get_indexed_files(self):
    #     try:
    #         tool_name = "get_indexed_file_names"
    #         token = self._token if self._token else ""
    #         async with Client(
    #             transport=StreamableHttpTransport(
    #                 self.server_url,
    #                 headers={"Authorization": token},
    #             ),
    #         ) as client:
    #             response = await client.call_tool(
    #                 name=tool_name
    #             )
    #         return response
    #     except Exception:
    #         print("An error occurred while processing your request.")
    #     return None
    
    # async def get_indexed_files(self):
    #     try:
    #         tool_name = "get_indexed_file_names"
    #         args = {
    #             "domain": self.industry,
    #             "kb_name": self.sub_industry
    #         }
    #         token = self._token if self._token else ""
    #         async with Client(
    #             transport=StreamableHttpTransport(
    #                 self.server_url,
    #                 headers={"Authorization": token},
    #             ),
    #         ) as client:
    #             response = await client.call_tool(
    #                 name=tool_name,
    #                 arguments=args
    #             )
    #         return response
    #     except Exception:
    #         print("An error occurred while processing your request.")
    #     return None

    def _has_non_empty_index_payload(self, response) -> bool:
        """Return True when MCP response content contains a non-empty index map."""
        try:
            content = getattr(response, "content", None)
            if not content:
                return False
            first_item = content[0] if len(content) > 0 else None
            text_json = getattr(first_item, "text", "")
            parsed = json.loads(text_json) if text_json else {}
            return isinstance(parsed, dict) and len(parsed) > 0
        except Exception:
            return False

    def _extract_index_payload(self, response) -> dict:
        """Extract {file_name: [doc_ids]} from MCP tool response content."""
        try:
            if isinstance(response, dict):
                return response if all(isinstance(v, list) for v in response.values()) else {}
            content = getattr(response, "content", None)
            if not content:
                return {}
            first_item = content[0] if len(content) > 0 else None
            text_json = getattr(first_item, "text", "")
            parsed = json.loads(text_json) if text_json else {}
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    async def get_indexed_files(self, workspace_id=None, role_id=None):
        try:
            tool_name = "get_indexed_file_names"

            # Construct kb_name same way as upload_rag does.
            role_id_str = str(role_id) if role_id is not None else ""
            workspace_id_alpha = ""
            if workspace_id:
                digit_map = {
                    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
                    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
                }
                result = []
                for c in str(workspace_id):
                    if c.isalpha():
                        result.append(c)
                    elif c.isdigit():
                        result.append(digit_map[c])
                workspace_id_alpha = ''.join(result)

            candidate_kb_names = []
            if workspace_id_alpha:
                candidate_kb_names.append(f"{self.sub_industry}/{workspace_id_alpha}")
            if workspace_id:
                # Backward compatibility for historical numeric-scoped writes.
                candidate_kb_names.append(f"{self.sub_industry}/{workspace_id}")
            candidate_kb_names.append(self.sub_industry)

            # De-duplicate while preserving order.
            seen = set()
            candidate_kb_names = [kb for kb in candidate_kb_names if not (kb in seen or seen.add(kb))]

            token = self._token if self._token else ""
            async with Client(
                transport=StreamableHttpTransport(
                    self.server_url,
                    headers={"Authorization": token},
                ),
            ) as client:
                merged: dict[str, list[str]] = {}
                last_response = None

                for kb_name in candidate_kb_names:
                    response = await client.call_tool(
                        name=tool_name,
                        arguments={
                            "domain": self.industry,
                            "kb_name": kb_name,
                        }
                    )
                    last_response = response
                    payload = self._extract_index_payload(response)
                    if not payload:
                        continue

                    for file_name, doc_ids in payload.items():
                        if file_name not in merged:
                            merged[file_name] = []
                        for doc_id in doc_ids or []:
                            if doc_id not in merged[file_name]:
                                merged[file_name].append(doc_id)

                if merged:
                    return merged

            return last_response
        except Exception:
            print("An error occurred while processing your request.")
        return None
        
    # async def delete_file(self, doc_id):
    #     try:
    #         tool_name = "delete_by_doc_id"
    #         args = {"doc_id": doc_id}
    #         token = self._token if self._token else ""
    #         async with Client(
    #             transport=StreamableHttpTransport(
    #                 self.server_url,
    #                 headers={"Authorization": token},
    #             ),
    #         ) as client:
    #             response = await client.call_tool(
    #                 name=tool_name,
    #                 arguments=args
    #             )
    #         return response
    #     except Exception:
    #         print("An error occurred while processing your request.")
    #     return None
    
    # async def delete_file(self, doc_id):
    #     try:
    #         tool_name = "delete_by_doc_id"
    #         args = {
    #         "doc_id": doc_id,
    #         "domain": self.industry,      # Add this
    #         "kb_name": self.sub_industry  # Add this
    #          }
    #         token = self._token if self._token else ""
    #         async with Client(
    #             transport=StreamableHttpTransport(
    #                 self.server_url,
    #                 headers={"Authorization": token},
    #             ),
    #         ) as client:
    #             response = await client.call_tool(
    #                 name=tool_name,
    #                 arguments=args
    #             )
    #         return response
    #     except Exception:
    #         print("An error occurred while processing your request.")
    #     return None

    async def delete_file(self, doc_id, workspace_id=None, role_id=None):
        try:
            tool_name = "delete_by_doc_id"

            # Use the same kb scoping used for indexing/query to target the correct workspace data.
            if role_id and str(role_id) != "34" and workspace_id:
                digit_map = {
                    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
                    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
                }
                result = []
                for c in str(workspace_id):
                    if c.isalpha():
                        result.append(c)
                    elif c.isdigit():
                        result.append(digit_map[c])
                workspace_id_alpha = ''.join(result)
                kb_name = f"{self.sub_industry}/{workspace_id_alpha}"
            else:
                kb_name = self.sub_industry

            args = {
                "doc_id": doc_id,
                "domain": self.industry,
                "kb_name": kb_name,
            }
            token = self._token if self._token else ""
            async with Client(
                transport=StreamableHttpTransport(
                    self.server_url,
                    headers={"Authorization": token},
                ),
            ) as client:
                response = await client.call_tool(
                    name=tool_name,
                    arguments=args
                )
            return response
        except Exception:
            print("An error occurred while processing your request.")
        return None

    async def delete_files_by_doc_ids(self, doc_ids, workspace_id=None, role_id=None):
        try:
            tool_name = "delete_by_doc_ids"

            if role_id and str(role_id) != "34" and workspace_id:
                digit_map = {
                    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
                    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
                }
                result = []
                for c in str(workspace_id):
                    if c.isalpha():
                        result.append(c)
                    elif c.isdigit():
                        result.append(digit_map[c])
                workspace_id_alpha = ''.join(result)
                kb_name = f"{self.sub_industry}/{workspace_id_alpha}"
            else:
                kb_name = self.sub_industry

            args = {
                "doc_ids": doc_ids,
                "domain": self.industry,
                "kb_name": kb_name,
                "skip_rebuild": True,
            }

            token = self._token if self._token else ""
            async with Client(
                transport=StreamableHttpTransport(
                    self.server_url,
                    headers={"Authorization": token},
                ),
            ) as client:
                response = await client.call_tool(
                    name=tool_name,
                    arguments=args
                )
            return response
        except Exception as e:
            err_msg = f"Bulk delete call failed: {type(e).__name__}: {e}"
            print(err_msg)
            return {
                "status": "client_exception",
                "error": str(e),
                "error_type": type(e).__name__,
                "doc_count": len(doc_ids or []),
            }

    async def delete_file_single_call(self, file_name, workspace_id=None, role_id=None, skip_rebuild=True):
        try:
            tool_name = "delete_file_single_call"
            args = {
                "domain": self.industry,
                "kb_name": self.sub_industry,
                "file_name": file_name,
                "workspace_id": workspace_id,
                "role_id": role_id,
                "skip_rebuild": skip_rebuild,
            }
            token = self._token if self._token else ""
            async with Client(
                transport=StreamableHttpTransport(
                    self.server_url,
                    headers={"Authorization": token},
                ),
            ) as client:
                response = await client.call_tool(
                    name=tool_name,
                    arguments=args,
                )

            if isinstance(response, dict):
                return response

            structured = getattr(response, "structured_content", None)
            if isinstance(structured, dict):
                return structured

            content = getattr(response, "content", None)
            if content and len(content) > 0:
                text_json = getattr(content[0], "text", "")
                if text_json:
                    parsed = json.loads(text_json)
                    if isinstance(parsed, dict):
                        return parsed

            return {"status": "error", "message": "delete_file_single_call returned unexpected response format"}
        except Exception as e:
            return {
                "status": "client_exception",
                "error": str(e),
                "error_type": type(e).__name__,
            }
    
    # async def delete_files_from_blob(self, filename_list):
    #     try:
    #         tool_name = "delete_files_from_blob"
    #         args = {"domain": self.industry,
    #                 "kbs": [self.sub_industry],
    #                 "file_names": filename_list}
    #         token = self._token if self._token else ""
    #         async with Client(
    #             transport=StreamableHttpTransport(
    #                 self.server_url,
    #                 headers={"Authorization": token},
    #             ),
    #         ) as client:
    #             response = await client.call_tool(
    #                 name=tool_name,
    #                 arguments=args
    #             )
    #         return response
    #     except Exception:
    #         print("An error occurred while processing your request.")
    #     return None

    # async def delete_files_from_blob(self, filename_list, workspace_id=None, role_id=None):
    #     try:
    #         tool_name = "delete_files_from_blob"
            
    #         # Match the upload_rag logic for blob path construction
    #         if role_id and str(role_id) != "34" and workspace_id:
    #             # User workspace - include workspace_id in path
    #             container_name = "workspace"
    #             upload_path = f"{self.industry}/{self.sub_industry}/{workspace_id}"
    #         else:
    #             # SME/knowledge curator - no workspace_id
    #             container_name = "knowledgecurator"
    #             upload_path = f"{self.industry}/{self.sub_industry}"
            
    #         args = {
    #             "container_name": container_name,
    #             "upload_path": upload_path,
    #             "file_names": filename_list
    #         }
            
    #         token = self._token if self._token else ""
    #         async with Client(
    #             transport=StreamableHttpTransport(
    #                 self.server_url,
    #                 headers={"Authorization": token},
    #             ),
    #         ) as client:
    #             response = await client.call_tool(
    #                 name=tool_name,
    #                 arguments=args
    #             )
    #         return response
    #     except Exception:
    #         print("An error occurred while processing your request.")
    #     return None

    async def delete_files_from_blob(self, filename_list, workspace_id=None, role_id=None):
        try:
            tool_name = "delete_files_from_blob"
            
            # Match the upload_rag logic but use the tool's expected parameter names
            if role_id and str(role_id) != "34" and workspace_id:
                # User workspace - include workspace_id in path
                container_name = "workspace"
                kbs = [f"{self.sub_industry}/{workspace_id}"]
            else:
                # SME/knowledge curator - no workspace_id
                container_name = "knowledgecurator"
                kbs = [self.sub_industry]
            
            args = {
                "domain": self.industry,
                "kbs": kbs,
                "file_names": filename_list,
                "container_name": container_name  # Add this
            }
            
            token = self._token if self._token else ""
            async with Client(
                transport=StreamableHttpTransport(
                    self.server_url,
                    headers={"Authorization": token},
                ),
            ) as client:
                response = await client.call_tool(
                    name=tool_name,
                    arguments=args
                )
            return response
        except Exception:
            print("An error occurred while processing your request.")
        return None