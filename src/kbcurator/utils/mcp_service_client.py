import socket
from kbcurator.client.mcp_client import MCPClient
import json
import asyncio
from dotenv import load_dotenv
import os
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

load_dotenv(os.path.abspath(os.path.join(os.getcwd(),'.env')))

class MCPServiceClient:

    # def __init__(self, host, port, industry, sub_industry):
    def __init__(self, server_url, industry, sub_industry, knowledge_bases, token: str | None = None):
        # self.host = host #host
        # self.port = port #port
        self.server_url = server_url #server_url
        self.industry = industry #industry
        self.sub_industry = sub_industry #sub_industry
        if knowledge_bases is None:
            self.knowledge_bases = [""] #knowledge_bases
        else:
            self.knowledge_bases = list(knowledge_bases)
            self.knowledge_bases.append("") #knowledge_bases
            
        self.obj = MCPClient(server_url=self.server_url, token=token)
        self._token = token

    # def set_token(self, token: str | None):
    #     """Update bearer token to be used for subsequent connections."""
    #     self._token = token
    #     if self.obj:
    #         self.obj.set_token(token)

    async def query_rag(self, intent, user_message, history, workspace_id, role_id):
        # Placeholder for MCP socket communication
        print(f"Calling MCP tool for intent: {intent} with message: {user_message}")
        # print(f"Host: {self.host}, Port: {self.port}, Industry: {self.industry}, Sub-industry: {self.sub_industry}")
        if str(role_id) == "34":
            print(f"SME query")
            pass
        else:
            digit_map = {
                '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
                '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
            }
            result = []
            for c in workspace_id:
                if c.isalpha():
                    result.append(c)
                elif c.isdigit():
                    result.append(digit_map[c])
                # skip non-alphanumeric characters
            workspace_id_alpha = ''.join(result)
            knowledge_bases = self.knowledge_bases
            knowledge_bases.append(workspace_id_alpha)
        try: 
            # arguments = {
            # "domain": "Banking & Financial Services",
            # "kb_name": "Asset & Wealth Management",
            # "question": "what is wealth management",
            # "history": [],
            # "user_prompt": "",
            # "mode": "mix"
            # }
            arguments = {
            "domain": self.industry,
            "kb_name": self.sub_industry,
            "question": user_message,
            "history": history,
            "knowledge_bases": knowledge_bases,
            "user_prompt": "",
            "mode": "mix"
            }

            tool_name = "query_rag"
            # print("Arguements for RAG query tool:", arguments)
            # await self.obj.connect_to_server()
            print("Connected to MCP server for RAG query.")
            # print(self._token)
            # response = await self.obj.session.call_tool(
            #     name=tool_name,
            #     arguments=arguments
            # )
            token = self._token if self._token else ""
            # async with Client(
            #     transport=StreamableHttpTransport(
            #         url=self.server_url,
            #         headers={"Authorization": token},
            #     ),
            # ) as client:
            #     response = await client.call_tool(
            #         name = tool_name,
            #         arguments = arguments
            #     )

            await self.obj.connect_to_server()
            print("Connected to MCP server for RAG query.")
            response = await self.obj.session.call_tool(
                name=tool_name,
                arguments=arguments 
            )
            print("Received response from MCP server for RAG query.")
            await self.obj.cleanup()

            text_value = response.structuredContent
            # await self.obj.cleanup()
            lightrag_text = text_value["LightRAG"] if text_value else ""
            print("Response from LightRAG:", lightrag_text[:10])
            return lightrag_text
        
            '''arguments = {
            "domain": self.industry,
            "kb_name": self.sub_industry,
            "question": user_message,
            "history": history
            }'''

            # call the aitl_reviewer tool with the response, state, and uuid
            # tool_name = "extract_text_from_query"
            # self.obj.connect_to_server()
            # response = await self.obj.session.call_tool(
            #    name=tool_name, 
            #    arguments=arguments
            # )
            # text_value = response.content[0].text
            # lightrag_text = json.loads(text_value)["LightRAG"]
            # self.obj.cleanup()

        except Exception as e:
            print(f"An error occurred while processing your request: {e}")
            return "Unable to process your request at this time. Please try again later."
    
    async def upload_rag(self, intent, file_names, workspace_id, user_id, role_id, file_contents):
        """
        Calls the unified MCP tool `upload_and_index_tool`
        which handles upload + indexing asynchronously in background.
        """
        print(f"Calling MCP upload_and_index_tool for intent: {intent}")

        if role_id == 34:
            container_name = "knowledgecurator"
            upload_path = f"{self.industry}/{self.sub_industry}"
            kb_name = self.sub_industry
        else:
            container_name = "workspace"
            upload_path = f"{self.industry}/{self.sub_industry}/{workspace_id}"
            digit_map = {
                '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
                '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
            }
            result = []
            for c in workspace_id:
                if c.isalpha():
                    result.append(c)
                elif c.isdigit():
                    result.append(digit_map[c])
                # skip non-alphanumeric characters
            workspace_id_alpha = ''.join(result)
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
            
            await self.obj.connect_to_server()
            print("Connected to MCP server for RAG upload and index.")
            response = await self.obj.session.call_tool(
                name=tool_name,
                arguments=upload_arguments 
            )
            print("Received response from MCP server for RAG upload and index.")
            await self.obj.cleanup()

            structured = response.structured_content if hasattr(response, "structured_content") else response.structuredContent or {}
            status = structured.get("status")
            tasks = structured.get("tasks")
            if isinstance(tasks, list) and tasks:
                print("Upload + Index tasks submitted successfully.")
                messages = []
                for t in tasks:
                    tid = t.get("task_id")
                    if tid is not None:
                        messages.append(
                            f"Upload and indexing started in background. Task ID: {tid} Use this task_id to poll status."
                        )
                if messages:
                    print(f"Status: {status}, Tasks: {[t.get('task_id') for t in tasks if t.get('task_id') is not None]}")
                    return {
                        "response": "\n".join(messages),
                        "task_ids": [t.get('task_id') for t in tasks if t.get('task_id') is not None]
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

    async def get_indexed_files(self):
        try:
            tool_name = "get_indexed_file_names"
            token = self._token if self._token else ""
            async with Client(
                transport=StreamableHttpTransport(
                    self.server_url,
                    headers={"Authorization": token},
                ),
            ) as client:
                response = await client.call_tool(
                    name=tool_name
                )
            return response
        except Exception:
            print("An error occurred while processing your request.")
        return None
        
    async def delete_file(self, doc_id):
        try:
            tool_name = "delete_by_doc_id"
            args = {"doc_id": doc_id}
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
    
    async def delete_files_from_blob(self, filename_list):
        try:
            tool_name = "delete_files_from_blob"
            args = {"domain": self.industry,
                    "kbs": [self.sub_industry],
                    "file_names": filename_list}
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