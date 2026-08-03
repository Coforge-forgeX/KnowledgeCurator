"""
Centralized API Models for KB REST Service

All request/response models with proper validation for Swagger/OpenAPI documentation.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum


# ============================================================================
# Common Models
# ============================================================================

class StatusResponse(BaseModel):
    """Generic success/error response"""
    success: bool
    message: str
    error: Optional[str] = None


class PaginationParams(BaseModel):
    """Pagination parameters"""
    page: int = Field(default=1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")


# ============================================================================
# Query Models
# ============================================================================

class QueryMode(str, Enum):
    """Query modes for LightRAG"""
    NAIVE = "naive"
    LOCAL = "local"
    GLOBAL = "global"
    HYBRID = "hybrid"
    MIX = "mix"


class KBQueryRequest(BaseModel):
    """POST /kb/query request"""
    workspace_id: int = Field(..., gt=0, description="Workspace ID")
    query: str = Field(..., min_length=1, max_length=5000, description="Search query")
    kb_id: Optional[int] = Field(None, description="Specific knowledge base ID")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to return")
    mode: QueryMode = Field(default=QueryMode.HYBRID, description="Query mode")
    only_need_context: bool = Field(default=False, description="Return only context without answer")

    class Config:
        json_schema_extra = {
            "example": {
                "workspace_id": 1,
                "query": "What is the company policy on remote work?",
                "kb_id": 123,
                "top_k": 5,
                "mode": "hybrid"
            }
        }


class KBQueryResponse(BaseModel):
    """POST /kb/query response"""
    success: bool = True
    response: str = Field(..., description="Generated answer")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Source documents")
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list, description="Retrieved text chunks")
    mode: str = Field(..., description="Query mode used")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "response": "The company allows remote work 2 days per week...",
                "sources": [
                    {"file_name": "hr_policy.pdf", "workspace_id": 1}
                ],
                "retrieved_chunks": [],
                "mode": "hybrid"
            }
        }


# ============================================================================
# Chat Models
# ============================================================================

class ChatMessage(BaseModel):
    """Single chat message"""
    role: str = Field(..., description="Message role: user, assistant, system")
    content: str = Field(..., min_length=1, description="Message content")

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ["user", "assistant", "system"]:
            raise ValueError("role must be one of: user, assistant, system")
        return v


class KBChatRequest(BaseModel):
    """POST /kb/chat request"""
    workspace_id: int = Field(..., gt=0, description="Workspace ID")
    question: str = Field(..., min_length=1, max_length=5000, description="User question")
    history: List[ChatMessage] = Field(default_factory=list, description="Chat history")
    mode: QueryMode = Field(default=QueryMode.HYBRID, description="Query mode")
    kb_id: Optional[int] = Field(None, description="Specific knowledge base ID")

    class Config:
        json_schema_extra = {
            "example": {
                "workspace_id": 1,
                "question": "What are the vacation policies?",
                "history": [
                    {"role": "user", "content": "Tell me about HR policies"},
                    {"role": "assistant", "content": "Here are the main HR policies..."}
                ],
                "mode": "hybrid"
            }
        }


class KBChatResponse(BaseModel):
    """POST /kb/chat response"""
    success: bool = True
    response: str = Field(..., description="AI assistant response")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Source documents")
    session_id: Optional[str] = Field(None, description="Conversation session ID")


# ============================================================================
# Indexing Models
# ============================================================================

class KBIndexRequest(BaseModel):
    """POST /kb/index request"""
    workspace_id: int = Field(..., gt=0, description="Workspace ID")
    text: str = Field(..., min_length=1, description="Text to index")
    file_metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "workspace_id": 1,
                "text": "This is a document about company policies...",
                "file_metadata": {
                    "file_name": "policy.txt",
                    "source": "sharepoint"
                }
            }
        }


class KBIndexResponse(BaseModel):
    """POST /kb/index response"""
    success: bool
    message: str
    task_id: Optional[int] = Field(None, description="Background task ID")
    document_id: Optional[str] = Field(None, description="Indexed document ID")


# ============================================================================
# Document Management Models
# ============================================================================

class DocumentStatus(str, Enum):
    """Document indexing status"""
    UPLOADING = "uploading"
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    COMPLETED = "completed"
    FAILED = "failed"


class IndexingStatusRequest(BaseModel):
    """GET /api/check-indexing-status request"""
    workspace_id: int = Field(..., gt=0, description="Workspace ID")
    task_ids: Optional[List[int]] = Field(None, description="Specific task IDs to check")


class TaskStatusInfo(BaseModel):
    """Individual task status"""
    task_id: int
    file_name: str
    status: DocumentStatus
    error_message: Optional[str] = None
    created_at: str
    updated_at: str


class IndexingStatusResponse(BaseModel):
    """GET /api/check-indexing-status response"""
    success: bool = True
    workspace_id: int
    tasks: List[TaskStatusInfo]


class ListDocumentsRequest(BaseModel):
    """GET /api/list-indexed-documents request"""
    workspace_id: int = Field(..., gt=0, description="Workspace ID")
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")
    status: Optional[DocumentStatus] = Field(None, description="Filter by status")


class DocumentInfo(BaseModel):
    """Individual document information"""
    id: int
    file_name: str
    file_path: str
    workspace_id: int
    status: str
    file_size: Optional[str] = None
    num_chunks: int = 0
    created_at: str
    updated_at: str


class ListDocumentsResponse(BaseModel):
    """GET /api/list-indexed-documents response"""
    success: bool = True
    documents: List[DocumentInfo]
    total: int
    page: int
    page_size: int


class DeleteDocumentsRequest(BaseModel):
    """DELETE /api/delete-documents request"""
    workspace_id: int = Field(..., gt=0, description="Workspace ID")
    task_ids: List[int] = Field(..., min_length=1, description="Task IDs to delete")

    @field_validator("task_ids")
    @classmethod
    def validate_task_ids_limit(cls, v: List[int]) -> List[int]:
        if len(v) > 100:
            raise ValueError("Maximum 100 documents can be deleted at once")
        return v


class DeleteDocumentsResponse(BaseModel):
    """DELETE /api/delete-documents response"""
    success: bool
    message: str
    deleted_count: int
    failed_ids: List[int] = Field(default_factory=list)


# ============================================================================
# Upload Models
# ============================================================================

class FileUpload(BaseModel):
    """Individual file upload"""
    file_name: str = Field(..., min_length=1, max_length=255, description="File name with extension")
    file_content: str = Field(..., description="Base64 encoded file content")


class UploadAndIndexRequest(BaseModel):
    """POST /api/upload-and-index request"""
    workspace_id: int = Field(..., gt=0, description="Workspace ID")
    files: List[FileUpload] = Field(..., min_length=1, max_length=10, description="Files to upload (max 10)")

    @field_validator("files")
    @classmethod
    def validate_duplicate_filenames(cls, v: List[FileUpload]) -> List[FileUpload]:
        file_names = [f.file_name for f in v]
        if len(file_names) != len(set(file_names)):
            raise ValueError("Duplicate file names not allowed")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "workspace_id": 1,
                "files": [
                    {
                        "file_name": "document.pdf",
                        "file_content": "JVBERi0xLjQKJeLjz9MKMSAwIG9iago..."
                    }
                ]
            }
        }


class FileTaskInfo(BaseModel):
    """Individual file task information"""
    task_id: int
    file_name: str
    file_path: str
    status: str


class UploadAndIndexResponse(BaseModel):
    """POST /api/upload-and-index response"""
    success: bool
    message: str
    workspace_id: int
    total_files: int
    tasks: List[FileTaskInfo]
    failed_files: List[str] = Field(default_factory=list)


# ============================================================================
# Knowledge Graph Models
# ============================================================================

class KnowledgeGraphRequest(BaseModel):
    """GET /api/get-knowledge-graph request"""
    workspace_id: int = Field(..., gt=0, description="Workspace ID")


class KnowledgeGraphResponse(BaseModel):
    """GET /api/get-knowledge-graph response"""
    success: bool = True
    workspace_id: int
    nodes: List[Dict[str, Any]] = Field(default_factory=list, description="Graph nodes")
    edges: List[Dict[str, Any]] = Field(default_factory=list, description="Graph edges")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Graph metadata")


# ============================================================================
# LLM Router Models
# ============================================================================

class LLMRouteRequest(BaseModel):
    """POST /llm/route request"""
    workspace_id: int = Field(..., gt=0, description="Workspace ID")
    prompt: str = Field(..., min_length=1, description="Prompt to route")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class LLMRouteResponse(BaseModel):
    """POST /llm/route response"""
    success: bool = True
    response: str
    model_used: str
    tokens_used: Optional[int] = None
