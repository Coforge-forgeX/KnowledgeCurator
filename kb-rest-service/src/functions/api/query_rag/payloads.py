"""
Query RAG API Payloads - UPDATED

Request and response models for the query_rag endpoint.
Simplified - domain and kb_name fetched from database.
"""
from typing import List, Optional

from pydantic import BaseModel, Field, validator


class QueryRAGRequest(BaseModel):
    """
    Minimal request model for query_rag endpoint.

    DESIGN: User provides minimal input - workspace_id is used to fetch:
    - domain (from workspace metadata)
    - kb_name (from workspace metadata)
    - role_id (from user-workspace membership)
    - all_kb_titles (for multi-KB queries)

    This prevents tampering and ensures data consistency.
    """

    # ============================================================================
    # Required Fields (User Must Provide)
    # ============================================================================

    query: str = Field(
        ...,
        description="User's question to the knowledge base",
        min_length=1,
        max_length=5000
    )
    workspace_id: int = Field(
        ...,
        description="Workspace ID - used to fetch domain, KB name, and access permissions",
        ge=1
    )

    # ============================================================================
    # Optional Fields (Defaults Provided)
    # ============================================================================

    mode: str = Field(
        default="hybrid",
        description="Query strategy - hybrid (recommended), local, global, naive, or mix"
    )
    history: Optional[List[dict]] = Field(
        default=None,
        description="Conversation history for context-aware responses (optional)"
    )
    agent_id: Optional[int] = Field(
        default=None,
        description="Agent ID for custom LLM routing (optional, uses workspace default if not provided)"
    )

    @validator("query")
    def validate_query(cls, v):
        """Validate query is not empty after stripping"""
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace")
        return v.strip()

    @validator("mode")
    def validate_mode(cls, v):
        """Validate mode is valid"""
        valid_modes = ["naive", "local", "global", "hybrid", "mix"]
        if v.lower() not in valid_modes:
            raise ValueError(f"Mode must be one of: {', '.join(valid_modes)}")
        return v.lower()

    class Config:
        schema_extra = {
            "example": {
                "query": "What is asset management?",
                "workspace_id": 123
            }
        }


class SourceInfo(BaseModel):
    """Source document information"""
    file_name: str = Field(..., description="File name with citation")
    download_url: str = Field(..., description="Download URL with SAS token")
    container_name: str = Field(..., description="Blob container name")
    blob_path: str = Field(..., description="Blob storage path")
    download_name: str = Field(..., description="Download file name")
    citation: Optional[str] = Field(None, description="Citation number (e.g., [1])")


class RetrievedChunkInfo(BaseModel):
    """Retrieved document chunk information"""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    content: str = Field(..., description="Chunk content")
    score: float = Field(..., description="Relevance score", ge=0.0, le=1.0)
    source: str = Field(..., description="Source identifier")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class QueryRAGResponse(BaseModel):
    """Response model for query_rag endpoint"""

    response: str = Field(..., description="Generated answer")
    sources: List[SourceInfo] = Field(
        default_factory=list,
        description="Source documents with download URLs"
    )
    retrieved_chunks: List[RetrievedChunkInfo] = Field(
        default_factory=list,
        description="Retrieved document chunks (for evaluation)"
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Additional metadata about the query"
    )

    # Legacy compatibility fields
    LightRAG: Optional[str] = Field(None, description="Legacy field - same as response")
    task_ids: List[int] = Field(default_factory=list, description="Legacy field - task IDs")

    def __init__(self, **data):
        """Initialize with legacy field compatibility"""
        super().__init__(**data)
        # Populate legacy LightRAG field for backward compatibility
        if not self.LightRAG and self.response:
            self.LightRAG = self.response

    class Config:
        schema_extra = {
            "example": {
                "response": "Asset management is...",
                "sources": [
                    {
                        "file_name": "[1] Portfolio_Analysis.pdf",
                        "download_url": "https://storage.blob.core.windows.net/...",
                        "container_name": "knowledgecurator",
                        "blob_path": "Banking/AssetManagement/Portfolio_Analysis.pdf",
                        "download_name": "Portfolio_Analysis.pdf",
                        "citation": "[1]"
                    }
                ],
                "retrieved_chunks": [
                    {
                        "chunk_id": "chunk_123",
                        "content": "Asset management involves...",
                        "score": 0.95,
                        "source": "Portfolio_Analysis.pdf",
                        "metadata": {}
                    }
                ],
                "metadata": {
                    "mode": "hybrid",
                    "workspace_id": 123,
                    "domain": "Banking",
                    "kb_name": "AssetManagement",
                    "reference_count": 3
                },
                "LightRAG": "Asset management is...",
                "task_ids": []
            }
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    details: Optional[dict] = Field(None, description="Additional error details")
    correlation_id: Optional[str] = Field(None, description="Request correlation ID")

    class Config:
        schema_extra = {
            "example": {
                "error": "Failed to execute RAG query",
                "error_code": "QUERY_FAILED",
                "details": {"workspace_id": 123},
                "correlation_id": "abc-123-def"
            }
        }
