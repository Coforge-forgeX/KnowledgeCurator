"""
Query RAG API Payloads - UPDATED

Request and response models for the query_rag endpoint.
Simplified - domain and kb_name fetched from database.
"""
from typing import Any, Dict, List, Optional

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
        default=1,
        description="Agent ID for custom LLM routing (default: 1)"
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
        json_schema_extra = {
            "example": {
                "query": "What is asset management?",
                "workspace_id": 123
            }
        }


class GraphDataModel(BaseModel):
    """Structured graph data per KB."""

    entities: List[Dict[str, Any]] = Field(default_factory=list)
    relationship: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KBChunkModel(BaseModel):
    """Chunk entry in per-KB result."""

    source: str = Field(..., description="Source file name or source identifier")
    chunks: str = Field(..., description="Chunk identifier")
    chunk: str = Field(..., description="Chunk data")


class KBResultModel(BaseModel):
    """Per-knowledge-base response section."""

    source: str = Field(..., description="KB name")
    graph_data: GraphDataModel = Field(default_factory=GraphDataModel)
    chunks: List[KBChunkModel] = Field(default_factory=list)


class QueryRAGResponse(BaseModel):
    """Compact response model for query_rag endpoint."""

    final_answer: str = Field(..., description="Final combined answer across all KBs")
    kb_results: List[KBResultModel] = Field(
        default_factory=list,
        description="Per-KB graph and chunk evidence"
    )
    requested_mode: str = Field(..., description="Requested query mode")
    effective_mode: str = Field(..., description="Effective query mode")

    class Config:
        json_schema_extra = {
            "example": {
                "final_answer": "Asset management is...",
                "kb_results": [
                    {
                        "source": "Banking/AssetManagement",
                        "graph_data": {
                            "entities": [],
                            "relationship": [],
                            "metadata": {}
                        },
                        "chunks": [
                            {
                                "source": "Portfolio_Analysis.pdf",
                                "chunks": "chunk_123",
                                "chunk": "Asset management involves..."
                            }
                        ]
                    }
                ],
                "requested_mode": "hybrid",
                "effective_mode": "hybrid"
            }
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    details: Optional[dict] = Field(None, description="Additional error details")
    correlation_id: Optional[str] = Field(None, description="Request correlation ID")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "Failed to execute RAG query",
                "error_code": "QUERY_FAILED",
                "details": {"workspace_id": 123},
                "correlation_id": "abc-123-def"
            }
        }
