"""
Query API Models

Pydantic models for LightRAG query REST endpoint.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """POST /query request"""

    workspace_id: int = Field(..., gt=0)
    user_id: int = Field(..., gt=0)
    role_id: int = Field(..., gt=0)
    agent_id: int = Field(..., gt=0)
    domain: str = Field(...)
    kb_name: str = Field(...)
    question: str = Field(..., min_length=1)
    knowledge_bases: Optional[List[str]] = None
    mode: str = Field(default="mix")
    history: Optional[List[Dict]] = None


class QueryResponse(BaseModel):
    """POST /query response"""

    response: str
    sources: List[Dict[str, str]] = Field(default_factory=list)
    retrieved_chunks: List[Dict] = Field(default_factory=list)
