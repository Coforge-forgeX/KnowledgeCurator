"""Standard API models for KB REST service"""
from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class APIResponse(BaseModel):
    """Standard API response model for Azure Functions"""

    success: bool
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    correlation_id: Optional[str] = None


class APIRequest(BaseModel):
    """Generic API request model"""

    data: Dict[str, Any]
    user_id: Optional[str] = None
    request_id: Optional[str] = None


class HealthResponse(APIResponse):
    """Health check response"""

    version: str
    environment: str
    status: str


class QueueMessage(BaseModel):
    """Queue message model"""

    type: str = "default"
    payload: Dict[str, Any]
    priority: int = Field(default=1, ge=1, le=10)
    retry_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=datetime.now)


class KBIndexRequest(BaseModel):
    """Knowledge Base index request model"""

    workspace_id: str
    kb_id: str
    documents: list[Dict[str, Any]]
    user_id: Optional[str] = None


class KBQueryRequest(BaseModel):
    """Knowledge Base query request model"""

    workspace_id: str
    kb_id: str
    query: str
    top_k: int = Field(default=5, ge=1, le=100)
    user_id: Optional[str] = None


class KBQueryResponse(APIResponse):
    """Knowledge Base query response"""

    results: Optional[list[Dict[str, Any]]] = None
    total_results: int = 0
