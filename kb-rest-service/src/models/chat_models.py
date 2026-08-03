"""
Chat API Models

Pydantic models for chatbot REST endpoints.
"""

from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, validator


class ChatMessage(BaseModel):
    """Individual chat message"""

    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)
    sources: List[Dict[str, str]] = Field(default_factory=list)

    @validator("role")
    def validate_role(cls, v):
        if v not in ["user", "assistant", "system"]:
            raise ValueError("Role must be 'user', 'assistant', or 'system'")
        return v


class ChatRequest(BaseModel):
    """POST /chat request"""

    workspace_id: int = Field(..., gt=0)
    user_id: int = Field(..., gt=0)
    role_id: int = Field(..., gt=0)
    agent_id: int = Field(..., gt=0)
    session_id: str = Field(...)
    user_message: str = Field(..., min_length=1)
    industry: str = Field(...)
    sub_industry: str = Field(...)
    mode: str = Field(default="SEARCH")
    knowledge_bases: Optional[List[str]] = None
    file_names: Optional[List[str]] = None
    file_contents: Optional[List[str]] = None

    @validator("mode")
    def validate_mode(cls, v):
        if v.upper() not in ["SEARCH", "UPDATE", "QUERY"]:
            raise ValueError("Mode must be SEARCH, UPDATE, or QUERY")
        return v.upper()


class ChatResponse(BaseModel):
    """POST /chat response"""

    response: str
    sources: List[Dict[str, str]] = Field(default_factory=list)
    task_ids: List[int] = Field(default_factory=list)
    session_id: str


class ConversationHistoryRequest(BaseModel):
    """GET /chat/history request"""

    workspace_id: int = Field(..., gt=0)
    user_id: int = Field(..., gt=0)
    limit: Optional[int] = Field(None, ge=1, le=100)


class SessionRenameRequest(BaseModel):
    """POST /chat/session/rename request"""

    workspace_id: int = Field(..., gt=0)
    user_id: int = Field(..., gt=0)
    session_id: str
    title: str = Field(..., min_length=1, max_length=200)


class SessionDeleteRequest(BaseModel):
    """POST /chat/session/delete request"""

    workspace_id: int = Field(..., gt=0)
    user_id: int = Field(..., gt=0)
    session_id: str
