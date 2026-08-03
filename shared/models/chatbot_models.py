"""
Chatbot Pydantic Models

Data models for chatbot operations following Pydantic best practices.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator


class ChatMessage(BaseModel):
    """Individual chat message in conversation history"""

    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")
    task_ids: List[int] = Field(default_factory=list, description="Associated task IDs")
    sources: List[Dict[str, str]] = Field(default_factory=list, description="Source documents with download URLs")

    @validator("role")
    def validate_role(cls, v):
        if v not in ["user", "assistant", "system"]:
            raise ValueError("Role must be 'user', 'assistant', or 'system'")
        return v


class ChatContext(BaseModel):
    """Session context for chatbot"""

    session_id: str = Field(..., description="Unique session identifier")
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list, description="Conversation history")
    pending_confirmation: Optional[Dict[str, Any]] = Field(None, description="Pending action confirmation")
    last_intent: Optional[str] = Field(None, description="Last detected user intent")

    class Config:
        """Pydantic config"""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ChatRequest(BaseModel):
    """Request payload for chatbot endpoint"""

    workspace_id: int = Field(..., description="Workspace identifier", gt=0)
    user_id: int = Field(..., description="User identifier", gt=0)
    role_id: int = Field(..., description="User role identifier", gt=0)
    agent_id: int = Field(..., description="LLM agent identifier", gt=0)
    session_id: str = Field(..., description="Session identifier")
    user_message: str = Field(..., min_length=1, description="User message")
    industry: str = Field(..., description="Domain/Industry name")
    sub_industry: str = Field(..., description="Knowledge base/Sub-industry name")
    mode: str = Field(default="SEARCH", description="Operation mode: SEARCH, UPDATE, QUERY")
    knowledge_bases: Optional[List[str]] = Field(None, description="List of knowledge base suffixes to query")
    file_names: Optional[List[str]] = Field(None, description="File names for upload")
    file_contents: Optional[List[str]] = Field(None, description="File contents (base64 encoded)")

    @validator("mode")
    def validate_mode(cls, v):
        if v.upper() not in ["SEARCH", "UPDATE", "QUERY"]:
            raise ValueError("Mode must be SEARCH, UPDATE, or QUERY")
        return v.upper()

    @validator("file_names", "file_contents")
    def validate_file_lists(cls, v, values):
        """Ensure file_names and file_contents have same length if both provided"""
        if v is not None and "file_names" in values and values["file_names"] is not None:
            if len(v) != len(values["file_names"]):
                raise ValueError("file_names and file_contents must have same length")
        return v


class ChatResponse(BaseModel):
    """Response payload for chatbot endpoint"""

    response: str = Field(..., description="Assistant response text")
    sources: List[Dict[str, str]] = Field(default_factory=list, description="Source documents with download URLs")
    task_ids: List[int] = Field(default_factory=list, description="Associated task IDs (for upload/indexing)")
    session_id: str = Field(..., description="Session identifier")

    class Config:
        """Pydantic config"""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ConversationSummary(BaseModel):
    """Summary of a conversation session"""

    session_id: str = Field(..., description="Session identifier")
    title: str = Field(..., description="Conversation title")
    created_at: datetime = Field(..., description="Session creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    message_count: int = Field(..., ge=0, description="Number of messages in session")

    class Config:
        """Pydantic config"""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SessionRenameRequest(BaseModel):
    """Request to rename a session"""

    workspace_id: int = Field(..., description="Workspace identifier", gt=0)
    user_id: int = Field(..., description="User identifier", gt=0)
    session_id: str = Field(..., description="Session identifier")
    title: str = Field(..., min_length=1, max_length=200, description="New session title")


class SessionDeleteRequest(BaseModel):
    """Request to delete a session"""

    workspace_id: int = Field(..., description="Workspace identifier", gt=0)
    user_id: int = Field(..., description="User identifier", gt=0)
    session_id: str = Field(..., description="Session identifier to delete")


class ConversationHistoryRequest(BaseModel):
    """Request to get conversation history"""

    workspace_id: int = Field(..., description="Workspace identifier", gt=0)
    user_id: int = Field(..., description="User identifier", gt=0)
    limit: Optional[int] = Field(None, ge=1, le=100, description="Max number of sessions to return")
