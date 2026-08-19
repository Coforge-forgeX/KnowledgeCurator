"""
Chat API Models

Pydantic models for chatbot REST endpoints.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, validator, field_validator

from src.functions.api.upload_and_index.payloads import FileUpload


class Mode(str, Enum):
    SEARCH = "SEARCH"
    UPDATE = "UPDATE"

class ChatSource(BaseModel):
    """
    One cited document behind an assistant answer.

    Mirrors `query_rag`'s `SourceReferenceModel` on purpose — both endpoints run
    the same retrieval path, so a source must look the same whichever one
    returned it. In particular `citation` is OPTIONAL: sources are extracted
    from the retrieved chunks and only carry a citation label when the answer
    actually cited them, so an uncited-but-retrieved source has `None` here. It
    was previously typed as a bare `Dict[str, str]`, which rejected that `None`
    and failed the whole response after the turn had already been persisted.
    """

    file_id: str = Field(..., description="Opaque token for GET /api/v2/files/{file_id}/download")
    file_name: str = Field(default="", description="Display file name")
    citation: Optional[str] = Field(default=None, description="Citation label from the answer, if cited")

    model_config = {"extra": "allow"}


class ChatMessage(BaseModel):
    """Individual chat message"""

    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)
    sources: List[ChatSource] = Field(default_factory=list)

    @validator("role")
    def validate_role(cls, v):
        if v not in ["user", "assistant", "system"]:
            raise ValueError("Role must be 'user', 'assistant', or 'system'")
        return v


class ChatRequest(BaseModel):
    """
    POST /chat request.

    Deliberately minimal: everything that is server-authoritative is resolved
    per request instead of being trusted from the client —
    - `user_id` comes from the authenticated Bearer token,
    - `role_id` / `can_curate_kb` from UserMap,
    - `industry` / `sub_industry` (domain / kb_name) and the workspace's
      knowledge bases from `get_workspace_storage_paths`,
    all inside `ChatAccessValidator`, which the orchestrator already runs once
    per request. Accepting them in the payload only created a second, weaker
    source of truth.
    """

    workspace_id: int = Field(..., gt=0)
    agent_id: int = Field(default=1, gt=0)
    session_id: str = Field(...)
    user_message: str = Field(..., min_length=1)
    mode: Mode = Field(default=Mode.SEARCH.value)
    files: Optional[List[FileUpload]] = None
    # file_names: Optional[List[str]] = None
    # file_contents: Optional[List[str]] = None

    @field_validator("files")
    @classmethod
    def validate_files_limit(cls, v: List[FileUpload]) -> List[FileUpload]:
        """Limit number of files per request"""
        max_files = 10
        if len(v) > max_files:
            raise ValueError(f"Maximum {max_files} files allowed per request")
        return v
    
    @field_validator("files")
    @classmethod
    def validate_duplicate_filenames(cls, v: List[FileUpload]) -> List[FileUpload]:
        """Check for duplicate file names"""
        file_names = [f.file_name for f in v]
        if len(file_names) != len(set(file_names)):
            raise ValueError("Duplicate file names not allowed")
        return v


class ChatResponse(BaseModel):
    """POST /chat response"""

    response: str
    sources: List[ChatSource] = Field(default_factory=list)
    task_ids: List[int] = Field(default_factory=list)
    session_id: str


class ConversationHistoryRequest(BaseModel):
    """
    GET /chat/history request (query string).

    Paginated: `page` is 1-indexed and `page_size` is capped so a caller cannot
    ask for an unbounded scan. `limit` is accepted as a deprecated alias for
    `page_size` so existing clients keep working.
    """

    workspace_id: int = Field(..., gt=0)
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)
    limit: Optional[int] = Field(default=None, ge=1, le=100, description="Deprecated: use page_size")

    @property
    def effective_page_size(self) -> int:
        return self.limit or self.page_size


class LoadConversationRequest(BaseModel):
    """
    GET /chat/load request (query string).

    The message list is paginated. `order` decides which end of the transcript
    page 1 sits at:
    - `desc` (default): page 1 is the newest messages, and paging forward walks
      backwards in time — what a chat UI wants when opening a conversation.
    - `asc`: page 1 is the oldest messages.

    Either way the messages *inside* a page are ordered oldest-first, so a page
    renders top-to-bottom as it was written.
    """

    workspace_id: int = Field(..., gt=0)
    session_id: str = Field(..., min_length=1)
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=200)
    order: str = Field(default="desc", description="'desc' (newest page first) or 'asc'")

    @validator("session_id")
    def strip_session_id(cls, v):
        v = (v or "").strip()
        if not v:
            raise ValueError("session_id cannot be empty")
        return v

    @validator("order")
    def validate_order(cls, v):
        v = (v or "").strip().lower()
        if v not in ("asc", "desc"):
            raise ValueError("order must be 'asc' or 'desc'")
        return v

    @property
    def newest_first(self) -> bool:
        return self.order == "desc"


class SessionRenameRequest(BaseModel):
    """POST /chat/session/rename request"""

    workspace_id: int = Field(..., gt=0)
    session_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1, max_length=200)

    @validator("session_id", "title")
    def strip_required_text(cls, v):
        v = (v or "").strip()
        if not v:
            raise ValueError("value cannot be empty")
        return v


class SessionDeleteRequest(BaseModel):
    """DELETE /chat/session/delete request"""

    workspace_id: int = Field(..., gt=0)
    session_id: str = Field(..., min_length=1)

    @validator("session_id")
    def strip_session_id(cls, v):
        v = (v or "").strip()
        if not v:
            raise ValueError("session_id cannot be empty")
        return v


class CancelChatRequest(BaseModel):
    """
    POST /chat/cancel request — cancels an in-flight message_gpt call.

    `user_id` is taken from the Bearer token, matching ChatRequest.
    """

    workspace_id: int = Field(..., gt=0)
    session_id: str = Field(..., min_length=1)
    reason: str = Field(default="user_requested")

    @validator("session_id")
    def strip_session_id(cls, v):
        v = (v or "").strip()
        if not v:
            raise ValueError("session_id cannot be empty")
        return v


class CancelChatResponse(BaseModel):
    """POST /chat/cancel response"""

    cancelled: bool
    session_id: str
