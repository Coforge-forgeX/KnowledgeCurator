"""
KB REST Service Models

Pydantic models for REST API request/response validation.
"""

from .chat_models import (
    ChatRequest,
    ChatResponse,
    ChatMessage,
    ChatSource,
    ConversationHistoryRequest,
    LoadConversationRequest,
    SessionRenameRequest,
    SessionDeleteRequest,
    StartConversationRequest,
)

from .upload_models import (
    UploadRequest,
    UploadResponse,
    FileMetadata,
)

from .query_models import (
    QueryRequest,
    QueryResponse,
)

__all__ = [
    # Chat models
    "ChatRequest",
    "ChatResponse",
    "ChatMessage",
    "ChatSource",
    "ConversationHistoryRequest",
    "LoadConversationRequest",
    "SessionRenameRequest",
    "SessionDeleteRequest",
    "StartConversationRequest",
    # Upload models
    "UploadRequest",
    "UploadResponse",
    "FileMetadata",
    # Query models
    "QueryRequest",
    "QueryResponse",
]
