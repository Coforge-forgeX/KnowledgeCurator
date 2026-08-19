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
    # Query models
    "QueryRequest",
    "QueryResponse",
]
