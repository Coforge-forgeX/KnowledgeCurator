"""Shared LightRAG helpers for KB and indexer services."""

from .builders import (
    build_azure_openai_chat_completion_func,
    build_azure_openai_embedding_func,
    RateLimitError,
)

__all__ = [
    "build_azure_openai_chat_completion_func",
    "build_azure_openai_embedding_func",
    "RateLimitError",
]
