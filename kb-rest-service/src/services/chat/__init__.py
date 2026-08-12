"""
Chat / message_gpt subsystem.

SOLID-oriented package that powers the `message_gpt` endpoint:
- access_validator: single-pass user + workspace + permission validation
- llm_provider: common_adapters-backed LLM completion used by intent detection & summarization
- context_middleware: pluggable context post-processing (summarization via common_adapters)
- file_context_extractor: extracts uploaded-file text for in-context (non-indexed) use
- intent_service: wires the existing intent_detection module for this service
- handlers: per-mode (search/update) message handlers
- orchestrator: top-level entry point used by the REST endpoint; persists via the
  same MongoDBService (chatbot database, "sessions"/"messages" collections) used
  by start_conversation/get_conversation_history/load_conversation/
  rename_conversation/delete_conversation, and runs handlers as cancellable
  tasks (common_adapters.cancel_convesation) for cancel_chat_message.
"""
from .access_validator import get_chat_access_validator
from .orchestrator import get_chat_orchestrator

__all__ = ["get_chat_access_validator", "get_chat_orchestrator"]
