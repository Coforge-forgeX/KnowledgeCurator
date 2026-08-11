"""
Handler Registry - Type-safe handler module mapping with lazy loading

This registry provides:
- Type-safe references (no string-based module paths in endpoint definitions)
- Lazy imports (preserves cold-start optimization - handlers load only when called)
- Validation on first use (catches missing/misconfigured handlers early)
- Single source of truth for all handler module paths

Usage:
    from src.registry import get_handler

    handler_module = get_handler("upload_and_index")
    result = await handler_module.main(request, context)
"""

import importlib
import os
import sys
from types import ModuleType
from typing import Dict


def _bootstrap_import_paths() -> None:
    """Ensure monorepo import paths are available in non-entrypoint contexts.

    `main.py` and `function_app.py` already set path precedence before importing
    handlers. Direct imports of `src.registry` (for validation scripts/tests)
    need the same setup so shared adapters resolve consistently.
    """
    src_dir = os.path.dirname(os.path.abspath(__file__))
    service_dir = os.path.dirname(src_dir)
    services_dir = os.path.dirname(service_dir)

    # Keep services before src so `shared.*` resolves to services/shared.
    sys.path = [p for p in sys.path if p not in {services_dir, service_dir}]
    sys.path.insert(0, service_dir)
    sys.path.insert(0, services_dir)


_bootstrap_import_paths()

# ============================================================================
# Handler Module Path Registry (String Mapping)
# ============================================================================
# These strings are validated on first access - typos fail immediately, not silently

HANDLER_MODULE_PATHS: Dict[str, str] = {
    # Document Upload & Indexing
    "upload_and_index": "src.functions.api.upload_and_index",
    "index_workspace_files": "src.functions.api.index_workspace_files",
    "file_tasks_status": "src.functions.api.file_tasks_status",
    "workspace_documents": "src.functions.api.workspace_documents_grouped",
    "delete_files_by_id": "src.functions.api.delete_files_by_id",
    # Knowledge Base Query
    "query_rag": "src.functions.api.query_rag",  # Optimized RAG query handler
    "query_source_download_url": "src.functions.api.query_source_download_url",
    "kb_index": "src.functions.api.kb_index",
    # Knowledge Graph
    "get_knowledge_graph": "src.functions.api.get_knowledge_graph",
    "fetch_graph": "src.functions.api.fetch_graph",
    "mutate_knowledge_graph": "src.functions.api.mutate_knowledge_graph",

    # Chat / Conversation session management
    "chat_start_conversation": "src.functions.api.start_conversation",
    "chat_get_conversation_history": "src.functions.api.get_conversation_history",
    "chat_load_conversation": "src.functions.api.load_conversation",
    "chat_rename_conversation": "src.functions.api.rename_conversation",
    "chat_delete_conversation": "src.functions.api.delete_conversation_history",
}

# Cache for loaded handlers (lazy loading + memoization)
_handler_cache: Dict[str, ModuleType] = {}


# ============================================================================
# Registry Access Functions
# ============================================================================


def get_handler(handler_name: str) -> ModuleType:
    """
    Get handler module by name with lazy loading and caching.

    Args:
        handler_name: Name of the handler (e.g., "upload_and_index", "query_kb")

    Returns:
        Handler module with a `main` function

    Raises:
        ValueError: If handler_name is not found in registry
        ImportError: If handler module cannot be imported
        AttributeError: If handler module doesn't have a `main` function

    Example:
        >>> handler = get_handler("upload_and_index")
        >>> result = await handler.main(request, context)
    """
    # Check if handler name exists in registry
    if handler_name not in HANDLER_MODULE_PATHS:
        available_handlers = ", ".join(sorted(HANDLER_MODULE_PATHS.keys()))
        raise ValueError(
            f"Handler '{handler_name}' not found in registry. "
            f"Available handlers: {available_handlers}"
        )

    # Return cached handler if already loaded
    if handler_name in _handler_cache:
        return _handler_cache[handler_name]

    # Lazy import the handler module
    module_path = HANDLER_MODULE_PATHS[handler_name]
    try:
        handler_module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Failed to import handler '{handler_name}' from '{module_path}': {e}"
        ) from e

    # Validate that handler has a main function
    if not hasattr(handler_module, "main"):
        raise AttributeError(
            f"Handler '{handler_name}' (module: {module_path}) "
            f"is missing required 'main' function"
        )

    # Cache and return
    _handler_cache[handler_name] = handler_module
    return handler_module


def list_handlers() -> list[str]:
    """
    List all registered handler names.

    Returns:
        Sorted list of handler names

    Example:
        >>> handlers = list_handlers()
        >>> print(f"Total handlers: {len(handlers)}")
    """
    return sorted(HANDLER_MODULE_PATHS.keys())


def validate_all_handlers() -> None:
    """
    Eagerly validate all handlers by attempting to load them.

    Useful for startup checks or testing to ensure all handlers are valid.
    In production, handlers are validated lazily on first use.

    Raises:
        ImportError: If any handler module cannot be imported
        AttributeError: If any handler is missing the `main` function
    """
    errors = []
    for handler_name in HANDLER_MODULE_PATHS.keys():
        try:
            get_handler(handler_name)
        except Exception as e:
            errors.append(f"  - {handler_name}: {e}")

    if errors:
        raise RuntimeError(
            f"Handler validation failed for {len(errors)} handler(s):\n"
            + "\n".join(errors)
        )
