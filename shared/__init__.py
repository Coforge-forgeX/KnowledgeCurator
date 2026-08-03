"""
Shared utilities for KB services.

Exports common workspace helpers and types.
"""

from .workspace_helpers import (
    workspace_id_to_alpha,
    get_workspace_identifier,
    get_workspace_working_dir,
    build_storage_paths,
)

__all__ = [
    "workspace_id_to_alpha",
    "get_workspace_identifier",
    "get_workspace_working_dir",
    "build_storage_paths",
]
