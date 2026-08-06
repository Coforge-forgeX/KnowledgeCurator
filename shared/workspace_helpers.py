"""
Shared workspace helper functions

Provides common utilities for workspace path management across services.
"""
from typing import Dict, Optional


def build_storage_paths(
    workspace_id: int,
    industry: str,
    sub_industry: str,
    knowledge_base: str,
    workspace_type: str,
    kg_container: str,
    workspace_container: str,
) -> Dict[str, str]:
    """
    Build storage paths for a workspace based on its type.

    Args:
        workspace_id: Workspace ID
        industry: Industry name (for KG workspaces)
        sub_industry: Sub-industry name (for KG workspaces)
        knowledge_base: KB title/name (for KG workspaces)
        workspace_type: Workspace type/keywords (e.g., "KG", "WORKSPACE")
        kg_container: Container name for KG workspaces
        workspace_container: Container name for regular workspaces

    Returns:
        Dict with:
            - container: Storage container name
            - upload_path: Base path for uploads
            - domain: Domain identifier
            - kb_name: Knowledge base name
            - is_kg: Boolean indicating KG workspace
    """
    is_kg = workspace_type == "KG"

    if is_kg:
        # KG workspace: use industry/sub_industry/kb structure
        # Path format: {industry}/{sub_industry}/{knowledge_base}
        upload_path = f"{industry}/{sub_industry}/{knowledge_base}".rstrip("/")

        return {
            "container": kg_container,
            "upload_path": upload_path,
            "domain": sub_industry,
            "kb_name": knowledge_base,
            "is_kg": True,
        }
    else:
        # Regular workspace: use workspace_id-based path
        # Path format: workspace_{workspace_id}
        upload_path = f"workspace_{workspace_id}"

        return {
            "container": workspace_container,
            "upload_path": upload_path,
            "domain": f"workspace_{workspace_id}",
            "kb_name": "",
            "is_kg": False,
        }
