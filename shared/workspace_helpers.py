"""Shared workspace helper functions for path and workspace naming."""
from typing import Dict, Optional


DIGIT_MAP = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}


def _clean_path_part(value: Optional[str]) -> str:
    """Normalize a path segment by trimming and removing duplicate separators."""
    if not value:
        return ""
    return str(value).strip().strip("/")


def workspace_id_to_alpha(workspace_id: Optional[int]) -> str:
    """Convert a workspace id to alpha words. Example: 12 -> onetwo."""
    if workspace_id is None:
        return ""

    parts = []
    for ch in str(workspace_id):
        if ch.isdigit():
            parts.append(DIGIT_MAP[ch])
        elif ch.isalpha():
            parts.append(ch)
    return "".join(parts)


def get_workspace_identifier(
    workspace_id: Optional[int],
    domain: Optional[str] = None,
    kb_name: Optional[str] = None,
) -> str:
    """Build alpha-only workspace identifier used by LightRAG workspace selection."""
    workspace_alpha = workspace_id_to_alpha(workspace_id)
    combined = f"{domain or ''}{kb_name or ''}{workspace_alpha}"
    return "".join(ch for ch in combined if ch.isalpha())


def get_workspace_working_dir(
    workspace_id: Optional[int],
    base_dir: str,
    domain: Optional[str] = None,
    kb_name: Optional[str] = None,
) -> str:
    """Build filesystem working dir from workspace identifier."""
    identifier = get_workspace_identifier(workspace_id, domain=domain, kb_name=kb_name)
    if not identifier:
        identifier = workspace_id_to_alpha(workspace_id)
    return f"{base_dir.rstrip('/')}" + f"/{identifier}"


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
    workspace_kind = (workspace_type or "").strip().lower()
    is_kg = workspace_kind == "kg"

    industry_part = _clean_path_part(industry)
    sub_industry_part = _clean_path_part(sub_industry)
    kb_part = _clean_path_part(knowledge_base)

    workspace_numeric = str(workspace_id)
    workspace_alpha = workspace_id_to_alpha(workspace_id)

    if is_kg:
        # KG workspace:
        # upload_path -> {industry}/{subindustry}/{kb_name}
        # domain -> {industry}
        # kb_name -> {subindustry}/{kb_name}
        upload_path = "/".join(
            part
            for part in [industry_part, sub_industry_part, kb_part]
            if part
        )
        kb_name_value = "/".join(
            part
            for part in [sub_industry_part, kb_part]
            if part
        )

        return {
            "container": kg_container,
            "upload_path": upload_path,
            "domain": industry_part,
            "kb_name": kb_name_value,
            "is_kg": True,
        }

    # Non-KG workspace:
    # upload_path -> {industry}/{subindustry}/{workspace_id_numeric}
    # domain -> {industry}
    # kb_name -> {subindustry}/{workspace_id_alpha}
    upload_path = "/".join(
        part
        for part in [industry_part, sub_industry_part, workspace_numeric]
        if part
    )
    kb_name_value = "/".join(
        part
        for part in [sub_industry_part, workspace_alpha]
        if part
    )

    return {
        "container": workspace_container,
        "upload_path": upload_path,
        "domain": industry_part,
        "kb_name": kb_name_value,
        "is_kg": False,
    }
