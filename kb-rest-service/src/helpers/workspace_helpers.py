"""
Workspace helpers for kb-rest-service

Wrapper around shared workspace_helpers module with database integration.
"""
from typing import Dict, Optional
from sqlalchemy import select

from src.core.database import (
    Workspace,
    WorkspaceIndustryIntentMap,
    Industry,
    SubIndustry,
    KnowledgeBase,
    get_async_session
)
from src.core.logging import get_logger
from src.core.config import settings
from shared.workspace_helpers import build_storage_paths as _build_storage_paths

logger = get_logger(__name__)


async def get_workspace_storage_paths(workspace_id: int) -> Optional[Dict[str, str]]:
    """
    Get storage paths for a workspace by ID.

    Fetches workspace from database and builds correct storage paths.
    For KG workspaces, also fetches industry, sub_industry, and kb details.
    For non-KG workspaces with multiple KBs, returns all KB titles for querying.

    Args:
        workspace_id: Workspace ID

    Returns:
        Dict with:
            - container: Storage container name
            - upload_path: Base path for uploads
            - domain: Domain identifier
            - kb_name: Primary knowledge base name (for indexing)
            - is_kg: Boolean indicating KG workspace
            - all_kb_titles: List of all KB titles (for querying across multiple KBs)
        or None if workspace not found
    """
    try:
        async with get_async_session() as session:
            stmt = select(Workspace).where(
                Workspace.workspace_id == workspace_id,
                Workspace.is_active == True
            )
            result = await session.execute(stmt)
            workspace = result.scalar_one_or_none()

            if not workspace:
                logger.warning("Workspace not found", workspace_id=workspace_id)
                return None

            # Initialize with default values
            industry_name = "general"
            sub_industry_name = "general"
            kb_title = ""
            all_kb_titles = []

            # Query for workspace-industry-kb mapping with joins
            # Fetch ALL mappings for non-KG workspaces with multiple KBs
            mapping_stmt = (
                select(
                    Industry.industry_name,
                    SubIndustry.subindustry_name,
                    KnowledgeBase.title
                )
                .select_from(WorkspaceIndustryIntentMap)
                .join(Industry, WorkspaceIndustryIntentMap.industry_id == Industry.industry_id)
                .join(SubIndustry, WorkspaceIndustryIntentMap.subindustry_id == SubIndustry.subindustry_id)
                .outerjoin(KnowledgeBase, WorkspaceIndustryIntentMap.kb_id == KnowledgeBase.id)
                .where(
                    WorkspaceIndustryIntentMap.workspace_id == workspace_id,
                    WorkspaceIndustryIntentMap.is_active == True
                )
            )
            mapping_result = await session.execute(mapping_stmt)
            mapping_rows = mapping_result.fetchall()

            if mapping_rows:
                # Use first row for primary paths (indexing)
                first_row = mapping_rows[0]
                industry_name = first_row.industry_name or "general"
                sub_industry_name = first_row.subindustry_name or "general"
                kb_title = first_row.title or ""

                # Collect all KB titles for multi-KB querying
                all_kb_titles = [row.title for row in mapping_rows if row.title]

                logger.debug(
                    "Fetched workspace metadata",
                    workspace_id=workspace_id,
                    industry=industry_name,
                    sub_industry=sub_industry_name,
                    kb=kb_title,
                    total_kbs=len(all_kb_titles)
                )
            else:
                logger.warning(
                    "No industry mapping found for workspace, using defaults",
                    workspace_id=workspace_id,
                    defaults={"industry": industry_name, "sub_industry": sub_industry_name}
                )

            # Build paths using shared function
            kg_container = settings.storage.STORAGE_CONTAINER_NAME
            workspace_container = settings.storage.WORKSPACE_CONTAINER_NAME

            paths = _build_storage_paths(
                workspace_id=workspace.workspace_id,
                industry=industry_name,
                sub_industry=sub_industry_name,
                knowledge_base=kb_title,
                workspace_type=workspace.keywords,
                kg_container=kg_container,
                workspace_container=workspace_container,
            )

            # Add list of all KB titles for multi-KB querying
            paths["all_kb_titles"] = all_kb_titles

            logger.info(
                "Built workspace storage paths",
                workspace_id=workspace_id,
                kb_count=len(all_kb_titles),
                **{k: v for k, v in paths.items() if k != "all_kb_titles"}
            )

            return paths

    except Exception as e:
        logger.error(
            "Failed to get workspace storage paths",
            error=e,
            workspace_id=workspace_id,
            exc_info=True
        )
        return None
