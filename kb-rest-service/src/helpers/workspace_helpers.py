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

logger = get_logger(__name__)

# Import shared workspace functions
import sys
from pathlib import Path

# Add shared to path (parent's parent's parent / shared)
shared_path = Path(__file__).parent.parent.parent.parent / "shared"
if str(shared_path) not in sys.path:
    sys.path.insert(0, str(shared_path))

from workspace_helpers import build_storage_paths as _build_storage_paths


async def get_workspace_storage_paths(workspace_id: int) -> Optional[Dict[str, str]]:
    """
    Get storage paths for a workspace by ID.

    Fetches workspace from database and builds correct storage paths.
    For KG workspaces, also fetches industry, sub_industry, and kb details.

    Args:
        workspace_id: Workspace ID

    Returns:
        Dict with container, upload_path, domain, kb_name, is_kg
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

            # Query for workspace-industry-kb mapping with joins
            # Try to fetch for all workspace types, not just KG
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
            mapping_row = mapping_result.first()

            if mapping_row:
                industry_name = mapping_row.industry_name or "general"
                sub_industry_name = mapping_row.subindustry_name or "general"
                kb_title = mapping_row.title or ""
                logger.debug(
                    "Fetched workspace metadata",
                    workspace_id=workspace_id,
                    industry=industry_name,
                    sub_industry=sub_industry_name,
                    kb=kb_title
                )
            else:
                logger.warning(
                    "No industry mapping found for workspace, using defaults",
                    workspace_id=workspace_id,
                    defaults={"industry": industry_name, "sub_industry": sub_industry_name}
                )

            # Build paths using shared function
            kg_container = settings.storage.AZURE_BLOB_STORAGE_CONTAINER_NAME or settings.storage.STORAGE_CONTAINER_NAME
            workspace_container = settings.storage.LOCAL_STORAGE_CONTAINER or settings.storage.STORAGE_CONTAINER_NAME

            # Keep legacy Azure workspace container override if provided.
            if getattr(settings.azure, "WORKSPACE_CONTAINER_NAME", None):
                workspace_container = settings.azure.WORKSPACE_CONTAINER_NAME

            paths = _build_storage_paths(
                workspace_id=workspace.workspace_id,
                industry=industry_name,
                sub_industry=sub_industry_name,
                knowledge_base=kb_title,
                workspace_type=workspace.keywords,
                kg_container=kg_container,
                workspace_container=workspace_container,
            )

            logger.info(
                "Built workspace storage paths",
                workspace_id=workspace_id,
                **paths
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
