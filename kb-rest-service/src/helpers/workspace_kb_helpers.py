"""
Workspace-Knowledge Base Helper Functions

Provides utilities for managing the relationship between workspaces and knowledge bases,
including determining workspace types and fetching linked KBs.
"""
from typing import List, Optional
from sqlalchemy import select

from src.core.database import Workspace, WorkspaceIndustryIntentMap, get_async_session
from src.core.logging import get_logger

logger = get_logger(__name__)


async def get_workspace_kb_ids(workspace_id: int) -> List[int]:
    """
    Get all knowledge base IDs linked to a workspace.

    Args:
        workspace_id: Workspace identifier

    Returns:
        List of KB IDs (empty list if none found)
    """
    try:
        async with get_async_session() as session:
            stmt = select(WorkspaceIndustryIntentMap.kb_id).where(
                WorkspaceIndustryIntentMap.workspace_id == workspace_id,
                WorkspaceIndustryIntentMap.is_active == True,
                WorkspaceIndustryIntentMap.kb_id.isnot(None)
            ).distinct()

            result = await session.execute(stmt)
            kb_ids = [row[0] for row in result.fetchall()]

            logger.debug(
                "Fetched KB IDs for workspace",
                workspace_id=workspace_id,
                kb_count=len(kb_ids),
            )

            return kb_ids

    except Exception as e:
        logger.error(
            "Failed to fetch KB IDs for workspace",
            error=e,
            workspace_id=workspace_id,
        )
        return []


async def get_workspace_type(workspace_id: int) -> Optional[str]:
    """
    Get workspace type from keywords field.

    Args:
        workspace_id: Workspace identifier

    Returns:
        Workspace type string (e.g., "Knowledge Graph", "Demo", "Trial", "Product")
        or None if workspace not found
    """
    try:
        async with get_async_session() as session:
            stmt = select(Workspace.keywords).where(
                Workspace.workspace_id == workspace_id,
                Workspace.is_active == True
            )

            result = await session.execute(stmt)
            row = result.first()

            if row:
                workspace_type = row[0]
                logger.debug(
                    "Fetched workspace type",
                    workspace_id=workspace_id,
                    workspace_type=workspace_type,
                )
                return workspace_type

            logger.warning("Workspace not found", workspace_id=workspace_id)
            return None

    except Exception as e:
        logger.error(
            "Failed to fetch workspace type",
            error=e,
            workspace_id=workspace_id,
        )
        return None


async def is_kg_workspace(workspace_id: int) -> bool:
    """
    Check if a workspace is a Knowledge Graph (KG) workspace.

    Args:
        workspace_id: Workspace identifier

    Returns:
        True if workspace is KG type, False otherwise
    """
    workspace_type = await get_workspace_type(workspace_id)

    if workspace_type is None:
        logger.warning(
            "Could not determine workspace type, defaulting to non-KG",
            workspace_id=workspace_id,
        )
        return False

    logger.debug(
        "Workspace KG check",
        workspace_id=workspace_id,
        workspace_type=workspace_type,
    )

    return workspace_type.lower() == "kg"


async def get_kb_id_for_upload(workspace_id: int) -> Optional[int]:
    """
    Get KB ID to associate with an uploaded document.

    Logic:
    - If workspace is KG type: return the KB ID (should be exactly one)
    - If workspace is Non-KG type: return None (document stays workspace-isolated)

    Args:
        workspace_id: Workspace identifier

    Returns:
        KB ID if KG workspace, None otherwise
    """
    try:
        # Check if workspace is KG type
        if not await is_kg_workspace(workspace_id):
            logger.debug(
                "Non-KG workspace - document will not be shared",
                workspace_id=workspace_id,
            )
            return None

        # Get KB ID for KG workspace
        kb_ids = await get_workspace_kb_ids(workspace_id)

        if not kb_ids:
            logger.warning(
                "KG workspace has no KB linked",
                workspace_id=workspace_id,
            )
            return None

        if len(kb_ids) > 1:
            logger.warning(
                "KG workspace has multiple KBs - using first",
                workspace_id=workspace_id,
                kb_ids=kb_ids,
            )

        kb_id = kb_ids[0]
        logger.info(
            "KG workspace - document will be shared via KB",
            workspace_id=workspace_id,
            kb_id=kb_id,
        )

        return kb_id

    except Exception as e:
        logger.error(
            "Failed to get KB ID for upload",
            error=e,
            workspace_id=workspace_id,
        )
        return None
