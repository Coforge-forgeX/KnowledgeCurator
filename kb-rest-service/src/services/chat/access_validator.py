"""
Single-pass access validation for the chat subsystem.

Design goal: validate the user's workspace membership/role AND resolve the
DB-sourced domain/kb_name/is_kg exactly once per request, concurrently,
instead of the pattern used by query_rag/upload_and_index/message_gpt today
where each downstream call independently re-validates the same facts.
"""
import asyncio
from typing import Optional

from src.core.database import UserMap, get_async_session
from src.core.exceptions import AuthorizationException, ValidationException
from src.core.logging import get_logger
from src.helpers.workspace_helpers import get_workspace_storage_paths
from sqlalchemy import select

from .models import AccessContext, WorkspaceMembership

logger = get_logger(__name__)


class ChatAccessValidator:
    """Resolves an :class:`AccessContext` for a (user, workspace) pair in one shot."""

    async def validate(self, user_id: int, workspace_id: int) -> AccessContext:
        user_map_row, storage_paths = await asyncio.gather(
            self._load_user_map(user_id, workspace_id),
            get_workspace_storage_paths(workspace_id),
        )

        if user_map_row is None:
            logger.warning(
                "User not authorized for workspace",
                user_id=user_id,
                workspace_id=workspace_id,
            )
            raise AuthorizationException(
                message=f"You are not authorized to access workspace {workspace_id}"
            )

        if not storage_paths:
            logger.error(
                "Failed to retrieve workspace storage paths",
                workspace_id=workspace_id,
            )
            raise ValidationException(
                message=f"Failed to retrieve workspace configuration for workspace {workspace_id}"
            )

        return AccessContext(
            user_id=user_id,
            workspace_id=workspace_id,
            role_id=user_map_row.role_id,
            can_curate_kb=bool(getattr(user_map_row, "can_curate_kb", False)),
            domain=storage_paths.get("domain", ""),
            kb_name=storage_paths.get("kb_name", ""),
            all_kb_titles=list(storage_paths.get("all_kb_titles", []) or []),
            is_kg=bool(storage_paths.get("is_kg")),
        )

    async def validate_membership(self, user_id: int, workspace_id: int) -> WorkspaceMembership:
        """
        Assert the user is an active member of the workspace, nothing more.

        Used by the session-management endpoints, which need authorization but
        not the workspace's domain/kb_name — see :class:`WorkspaceMembership`.
        """
        user_map_row = await self._load_user_map(user_id, workspace_id)

        if user_map_row is None:
            logger.warning(
                "User not authorized for workspace",
                user_id=user_id,
                workspace_id=workspace_id,
            )
            raise AuthorizationException(
                message=f"You are not authorized to access workspace {workspace_id}"
            )

        return WorkspaceMembership(
            user_id=user_id,
            workspace_id=workspace_id,
            role_id=user_map_row.role_id,
            can_curate_kb=bool(getattr(user_map_row, "can_curate_kb", False)),
        )

    @staticmethod
    async def _load_user_map(user_id: int, workspace_id: int) -> Optional[UserMap]:
        async with get_async_session() as session:
            stmt = select(UserMap).where(
                UserMap.user_id == user_id,
                UserMap.workspace_id == workspace_id,
                UserMap.is_active == True,  # noqa: E712
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none()


_validator_instance: Optional[ChatAccessValidator] = None


def get_chat_access_validator() -> ChatAccessValidator:
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = ChatAccessValidator()
    return _validator_instance
