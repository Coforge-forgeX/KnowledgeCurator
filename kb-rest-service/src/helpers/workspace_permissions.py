"""Workspace permission helper utilities."""

from sqlalchemy import select

from src.core.database import UserMap, get_async_session
from src.core.exceptions import AuthorizationException

WORKSPACE_ADMIN_ROLE_ID = 3


async def require_workspace_admin_curator(
    user_id: int,
    workspace_id: int,
    action_description: str = "perform this action",
) -> UserMap:
    """
    Require active workspace membership with admin role and can_curate_kb=True.

    Returns the active UserMap row when authorized.
    """
    async with get_async_session() as session:
        stmt = select(UserMap).where(
            UserMap.user_id == user_id,
            UserMap.workspace_id == workspace_id,
            UserMap.is_active == True,
        )
        result = await session.execute(stmt)
        user_map = result.scalar_one_or_none()

        if not user_map:
            raise AuthorizationException(
                message=f"You are not authorized to access workspace {workspace_id}"
            )

        # is_admin = bool(getattr(user_map, "role_id", None) == WORKSPACE_ADMIN_ROLE_ID)
        can_curate = bool(getattr(user_map, "can_curate_kb", False))

        if not can_curate:
            raise AuthorizationException(
                message=(
                    "Only workspace admin with can_curate_kb=true can "
                    f"{action_description}"
                )
            )

        return user_map