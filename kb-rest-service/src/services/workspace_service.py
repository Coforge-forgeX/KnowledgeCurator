"""
Workspace Service - Workspace Configuration & Authorization

Handles workspace metadata retrieval and user authorization.
Follows Single Responsibility Principle.
"""
from typing import Optional, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_async_session
from src.core.exceptions import AuthorizationException, ValidationException
from src.core.logging import get_logger

logger = get_logger(__name__)


# Database models (assuming these exist in your schema)
# If not, these need to be added to src/core/database.py
try:
    from src.core.database import Workspace, UserMap
except ImportError:
    # Fallback - define minimal models here
    from sqlalchemy import Boolean, Column, Integer, String
    from sqlalchemy.ext.declarative import declarative_base

    Base = declarative_base()

    class Workspace(Base):
        __tablename__ = "workspaces"

        workspace_id = Column(Integer, primary_key=True)
        domain = Column(String, nullable=False)
        kb_name = Column(String, nullable=False)
        is_active = Column(Boolean, default=True)
        keywords = Column(String)  # Workspace type: KG, DM, etc.

    class UserMap(Base):
        __tablename__ = "user_map"

        id = Column(Integer, primary_key=True)
        user_id = Column(Integer, nullable=False)
        workspace_id = Column(Integer, nullable=False)
        role_id = Column(Integer, nullable=False)
        is_active = Column(Boolean, default=True)


class WorkspaceConfig:
    """Workspace configuration data"""
    def __init__(
        self,
        workspace_id: int,
        domain: str,
        kb_name: str,
        workspace_type: Optional[str] = None
    ):
        self.workspace_id = workspace_id
        self.domain = domain
        self.kb_name = kb_name
        self.workspace_type = workspace_type


class WorkspaceService:
    """
    Service for workspace operations.

    Responsibilities:
    - Fetch workspace configuration from database
    - Validate user-workspace membership
    - Check workspace permissions
    """

    async def get_workspace_config(
        self,
        workspace_id: int
    ) -> WorkspaceConfig:
        """
        Get workspace configuration from database.

        Args:
            workspace_id: Workspace identifier

        Returns:
            WorkspaceConfig with domain, kb_name, etc.

        Raises:
            ValidationException: If workspace not found or inactive
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
                    logger.warning(
                        f"Workspace not found or inactive",
                        workspace_id=workspace_id
                    )
                    raise ValidationException(
                        message=f"Workspace {workspace_id} not found or is inactive"
                    )

                # Extract workspace type from keywords
                workspace_type = None
                if workspace.keywords:
                    # Keywords format: "KG,DM,TR" etc.
                    workspace_type = workspace.keywords.split(',')[0].strip().upper()

                config = WorkspaceConfig(
                    workspace_id=workspace.workspace_id,
                    domain=workspace.domain,
                    kb_name=workspace.kb_name,
                    workspace_type=workspace_type
                )

                logger.info(
                    f"Retrieved workspace config",
                    workspace_id=workspace_id,
                    domain=config.domain,
                    kb_name=config.kb_name
                )

                return config

        except ValidationException:
            raise
        except Exception as e:
            logger.error(
                f"Failed to retrieve workspace config",
                error=e,
                workspace_id=workspace_id
            )
            raise ValidationException(
                message=f"Failed to retrieve workspace configuration: {str(e)}"
            )

    async def validate_user_workspace_access(
        self,
        user_id: int,
        workspace_id: int
    ) -> Tuple[bool, int]:
        """
        Validate that user has access to workspace.

        Args:
            user_id: User identifier
            workspace_id: Workspace identifier

        Returns:
            Tuple of (is_authorized: bool, role_id: int)

        Raises:
            AuthorizationException: If user not authorized
        """
        try:
            async with get_async_session() as session:
                stmt = select(UserMap).where(
                    UserMap.user_id == user_id,
                    UserMap.workspace_id == workspace_id,
                    UserMap.is_active == True
                )
                result = await session.execute(stmt)
                user_map = result.scalar_one_or_none()

                if not user_map:
                    logger.warning(
                        f"User not authorized for workspace",
                        user_id=user_id,
                        workspace_id=workspace_id
                    )
                    raise AuthorizationException(
                        message=f"You are not authorized to access workspace {workspace_id}"
                    )

                logger.info(
                    f"User authorized for workspace",
                    user_id=user_id,
                    workspace_id=workspace_id,
                    role_id=user_map.role_id
                )

                return True, user_map.role_id

        except AuthorizationException:
            raise
        except Exception as e:
            logger.error(
                f"Failed to validate user workspace access",
                error=e,
                user_id=user_id,
                workspace_id=workspace_id
            )
            raise AuthorizationException(
                message=f"Authorization check failed: {str(e)}"
            )

    async def get_user_role_in_workspace(
        self,
        user_id: int,
        workspace_id: int
    ) -> int:
        """
        Get user's role ID in a workspace.

        Args:
            user_id: User identifier
            workspace_id: Workspace identifier

        Returns:
            Role ID

        Raises:
            AuthorizationException: If user not in workspace
        """
        _, role_id = await self.validate_user_workspace_access(
            user_id,
            workspace_id
        )
        return role_id

    async def check_workspace_type(
        self,
        workspace_id: int,
        expected_type: str
    ) -> bool:
        """
        Check if workspace is of a specific type.

        Args:
            workspace_id: Workspace identifier
            expected_type: Expected type (e.g., "KG", "DM")

        Returns:
            True if workspace matches type
        """
        config = await self.get_workspace_config(workspace_id)
        return config.workspace_type == expected_type.upper()


# ============================================================================
# Singleton Instance
# ============================================================================

_workspace_service_instance: Optional[WorkspaceService] = None


def get_workspace_service() -> WorkspaceService:
    """Get or create singleton workspace service instance"""
    global _workspace_service_instance
    if _workspace_service_instance is None:
        _workspace_service_instance = WorkspaceService()
    return _workspace_service_instance
