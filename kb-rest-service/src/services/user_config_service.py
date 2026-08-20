"""
User Configuration Service

Provides high-level async operations for user configuration management:
- Fetching user configuration from MongoDB
- Updating user configuration in MongoDB
- Optional SharePoint config adapter pre-processing
- Real-time Redis cache update for active user sessions
"""

import json
import os
from typing import Any, Dict, List, Optional, Union

from src.core.config import settings
from src.core.logging import get_logger
from src.core.redis import get_redis_client, is_redis_available
from src.services.mongodb_service import get_mongodb_service

logger = get_logger(__name__)


class UserConfigService:
    """Service layer managing user configuration persistence and caching."""

    async def get_config(
        self,
        workspace_id: Union[int, str],
        user_id: Union[int, str],
        fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve configuration fields for an authenticated user and workspace.

        Args:
            workspace_id: Unique workspace identifier
            user_id: Authenticated user identifier (extracted from JWT)
            fields: Optional list of specific configuration fields to retrieve

        Returns:
            Dict containing status and configuration key-values
        """
        mongo_service = get_mongodb_service()
        await mongo_service.initialize()

        result = await mongo_service.get_user_config(
            workspace_id=workspace_id,
            user_id=user_id,
            fields=fields,
        )

        return {
            "status": "success",
            **result,
        }

    async def update_config(
        self,
        workspace_id: Union[int, str],
        user_id: Union[int, str],
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Update user configuration in MongoDB and synchronize active session caches in Redis.

        Args:
            workspace_id: Unique workspace identifier
            user_id: Authenticated user identifier (extracted from JWT)
            data: Key-value configuration dictionary to update

        Returns:
            Dict containing operation response and updated status
        """
        if not data:
            return {"status": "success", "message": "No data provided for update"}

        # Attempt to run sharepoint_update_config adapter if available
        try:
            from common_adapters.sharepoint import sharepoint_update_config
            processed_data = sharepoint_update_config(data)
            if processed_data:
                data = processed_data
        except ImportError:
            logger.debug("common_adapters.sharepoint not installed, proceeding with raw config data")
        except Exception as ex:
            logger.warning(f"sharepoint_update_config hook error: {ex}")

        mongo_service = get_mongodb_service()
        await mongo_service.initialize()

        # Update in MongoDB
        response = await mongo_service.set_user_config(
            workspace_id=workspace_id,
            user_id=user_id,
            config=data,
        )

        # Retrieve updated full configuration doc
        updated_user_config = await mongo_service.get_user_config(
            workspace_id=workspace_id,
            user_id=user_id,
        )

        # Synchronize active sessions in Redis if Redis is connected
        if is_redis_available():
            redis_client = get_redis_client()
            if redis_client:
                try:
                    ttl_seconds = float(
                        os.getenv(
                            "REDIS_EXPIRY_SECONDS",
                            str(settings.cache.CONVERSATION_HISTORY_CACHE_TTL),
                        )
                    )
                    recent_sessions = await mongo_service.get_recent_sessions_by_ttl(
                        workspace_id=workspace_id,
                        user_id=user_id,
                        ttl_seconds=ttl_seconds,
                    )
                    config_json = json.dumps(updated_user_config)
                    for sess_id in recent_sessions:
                        cache_key = f"{sess_id}-config"
                        redis_client.setex(cache_key, int(ttl_seconds), config_json)
                except Exception as cache_err:
                    logger.warning(f"Failed to update session configs in Redis: {cache_err}")

        return response


_service_instance: Optional[UserConfigService] = None


def get_user_config_service() -> UserConfigService:
    """Get singleton instance of UserConfigService."""
    global _service_instance
    if _service_instance is None:
        _service_instance = UserConfigService()
    return _service_instance
 