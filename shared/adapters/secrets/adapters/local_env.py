"""Local environment variables adapter for secrets"""

import os
import logging
from typing import Optional

from ..protocols import SecretsAdapter

logger = logging.getLogger(__name__)


class LocalEnvAdapter(SecretsAdapter):
    """Local environment variables implementation"""

    def __init__(self) -> None:
        """Initialize local environment adapter"""
        logger.info("Local environment secrets adapter initialized")

    @property
    def provider_name(self) -> str:
        return "local"

    async def get_secret(self, secret_name: str, default_value: Optional[str] = None) -> Optional[str]:
        """Get secret from environment variables"""
        value = os.getenv(secret_name, default_value)
        if value is None:
            logger.debug(f"Secret not found in environment: {secret_name}")
        return value

    async def set_secret(self, secret_name: str, secret_value: str) -> bool:
        """Set environment variable (runtime only - not persistent)"""
        try:
            os.environ[secret_name] = secret_value
            logger.info(f"Environment variable set: {secret_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to set environment variable {secret_name}: {e}")
            return False

    async def delete_secret(self, secret_name: str) -> bool:
        """Delete environment variable (runtime only)"""
        try:
            if secret_name in os.environ:
                del os.environ[secret_name]
                logger.info(f"Environment variable deleted: {secret_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete environment variable {secret_name}: {e}")
            return False

    async def secret_exists(self, secret_name: str) -> bool:
        """Check if environment variable exists"""
        return secret_name in os.environ
