"""GCP Secret Manager adapter for secrets"""

import logging
from typing import Optional

from core.exceptions import ConfigurationException

from ..protocols import SecretsAdapter

logger = logging.getLogger(__name__)


class GCPSecretManagerAdapter(SecretsAdapter):
    """GCP Secret Manager implementation"""

    def __init__(self, project_id: Optional[str] = None) -> None:
        """
        Initialize GCP Secret Manager adapter.

        Args:
            project_id: GCP project ID
                       Falls back to GCP_PROJECT_ID or GOOGLE_CLOUD_PROJECT env vars if not provided

        Required:
            - project_id or GCP_PROJECT_ID/GOOGLE_CLOUD_PROJECT environment variable
            - GCP credentials (via GOOGLE_APPLICATION_CREDENTIALS or default credentials)
        """
        try:
            from google.cloud import secretmanager
        except ImportError:
            raise ConfigurationException(
                "google-cloud-secret-manager not installed. "
                "Install with: pip install google-cloud-secret-manager",
                config_key="google-cloud-secret-manager",
            )

        import os

        self._project_id = (
            project_id
            or os.getenv("GCP_PROJECT_ID")
            or os.getenv("GOOGLE_CLOUD_PROJECT")
        )
        if not self._project_id:
            raise ConfigurationException(
                "project_id or GCP_PROJECT_ID/GOOGLE_CLOUD_PROJECT environment variable required",
                config_key="GCP_PROJECT_ID",
            )

        self._client = secretmanager.SecretManagerServiceClient()
        self._parent = f"projects/{self._project_id}"

        logger.info(
            "GCP Secret Manager adapter initialized",
            project_id=self._project_id,
        )

    @property
    def provider_name(self) -> str:
        return "gcp"

    def _get_secret_path(self, secret_name: str) -> str:
        """Get full secret path"""
        return f"{self._parent}/secrets/{secret_name}/versions/latest"

    def _get_secret_name_path(self, secret_name: str) -> str:
        """Get secret name path (without version)"""
        return f"{self._parent}/secrets/{secret_name}"

    async def get_secret(self, secret_name: str, default_value: Optional[str] = None) -> Optional[str]:
        """Get secret from GCP Secret Manager"""
        import asyncio

        try:
            secret_path = self._get_secret_path(secret_name)
            response = await asyncio.to_thread(
                self._client.access_secret_version,
                request={"name": secret_path}
            )

            secret_value = response.payload.data.decode("utf-8")
            logger.debug(f"Secret retrieved from GCP Secret Manager: {secret_name}")
            return secret_value

        except Exception as e:
            logger.warning(f"Secret not found in GCP Secret Manager: {secret_name}, error: {e}")
            return default_value

    async def set_secret(self, secret_name: str, secret_value: str) -> bool:
        """Set/update secret in GCP Secret Manager"""
        import asyncio

        try:
            secret_name_path = self._get_secret_name_path(secret_name)

            # Check if secret exists
            try:
                await asyncio.to_thread(
                    self._client.get_secret,
                    request={"name": secret_name_path}
                )
                secret_exists = True
            except Exception:
                secret_exists = False

            # Create secret if it doesn't exist
            if not secret_exists:
                await asyncio.to_thread(
                    self._client.create_secret,
                    request={
                        "parent": self._parent,
                        "secret_id": secret_name,
                        "secret": {"replication": {"automatic": {}}},
                    }
                )
                logger.info(f"Secret created in GCP Secret Manager: {secret_name}")

            # Add new version with the value
            await asyncio.to_thread(
                self._client.add_secret_version,
                request={
                    "parent": secret_name_path,
                    "payload": {"data": secret_value.encode("utf-8")},
                }
            )

            logger.info(f"Secret version added in GCP Secret Manager: {secret_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to set secret in GCP Secret Manager {secret_name}: {e}")
            return False

    async def delete_secret(self, secret_name: str) -> bool:
        """Delete secret from GCP Secret Manager"""
        import asyncio

        try:
            secret_name_path = self._get_secret_name_path(secret_name)
            await asyncio.to_thread(
                self._client.delete_secret,
                request={"name": secret_name_path}
            )
            logger.info(f"Secret deleted from GCP Secret Manager: {secret_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete secret from GCP Secret Manager {secret_name}: {e}")
            return False

    async def secret_exists(self, secret_name: str) -> bool:
        """Check if secret exists in GCP Secret Manager"""
        import asyncio

        try:
            secret_name_path = self._get_secret_name_path(secret_name)
            await asyncio.to_thread(
                self._client.get_secret,
                request={"name": secret_name_path}
            )
            return True
        except Exception:
            return False
