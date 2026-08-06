"""Azure Key Vault adapter for secrets"""

import asyncio
import logging
from typing import Optional

from core.exceptions import ConfigurationException

from ..protocols import SecretsAdapter

logger = logging.getLogger(__name__)


class AzureKeyVaultAdapter(SecretsAdapter):
    """Azure Key Vault implementation"""

    def __init__(self, vault_url: Optional[str] = None) -> None:
        """
        Initialize Azure Key Vault adapter.

        Args:
            vault_url: Azure Key Vault URL (e.g., https://my-vault.vault.azure.net/)
                      Falls back to AZURE_KEY_VAULT_URL env var if not provided

        Required:
            - vault_url or AZURE_KEY_VAULT_URL environment variable
            - Azure credentials (DefaultAzureCredential: managed identity, service principal, etc.)
        """
        try:
            from azure.identity.aio import DefaultAzureCredential
            from azure.keyvault.secrets.aio import SecretClient
        except ImportError:
            raise ConfigurationException(
                "azure-keyvault-secrets and azure-identity not installed. "
                "Install with: pip install azure-keyvault-secrets azure-identity",
                config_key="azure-keyvault-secrets",
            )

        import os

        self._vault_url = vault_url or os.getenv("AZURE_KEY_VAULT_URL")
        if not self._vault_url:
            raise ConfigurationException(
                "vault_url or AZURE_KEY_VAULT_URL environment variable required",
                config_key="AZURE_KEY_VAULT_URL",
            )

        self._credential = DefaultAzureCredential()
        self._client = SecretClient(vault_url=self._vault_url, credential=self._credential)

        logger.info(
            "Azure Key Vault adapter initialized",
            vault_url=self._vault_url,
        )

    @property
    def provider_name(self) -> str:
        return "azure"

    async def get_secret(self, secret_name: str, default_value: Optional[str] = None) -> Optional[str]:
        """Get secret from Azure Key Vault"""
        try:
            secret = await self._client.get_secret(secret_name)
            logger.debug(f"Secret retrieved from Azure Key Vault: {secret_name}")
            return secret.value
        except Exception as e:
            logger.warning(f"Secret not found in Azure Key Vault: {secret_name}, error: {e}")
            return default_value

    async def set_secret(self, secret_name: str, secret_value: str) -> bool:
        """Set/update secret in Azure Key Vault"""
        try:
            await self._client.set_secret(secret_name, secret_value)
            logger.info(f"Secret set in Azure Key Vault: {secret_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to set secret in Azure Key Vault {secret_name}: {e}")
            return False

    async def delete_secret(self, secret_name: str) -> bool:
        """Delete secret from Azure Key Vault (begins deletion process)"""
        try:
            await self._client.begin_delete_secret(secret_name)
            logger.info(f"Secret deletion initiated in Azure Key Vault: {secret_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete secret from Azure Key Vault {secret_name}: {e}")
            return False

    async def secret_exists(self, secret_name: str) -> bool:
        """Check if secret exists in Azure Key Vault"""
        try:
            await self._client.get_secret(secret_name)
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """Close client connections"""
        try:
            await self._client.close()
            await self._credential.close()
        except Exception as e:
            logger.warning(f"Error closing Azure Key Vault client: {e}")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
