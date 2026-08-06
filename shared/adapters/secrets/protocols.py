"""Abstract secrets adapter interface"""

from abc import ABC, abstractmethod
from typing import Optional


class SecretsAdapter(ABC):
    """Abstract interface for secrets/vault operations"""

    @abstractmethod
    async def get_secret(self, secret_name: str, default_value: Optional[str] = None) -> Optional[str]:
        """
        Get a secret value by name.

        Args:
            secret_name: Name/key of the secret
            default_value: Value to return if secret not found

        Returns:
            Secret value or default_value if not found

        Raises:
            Exception: If there's an error accessing the vault (not for missing secrets)
        """
        pass

    @abstractmethod
    async def set_secret(self, secret_name: str, secret_value: str) -> bool:
        """
        Set/update a secret value.

        Args:
            secret_name: Name/key of the secret
            secret_value: Secret value to store

        Returns:
            True if successful, False otherwise

        Note:
            Local adapter supports this; cloud vaults may not allow programmatic writes
        """
        pass

    @abstractmethod
    async def delete_secret(self, secret_name: str) -> bool:
        """
        Delete a secret.

        Args:
            secret_name: Name/key of the secret to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def secret_exists(self, secret_name: str) -> bool:
        """
        Check if a secret exists.

        Args:
            secret_name: Name/key of the secret

        Returns:
            True if exists, False otherwise
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get secrets provider name (azure/aws/gcp/local)"""
        pass
