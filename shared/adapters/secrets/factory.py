"""
Factory for creating secrets adapters (Factory Pattern).

Provides simple interface for creating appropriate secrets adapter based on configuration.
"""
import logging
import os
from enum import Enum
from typing import Optional

from .protocols import SecretsAdapter

logger = logging.getLogger(__name__)


class SecretsProvider(str, Enum):
    """Supported secrets providers"""

    AZURE = "azure"
    AWS = "aws"
    GCP = "gcp"
    LOCAL = "local"


def _detect_cloud_provider() -> str:
    """
    Auto-detect cloud provider from environment.

    Returns:
        Provider name (azure/aws/gcp/local)
    """
    # Check explicit configuration
    provider = os.getenv("SECRETS_PROVIDER", "").strip().lower()
    if provider in {"azure", "aws", "gcp", "local"}:
        return provider

    # Auto-detect from cloud environment variables
    # AWS detection
    if os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"):
        return "aws"

    # Azure detection
    if os.getenv("WEBSITE_SITE_NAME") or os.getenv("AzureWebJobsStorage"):
        return "azure"

    # GCP detection
    if os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT_ID"):
        return "gcp"

    # Check if vault URLs are configured
    if os.getenv("AZURE_KEY_VAULT_URL"):
        return "azure"

    # Default to local
    return "local"


class SecretsFactory:
    """
    Factory for creating secrets adapters.

    Usage:
        # Auto-detect provider
        secrets = SecretsFactory.create()

        # Explicit Azure Key Vault
        secrets = SecretsFactory.create(
            provider="azure",
            vault_url="https://my-vault.vault.azure.net/"
        )

        # Explicit AWS Secrets Manager
        secrets = SecretsFactory.create(
            provider="aws",
            region_name="us-east-1"
        )

        # Explicit GCP Secret Manager
        secrets = SecretsFactory.create(
            provider="gcp",
            project_id="my-project"
        )
    """

    @staticmethod
    def create(
        provider: Optional[str] = None,
        **kwargs
    ) -> SecretsAdapter:
        """
        Create secrets adapter based on provider.

        Args:
            provider: Secrets provider ("azure", "aws", "gcp", "local")
                     If None, auto-detects from environment
            **kwargs: Provider-specific arguments:
                - Azure: vault_url
                - AWS: region_name
                - GCP: project_id
                - Local: (no args needed)

        Returns:
            SecretsAdapter instance

        Raises:
            ValueError: If provider is unknown or required config is missing
        """
        # Auto-detect if not specified
        if not provider:
            provider = _detect_cloud_provider()
            logger.info(f"Auto-detected secrets provider: {provider}")

        # Normalize provider name
        provider = provider.lower().strip()

        # Convert string to enum
        try:
            provider_enum = SecretsProvider(provider)
        except ValueError:
            logger.warning(
                f"Unknown secrets provider '{provider}', falling back to local"
            )
            provider_enum = SecretsProvider.LOCAL

        logger.info(f"Creating secrets adapter for provider: {provider_enum.value}")

        # Create adapter based on provider
        if provider_enum == SecretsProvider.AZURE:
            from .adapters.azure_keyvault import AzureKeyVaultAdapter
            vault_url = kwargs.get("vault_url") or os.getenv("AZURE_KEY_VAULT_URL")
            return AzureKeyVaultAdapter(vault_url=vault_url)

        elif provider_enum == SecretsProvider.AWS:
            from .adapters.aws_secrets_manager import AWSSecretsManagerAdapter
            region_name = (
                kwargs.get("region_name")
                or os.getenv("AWS_REGION")
                or os.getenv("AWS_DEFAULT_REGION")
            )
            return AWSSecretsManagerAdapter(region_name=region_name)

        elif provider_enum == SecretsProvider.GCP:
            from .adapters.gcp_secret_manager import GCPSecretManagerAdapter
            project_id = (
                kwargs.get("project_id")
                or os.getenv("GCP_PROJECT_ID")
                or os.getenv("GOOGLE_CLOUD_PROJECT")
            )
            return GCPSecretManagerAdapter(project_id=project_id)

        elif provider_enum == SecretsProvider.LOCAL:
            from .adapters.local_env import LocalEnvAdapter
            return LocalEnvAdapter()

        else:
            # Should never reach here due to enum validation
            logger.error(f"Unhandled secrets provider: {provider_enum}")
            raise ValueError(f"Unsupported provider: {provider}")


def get_secrets_adapter(
    provider: Optional[str] = None,
    **kwargs
) -> SecretsAdapter:
    """
    Get a secrets adapter instance.

    Args:
        provider: Secrets provider ("azure", "aws", "gcp", "local")
                 If None, auto-detects from environment
        **kwargs: Provider-specific arguments

    Returns:
        SecretsAdapter instance

    Example:
        # Auto-detect from environment
        secrets = get_secrets_adapter()
        api_key = await secrets.get_secret("API_KEY")

        # Explicit Azure
        secrets = get_secrets_adapter(
            provider="azure",
            vault_url="https://my-vault.vault.azure.net/"
        )
        db_password = await secrets.get_secret("DB_PASSWORD")
    """
    return SecretsFactory.create(provider=provider, **kwargs)
