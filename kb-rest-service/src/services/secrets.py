"""
Secrets management service for kb-rest-service.

Provides a singleton secrets adapter configured from settings.
"""
from typing import Optional

from shared.adapters.secrets import get_secrets_adapter as _get_secrets_adapter
from shared.adapters.secrets import SecretsAdapter


_secrets_adapter: Optional[SecretsAdapter] = None


def get_secrets_adapter(force_recreate: bool = False) -> SecretsAdapter:
    """
    Get secrets adapter configured from kb-rest-service settings.

    Args:
        force_recreate: If True, recreate the adapter even if one exists

    Returns:
        SecretsAdapter instance (auto-detects provider from environment)

    Example:
        secrets = get_secrets_adapter()
        openai_key = await secrets.get_secret("OPENAI_API_KEY")
        db_password = await secrets.get_secret("DATABASE_PASSWORD", default_value="dev-pass")
    """
    global _secrets_adapter

    if _secrets_adapter is None or force_recreate:
        # Auto-detect provider from environment
        # Set SECRETS_PROVIDER, AZURE_KEY_VAULT_URL, AWS_REGION, or GCP_PROJECT_ID
        # to configure specific provider
        _secrets_adapter = _get_secrets_adapter()

    return _secrets_adapter


__all__ = [
    "get_secrets_adapter",
    "SecretsAdapter",
]
