"""Secrets/Vault adapters for Azure, AWS, GCP, and Local"""

from .protocols import SecretsAdapter
from .models import SecretInfo
from .factory import (
    SecretsFactory,
    SecretsProvider,
    get_secrets_adapter,
)
from .adapters import (
    AzureKeyVaultAdapter,
    AWSSecretsManagerAdapter,
    GCPSecretManagerAdapter,
    LocalEnvAdapter,
)

__all__ = [
    # Core
    "SecretsAdapter",
    "SecretInfo",
    # Factory
    "SecretsFactory",
    "SecretsProvider",
    "get_secrets_adapter",
    # Adapters
    "AzureKeyVaultAdapter",
    "AWSSecretsManagerAdapter",
    "GCPSecretManagerAdapter",
    "LocalEnvAdapter",
]
