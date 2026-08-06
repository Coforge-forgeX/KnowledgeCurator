"""Secrets models and data classes"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SecretInfo:
    """
    Information about a secret.

    Attributes:
        name: Secret name/key
        provider: Provider name (azure/aws/gcp/local)
        version: Secret version (if supported by provider)
        created_at: Creation timestamp (if available)
    """

    name: str
    provider: str
    version: Optional[str] = None
    created_at: Optional[str] = None
