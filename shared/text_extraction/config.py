"""Configuration objects for text extraction.

`shared` is consumed by services with their own settings frameworks
(pydantic-settings in both indexer-service and kb-rest-service), so this module
never reaches into a service's settings object. Callers pass credentials in
explicitly; reading the process environment is only a fallback for callers that
have no config of their own.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DocIntelligenceConfig:
    """Azure Document Intelligence credentials used by the OCR fallbacks."""

    endpoint: Optional[str] = None
    api_key: Optional[str] = None

    @classmethod
    def from_env(cls) -> "DocIntelligenceConfig":
        """Read credentials from the process environment.

        Supports both the long (`AZURE_DOCUMENT_INTELLIGENCE_*`) and short
        (`AZURE_DOC_INTELLIGENCE_*`) names that exist across the services'
        .env files and Azure app settings.
        """
        return cls(
            endpoint=(
                os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
                or os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT")
            ),
            api_key=(
                os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
                or os.getenv("AZURE_DOC_INTELLIGENCE_KEY")
            ),
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.endpoint and self.api_key)
