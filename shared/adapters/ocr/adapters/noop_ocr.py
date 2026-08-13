"""No-op OCR adapter for environments without OCR service."""

import logging

logger = logging.getLogger(__name__)


class NoOpOCRAdapter:
    """
    No-operation OCR adapter that always fails.

    Used when no OCR provider is configured or when OCR is explicitly disabled.
    This allows the service to function for non-OCR operations.
    """

    @property
    def is_configured(self) -> bool:
        """Always returns False - no OCR configured."""
        return False

    @property
    def provider_name(self) -> str:
        return "none"

    async def extract_text(self, file_bytes: bytes, file_path: str) -> str:
        """
        Raises an error indicating OCR is not available.

        Args:
            file_bytes: Raw file bytes (unused)
            file_path: File name for error message

        Raises:
            ValueError: Always, indicating OCR is not configured
        """
        raise ValueError(
            f"OCR not configured. Cannot extract text from {file_path}. "
            "Set OCR_PROVIDER environment variable to 'azure', 'aws', or 'gcp' "
            "and provide the necessary credentials."
        )
