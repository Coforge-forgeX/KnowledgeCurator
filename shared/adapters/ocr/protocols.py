"""OCR adapter protocol - provider-agnostic interface for document OCR."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class OCRAdapter(Protocol):
    """
    Protocol for OCR adapters that extract text from document images.

    Used as a fallback when basic PDF/DOC text extraction returns insufficient
    content, typically for scanned documents or image-heavy files.
    """

    async def extract_text(self, file_bytes: bytes, file_path: str) -> str:
        """
        Extract text from document bytes using OCR.

        Args:
            file_bytes: Raw file bytes
            file_path: File name/path (used for logging and error messages)

        Returns:
            Extracted text content

        Raises:
            Exception: If OCR extraction fails
        """
        ...

    @property
    def is_configured(self) -> bool:
        """Check if this adapter has valid credentials."""
        ...

    @property
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'azure', 'aws', 'gcp')."""
        ...
