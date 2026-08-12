"""
Extracts plain text from files uploaded alongside a SEARCH-mode chat message.

This is deliberately separate from the indexing pipeline (upload_and_index /
indexer-service): files handled here are never persisted or indexed — their
text is only added to the LLM context for the current turn, per the
"context, not indexed" requirement.

Delegates actual per-file-type extraction to `shared.text_extraction`, the
same module indexer-service uses for its indexing pipeline, so both services
share one implementation instead of maintaining duplicate PDF/DOCX parsing.
"""
import base64
from typing import List, Optional

from shared.text_extraction import (
    DocIntelligenceConfig,
    TextExtractionError,
    TextExtractionService,
)

from src.core.config import settings
from src.core.exceptions import ValidationException
from src.core.logging import get_logger

logger = get_logger(__name__)


def _decode_base64(file_content: str) -> bytes:
    normalized = file_content.split(",", 1)[1] if file_content.startswith("data:") else file_content
    return base64.b64decode(normalized)


class FileContextExtractor:
    """Turns uploaded (file_name, file_content) pairs into a single context string."""

    def __init__(self) -> None:
        # Credentials come from this service's settings (not the process
        # environment) so the OCR fallback behaves the same locally, where
        # values are only present in .env, as it does when deployed.
        self._extraction_service = TextExtractionService(
            doc_intelligence=DocIntelligenceConfig(
                endpoint=settings.azure.AZURE_DOC_INTELLIGENCE_ENDPOINT,
                api_key=settings.azure.AZURE_DOC_INTELLIGENCE_KEY,
            )
        )

    async def extract(self, file_names: List[str], file_contents: List[str]) -> str:
        if len(file_names) != len(file_contents):
            raise ValidationException(message="file_names and file_contents must have the same length")

        sections: List[str] = []
        for file_name, file_content in zip(file_names, file_contents):
            try:
                file_bytes = _decode_base64(file_content)
            except Exception as e:
                logger.warning("Skipping file with invalid base64 content", file_name=file_name, error=str(e))
                continue

            try:
                result = await self._extraction_service.extract_text(file_bytes=file_bytes, file_path=file_name)
            except TextExtractionError as e:
                logger.warning("Skipping file for context extraction", file_name=file_name, error=e.message)
                continue

            if result.text.strip():
                sections.append(f"--- Content of {file_name} ---\n{result.text.strip()}")

        return "\n\n".join(sections)


_extractor_instance: Optional[FileContextExtractor] = None


def get_file_context_extractor() -> FileContextExtractor:
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = FileContextExtractor()
    return _extractor_instance
