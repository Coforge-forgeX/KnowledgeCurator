"""
Extracts plain text from files uploaded alongside a SEARCH-mode chat message.

This is deliberately separate from the indexing pipeline (upload_and_index /
indexer-service): files handled here are never persisted or indexed — their
text is only added to the LLM context for the current turn, per the
"context, not indexed" requirement.

Now uses multi-provider OCR adapters for cloud-agnostic deployment.
"""
import base64
from typing import List, Optional

from shared.adapters.ocr import get_ocr_adapter
from shared.text_extraction.extractor_v2 import TextExtractionService
from shared.text_extraction import TextExtractionError, TextExtractionResult

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
        # Build OCR adapter based on configuration
        ocr_provider = settings.ocr.OCR_PROVIDER or settings.CLOUD_PROVIDER

        try:
            ocr_adapter = get_ocr_adapter(
                provider=ocr_provider,
                # Azure Document Intelligence
                azure_endpoint=settings.ocr.AZURE_DOC_INTELLIGENCE_ENDPOINT,
                azure_api_key=settings.ocr.AZURE_DOC_INTELLIGENCE_KEY,
                # AWS Textract
                aws_region=settings.ocr.AWS_TEXTRACT_REGION or settings.AWS_REGION,
                aws_access_key_id=settings.ocr.AWS_TEXTRACT_ACCESS_KEY_ID,
                aws_secret_access_key=settings.ocr.AWS_TEXTRACT_SECRET_ACCESS_KEY,
                # GCP Document AI
                gcp_project_id=settings.ocr.GCP_DOCUMENT_AI_PROJECT_ID,
                gcp_location=settings.ocr.GCP_DOCUMENT_AI_LOCATION,
                gcp_processor_id=settings.ocr.GCP_DOCUMENT_AI_PROCESSOR_ID,
                gcp_credentials_path=settings.ocr.GCP_DOCUMENT_AI_CREDENTIALS_PATH,
            )
            logger.info(
                "FileContextExtractor initialized",
                ocr_provider=ocr_provider,
                ocr_configured=ocr_adapter.is_configured,
            )
        except Exception as e:
            logger.warning(
                "Failed to initialize OCR adapter, falling back to no-op",
                ocr_provider=ocr_provider,
                error=str(e),
            )
            from shared.adapters.ocr import NoOpOCRAdapter
            ocr_adapter = NoOpOCRAdapter()

        self._extraction_service = TextExtractionService(ocr_adapter=ocr_adapter)

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
