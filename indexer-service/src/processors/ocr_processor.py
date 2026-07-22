"""
OCR Document Processor using Azure Document Intelligence

Handles scanned PDFs, images, and documents requiring OCR.
"""
import os
from typing import Optional

from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

from core.exceptions import ConfigurationException
from core.logging import get_logger

from .base import DocumentProcessor, ProcessedDocument, ProcessingException

logger = get_logger(__name__)


class AzureDocumentIntelligenceProcessor(DocumentProcessor):
    """
    Azure Document Intelligence (Form Recognizer) processor.

    Uses Azure's OCR and document understanding capabilities to extract
    text from scanned documents, images, and complex PDFs.
    """

    SUPPORTED_FORMATS = {
        ".pdf",
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tiff",
        ".tif",
    }

    def __init__(self, **config):
        """
        Initialize Azure Document Intelligence processor.

        Environment Variables:
            AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT: Azure endpoint URL
            AZURE_DOCUMENT_INTELLIGENCE_KEY: Azure subscription key
        """
        super().__init__(**config)

        from core.config import settings

        self.endpoint = config.get("endpoint") or settings.azure.AZURE_DOC_INTELLIGENCE_ENDPOINT
        self.key = config.get("key") or settings.azure.AZURE_DOC_INTELLIGENCE_KEY

        if not self.endpoint or not self.key:
            raise ConfigurationException(
                message="Azure Document Intelligence credentials not configured",
                config_key="AZURE_DOCUMENT_INTELLIGENCE",
            )

        self.client = DocumentAnalysisClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.key),
        )

        logger.info("Azure Document Intelligence processor initialized")

    async def can_process(self, file_name: str, content_type: Optional[str] = None) -> bool:
        """Check if file can be processed with Azure Document Intelligence"""
        file_ext = os.path.splitext(file_name.lower())[1]
        return file_ext in self.SUPPORTED_FORMATS

    async def process(self, content: bytes, file_name: str) -> ProcessedDocument:
        """Process document using Azure Document Intelligence"""
        try:
            logger.info(
                "Processing document with Azure Document Intelligence",
                file_name=file_name,
                size=len(content),
            )

            # Use prebuilt-read model for general text extraction
            poller = self.client.begin_analyze_document(
                model_id="prebuilt-read",
                document=content,
            )

            result = poller.result()

            # Extract text content
            text_parts = []
            page_count = len(result.pages)

            # Extract text from all pages
            for page in result.pages:
                page_lines = []
                for line in page.lines:
                    page_lines.append(line.content)

                if page_lines:
                    text_parts.append("\n".join(page_lines))

            full_text = "\n\n".join(text_parts)

            # Calculate average confidence
            confidences = []
            for page in result.pages:
                for line in page.lines:
                    if hasattr(line, "confidence") and line.confidence:
                        confidences.append(line.confidence)

            avg_confidence = (
                sum(confidences) / len(confidences) if confidences else None
            )

            # Detect language (if available)
            languages = []
            if hasattr(result, "languages") and result.languages:
                languages = [lang.locale for lang in result.languages]

            logger.info(
                "Document processed successfully with Azure Document Intelligence",
                file_name=file_name,
                text_length=len(full_text),
                page_count=page_count,
                confidence=avg_confidence,
            )

            return ProcessedDocument(
                text=full_text,
                metadata={
                    "file_name": file_name,
                    "file_type": os.path.splitext(file_name)[1].lstrip("."),
                    "processor": "azure_document_intelligence",
                    "model": "prebuilt-read",
                    "languages": languages,
                },
                page_count=page_count,
                language=languages[0] if languages else None,
                confidence=avg_confidence,
            )

        except Exception as e:
            logger.error(
                "Failed to process document with Azure Document Intelligence",
                error=e,
                file_name=file_name,
            )
            raise ProcessingException(
                message=f"Failed to process with Azure Document Intelligence: {str(e)}",
                file_name=file_name,
                cause=e,
            )

    async def extract_text(self, content: bytes) -> str:
        """Extract plain text from document"""
        poller = self.client.begin_analyze_document(
            model_id="prebuilt-read",
            document=content,
        )

        result = poller.result()

        text_parts = []
        for page in result.pages:
            page_lines = []
            for line in page.lines:
                page_lines.append(line.content)

            if page_lines:
                text_parts.append("\n".join(page_lines))

        return "\n\n".join(text_parts)
