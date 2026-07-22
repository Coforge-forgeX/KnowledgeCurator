"""
Document Processor Factory

Factory for creating document processors based on file type.
Implements the Factory and Chain of Responsibility patterns.
"""
from typing import List, Optional

from core.logging import get_logger

from .base import DocumentProcessor, ProcessingException
from .ocr_processor import AzureDocumentIntelligenceProcessor
from .pdf_processor import PDFProcessor

logger = get_logger(__name__)


class DocumentProcessorFactory:
    """
    Factory for creating and managing document processors.

    Design Pattern: Factory + Chain of Responsibility
    Purpose: Route documents to appropriate processors based on file type
    """

    def __init__(self):
        """Initialize document processor factory with all available processors"""
        self.processors: List[DocumentProcessor] = []

        # Register processors in priority order
        self._register_processors()

        logger.info(
            "Document processor factory initialized",
            processor_count=len(self.processors),
        )

    def _register_processors(self) -> None:
        """Register all available document processors"""
        # PDF processor (uses OCR fallback)
        self.processors.append(PDFProcessor(use_azure_ocr_fallback=True))

        # DOCX processor
        from .docx_processor import DOCXProcessor
        self.processors.append(DOCXProcessor())

        # Plain text processor
        from .text_processor import TextProcessor
        self.processors.append(TextProcessor())

        # Image processor (uses Azure Document Intelligence)
        self.processors.append(AzureDocumentIntelligenceProcessor())

    async def get_processor(
        self,
        file_name: str,
        content_type: Optional[str] = None,
    ) -> DocumentProcessor:
        """
        Get appropriate processor for the file.

        Args:
            file_name: Name of the file
            content_type: MIME type of the file

        Returns:
            DocumentProcessor: Appropriate processor

        Raises:
            ProcessingException: If no processor can handle the file
        """
        for processor in self.processors:
            if await processor.can_process(file_name, content_type):
                logger.info(
                    "Found processor for file",
                    file_name=file_name,
                    processor=processor.__class__.__name__,
                )
                return processor

        raise ProcessingException(
            message=f"No processor found for file type",
            file_name=file_name,
        )

    async def process_document(
        self,
        content: bytes,
        file_name: str,
        content_type: Optional[str] = None,
    ):
        """
        Process document using appropriate processor.

        Args:
            content: File content as bytes
            file_name: Original file name
            content_type: MIME type

        Returns:
            ProcessedDocument: Processed document with extracted text

        Raises:
            ProcessingException: If processing fails
        """
        processor = await self.get_processor(file_name, content_type)
        return await processor.process(content, file_name)

    async def cleanup(self) -> None:
        """Cleanup all processors"""
        for processor in self.processors:
            try:
                await processor.cleanup()
            except Exception as e:
                logger.warning(
                    "Error cleaning up processor",
                    error=e,
                    processor=processor.__class__.__name__,
                )


# Singleton instance
_processor_factory_instance: Optional[DocumentProcessorFactory] = None


def get_document_processor_factory() -> DocumentProcessorFactory:
    """Get or create singleton document processor factory"""
    global _processor_factory_instance
    if _processor_factory_instance is None:
        _processor_factory_instance = DocumentProcessorFactory()
    return _processor_factory_instance
