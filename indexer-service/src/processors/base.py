"""
Base Document Processor Interface

Defines the contract for document processors that extract text from various file formats.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ProcessedDocument:
    """Result of document processing"""
    text: str
    metadata: Dict[str, any]
    page_count: Optional[int] = None
    language: Optional[str] = None
    confidence: Optional[float] = None


class DocumentProcessor(ABC):
    """
    Abstract base class for document processors.

    Each processor handles specific document types (PDF, DOCX, images, etc.)
    and extracts text content with metadata.

    Design Pattern: Strategy Pattern
    Purpose: Provide pluggable document processing strategies
    """

    def __init__(self, **config):
        """
        Initialize document processor.

        Args:
            **config: Processor-specific configuration
        """
        self.config = config

    @abstractmethod
    async def can_process(self, file_name: str, content_type: Optional[str] = None) -> bool:
        """
        Check if this processor can handle the given file.

        Args:
            file_name: Name of the file
            content_type: MIME type of the file

        Returns:
            bool: True if this processor can handle the file
        """
        pass

    @abstractmethod
    async def process(self, content: bytes, file_name: str) -> ProcessedDocument:
        """
        Process document and extract text.

        Args:
            content: File content as bytes
            file_name: Original file name

        Returns:
            ProcessedDocument: Extracted text and metadata

        Raises:
            ProcessingException: If processing fails
        """
        pass

    @abstractmethod
    async def extract_text(self, content: bytes) -> str:
        """
        Extract plain text from document.

        Args:
            content: File content as bytes

        Returns:
            str: Extracted text
        """
        pass

    async def cleanup(self) -> None:
        """
        Cleanup resources if needed.
        Override if processor needs cleanup.
        """
        pass


class ProcessingException(Exception):
    """Exception raised during document processing"""

    def __init__(self, message: str, file_name: Optional[str] = None, cause: Optional[Exception] = None):
        self.message = message
        self.file_name = file_name
        self.cause = cause
        super().__init__(self.message)
