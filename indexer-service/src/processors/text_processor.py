"""
Text Document Processor

Handles plain text files.
"""
from typing import Optional

from core.logging import get_logger

from .base import DocumentProcessor, ProcessedDocument, ProcessingException

logger = get_logger(__name__)


class TextProcessor(DocumentProcessor):
    """
    Plain text document processor.

    Handles .txt, .md, .csv, and other text-based files.
    """

    TEXT_EXTENSIONS = {
        ".txt",
        ".md",
        ".markdown",
        ".csv",
        ".json",
        ".xml",
        ".yaml",
        ".yml",
        ".log",
        ".html",
        ".htm",
    }

    async def can_process(self, file_name: str, content_type: Optional[str] = None) -> bool:
        """Check if file is plain text"""
        if content_type and "text" in content_type.lower():
            return True

        file_ext = "." + file_name.lower().split(".")[-1] if "." in file_name else ""
        return file_ext in self.TEXT_EXTENSIONS

    async def process(self, content: bytes, file_name: str) -> ProcessedDocument:
        """Process text file and extract content"""
        try:
            logger.info("Processing text document", file_name=file_name)

            text = await self.extract_text(content)

            if not text or len(text.strip()) < 10:
                raise ProcessingException(
                    message="Text file is empty or too short",
                    file_name=file_name,
                )

            # Estimate pages based on line count
            line_count = len(text.split("\n"))
            page_estimate = max(1, line_count // 50)

            logger.info(
                "Text processed successfully",
                file_name=file_name,
                text_length=len(text),
                line_count=line_count,
            )

            return ProcessedDocument(
                text=text,
                metadata={
                    "file_name": file_name,
                    "file_type": "text",
                    "processor": "text_processor",
                    "line_count": line_count,
                },
                page_count=page_estimate,
            )

        except Exception as e:
            logger.error("Failed to process text file", error=e, file_name=file_name)
            raise ProcessingException(
                message=f"Failed to process text file: {str(e)}",
                file_name=file_name,
                cause=e,
            )

    async def extract_text(self, content: bytes) -> str:
        """Extract text from file"""
        try:
            # Try UTF-8 first
            try:
                return content.decode("utf-8")
            except UnicodeDecodeError:
                # Try other common encodings
                for encoding in ["latin-1", "cp1252", "ascii"]:
                    try:
                        return content.decode(encoding)
                    except UnicodeDecodeError:
                        continue

                # If all fail, decode with errors ignored
                return content.decode("utf-8", errors="ignore")

        except Exception as e:
            logger.warning("Text extraction failed", error=e)
            raise
