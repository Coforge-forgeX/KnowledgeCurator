"""
PDF Document Processor

Extracts text from PDF files using PyPDF2 and pdfplumber.
Falls back to Azure Document Intelligence for scanned PDFs.
"""
import io
from typing import Optional

import PyPDF2
import pdfplumber

from core.config import settings
from core.logging import get_logger

from .base import DocumentProcessor, ProcessedDocument, ProcessingException

logger = get_logger(__name__)

# Configuration from settings
MIN_TEXT_THRESHOLD = settings.processing.PDF_MIN_TEXT_CHARS
MIN_TEXT_PER_PAGE = settings.processing.PDF_MIN_TEXT_PER_PAGE
USE_PER_PAGE_OCR = settings.processing.PDF_PER_PAGE_OCR


class PDFProcessor(DocumentProcessor):
    """
    PDF document processor.

    Uses PyPDF2 for text-based PDFs and pdfplumber for better layout preservation.
    Falls back to Azure Document Intelligence for scanned PDFs.
    """

    def __init__(self, use_azure_ocr_fallback: bool = True, **config):
        """
        Initialize PDF processor.

        Args:
            use_azure_ocr_fallback: Use Azure Document Intelligence for OCR fallback
            **config: Additional configuration
        """
        super().__init__(**config)
        self.use_azure_ocr_fallback = use_azure_ocr_fallback

    async def can_process(self, file_name: str, content_type: Optional[str] = None) -> bool:
        """Check if file is a PDF"""
        if content_type and "pdf" in content_type.lower():
            return True

        return file_name.lower().endswith(".pdf")

    async def process(self, content: bytes, file_name: str) -> ProcessedDocument:
        """Process PDF and extract text"""
        try:
            logger.info("Processing PDF document", file_name=file_name)

            # Count pages first
            page_count = await self._count_pages(content)

            logger.info("PDF page count", file_name=file_name, page_count=page_count)

            # Choose extraction strategy
            if USE_PER_PAGE_OCR and page_count > 1:
                # Per-page fallback (more efficient)
                text, extraction_method = await self._extract_with_per_page_fallback(
                    content, file_name, page_count
                )
            else:
                # Whole-document fallback (simpler)
                text, extraction_method = await self._extract_with_whole_document_fallback(
                    content, file_name, page_count
                )

            if not text or len(text.strip()) < 50:
                raise ProcessingException(
                    message="Could not extract meaningful text from PDF",
                    file_name=file_name,
                )

            logger.info(
                "PDF processed successfully",
                file_name=file_name,
                text_length=len(text),
                page_count=page_count,
                extraction_method=extraction_method,
            )

            return ProcessedDocument(
                text=text,
                metadata={
                    "file_name": file_name,
                    "file_type": "pdf",
                    "processor": "pdf_processor",
                    "extraction_method": extraction_method,
                },
                page_count=page_count,
            )

        except Exception as e:
            logger.error("Failed to process PDF", error_msg=e, file_name=file_name)
            raise ProcessingException(
                message=f"Failed to process PDF: {str(e)}",
                file_name=file_name,
                cause=e,
            )

    async def extract_text(self, content: bytes) -> str:
        """Extract plain text from PDF"""
        text = await self._extract_with_pdfplumber(content)

        if not text or len(text.strip()) < 50:
            text = await self._extract_with_pypdf2(content)

        return text

    async def _extract_with_pdfplumber(self, content: bytes) -> str:
        """Extract text using pdfplumber"""
        try:
            pdf_file = io.BytesIO(content)
            text_parts = []

            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)

            return "\n\n".join(text_parts)

        except Exception as e:
            logger.warning("pdfplumber extraction failed", error_msg=e)
            return ""

    async def _extract_with_pypdf2(self, content: bytes) -> str:
        """Extract text using PyPDF2"""
        try:
            pdf_file = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            text_parts = []
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

            return "\n\n".join(text_parts)

        except Exception as e:
            logger.warning("PyPDF2 extraction failed", error_msg=e)
            return ""

    async def _count_pages(self, content: bytes) -> int:
        """Count number of pages in PDF"""
        try:
            pdf_file = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            return len(pdf_reader.pages)
        except Exception:
            return 0

    async def _extract_with_whole_document_fallback(
        self, content: bytes, file_name: str, page_count: int
    ) -> tuple[str, str]:
        """
        Extract text with whole-document fallback strategy.

        Returns:
            tuple[str, str]: (extracted_text, extraction_method)
        """
        # Try pdfplumber first
        text = await self._extract_with_pdfplumber(content)

        if self._should_use_ocr(text, page_count):
            # Try PyPDF2 as fallback
            logger.info("Trying PyPDF2 fallback", file_name=file_name)
            text = await self._extract_with_pypdf2(content)

        if self._should_use_ocr(text, page_count):
            if self.use_azure_ocr_fallback:
                # Use Azure Document Intelligence for entire document
                logger.info(
                    "Using Azure OCR for entire document",
                    file_name=file_name,
                    reason="insufficient_text_extraction",
                )
                from .ocr_processor import AzureDocumentIntelligenceProcessor

                ocr_processor = AzureDocumentIntelligenceProcessor()
                processed = await ocr_processor.process(content, file_name)
                return processed.text, "azure_document_intelligence"
            else:
                return text, "pdfplumber_with_warnings"

        return text, "pdfplumber"

    async def _extract_with_per_page_fallback(
        self, content: bytes, file_name: str, page_count: int
    ) -> tuple[str, str]:
        """
        Extract text with per-page fallback strategy.
        More efficient - only OCRs pages that need it.

        Returns:
            tuple[str, str]: (extracted_text, extraction_method)
        """
        pdf_file = io.BytesIO(content)
        text_parts = []
        ocr_pages = []
        extraction_methods = []

        try:
            with pdfplumber.open(pdf_file) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    # Try pdfplumber for this page
                    page_text = page.extract_text() or ""

                    # Check if this page needs OCR
                    if len(page_text.strip()) < MIN_TEXT_PER_PAGE:
                        # Try PyPDF2 for this page
                        try:
                            pdf_file_pypdf = io.BytesIO(content)
                            pdf_reader = PyPDF2.PdfReader(pdf_file_pypdf)
                            page_text = pdf_reader.pages[page_num - 1].extract_text() or ""
                        except Exception as e:
                            logger.warning(
                                "PyPDF2 failed for page",
                                page_num=page_num,
                                error_msg=e,
                            )

                    # Still insufficient text? Mark for OCR
                    if len(page_text.strip()) < MIN_TEXT_PER_PAGE:
                        ocr_pages.append(page_num)
                        if self.use_azure_ocr_fallback:
                            # Need to OCR entire document (Azure doesn't support per-page)
                            logger.info(
                                "Page needs OCR, will OCR entire document",
                                page_num=page_num,
                                file_name=file_name,
                            )
                            extraction_methods.append("requires_ocr")
                        else:
                            extraction_methods.append("pdfplumber_low_confidence")
                    else:
                        extraction_methods.append("pdfplumber")

                    text_parts.append(page_text)

            # If any pages need OCR, OCR entire document
            if ocr_pages and self.use_azure_ocr_fallback:
                logger.info(
                    "Using Azure OCR for entire document",
                    file_name=file_name,
                    pages_needing_ocr=ocr_pages,
                    total_pages=page_count,
                )
                from .ocr_processor import AzureDocumentIntelligenceProcessor

                ocr_processor = AzureDocumentIntelligenceProcessor()
                processed = await ocr_processor.process(content, file_name)
                return processed.text, "azure_document_intelligence_partial"

            # Determine overall extraction method
            if "requires_ocr" in extraction_methods:
                method = "pdfplumber_with_missing_pages"
            elif "pdfplumber_low_confidence" in extraction_methods:
                method = "pdfplumber_mixed_confidence"
            else:
                method = "pdfplumber"

            return "\n\n".join(text_parts), method

        except Exception as e:
            logger.warning("Per-page extraction failed, falling back to whole-document", error_msg=e)
            # Fall back to whole-document strategy
            return await self._extract_with_whole_document_fallback(content, file_name, page_count)

    def _should_use_ocr(self, text: str, page_count: int) -> bool:
        """
        Determine if OCR is needed based on extracted text quality.

        Args:
            text: Extracted text
            page_count: Number of pages in PDF

        Returns:
            bool: True if OCR should be used
        """
        if not text:
            return True

        text_stripped = text.strip()

        # Check 1: Insufficient total text
        if len(text_stripped) < MIN_TEXT_THRESHOLD:
            logger.info(
                "OCR needed: insufficient total text",
                text_length=len(text_stripped),
                threshold=MIN_TEXT_THRESHOLD,
            )
            return True

        # Check 2: Insufficient text per page
        if page_count > 0:
            text_per_page = len(text_stripped) / page_count
            if text_per_page < MIN_TEXT_PER_PAGE:
                logger.info(
                    "OCR needed: insufficient text per page",
                    text_per_page=text_per_page,
                    threshold=MIN_TEXT_PER_PAGE,
                )
                return True

        # Check 3: Mostly page numbers/headers (heuristic)
        words = text_stripped.split()
        if words:
            # If more than 50% of text is numbers, likely just page numbers
            number_words = sum(1 for w in words if w.isdigit())
            if number_words / len(words) > 0.5:
                logger.info(
                    "OCR needed: mostly numbers (likely page headers)",
                    number_ratio=number_words / len(words),
                )
                return True

            # If average word length < 3, likely garbage
            avg_word_length = sum(len(w) for w in words) / len(words)
            if avg_word_length < 3:
                logger.info(
                    "OCR needed: short average word length",
                    avg_word_length=avg_word_length,
                )
                return True

        return False
