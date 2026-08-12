"""
Dedicated service for robust text extraction by file type.

Shared between indexer-service (indexing pipeline) and kb-rest-service
(chat file-context extraction) so both use one implementation instead of
maintaining duplicate extraction logic. This module is intentionally
decoupled from any single service's settings/logging framework: Azure
Document Intelligence credentials are injected by the caller as a
`DocIntelligenceConfig`, so each service keeps ownership of its own config
loading (see `config.py` for the environment fallback).
"""

import asyncio
import io
import logging
import os
import zipfile
from typing import Optional

import PyPDF2
import pdfplumber
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential
from docx import Document

from .config import DocIntelligenceConfig
from .decoders import normalize_file_bytes
from .models import TextExtractionError, TextExtractionResult

logger = logging.getLogger(__name__)


class BaseExtractor:
    """Base extractor contract."""

    supported_extensions: tuple[str, ...] = ()

    def can_extract(self, extension: str) -> bool:
        return extension.lower() in self.supported_extensions

    async def extract(self, file_bytes: bytes, file_path: str) -> TextExtractionResult:
        raise NotImplementedError


class PlainTextExtractor(BaseExtractor):
    supported_extensions = (".txt", ".md", ".csv", ".json", ".xml", ".yaml", ".yml", ".log")

    async def extract(self, file_bytes: bytes, file_path: str) -> TextExtractionResult:
        for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            try:
                text = file_bytes.decode(encoding)
                return TextExtractionResult(
                    text=text,
                    extractor="plain_text",
                    metadata={"encoding": encoding},
                )
            except UnicodeDecodeError:
                continue

        raise TextExtractionError(
            "Unable to decode text file with supported encodings",
            file_path=file_path,
        )


class PdfExtractor(BaseExtractor):
    supported_extensions = (".pdf",)

    def __init__(self, doc_intelligence: Optional[DocIntelligenceConfig] = None) -> None:
        self._doc_intelligence = doc_intelligence or DocIntelligenceConfig()

    async def _extract_pdfplumber(self, file_bytes: bytes) -> str:
        def _read() -> str:
            chunks = []
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    chunks.append(page.extract_text() or "")
            return "\n\n".join(chunks)

        return await asyncio.to_thread(_read)

    async def _extract_pypdf2(self, file_bytes: bytes) -> str:
        def _read() -> str:
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            chunks = []
            for page in reader.pages:
                chunks.append(page.extract_text() or "")
            return "\n\n".join(chunks)

        return await asyncio.to_thread(_read)

    async def _extract_azure_doc_intelligence(self, file_bytes: bytes) -> str:
        config = self._doc_intelligence

        if not config.is_configured:
            raise TextExtractionError("Document Intelligence not configured")

        client = DocumentIntelligenceClient(config.endpoint, AzureKeyCredential(config.api_key))
        poller = await asyncio.to_thread(
            client.begin_analyze_document,
            "prebuilt-read",
            body=AnalyzeDocumentRequest(bytes_source=file_bytes),
            locale="en-US",
        )
        result = await asyncio.to_thread(poller.result)
        return result.content or ""

    async def extract(self, file_bytes: bytes, file_path: str) -> TextExtractionResult:
        first_pass = await self._extract_pdfplumber(file_bytes)
        if len(first_pass.strip()) >= 100:
            return TextExtractionResult(text=first_pass, extractor="pdfplumber")

        second_pass = await self._extract_pypdf2(file_bytes)
        if len(second_pass.strip()) >= 100:
            return TextExtractionResult(text=second_pass, extractor="pypdf2")

        fallback = await self._extract_azure_doc_intelligence(file_bytes)
        if len(fallback.strip()) >= 10:
            return TextExtractionResult(text=fallback, extractor="azure_document_intelligence")

        raise TextExtractionError("Could not extract meaningful text from PDF", file_path=file_path)


class DocxExtractor(BaseExtractor):
    supported_extensions = (".docx",)

    @staticmethod
    def _validate_docx_payload(file_bytes: bytes, file_path: str) -> None:
        if not file_bytes.startswith(b"PK"):
            raise TextExtractionError(
                "Invalid DOCX payload: expected zipped Office document",
                file_path=file_path,
            )

        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes), "r") as archive:
                if "word/document.xml" not in archive.namelist():
                    raise TextExtractionError(
                        "Invalid DOCX payload: missing word/document.xml",
                        file_path=file_path,
                    )
        except zipfile.BadZipFile as exc:
            raise TextExtractionError(
                "Invalid DOCX payload: corrupt zip structure",
                file_path=file_path,
                cause=exc,
            ) from exc

    async def extract(self, file_bytes: bytes, file_path: str) -> TextExtractionResult:
        self._validate_docx_payload(file_bytes, file_path)

        def _read() -> str:
            doc = Document(io.BytesIO(file_bytes))
            parts: list[str] = []

            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    parts.append(paragraph.text)

            for table in doc.tables:
                for row in table.rows:
                    row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if row_cells:
                        parts.append(" | ".join(row_cells))

            for section in doc.sections:
                for paragraph in section.header.paragraphs:
                    if paragraph.text.strip():
                        parts.append(paragraph.text)
                for paragraph in section.footer.paragraphs:
                    if paragraph.text.strip():
                        parts.append(paragraph.text)

            return "\n\n".join(parts)

        text = await asyncio.to_thread(_read)
        if len(text.strip()) < 10:
            raise TextExtractionError("DOCX text is empty after extraction", file_path=file_path)

        return TextExtractionResult(text=text, extractor="docx")


class DocExtractor(BaseExtractor):
    supported_extensions = (".doc",)

    def __init__(self, doc_intelligence: Optional[DocIntelligenceConfig] = None) -> None:
        self._doc_intelligence = doc_intelligence or DocIntelligenceConfig()

    async def extract(self, file_bytes: bytes, file_path: str) -> TextExtractionResult:
        # Legacy Word binary files should start with OLE signature bytes.
        # If they do not, fail fast instead of sending invalid bytes to OCR.
        is_binary_doc = file_bytes.startswith(bytes.fromhex("D0CF11E0A1B11AE1"))

        if file_bytes.startswith(b"{\\rtf"):
            for encoding in ("utf-8", "cp1252", "latin-1"):
                try:
                    text = file_bytes.decode(encoding, errors="ignore")
                    if text.strip():
                        return TextExtractionResult(text=text, extractor="rtf_in_doc")
                except Exception:
                    continue

        if not is_binary_doc:
            raise TextExtractionError(
                "Legacy .doc extraction failed. Please convert the file to .docx or PDF.",
                file_path=file_path,
            )

        config = self._doc_intelligence

        if config.is_configured:
            client = DocumentIntelligenceClient(config.endpoint, AzureKeyCredential(config.api_key))
            try:
                poller = await asyncio.to_thread(
                    client.begin_analyze_document,
                    "prebuilt-read",
                    body=AnalyzeDocumentRequest(bytes_source=file_bytes),
                    locale="en-US",
                )
                result = await asyncio.to_thread(poller.result)
                content = (result.content or "").strip()
                if content:
                    return TextExtractionResult(text=content, extractor="azure_document_intelligence")
            except Exception as exc:
                logger.warning(
                    "Legacy .doc OCR extraction failed for %s: %s",
                    file_path,
                    exc,
                )

        raise TextExtractionError(
            "Legacy .doc extraction failed. Please convert the file to .docx or PDF.",
            file_path=file_path,
        )


class TextExtractionService:
    """Facade that routes extraction through dedicated extractor classes.

    Args:
        doc_intelligence: Azure Document Intelligence credentials used by the
            PDF and legacy .doc OCR fallbacks. Services should pass their own
            settings; when omitted the credentials are read from the process
            environment, which only works where they are exported as real
            environment variables (pydantic-settings' `env_file` does not
            populate `os.environ`).
    """

    def __init__(self, doc_intelligence: Optional[DocIntelligenceConfig] = None) -> None:
        config = doc_intelligence if doc_intelligence is not None else DocIntelligenceConfig.from_env()
        self._doc_intelligence = config
        self._extractors = (
            PlainTextExtractor(),
            PdfExtractor(config),
            DocxExtractor(),
            DocExtractor(config),
        )

    async def extract_text(self, file_bytes: bytes, file_path: str) -> TextExtractionResult:
        normalized_bytes = normalize_file_bytes(file_bytes=file_bytes, file_path=file_path)
        extension = os.path.splitext(file_path)[1].lower()

        for extractor in self._extractors:
            if extractor.can_extract(extension):
                return await extractor.extract(normalized_bytes, file_path)

        raise TextExtractionError(f"Unsupported file type: {extension}", file_path=file_path)
