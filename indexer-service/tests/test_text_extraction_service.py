"""Tests for robust text extraction and payload normalization."""

import asyncio
import base64

import pytest

from services.text_extraction.decoders import DOCX_SIGNATURE, normalize_file_bytes
from services.text_extraction.extractor import DocExtractor, DocxExtractor
from services.text_extraction.models import TextExtractionError


def test_docx_extractor_rejects_non_docx_payload() -> None:
    extractor = DocxExtractor()

    with pytest.raises(TextExtractionError):
        asyncio.run(extractor.extract(b"not-a-docx-payload", "broken.docx"))


def test_doc_extractor_fails_without_supported_legacy_payload() -> None:
    extractor = DocExtractor()

    with pytest.raises(TextExtractionError):
        asyncio.run(extractor.extract(b"plain-text-that-is-not-doc", "legacy.doc"))


def test_normalize_file_bytes_decodes_base64_data_uri_for_docx() -> None:
    original = DOCX_SIGNATURE + b"fake-docx-binary"
    payload = (
        "data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,"
        + base64.b64encode(original).decode("ascii")
    )

    normalized = normalize_file_bytes(payload.encode("utf-8"), "resume.docx")

    assert normalized == original


def test_normalize_file_bytes_keeps_raw_binary_for_non_matching_signature() -> None:
    # This base64 string decodes, but not to a valid .docx signature.
    payload = base64.b64encode(b"definitely-not-a-docx").decode("ascii")

    normalized = normalize_file_bytes(payload.encode("utf-8"), "resume.docx")

    assert normalized == payload.encode("utf-8")
