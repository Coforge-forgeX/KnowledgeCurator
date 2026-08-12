"""Tests for robust text extraction and payload normalization."""

import asyncio
import base64

import pytest

from shared.text_extraction.config import DocIntelligenceConfig
from shared.text_extraction.decoders import DOCX_SIGNATURE, normalize_file_bytes
from shared.text_extraction.extractor import DocExtractor, DocxExtractor, TextExtractionService
from shared.text_extraction.models import TextExtractionError


def test_doc_intelligence_config_requires_both_values() -> None:
    assert not DocIntelligenceConfig().is_configured
    assert not DocIntelligenceConfig(endpoint="https://example.net/").is_configured
    assert not DocIntelligenceConfig(api_key="secret").is_configured
    assert DocIntelligenceConfig(endpoint="https://example.net/", api_key="secret").is_configured


def test_doc_intelligence_config_from_env_prefers_long_names(monkeypatch) -> None:
    monkeypatch.setenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "https://long.example.net/")
    monkeypatch.setenv("AZURE_DOC_INTELLIGENCE_ENDPOINT", "https://short.example.net/")
    monkeypatch.setenv("AZURE_DOC_INTELLIGENCE_KEY", "short-key")

    config = DocIntelligenceConfig.from_env()

    assert config.endpoint == "https://long.example.net/"
    assert config.api_key == "short-key"


def test_injected_config_is_used_over_environment(monkeypatch) -> None:
    # Explicit config must win, so a service whose credentials live only in its
    # own settings object still gets the OCR fallbacks.
    monkeypatch.setenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "https://from-env.example.net/")
    monkeypatch.setenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", "env-key")
    injected = DocIntelligenceConfig(endpoint="https://from-settings.example.net/", api_key="settings-key")

    service = TextExtractionService(doc_intelligence=injected)

    assert service._doc_intelligence == injected


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
