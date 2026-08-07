"""Helpers for normalizing file payloads before extraction."""

import base64
import os

DOCX_SIGNATURE = b"PK\x03\x04"
PDF_SIGNATURE = b"%PDF-"
DOC_SIGNATURE = bytes.fromhex("D0CF11E0A1B11AE1")


def _strip_data_uri_prefix(value: str) -> str:
    if value.startswith("data:") and "," in value:
        return value.split(",", 1)[1]
    return value


def _try_base64_decode(value: str) -> bytes | None:
    try:
        return base64.b64decode(value, validate=True)
    except Exception:
        return None


def _expected_signature_matches(decoded: bytes, extension: str) -> bool:
    ext = extension.lower()
    if ext == ".pdf":
        return decoded.startswith(PDF_SIGNATURE)
    if ext == ".docx":
        return decoded.startswith(DOCX_SIGNATURE)
    if ext == ".doc":
        return decoded.startswith(DOC_SIGNATURE) or decoded.startswith(b"{\\rtf")
    return False


def normalize_file_bytes(file_bytes: bytes, file_path: str) -> bytes:
    """Normalize file payload to raw binary content.

    Some upstream flows accidentally persist base64 strings instead of raw bytes.
    This method decodes only when the decoded payload matches the expected file
    signature for the extension.
    """
    if not file_bytes:
        return file_bytes

    ext = os.path.splitext(file_path)[1].lower()

    if ext not in {".pdf", ".docx", ".doc"}:
        return file_bytes

    try:
        text_candidate = file_bytes.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError:
        return file_bytes

    normalized = _strip_data_uri_prefix(text_candidate)
    decoded = _try_base64_decode(normalized)
    if not decoded:
        return file_bytes

    if _expected_signature_matches(decoded, ext):
        return decoded

    return file_bytes
