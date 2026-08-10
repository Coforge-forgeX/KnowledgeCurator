"""Signed file token helpers for deterministic, non-guessable file IDs."""

import base64
import hashlib
import hmac
import json
from typing import Any, Dict, Optional

from src.core.config import settings

TOKEN_PREFIX = "qfs1_"


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * ((4 - (len(data) % 4)) % 4)
    return base64.urlsafe_b64decode((data + padding).encode("ascii"))


def _sign(payload_bytes: bytes) -> bytes:
    secret = settings.security.JWT_SECRET_KEY.encode("utf-8")
    return hmac.new(secret, payload_bytes, hashlib.sha256).digest()


def create_signed_file_id(
    *,
    workspace_id: int,
    container_name: str,
    blob_path: str,
    provider: str,
    file_name: str,
) -> str:
    """Create deterministic signed token for a file reference."""
    payload: Dict[str, Any] = {
        "v": 1,
        "workspace_id": int(workspace_id),
        "container_name": str(container_name or "").strip(),
        "blob_path": str(blob_path or "").strip(),
        "provider": str(provider or "azure").strip().lower(),
        "file_name": str(file_name or "").strip(),
    }
    payload_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    signature = _sign(payload_bytes)
    return f"{TOKEN_PREFIX}{_b64url_encode(payload_bytes)}.{_b64url_encode(signature)}"


def decode_signed_file_id(file_id: str) -> Optional[Dict[str, Any]]:
    """Verify and decode signed file token. Returns None when token is not valid."""
    if not isinstance(file_id, str) or not file_id.startswith(TOKEN_PREFIX):
        return None

    token_body = file_id[len(TOKEN_PREFIX):]
    if "." not in token_body:
        return None

    payload_part, signature_part = token_body.split(".", 1)
    try:
        payload_bytes = _b64url_decode(payload_part)
        signature = _b64url_decode(signature_part)
    except Exception:
        return None

    expected = _sign(payload_bytes)
    if not hmac.compare_digest(signature, expected):
        return None

    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    required = ["workspace_id", "container_name", "blob_path", "provider", "file_name"]
    if any(key not in payload for key in required):
        return None

    return payload
