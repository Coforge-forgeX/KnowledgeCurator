"""
Helper for minting fresh Azure Blob Storage SAS download URLs on demand.

Background
----------
SAS download URLs are time-limited (read-only, ~7 day expiry). Previously the
fully-formed URL (with the SAS token baked in) was persisted into chat history.
When a user reopened an old session and clicked a stored link, the embedded SAS
token had long expired and Azure returned:

    AuthenticationFailed - Signed expiry time [...] must be after signed start time [...]

The fix is to persist only the blob *coordinates* (container_name + blob_path)
and mint a brand-new SAS token at the moment the link is served. This module is
that single, reusable minting point.
"""

import os
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse
from typing import Optional

from azure.storage.blob import generate_blob_sas, BlobSasPermissions, BlobServiceClient

# Absorb small clock skew between this app server and Azure so a freshly minted
# token is never rejected for starting "in the future".
_START_SKEW = timedelta(minutes=5)


def build_sas_url(
    container_name: str,
    blob_path: str,
    file_name: Optional[str] = None,
    expiry_days: int = 7,
) -> Optional[str]:
    """
    Mint a fresh read-only SAS download URL for a blob.

    Args:
        container_name: Azure blob container.
        blob_path: Full path of the blob within the container.
        file_name: Optional name forced via content-disposition (browser download).
        expiry_days: Days until the SAS token expires.

    Returns:
        A fully-formed download URL, or None if configuration/coordinates are missing.
    """
    if not container_name or not blob_path:
        return None

    connection_string = os.getenv("AZURE_BLOB_STORAGE_CONNECTION_STRING")
    if not connection_string:
        return None

    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        account_name = blob_service_client.account_name
        account_key = blob_service_client.credential.account_key
        if not account_name or not account_key:
            return None

        # Timezone-aware UTC avoids the naive datetime.now() (local time) bug:
        # Azure interprets the value as UTC, so local time skews the expiry.
        now_utc = datetime.now(timezone.utc)
        download_name = file_name or os.path.basename(blob_path)

        sas_token = generate_blob_sas(
            account_name=account_name,
            container_name=container_name,
            blob_name=blob_path,
            account_key=account_key,
            permission=BlobSasPermissions(read=True),
            start=now_utc - _START_SKEW,
            expiry=now_utc + timedelta(days=expiry_days),
            content_disposition=f'attachment; filename="{download_name}"',
        )
        return (
            f"https://{account_name}.blob.core.windows.net/"
            f"{container_name}/{blob_path}?{sas_token}"
        )
    except Exception:
        return None


def refresh_source_url(source: dict, expiry_days: int = 7) -> dict:
    """
    Return a copy of a persisted "source" dict with a freshly minted download_url.

    The source must carry blob coordinates (``container_name`` + ``blob_path``)
    persisted at creation time. If those are missing (legacy records saved before
    this fix), the original dict is returned unchanged.
    """
    if not isinstance(source, dict):
        return source

    container_name = source.get("container_name")
    blob_path = source.get("blob_path")
    if not container_name or not blob_path:
        # Legacy record without coordinates: nothing we can safely regenerate.
        return source

    # If we already have a SAS URL that is still valid for long enough, reuse it
    # instead of minting a new one on every history read.
    existing_url = source.get("download_url")
    if isinstance(existing_url, str) and existing_url:
        try:
            parsed = urlparse(existing_url)
            qs = parse_qs(parsed.query)
            # Azure SAS expiry is in `se` (signed expiry). Example format:
            # 2026-07-01T11:42:12Z
            se = (qs.get("se") or [None])[0]
            sig = (qs.get("sig") or [None])[0]
            if se and sig:
                expiry_utc = datetime.strptime(se, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                now_utc = datetime.now(timezone.utc)

                # Refresh only when expired or close to expiry.
                # Keep a buffer larger than the minting start-skew so clients
                # don't get a URL that is about to expire mid-download.
                refresh_buffer = max(timedelta(minutes=15), _START_SKEW * 2)
                if expiry_utc - now_utc > refresh_buffer:
                    return source
        except Exception:
            # Any parsing error falls back to minting a fresh URL.
            pass

    fresh_url = build_sas_url(
        container_name=container_name,
        blob_path=blob_path,
        file_name=source.get("download_name") or os.path.basename(blob_path),
        expiry_days=expiry_days,
    )
    if not fresh_url:
        return source

    refreshed = dict(source)
    refreshed["download_url"] = fresh_url
    return refreshed
