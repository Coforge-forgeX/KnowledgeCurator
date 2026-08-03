"""Storage models and data classes"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BlobInfo:
    """
    Information about an uploaded blob/file.

    Attributes:
        container: Container/bucket name
        blob_name: Full blob path (with prefix if applicable)
        blob_url: Direct URL to the blob
        provider: Storage provider (azure/aws/gcp/local)
        size_bytes: File size in bytes (optional)
    """

    container: str
    blob_name: str
    blob_url: str
    provider: str
    size_bytes: Optional[int] = None
