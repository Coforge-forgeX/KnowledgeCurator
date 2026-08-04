"""Storage adapter implementations for different cloud providers"""

from .azure_blob import AzureBlobStorageAdapter

# Provider-specific adapters with optional dependencies are imported lazily.
try:
    from .aws_s3 import AWSS3StorageAdapter
except Exception:  # pragma: no cover - optional dependency path
    AWSS3StorageAdapter = None

try:
    from .gcp_storage import GCPStorageAdapter
except Exception:  # pragma: no cover - optional dependency path
    GCPStorageAdapter = None

try:
    from .local_storage import LocalStorageAdapter
except Exception:  # pragma: no cover - optional dependency path
    LocalStorageAdapter = None

__all__ = [
    "AzureBlobStorageAdapter",
    "AWSS3StorageAdapter",
    "GCPStorageAdapter",
    "LocalStorageAdapter",
]
