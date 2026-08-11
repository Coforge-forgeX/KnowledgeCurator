"""Storage abstraction layer for multi-cloud deployment"""

from .factory import StorageFactory, StorageProvider, get_storage_adapter
from .models import BlobInfo
from .protocols import StorageAdapter

__all__ = [
    "StorageAdapter",
    "StorageFactory",
    "StorageProvider",
    "BlobInfo",
    "get_storage_adapter",
]
