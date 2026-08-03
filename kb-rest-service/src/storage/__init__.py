"""Storage abstraction layer for multi-cloud deployment"""

from .factory import StorageFactory, get_storage_adapter
from .models import BlobInfo
from .protocols import StorageAdapter

__all__ = [
    "StorageAdapter",
    "StorageFactory",
    "BlobInfo",
    "get_storage_adapter",
]
