"""Storage adapter implementations for different cloud providers"""

from .azure_blob import AzureBlobStorageAdapter
from .aws_s3 import AWSS3StorageAdapter
from .gcp_storage import GCPStorageAdapter
from .local_storage import LocalStorageAdapter

__all__ = [
    "AzureBlobStorageAdapter",
    "AWSS3StorageAdapter",
    "GCPStorageAdapter",
    "LocalStorageAdapter",
]
