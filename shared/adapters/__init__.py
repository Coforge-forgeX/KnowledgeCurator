"""
Shared adapters package for multi-cloud operations.

This package provides unified interfaces for storage and queue operations
across different cloud providers (Azure, AWS, GCP).

Both kb-rest-service and indexer-service use these adapters to remain
cloud-agnostic.
"""
from .storage import get_storage_adapter, StorageAdapter, StorageFactory
from .queue import get_queue_adapter, QueueAdapter, QueueFactory

__all__ = [
    # Storage
    "StorageAdapter",
    "StorageFactory",
    "get_storage_adapter",
    # Queue
    "QueueAdapter",
    "QueueFactory",
    "get_queue_adapter",
]
