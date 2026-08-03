"""
Shared Models (Used by both indexer-service and kb-rest-service)

Only models that are shared across services should be here.
Service-specific models belong in that service's directory.
"""

from .indexing_models import IndexingJob, IndexingStatusEnum

__all__ = [
    "IndexingJob",
    "IndexingStatusEnum",
]
