# Shared Adapters

Multi-cloud storage and queue adapters shared between kb-rest-service and indexer-service.

## Overview

This package provides provider-agnostic interfaces for:
- **Storage** - Azure Blob, AWS S3, GCP Cloud Storage, Local filesystem
- **Queue** - Azure Storage Queue, AWS SQS, Redis Queue

Both services use these adapters to remain cloud-independent and avoid code duplication.

## Directory Structure

```
shared/adapters/
├── __init__.py
├── README.md
├── storage/
│   ├── __init__.py
│   ├── protocols.py         # StorageAdapter interface
│   ├── models.py            # BlobInfo data class
│   ├── factory.py           # StorageFactory
│   └── adapters/
│       ├── azure_blob.py    # Azure Blob Storage
│       ├── aws_s3.py        # AWS S3
│       ├── gcp_storage.py   # GCP Cloud Storage
│       └── local_storage.py # Local filesystem
└── queue/
    ├── __init__.py
    ├── protocols.py         # QueueAdapter interface
    ├── models.py            # QueueMessage data class
    ├── factory.py           # QueueFactory
    └── adapters/
        ├── azure_queue.py   # Azure Storage Queue
        ├── aws_sqs.py       # AWS SQS
        └── redis_queue.py   # Redis Queue
```

## Usage from kb-rest-service

```python
# Add shared to Python path in kb-rest-service
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))

from adapters.storage import get_storage_adapter
from adapters.queue import get_queue_adapter

# Create storage adapter
storage = get_storage_adapter(
    provider="azure",
    connection_string=settings.storage.AZURE_BLOB_STORAGE_CONNECTION_STRING,
    container_name=settings.storage.STORAGE_CONTAINER_NAME
)

# Create queue adapter
queue = get_queue_adapter(
    provider="azure",
    connection_string=settings.azure.AZURE_STORAGE_CONNECTION_STRING,
    queue_name=settings.azure.INDEXING_QUEUE_NAME
)
```

## Usage from indexer-service

```python
# Add shared to Python path in indexer-service
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))

from adapters.storage import get_storage_adapter
from adapters.queue import get_queue_adapter
from src.core.config import settings

# Create storage adapter
storage = get_storage_adapter(
    provider=settings.storage.STORAGE_PROVIDER,
    connection_string=settings.storage.AZURE_BLOB_STORAGE_CONNECTION_STRING,
    container_name=settings.storage.STORAGE_CONTAINER_NAME
)

# Create queue adapter
queue = get_queue_adapter(
    provider=settings.queue.QUEUE_PROVIDER,
    connection_string=settings.azure.AZURE_STORAGE_CONNECTION_STRING,
    queue_name=settings.azure.INDEXING_QUEUE_NAME
)
```

## Wrapper Pattern (Recommended)

Instead of modifying all imports across both services, create service-specific wrappers:

### kb-rest-service/src/storage/factory.py

```python
"""Wrapper for shared storage adapters"""
import sys
from pathlib import Path

# Add shared to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))

from adapters.storage import get_storage_adapter as _get_storage_adapter, StorageFactory as _StorageFactory
from core.config import settings

def get_storage_adapter(force_recreate: bool = False):
    """Get storage adapter configured from settings"""
    return _get_storage_adapter(
        provider=settings.storage.STORAGE_PROVIDER,
        connection_string=settings.storage.AZURE_BLOB_STORAGE_CONNECTION_STRING,
        container_name=settings.storage.STORAGE_CONTAINER_NAME
    )

# Re-export for compatibility
StorageFactory = _StorageFactory
```

### indexer-service/src/storage/factory.py

```python
"""Wrapper for shared storage adapters"""
import sys
from pathlib import Path

# Add shared to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))

from adapters.storage import get_storage_adapter as _get_storage_adapter, StorageFactory as _StorageFactory
from src.core.config import settings

def get_storage_adapter(force_recreate: bool = False):
    """Get storage adapter configured from settings"""
    return _get_storage_adapter(
        provider=settings.storage.STORAGE_PROVIDER,
        connection_string=settings.storage.AZURE_BLOB_STORAGE_CONNECTION_STRING,
        container_name=settings.storage.STORAGE_CONTAINER_NAME
    )

# Re-export for compatibility
StorageFactory = _StorageFactory
```

## Benefits

✅ **Single Source of Truth** - One implementation shared by both services  
✅ **No Code Duplication** - Changes propagate to both services automatically  
✅ **Provider Agnostic** - Switch clouds without code changes  
✅ **Testable** - Easy to mock in unit tests  
✅ **Maintainable** - Fix bugs once, benefits both services  

## Migration Steps

### Option 1: Direct Import (Requires path setup in each file)

1. Add shared to Python path
2. Import from `adapters.storage` or `adapters.queue`
3. Pass configuration explicitly

### Option 2: Wrapper Pattern (Recommended)

1. Update service-specific `factory.py` to wrap shared adapters
2. Existing code continues to work with `from storage.factory import get_storage_adapter`
3. No changes needed in business logic

## Adding New Providers

1. Implement the `StorageAdapter` or `QueueAdapter` protocol
2. Add to factory's provider enum
3. Update factory's `create()` method
4. Both services automatically support the new provider!

Example:

```python
# shared/adapters/storage/adapters/my_cloud.py
from ..protocols import StorageAdapter
from ..models import BlobInfo

class MyCloudStorageAdapter(StorageAdapter):
    async def upload(self, filename: str, data: bytes, content_type=None) -> BlobInfo:
        # Implementation
        pass
    
    # ... implement other methods
```

Then update `factory.py`:

```python
elif provider_enum == StorageProvider.MYCLOUD:
    from .adapters.my_cloud import MyCloudStorageAdapter
    return MyCloudStorageAdapter(**kwargs)
```

## Testing

Create mock adapters for testing:

```python
from shared.adapters.storage.protocols import StorageAdapter
from shared.adapters.storage.models import BlobInfo

class MockStorageAdapter(StorageAdapter):
    def __init__(self):
        self.uploads = {}
    
    async def upload(self, filename: str, data: bytes, content_type=None) -> BlobInfo:
        self.uploads[filename] = data
        return BlobInfo(
            container="test",
            blob_name=filename,
            blob_url=f"https://test/{filename}",
            provider="mock",
            size_bytes=len(data)
        )
    
    # ... implement other methods
```

## Future Enhancements

- [ ] Add caching layer for frequently accessed blobs
- [ ] Implement retry logic with exponential backoff
- [ ] Add circuit breaker pattern
- [ ] Add metrics/telemetry
- [ ] Support for streaming large files
- [ ] Batch operations for efficiency
