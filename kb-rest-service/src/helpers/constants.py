"""Application constants for kb-rest-service"""
from enum import Enum


class QueryMode(str, Enum):
    """LightRAG query modes"""

    NAIVE = "naive"
    LOCAL = "local"
    GLOBAL = "global"
    HYBRID = "hybrid"


class IndexingStatus(str, Enum):
    """Document indexing status"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class StorageType(str, Enum):
    """Storage backend types"""

    FAISS = "FaissVectorDBStorage"
    NEO4J = "Neo4JStorage"
    MILVUS = "MilvusVectorDBStorage"
    CHROMA = "ChromaVectorDBStorage"


class FileType(str, Enum):
    """Supported file types for indexing"""

    PDF = "application/pdf"
    DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    TXT = "text/plain"
    MD = "text/markdown"
    HTML = "text/html"
    JSON = "application/json"


# Query defaults
DEFAULT_QUERY_MODE = QueryMode.HYBRID
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100

# Indexing defaults
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
MAX_DOCUMENT_SIZE_MB = 10
MAX_BATCH_SIZE = 100

# Queue names
INDEXING_QUEUE_NAME = "indexing-queue"
QUERY_QUEUE_NAME = "query-queue"

# Timeouts (in seconds)
QUERY_TIMEOUT = 30
INDEX_TIMEOUT = 300
QUEUE_VISIBILITY_TIMEOUT = 300

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 5

# Cache TTL (in seconds)
QUERY_CACHE_TTL = 3600
RESULT_CACHE_TTL = 1800
