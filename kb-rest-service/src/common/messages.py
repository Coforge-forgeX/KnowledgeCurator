"""Standardized messages for API responses"""


class ErrorMessages:
    """Error message constants"""

    # Authentication & Authorization
    AUTHENTICATION_REQUIRED = "Authentication required. Please provide a valid token."
    INVALID_TOKEN = "Invalid or expired token"
    UNAUTHORIZED_ACCESS = "You are not authorized to access this resource"
    INSUFFICIENT_PERMISSIONS = "You do not have sufficient permissions for this operation"

    # Validation
    INVALID_REQUEST = "Invalid request data"
    MISSING_REQUIRED_FIELD = "Missing required field: {field}"
    INVALID_FIELD_VALUE = "Invalid value for field: {field}"
    INVALID_QUERY_PARAMETERS = "Invalid query parameters provided"

    # Database
    DATABASE_ERROR = "A database error occurred"
    RECORD_NOT_FOUND = "{resource} not found"
    DUPLICATE_RECORD = "{resource} already exists"
    DATABASE_CONNECTION_ERROR = "Failed to connect to database"

    # LightRAG & Knowledge Base
    LIGHTRAG_INITIALIZATION_ERROR = "Failed to initialize LightRAG service"
    QUERY_EXECUTION_ERROR = "Failed to execute query"
    DOCUMENT_INDEXING_ERROR = "Failed to index document"
    DOCUMENT_DELETION_ERROR = "Failed to delete document"
    INVALID_QUERY_MODE = "Invalid query mode. Must be one of: naive, local, global, hybrid"

    # Queue Operations
    QUEUE_SEND_ERROR = "Failed to send message to queue"
    QUEUE_RECEIVE_ERROR = "Failed to receive message from queue"
    INDEXING_QUEUE_ERROR = "Failed to queue document for indexing"

    # File Operations
    FILE_NOT_FOUND = "File not found: {filename}"
    FILE_UPLOAD_ERROR = "Failed to upload file"
    FILE_DELETE_ERROR = "Failed to delete file"
    INVALID_FILE_TYPE = "Invalid file type. Allowed types: {types}"
    FILE_TOO_LARGE = "File size exceeds maximum allowed size of {max_size}"

    # Blob Storage
    BLOB_UPLOAD_ERROR = "Failed to upload to blob storage"
    BLOB_DOWNLOAD_ERROR = "Failed to download from blob storage"
    BLOB_DELETE_ERROR = "Failed to delete from blob storage"
    BLOB_NOT_FOUND = "Blob not found: {blob_name}"

    # Neo4j
    NEO4J_CONNECTION_ERROR = "Failed to connect to Neo4j database"
    NEO4J_QUERY_ERROR = "Failed to execute Neo4j query"

    # Configuration
    CONFIGURATION_ERROR = "Configuration error: {detail}"
    MISSING_CONFIGURATION = "Missing required configuration: {config_key}"

    # General
    INTERNAL_SERVER_ERROR = "An internal server error occurred"
    SERVICE_UNAVAILABLE = "Service temporarily unavailable"
    REQUEST_TIMEOUT = "Request timed out"
    RATE_LIMIT_EXCEEDED = "Rate limit exceeded. Please try again later"


class SuccessMessages:
    """Success message constants"""

    # Query Operations
    QUERY_EXECUTED = "Query executed successfully"
    RESULTS_RETRIEVED = "Results retrieved successfully"

    # Document Operations
    DOCUMENT_INDEXED = "Document indexed successfully"
    DOCUMENT_DELETED = "Document deleted successfully"
    DOCUMENTS_INDEXED = "{count} documents indexed successfully"
    DOCUMENTS_DELETED = "{count} documents deleted successfully"

    # File Operations
    FILE_UPLOADED = "File uploaded successfully"
    FILE_DELETED = "File deleted successfully"
    FILES_UPLOADED = "{count} files uploaded successfully"

    # Indexing Queue
    INDEXING_QUEUED = "Document queued for indexing"
    INDEXING_STARTED = "Indexing process started"
    INDEXING_COMPLETED = "Indexing completed successfully"

    # General
    OPERATION_SUCCESSFUL = "Operation completed successfully"
    REQUEST_PROCESSED = "Request processed successfully"


class InfoMessages:
    """Informational message constants"""

    # Processing Status
    PROCESSING = "Processing your request..."
    INDEXING_IN_PROGRESS = "Document indexing in progress"
    QUERY_PROCESSING = "Processing query..."

    # Warnings
    NO_RESULTS_FOUND = "No results found for your query"
    PARTIAL_SUCCESS = "Operation completed with warnings"
    DEPRECATED_ENDPOINT = "This endpoint is deprecated. Please use {new_endpoint} instead"

    # Configuration
    USING_DEFAULT_CONFIGURATION = "Using default configuration for {component}"
    CONFIGURATION_LOADED = "Configuration loaded successfully"
