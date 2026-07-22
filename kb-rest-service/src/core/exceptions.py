"""Custom exceptions for kb-rest-service"""
from typing import Any, Dict, Optional


class APIException(Exception):
    """Base API exception"""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)


class ValidationException(APIException):
    """Validation error"""

    def __init__(
        self,
        message: str = "Validation failed",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            status_code=400,
            error_code="VALIDATION_ERROR",
            details=details,
        )


class AuthenticationException(APIException):
    """Authentication error"""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(
            message=message, status_code=401, error_code="AUTHENTICATION_ERROR"
        )


class AuthorizationException(APIException):
    """Authorization error"""

    def __init__(self, message: str = "Access forbidden"):
        super().__init__(
            message=message, status_code=403, error_code="AUTHORIZATION_ERROR"
        )


class NotFoundException(APIException):
    """Resource not found"""

    def __init__(
        self, message: str = "Resource not found", resource_type: str = "resource"
    ):
        super().__init__(
            message=message,
            status_code=404,
            error_code="NOT_FOUND",
            details={"resource_type": resource_type},
        )


class ConflictException(APIException):
    """Resource conflict"""

    def __init__(
        self, message: str = "Resource conflict", resource_type: str = "resource"
    ):
        super().__init__(
            message=message,
            status_code=409,
            error_code="CONFLICT",
            details={"resource_type": resource_type},
        )


class DatabaseException(APIException):
    """Database error"""

    def __init__(self, message: str = "Database error", operation: str = "unknown"):
        super().__init__(
            message=message,
            status_code=500,
            error_code="DATABASE_ERROR",
            details={"operation": operation},
        )


class ConfigurationException(APIException):
    """Configuration error"""

    def __init__(
        self, message: str = "Configuration error", config_key: str = "unknown"
    ):
        super().__init__(
            message=message,
            status_code=500,
            error_code="CONFIGURATION_ERROR",
            details={"config_key": config_key},
        )


class BusinessLogicException(APIException):
    """Business logic error"""

    def __init__(self, message: str = "Business logic error", rule: str = "unknown"):
        super().__init__(
            message=message,
            status_code=422,
            error_code="BUSINESS_LOGIC_ERROR",
            details={"rule": rule},
        )


class ExternalServiceException(APIException):
    """External service error"""

    def __init__(
        self, message: str = "External service error", service: str = "unknown"
    ):
        super().__init__(
            message=message,
            status_code=502,
            error_code="EXTERNAL_SERVICE_ERROR",
            details={"service": service},
        )


class LightRAGException(APIException):
    """LightRAG operation error"""

    def __init__(self, message: str = "LightRAG operation failed", operation: str = "unknown"):
        super().__init__(
            message=message,
            status_code=500,
            error_code="LIGHTRAG_ERROR",
            details={"operation": operation},
        )


class QueueException(APIException):
    """Queue operation error"""

    def __init__(self, message: str = "Queue operation failed", operation: str = "unknown"):
        super().__init__(
            message=message,
            status_code=500,
            error_code="QUEUE_ERROR",
            details={"operation": operation},
        )
