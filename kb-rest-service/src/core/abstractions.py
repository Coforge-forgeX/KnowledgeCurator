"""
Cloud provider-agnostic request/response/context abstractions.

This module defines abstract interfaces that allow business logic to be
independent of the underlying cloud provider (Azure, AWS, GCP).

The FastAPI adapter implements these interfaces, enabling the same handler
code to run on Azure Container Apps, AWS App Runner/Lambda, GCP Cloud Run,
or any Docker-based platform.

Note: AbstractResponse is a concrete class (not abstract) that can be
instantiated directly by response utilities.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class AbstractRequest(ABC):
    """
    Abstract HTTP request interface.

    Provides a unified interface for accessing HTTP request data
    regardless of the underlying platform (Azure Functions, FastAPI, etc.)
    """

    @abstractmethod
    def get_header(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get HTTP header value by name (case-insensitive).

        Args:
            name: Header name (e.g., "Authorization", "Content-Type")
            default: Default value if header not found

        Returns:
            Header value or default if not found
        """
        pass

    @abstractmethod
    def get_json(self) -> Dict[str, Any]:
        """
        Parse and return request body as JSON.

        Returns:
            Dictionary containing parsed JSON body

        Raises:
            ValueError: If body is not valid JSON
        """
        pass

    @abstractmethod
    def get_query_param(
        self, name: str, default: Optional[str] = None
    ) -> Optional[str]:
        """
        Get query parameter value by name.

        Args:
            name: Query parameter name
            default: Default value if parameter not found

        Returns:
            Query parameter value or default if not found
        """
        pass

    @abstractmethod
    def get_query_params(self) -> Dict[str, str]:
        """
        Get all query parameters as a dictionary.

        Returns:
            Dictionary of all query parameters
        """
        pass

    @abstractmethod
    def get_cookies(self) -> Dict[str, str]:
        """
        Get request cookies.

        Returns:
            Dictionary of all cookies
        """
        pass

    @property
    @abstractmethod
    def method(self) -> str:
        """
        HTTP method (GET, POST, PUT, DELETE, etc.)

        Returns:
            HTTP method in uppercase
        """
        pass

    @property
    @abstractmethod
    def url(self) -> str:
        """
        Full request URL including protocol, host, path, and query string.

        Returns:
            Complete request URL
        """
        pass

    @property
    @abstractmethod
    def path(self) -> str:
        """
        URL path component without query string.

        Returns:
            Request path (e.g., "/api/upload-and-index")
        """
        pass


class AbstractResponse:
    """
    Provider-agnostic HTTP response.

    This is a concrete class (not abstract) that can be instantiated directly.
    It stores response data in a platform-independent format.

    The FastAPI adapter converts it to JSONResponse via to_platform_response().
    """

    def __init__(
        self,
        body: Any,
        status_code: int = 200,
        headers: Optional[Dict[str, str]] = None,
        mimetype: str = "application/json",
    ):
        """
        Initialize abstract response.

        Args:
            body: Response body (string, dict, or list)
            status_code: HTTP status code (default: 200)
            headers: Optional response headers
            mimetype: Content type (default: "application/json")
        """
        self.body = body
        self.status_code = status_code
        self.headers = headers or {}
        self.mimetype = mimetype

    def to_platform_response(self) -> Any:
        """
        Convert to platform-specific response.

        Default implementation returns self. Adapters can override
        this method to convert to their native response type.

        Returns:
            Platform-specific response object (or self for generic use)
        """
        return self


class AbstractContext(ABC):
    """
    Abstract execution context interface.

    Provides access to request metadata and execution environment
    regardless of the underlying platform.
    """

    @property
    @abstractmethod
    def request_id(self) -> str:
        """
        Unique identifier for this request.

        Used for correlation across distributed systems and logging.

        Returns:
            Unique request ID (UUID format)
        """
        pass

    @property
    @abstractmethod
    def function_name(self) -> str:
        """
        Name of the function/endpoint being executed.

        Returns:
            Function/endpoint name (e.g., "upload_and_index", "kb_query")
        """
        pass

    @property
    def correlation_id(self) -> str:
        """
        Correlation ID for distributed tracing.

        Defaults to request_id for backward compatibility with Azure Functions.

        Returns:
            Correlation ID (same as request_id)
        """
        return self.request_id
