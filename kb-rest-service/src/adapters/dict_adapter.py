"""
Dictionary-based request/context adapters for simple function calls.

These adapters allow calling AbstractRequest/AbstractContext-based handlers
with simple dictionary data, useful for testing or simplified wrappers.
"""
import uuid
from typing import Any, Dict, Optional

from src.core.abstractions import AbstractContext, AbstractRequest


class DictRequest(AbstractRequest):
    """
    Simple dictionary-based request implementation.

    Useful for wrapping dict-based function calls to work with
    AbstractRequest-based handlers.
    """

    def __init__(
        self,
        body: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        query_params: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        method: str = "POST",
        url: str = "",
        path: str = "/api",
    ):
        """
        Initialize dictionary-based request.

        Args:
            body: Request body as dictionary
            headers: Request headers
            query_params: Query parameters
            cookies: Request cookies
            method: HTTP method
            url: Full URL
            path: URL path
        """
        self._body = body or {}
        self._headers = {k.lower(): v for k, v in (headers or {}).items()}
        self._query_params = query_params or {}
        self._cookies = cookies or {}
        self._method = method
        self._url = url
        self._path = path

    def get_header(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """Get HTTP header value by name (case-insensitive)"""
        return self._headers.get(name.lower(), default)

    def get_json(self) -> Dict[str, Any]:
        """Return request body as JSON dictionary"""
        return self._body

    def get_query_param(
        self, name: str, default: Optional[str] = None
    ) -> Optional[str]:
        """Get query parameter value by name"""
        return self._query_params.get(name, default)

    def get_query_params(self) -> Dict[str, str]:
        """Get all query parameters as dictionary"""
        return self._query_params

    def get_cookies(self) -> Dict[str, str]:
        """Get request cookies"""
        return self._cookies

    @property
    def method(self) -> str:
        """HTTP method"""
        return self._method

    @property
    def url(self) -> str:
        """Full request URL"""
        return self._url

    @property
    def path(self) -> str:
        """URL path component"""
        return self._path

    @property
    def headers(self) -> Dict[str, str]:
        """Request headers (compatibility)"""
        return self._headers


class DictContext(AbstractContext):
    """
    Simple dictionary-based context implementation.

    Provides basic context information for AbstractContext-based handlers.
    """

    def __init__(
        self,
        request_id: Optional[str] = None,
        function_name: str = "function",
    ):
        """
        Initialize dictionary-based context.

        Args:
            request_id: Unique request ID (generates UUID if not provided)
            function_name: Name of the function being executed
        """
        self._request_id = request_id or str(uuid.uuid4())
        self._function_name = function_name

    @property
    def request_id(self) -> str:
        """Unique identifier for this request"""
        return self._request_id

    @property
    def function_name(self) -> str:
        """Name of the function/endpoint being executed"""
        return self._function_name

    @property
    def correlation_id(self) -> str:
        """Correlation ID for distributed tracing"""
        return self._request_id
