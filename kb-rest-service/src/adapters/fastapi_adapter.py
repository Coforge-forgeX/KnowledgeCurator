"""
FastAPI adapter for multi-cloud deployment.

This is the ONLY adapter needed for all three cloud providers:
- Azure Container Apps
- AWS App Runner / Lambda Container
- GCP Cloud Run

The adapter wraps Starlette Request objects (used by FastAPI) and
converts them to our abstract request/response/context interfaces.
"""
import json
import uuid
from typing import Any, Dict, Optional

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from src.core.abstractions import AbstractContext, AbstractRequest, AbstractResponse
from src.core.logging import get_logger

logger = get_logger(__name__)


class FastAPIRequest(AbstractRequest):
    """
    Wraps Starlette Request to AbstractRequest interface.

    This adapter allows FastAPI handlers to use the same interface
    as Azure Functions handlers, enabling code reuse across platforms.
    """

    def __init__(self, request: Request):
        """
        Initialize FastAPI request wrapper.

        Args:
            request: Starlette Request object from FastAPI
        """
        self._request = request
        self._cached_json: Optional[Dict[str, Any]] = None

    def get_header(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get request header (case-insensitive).

        Starlette headers are case-insensitive, so this works correctly
        regardless of the case used in the header name.

        Args:
            name: Header name (e.g., "Authorization", "authorization")
            default: Default value if header not found

        Returns:
            Header value or default if not found
        """
        return self._request.headers.get(name.lower(), default)

    def get_json(self) -> Dict[str, Any]:
        """
        Parse and return request body as JSON.

        The body is cached after first parse to avoid re-parsing on
        subsequent calls (important for performance).

        If FastAPI has already parsed the body (stored in request.state.parsed_payload),
        use that instead of re-parsing to avoid double-read issues.

        Returns:
            Dictionary containing parsed JSON body

        Raises:
            ValueError: If body is not valid JSON
        """
        if self._cached_json is None:
            # Check if FastAPI already parsed the payload (for Swagger/validation)
            if hasattr(self._request.state, "parsed_payload"):
                from pydantic import BaseModel
                parsed_payload = self._request.state.parsed_payload

                # Convert Pydantic model to dict
                if isinstance(parsed_payload, BaseModel):
                    self._cached_json = parsed_payload.model_dump()
                else:
                    self._cached_json = parsed_payload
            else:
                # Fall back to reading raw body
                try:
                    # In FastAPI, the body is already parsed during request handling
                    # Access the raw body bytes
                    body_bytes = self._request._body
                    if body_bytes:
                        self._cached_json = json.loads(body_bytes.decode("utf-8"))
                    else:
                        self._cached_json = {}
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    logger.error(
                        "Failed to parse JSON body",
                        error=e,
                        error_type=type(e).__name__,
                    )
                    raise ValueError(f"Invalid JSON in request body: {e}") from e
                except Exception as e:
                    logger.error(
                        "Unexpected error parsing JSON body",
                        error=e,
                        error_type=type(e).__name__,
                    )
                    self._cached_json = {}

        # Type checker needs assurance that _cached_json is not None here
        assert self._cached_json is not None
        return self._cached_json

    def get_query_param(
        self, name: str, default: Optional[str] = None
    ) -> Optional[str]:
        """
        Get single query parameter by name.

        Args:
            name: Query parameter name
            default: Default value if parameter not found

        Returns:
            Query parameter value or default if not found
        """
        return self._request.query_params.get(name, default)

    def get_query_params(self) -> Dict[str, str]:
        """
        Get all query parameters as dictionary.

        Returns:
            Dictionary of all query parameters
        """
        return dict(self._request.query_params)

    def get_cookies(self) -> Dict[str, str]:
        """
        Get request cookies.

        Returns:
            Dictionary of all cookies
        """
        return dict(self._request.cookies)

    @property
    def method(self) -> str:
        """
        HTTP method (GET, POST, PUT, DELETE, etc.)

        Returns:
            HTTP method in uppercase
        """
        return self._request.method

    @property
    def url(self) -> str:
        """
        Full request URL including protocol, host, path, and query string.

        Returns:
            Complete request URL
        """
        return str(self._request.url)

    @property
    def path(self) -> str:
        """
        URL path component without query string.

        Returns:
            Request path (e.g., "/api/login")
        """
        return self._request.url.path


class FastAPIResponse(AbstractResponse):
    """
    Converts AbstractResponse to FastAPI JSONResponse.

    This adapter allows handlers to return abstract responses
    that get automatically converted to FastAPI-compatible responses.
    """

    def to_platform_response(self) -> Response:
        """
        Convert to FastAPI JSONResponse.

        Handles both string and dict/list response bodies,
        automatically JSON-serializing structured data.

        Returns:
            FastAPI JSONResponse object
        """
        # Determine content type based on body
        content: str | dict | list
        if isinstance(self.body, str):
            content = self.body
        elif isinstance(self.body, (dict, list)):
            content = self.body
        else:
            # Convert other types to string
            content = str(self.body)

        return JSONResponse(
            content=content,
            status_code=self.status_code,
            headers=self.headers,
            media_type=self.mimetype,
        )


class FastAPIContext(AbstractContext):
    """
    Synthetic execution context for FastAPI.

    Since FastAPI doesn't have a built-in execution context like
    Azure Functions, we create a synthetic one that provides the
    same interface for consistency across platforms.
    """

    def __init__(self, request: Request):
        """
        Initialize FastAPI execution context.

        Args:
            request: Starlette Request object
        """
        self._request = request

        # Try to get correlation ID from headers or state (set by middleware)
        self._request_id = (
            getattr(request.state, "correlation_id", None)
            or request.headers.get("x-correlation-id")
            or request.headers.get("X-Correlation-ID")
            or request.headers.get("x-request-id")
            or request.headers.get("X-Request-ID")
            or str(uuid.uuid4())
        )

        # Extract function name from URL path
        # For "/login" -> "login"
        # For "/api/workspace-create" -> "workspace-create"
        path = request.url.path.lstrip("/")
        self._function_name = path.split("/")[0] if path else "root"

    @property
    def request_id(self) -> str:
        """
        Unique identifier for this request.

        Returns:
            Request ID (from header or generated UUID)
        """
        return self._request_id

    @property
    def function_name(self) -> str:
        """
        Name of the function/endpoint being executed.

        Returns:
            Function name derived from URL path
        """
        return self._function_name

    @property
    def correlation_id(self) -> str:
        """
        Correlation ID for distributed tracing.

        Returns:
            Same as request_id
        """
        return self._request_id
