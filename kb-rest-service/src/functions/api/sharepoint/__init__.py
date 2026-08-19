"""SharePoint V2 API package."""

from .payloads import (
    TestSharePointConnectionRequest,
    TestSharePointConnectionResponse,
    ToggleSharePointConnectionRequest,
    ToggleSharePointConnectionResponse,
    ExtractSharePointDataRequest,
    ExtractSharePointDataResponse,
)

__all__ = [
    "TestSharePointConnectionRequest",
    "TestSharePointConnectionResponse",
    "ToggleSharePointConnectionRequest",
    "ToggleSharePointConnectionResponse",
    "ExtractSharePointDataRequest",
    "ExtractSharePointDataResponse",
]
