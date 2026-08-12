"""Data models and exceptions for text extraction."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TextExtractionResult:
    """Normalized text extraction result."""

    text: str
    extractor: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class TextExtractionError(Exception):
    """Raised when text extraction fails."""

    def __init__(
        self,
        message: str,
        *,
        file_path: Optional[str] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        self.message = message
        self.file_path = file_path
        self.cause = cause
        super().__init__(message)
