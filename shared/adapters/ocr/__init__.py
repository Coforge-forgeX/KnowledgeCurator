"""OCR adapters - multi-cloud document OCR support."""

from .adapters import (
    AWSTextractAdapter,
    AzureDocumentIntelligenceAdapter,
    GCPDocumentAIAdapter,
    NoOpOCRAdapter,
)
from .factory import get_ocr_adapter
from .protocols import OCRAdapter

__all__ = [
    "OCRAdapter",
    "get_ocr_adapter",
    "AzureDocumentIntelligenceAdapter",
    "AWSTextractAdapter",
    "GCPDocumentAIAdapter",
    "NoOpOCRAdapter",
]
