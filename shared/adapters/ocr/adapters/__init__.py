"""OCR adapter implementations."""

from .azure_document_intelligence import AzureDocumentIntelligenceAdapter
from .aws_textract import AWSTextractAdapter
from .gcp_document_ai import GCPDocumentAIAdapter
from .noop_ocr import NoOpOCRAdapter

__all__ = [
    "AzureDocumentIntelligenceAdapter",
    "AWSTextractAdapter",
    "GCPDocumentAIAdapter",
    "NoOpOCRAdapter",
]
