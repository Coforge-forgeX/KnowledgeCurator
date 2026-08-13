"""Azure Document Intelligence OCR adapter."""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AzureDocumentIntelligenceAdapter:
    """Azure Document Intelligence OCR implementation."""

    def __init__(self, endpoint: Optional[str] = None, api_key: Optional[str] = None) -> None:
        """
        Initialize Azure Document Intelligence adapter.

        Args:
            endpoint: Azure Document Intelligence endpoint
            api_key: Azure Document Intelligence API key
        """
        self._endpoint = endpoint
        self._api_key = api_key

    @property
    def is_configured(self) -> bool:
        """Check if Azure credentials are configured."""
        return bool(self._endpoint and self._api_key)

    @property
    def provider_name(self) -> str:
        return "azure"

    async def extract_text(self, file_bytes: bytes, file_path: str) -> str:
        """
        Extract text using Azure Document Intelligence.

        Args:
            file_bytes: Raw file bytes
            file_path: File name for logging

        Returns:
            Extracted text content

        Raises:
            ImportError: If azure-ai-documentintelligence not installed
            Exception: If OCR fails
        """
        if not self.is_configured:
            raise ValueError("Azure Document Intelligence credentials not configured")

        try:
            from azure.ai.documentintelligence import DocumentIntelligenceClient
            from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
            from azure.core.credentials import AzureKeyCredential
        except ImportError as e:
            raise ImportError(
                "azure-ai-documentintelligence not installed. "
                "Install with: pip install azure-ai-documentintelligence"
            ) from e

        client = DocumentIntelligenceClient(self._endpoint, AzureKeyCredential(self._api_key))

        poller = await asyncio.to_thread(
            client.begin_analyze_document,
            "prebuilt-read",
            body=AnalyzeDocumentRequest(bytes_source=file_bytes),
            locale="en-US",
        )
        result = await asyncio.to_thread(poller.result)

        return result.content or ""
