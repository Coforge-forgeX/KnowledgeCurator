"""GCP Document AI OCR adapter."""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class GCPDocumentAIAdapter:
    """GCP Document AI OCR implementation."""

    def __init__(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        processor_id: Optional[str] = None,
        credentials_path: Optional[str] = None,
    ) -> None:
        """
        Initialize GCP Document AI adapter.

        Args:
            project_id: GCP project ID
            location: Processor location (e.g., 'us', 'eu')
            processor_id: Document AI processor ID (if None, creates OCR processor)
            credentials_path: Path to service account JSON
        """
        self._project_id = project_id
        self._location = location or "us"
        self._processor_id = processor_id
        self._credentials_path = credentials_path

    @property
    def is_configured(self) -> bool:
        """Check if GCP credentials are configured."""
        return bool(self._project_id)

    @property
    def provider_name(self) -> str:
        return "gcp"

    async def extract_text(self, file_bytes: bytes, file_path: str) -> str:
        """
        Extract text using GCP Document AI.

        Args:
            file_bytes: Raw file bytes
            file_path: File name for logging

        Returns:
            Extracted text content

        Raises:
            ImportError: If google-cloud-documentai not installed
            Exception: If OCR fails
        """
        if not self.is_configured:
            raise ValueError("GCP Document AI credentials not configured (missing project_id)")

        try:
            from google.cloud import documentai_v1 as documentai
        except ImportError as e:
            raise ImportError(
                "google-cloud-documentai not installed. "
                "Install with: pip install google-cloud-documentai"
            ) from e

        def _extract() -> str:
            # Initialize client
            if self._credentials_path:
                from google.oauth2 import service_account
                credentials = service_account.Credentials.from_service_account_file(
                    self._credentials_path
                )
                client = documentai.DocumentProcessorServiceClient(credentials=credentials)
            else:
                # Use default credentials (GOOGLE_APPLICATION_CREDENTIALS env var)
                client = documentai.DocumentProcessorServiceClient()

            # If no processor_id provided, use general OCR processor
            if self._processor_id:
                processor_name = client.processor_path(
                    self._project_id, self._location, self._processor_id
                )
            else:
                # Use the general processor type for OCR
                processor_name = client.processor_path(
                    self._project_id, self._location, "OCR_PROCESSOR"
                )

            # Determine mime type from file extension
            mime_type = "application/pdf"
            if file_path.lower().endswith((".jpg", ".jpeg")):
                mime_type = "image/jpeg"
            elif file_path.lower().endswith(".png"):
                mime_type = "image/png"
            elif file_path.lower().endswith(".tiff"):
                mime_type = "image/tiff"

            # Create request
            raw_document = documentai.RawDocument(content=file_bytes, mime_type=mime_type)
            request = documentai.ProcessRequest(name=processor_name, raw_document=raw_document)

            # Process document
            result = client.process_document(request=request)

            return result.document.text or ""

        return await asyncio.to_thread(_extract)
