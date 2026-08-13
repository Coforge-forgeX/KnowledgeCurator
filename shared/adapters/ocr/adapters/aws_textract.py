"""AWS Textract OCR adapter."""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AWSTextractAdapter:
    """AWS Textract OCR implementation."""

    def __init__(
        self,
        region_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ) -> None:
        """
        Initialize AWS Textract adapter.

        Args:
            region_name: AWS region (e.g., us-east-1)
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
        """
        self._region_name = region_name or "us-east-1"
        self._aws_access_key_id = aws_access_key_id
        self._aws_secret_access_key = aws_secret_access_key

    @property
    def is_configured(self) -> bool:
        """Check if AWS credentials are configured."""
        # AWS SDK can use instance roles, so we consider it configured if region exists
        # or if explicit credentials are provided
        return bool(self._region_name or (self._aws_access_key_id and self._aws_secret_access_key))

    @property
    def provider_name(self) -> str:
        return "aws"

    async def extract_text(self, file_bytes: bytes, file_path: str) -> str:
        """
        Extract text using AWS Textract.

        Args:
            file_bytes: Raw file bytes
            file_path: File name for logging

        Returns:
            Extracted text content

        Raises:
            ImportError: If boto3 not installed
            Exception: If OCR fails
        """
        try:
            import boto3
        except ImportError as e:
            raise ImportError(
                "boto3 not installed. Install with: pip install boto3"
            ) from e

        def _extract() -> str:
            # Create Textract client
            if self._aws_access_key_id and self._aws_secret_access_key:
                client = boto3.client(
                    "textract",
                    region_name=self._region_name,
                    aws_access_key_id=self._aws_access_key_id,
                    aws_secret_access_key=self._aws_secret_access_key,
                )
            else:
                # Use default credentials (instance role, environment, etc.)
                client = boto3.client("textract", region_name=self._region_name)

            # Detect document text
            response = client.detect_document_text(Document={"Bytes": file_bytes})

            # Extract text blocks
            text_parts = []
            for block in response.get("Blocks", []):
                if block["BlockType"] == "LINE":
                    text_parts.append(block.get("Text", ""))

            return "\n".join(text_parts)

        return await asyncio.to_thread(_extract)
