"""OCR adapter factory - creates the appropriate OCR adapter based on configuration."""

import logging
from typing import Optional

from .adapters import (
    AWSTextractAdapter,
    AzureDocumentIntelligenceAdapter,
    GCPDocumentAIAdapter,
    NoOpOCRAdapter,
)
from .protocols import OCRAdapter

logger = logging.getLogger(__name__)


def get_ocr_adapter(
    provider: str,
    # Azure Document Intelligence
    azure_endpoint: Optional[str] = None,
    azure_api_key: Optional[str] = None,
    # AWS Textract
    aws_region: Optional[str] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    # GCP Document AI
    gcp_project_id: Optional[str] = None,
    gcp_location: Optional[str] = None,
    gcp_processor_id: Optional[str] = None,
    gcp_credentials_path: Optional[str] = None,
) -> OCRAdapter:
    """
    Create an OCR adapter based on provider.

    Args:
        provider: OCR provider ('azure', 'aws', 'gcp', or 'none')
        azure_endpoint: Azure Document Intelligence endpoint
        azure_api_key: Azure Document Intelligence API key
        aws_region: AWS region
        aws_access_key_id: AWS access key ID
        aws_secret_access_key: AWS secret access key
        gcp_project_id: GCP project ID
        gcp_location: GCP processor location
        gcp_processor_id: GCP Document AI processor ID
        gcp_credentials_path: Path to GCP service account JSON

    Returns:
        OCRAdapter instance

    Raises:
        ValueError: If provider is unknown

    Example:
        # Azure
        ocr = get_ocr_adapter(
            provider="azure",
            azure_endpoint="https://....cognitiveservices.azure.com/",
            azure_api_key="your-key"
        )

        # AWS
        ocr = get_ocr_adapter(
            provider="aws",
            aws_region="us-east-1",
            aws_access_key_id="AKIA...",
            aws_secret_access_key="..."
        )

        # GCP
        ocr = get_ocr_adapter(
            provider="gcp",
            gcp_project_id="my-project",
            gcp_credentials_path="/path/to/service-account.json"
        )
    """
    provider = provider.lower().strip()

    if provider == "azure":
        adapter = AzureDocumentIntelligenceAdapter(
            endpoint=azure_endpoint,
            api_key=azure_api_key,
        )
        if adapter.is_configured:
            logger.info("OCR adapter initialized", provider="azure")
        else:
            logger.warning(
                "Azure Document Intelligence credentials not configured. "
                "OCR fallback will not be available."
            )
        return adapter

    elif provider == "aws":
        adapter = AWSTextractAdapter(
            region_name=aws_region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        if adapter.is_configured:
            logger.info("OCR adapter initialized", provider="aws", region=aws_region)
        else:
            logger.warning(
                "AWS Textract credentials not configured. "
                "OCR fallback will not be available."
            )
        return adapter

    elif provider == "gcp":
        adapter = GCPDocumentAIAdapter(
            project_id=gcp_project_id,
            location=gcp_location,
            processor_id=gcp_processor_id,
            credentials_path=gcp_credentials_path,
        )
        if adapter.is_configured:
            logger.info("OCR adapter initialized", provider="gcp", project=gcp_project_id)
        else:
            logger.warning(
                "GCP Document AI credentials not configured. "
                "OCR fallback will not be available."
            )
        return adapter

    elif provider in ("none", "disabled", ""):
        logger.info("OCR adapter disabled")
        return NoOpOCRAdapter()

    else:
        raise ValueError(
            f"Unknown OCR provider: {provider}. "
            f"Must be one of: 'azure', 'aws', 'gcp', 'none'"
        )
