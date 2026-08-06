"""Cloud provider detection utility for progress module"""
from __future__ import annotations

from enum import Enum
from typing import Optional


class CloudProvider(str, Enum):
    AZURE = "azure"
    AWS = "aws"
    GCP = "gcp"
    LOCAL = "local"


def resolve_cloud_provider(
    explicit_provider: Optional[str] = None,
    aws_region: Optional[str] = None,
    azure_website_name: Optional[str] = None,
    azure_webjobs_storage: Optional[str] = None,
    gcp_project_id: Optional[str] = None,
) -> CloudProvider:
    """
    Auto-detect cloud provider from configuration.

    Args:
        explicit_provider: Explicit provider name (azure/aws/gcp/local)
        aws_region: AWS region (indicates AWS environment)
        azure_website_name: Azure website name (indicates Azure environment)
        azure_webjobs_storage: Azure WebJobs storage (indicates Azure environment)
        gcp_project_id: GCP project ID (indicates GCP environment)

    Returns:
        CloudProvider enum

    Checks in order:
    1. Explicit provider if specified
    2. AWS-specific config (region)
    3. Azure-specific config (website name, webjobs storage)
    4. GCP-specific config (project ID)
    5. Defaults to LOCAL
    """
    # Explicit provider takes precedence
    if explicit_provider:
        provider = explicit_provider.strip().lower()
        if provider in {"azure", "aws", "gcp", "local"}:
            return CloudProvider(provider)

    # Auto-detect from environment indicators
    if aws_region:
        return CloudProvider.AWS

    if azure_website_name or azure_webjobs_storage:
        return CloudProvider.AZURE

    if gcp_project_id:
        return CloudProvider.GCP

    return CloudProvider.LOCAL
