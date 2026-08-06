"""AWS Secrets Manager adapter for secrets"""

import logging
from typing import Optional
import json

from core.exceptions import ConfigurationException

from ..protocols import SecretsAdapter

logger = logging.getLogger(__name__)


class AWSSecretsManagerAdapter(SecretsAdapter):
    """AWS Secrets Manager implementation"""

    def __init__(self, region_name: Optional[str] = None) -> None:
        """
        Initialize AWS Secrets Manager adapter.

        Args:
            region_name: AWS region (e.g., us-east-1)
                        Falls back to AWS_REGION or AWS_DEFAULT_REGION env vars if not provided

        Required:
            - AWS credentials (via boto3: IAM role, credentials file, env vars, etc.)
            - region_name or AWS_REGION/AWS_DEFAULT_REGION environment variable
        """
        try:
            import boto3
        except ImportError:
            raise ConfigurationException(
                "boto3 not installed. Install with: pip install boto3",
                config_key="boto3",
            )

        import os

        self._region_name = region_name or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
        if not self._region_name:
            raise ConfigurationException(
                "region_name or AWS_REGION/AWS_DEFAULT_REGION environment variable required",
                config_key="AWS_REGION",
            )

        self._client = boto3.client("secretsmanager", region_name=self._region_name)

        logger.info(
            "AWS Secrets Manager adapter initialized",
            region=self._region_name,
        )

    @property
    def provider_name(self) -> str:
        return "aws"

    async def get_secret(self, secret_name: str, default_value: Optional[str] = None) -> Optional[str]:
        """Get secret from AWS Secrets Manager"""
        import asyncio

        try:
            response = await asyncio.to_thread(
                self._client.get_secret_value,
                SecretId=secret_name
            )

            # Handle both string and binary secrets
            if "SecretString" in response:
                secret_value = response["SecretString"]
            elif "SecretBinary" in response:
                secret_value = response["SecretBinary"].decode("utf-8")
            else:
                logger.warning(f"Secret found but no value: {secret_name}")
                return default_value

            logger.debug(f"Secret retrieved from AWS Secrets Manager: {secret_name}")
            return secret_value

        except self._client.exceptions.ResourceNotFoundException:
            logger.warning(f"Secret not found in AWS Secrets Manager: {secret_name}")
            return default_value
        except Exception as e:
            logger.error(f"Error retrieving secret from AWS Secrets Manager {secret_name}: {e}")
            return default_value

    async def set_secret(self, secret_name: str, secret_value: str) -> bool:
        """Set/update secret in AWS Secrets Manager"""
        import asyncio

        try:
            # Try to update existing secret first
            try:
                await asyncio.to_thread(
                    self._client.update_secret,
                    SecretId=secret_name,
                    SecretString=secret_value
                )
                logger.info(f"Secret updated in AWS Secrets Manager: {secret_name}")
                return True
            except self._client.exceptions.ResourceNotFoundException:
                # Secret doesn't exist, create it
                await asyncio.to_thread(
                    self._client.create_secret,
                    Name=secret_name,
                    SecretString=secret_value
                )
                logger.info(f"Secret created in AWS Secrets Manager: {secret_name}")
                return True

        except Exception as e:
            logger.error(f"Failed to set secret in AWS Secrets Manager {secret_name}: {e}")
            return False

    async def delete_secret(self, secret_name: str) -> bool:
        """Delete secret from AWS Secrets Manager (with recovery window)"""
        import asyncio

        try:
            await asyncio.to_thread(
                self._client.delete_secret,
                SecretId=secret_name,
                RecoveryWindowInDays=7  # 7-day recovery window
            )
            logger.info(f"Secret deletion scheduled in AWS Secrets Manager: {secret_name}")
            return True
        except self._client.exceptions.ResourceNotFoundException:
            logger.warning(f"Secret not found for deletion: {secret_name}")
            return False
        except Exception as e:
            logger.error(f"Failed to delete secret from AWS Secrets Manager {secret_name}: {e}")
            return False

    async def secret_exists(self, secret_name: str) -> bool:
        """Check if secret exists in AWS Secrets Manager"""
        import asyncio

        try:
            await asyncio.to_thread(
                self._client.describe_secret,
                SecretId=secret_name
            )
            return True
        except self._client.exceptions.ResourceNotFoundException:
            return False
        except Exception as e:
            logger.warning(f"Error checking secret existence: {secret_name}, error: {e}")
            return False
