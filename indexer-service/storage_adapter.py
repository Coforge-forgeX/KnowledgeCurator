"""Storage adapter factory for indexer service"""
import os
import sys
from azure.storage.blob import BlobServiceClient

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)


class AzureBlobAdapter:
    """Azure Blob Storage adapter for indexer service"""

    def __init__(self, connection_string: str, container_name: str):
        self.provider_name = "azure"
        self.connection_string = connection_string
        self.container_name = container_name
        self.client = None

    def _get_client(self):
        """Get or create blob service client"""
        if not self.client:
            self.client = BlobServiceClient.from_connection_string(
                self.connection_string
            )
        return self.client

    def download(self, blob_path: str) -> bytes:
        """
        Download blob content (sync method, wrap with asyncio.to_thread for async).

        Args:
            blob_path: Path to blob in container

        Returns:
            Blob content as bytes
        """
        try:
            client = self._get_client()
            container_client = client.get_container_client(self.container_name)
            blob_client = container_client.get_blob_client(blob_path)

            blob_data = blob_client.download_blob().readall()
            logger.info(f"Downloaded blob: {blob_path}")
            return blob_data

        except Exception as e:
            logger.error(f"Failed to download blob: {blob_path}", error_msg=e)
            raise

    def upload(self, blob_path: str, data: bytes) -> None:
        """
        Upload data to blob (sync method, wrap with asyncio.to_thread for async).

        Args:
            blob_path: Path to blob in container
            data: Data to upload
        """
        try:
            client = self._get_client()
            container_client = client.get_container_client(self.container_name)
            blob_client = container_client.get_blob_client(blob_path)

            blob_client.upload_blob(data, overwrite=True)
            logger.info(f"Uploaded blob: {blob_path}")

        except Exception as e:
            logger.error(f"Failed to upload blob: {blob_path}", error_msg=e)
            raise

    def delete(self, blob_path: str) -> None:
        """
        Delete blob (sync method, wrap with asyncio.to_thread for async).

        Args:
            blob_path: Path to blob in container
        """
        try:
            client = self._get_client()
            container_client = client.get_container_client(self.container_name)
            blob_client = container_client.get_blob_client(blob_path)

            blob_client.delete_blob()
            logger.info(f"Deleted blob: {blob_path}")

        except Exception as e:
            logger.error(f"Failed to delete blob: {blob_path}", error_msg=e)
            raise


def get_storage_adapter():
    """
    Get storage adapter based on configuration.

    Returns configured storage client (Azure Blob)
    """
    provider = settings.storage.STORAGE_PROVIDER.lower()

    if provider == "azure":
        # Use Azure Storage connection string from config
        connection_string = settings.azure.AZURE_STORAGE_CONNECTION_STRING
        container_name = settings.storage.STORAGE_CONTAINER_NAME

        logger.info(
            "Creating Azure Blob storage adapter",
            container=container_name
        )

        return AzureBlobAdapter(
            connection_string=connection_string,
            container_name=container_name
        )
    else:
        raise ValueError(f"Unsupported storage provider: {provider}")
