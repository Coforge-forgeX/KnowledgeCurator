"""GCP Cloud Storage adapter"""

from datetime import timedelta
import os
from typing import Optional

from core.exceptions import ConfigurationException
from core.logging import get_logger

from ..models import BlobInfo
from ..protocols import StorageAdapter

logger = get_logger(__name__)


class GCPStorageAdapter(StorageAdapter):
    """GCP Cloud Storage implementation"""

    def __init__(self) -> None:
        """
        Initialize GCP Storage adapter.

        Required settings:
            GCP_PROJECT_ID: GCP project ID
            GCP_CREDENTIALS_PATH or GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON
            GCS_BUCKET_NAME: GCS bucket name
            GCS_PATH_PREFIX: Optional path prefix
        """
        try:
            from google.cloud import storage
        except ImportError:
            raise ConfigurationException(
                "google-cloud-storage not installed. Install with: pip install google-cloud-storage",
                config_key="google-cloud-storage",
            )

        from core.config import settings

        project_id = getattr(settings.storage, 'GCP_PROJECT_ID', None)
        # GOOGLE_APPLICATION_CREDENTIALS is a standard GCP env var, keep os.getenv as fallback
        credentials_path = (
            getattr(settings.storage, 'GCP_CREDENTIALS_PATH', None) or
            os.getenv("GOOGLE_APPLICATION_CREDENTIALS")  # Standard GCP environment variable
        )

        if not project_id:
            raise ConfigurationException(
                "GCP_PROJECT_ID not configured",
                config_key="GCP_PROJECT_ID",
            )

        self._bucket_name = getattr(settings.storage, 'GCS_BUCKET_NAME', None)
        if not self._bucket_name:
            raise ConfigurationException(
                "GCS_BUCKET_NAME not configured",
                config_key="GCS_BUCKET_NAME",
            )

        self._path_prefix = (getattr(settings.storage, 'GCS_PATH_PREFIX', None) or "").strip("/")
        self._expiry_minutes = int(getattr(settings.storage, 'GCS_URL_EXPIRY_MINUTES', None) or 60)

        # Create GCS client
        if credentials_path:
            self._client = storage.Client.from_service_account_json(
                credentials_path, project=project_id
            )
        else:
            self._client = storage.Client(project=project_id)

        self._bucket = self._client.bucket(self._bucket_name)

        logger.info(
            "GCP Cloud Storage adapter initialized",
            bucket=self._bucket_name,
            project=project_id,
            prefix=self._path_prefix,
        )

    @property
    def provider_name(self) -> str:
        return "gcp"

    @property
    def container_name(self) -> str:
        return self._bucket_name

    def _build_blob_name(self, filename: str) -> str:
        """Build full blob path with prefix"""
        if not self._path_prefix:
            return filename
        return f"{self._path_prefix}/{filename}"

    async def upload(
        self, filename: str, data: bytes, content_type: Optional[str] = None
    ) -> BlobInfo:
        """Upload file to GCP Cloud Storage"""
        import asyncio

        if not filename or not filename.strip():
            raise ValueError("filename cannot be empty")

        blob_name = self._build_blob_name(filename.strip())
        blob = self._bucket.blob(blob_name)

        try:
            # Upload blob (using asyncio.to_thread for sync SDK)
            await asyncio.to_thread(
                blob.upload_from_string,
                data,
                content_type=content_type or "application/octet-stream",
            )

            size_bytes = len(data)

            logger.info(
                "File uploaded to GCS",
                blob_name=blob_name,
                size_bytes=size_bytes,
                content_type=content_type,
            )

            return BlobInfo(
                container=self._bucket_name,
                blob_name=blob_name,
                blob_url=blob.public_url,
                provider="gcp",
                size_bytes=size_bytes,
            )

        except Exception as e:
            logger.error(f"Failed to upload to GCS: {e}")
            raise

    async def generate_download_url(
        self, filename: str, expiry_minutes: Optional[int] = None
    ) -> str:
        """Generate signed URL for GCS blob"""
        import asyncio

        blob_name = self._build_blob_name(filename.strip())
        blob = self._bucket.blob(blob_name)

        try:
            expiry_delta = timedelta(minutes=expiry_minutes or self._expiry_minutes)
            url = await asyncio.to_thread(blob.generate_signed_url, expiration=expiry_delta)
            return url
        except Exception as e:
            logger.error(f"Failed to generate signed URL: {e}")
            raise

    async def blob_exists(self, filename: str) -> bool:
        """Check if blob exists in GCS"""
        import asyncio

        blob_name = self._build_blob_name(filename.strip())
        blob = self._bucket.blob(blob_name)
        try:
            exists = await asyncio.to_thread(blob.exists)
            return exists
        except Exception as e:
            logger.warning(f"Error checking blob existence: {e}")
            return False

    async def delete(self, filename: str) -> bool:
        """Delete blob from GCS"""
        import asyncio

        blob_name = self._build_blob_name(filename.strip())
        blob = self._bucket.blob(blob_name)
        try:
            await asyncio.to_thread(blob.delete)
            logger.info(f"Deleted GCS blob: {blob_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete GCS blob {blob_name}: {e}")
            return False

    async def download(self, filename: str) -> bytes:
        """Download blob content from GCS"""
        import asyncio

        blob_name = self._build_blob_name(filename.strip())
        blob = self._bucket.blob(blob_name)
        try:
            if not await asyncio.to_thread(blob.exists):
                raise FileNotFoundError(f"GCS blob not found: {blob_name}")

            content = await asyncio.to_thread(blob.download_as_bytes)
            logger.info(f"Downloaded GCS blob: {blob_name}, size: {len(content)} bytes")
            return content
        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to download GCS blob {blob_name}: {e}")
            raise

    async def list_files(self, prefix: Optional[str] = None) -> list[str]:
        """List all blobs in GCS bucket with optional prefix"""
        import asyncio

        try:
            # Build the full prefix including path_prefix
            if prefix:
                full_prefix = self._build_blob_name(prefix.strip())
            elif self._path_prefix:
                full_prefix = self._path_prefix.rstrip("/") + "/"
            else:
                full_prefix = None

            file_paths: list[str] = []

            # List blobs
            def _list_blobs():
                paths = []
                blobs = self._bucket.list_blobs(prefix=full_prefix) if full_prefix else self._bucket.list_blobs()

                for blob in blobs:
                    # Skip directory markers (blobs ending with /)
                    if not blob.name.endswith('/'):
                        # Remove path_prefix if present to return relative paths
                        if self._path_prefix and blob.name.startswith(self._path_prefix):
                            relative_name = blob.name[len(self._path_prefix):].lstrip('/')
                            paths.append(relative_name)
                        else:
                            paths.append(blob.name)
                return paths

            file_paths = await asyncio.to_thread(_list_blobs)

            logger.info(
                f"Listed {len(file_paths)} files from GCS",
                prefix=prefix or "(all)",
            )

            return file_paths

        except Exception as e:
            logger.error(f"Failed to list files from GCS: {e}", exc_info=True)
            raise

    @property
    def provider_name(self) -> str:
        return "gcp"

    @property
    def container_name(self) -> str:
        return self._bucket_name
