"""Local filesystem storage adapter (for development/testing)"""

from pathlib import Path
from typing import Optional
from urllib.parse import quote

from core.config import settings
from core.exceptions import ConfigurationException
from core.logging import get_logger

from ..models import BlobInfo
from ..protocols import StorageAdapter

logger = get_logger(__name__)


class LocalStorageAdapter(StorageAdapter):
    """Local filesystem storage implementation (for development/testing)"""

    def __init__(self) -> None:
        """
        Initialize local storage adapter.

        Uses settings from config:
            LOCAL_STORAGE_PATH: Base directory for file storage (default: ./local_storage)
            LOCAL_STORAGE_CONTAINER: Container/folder name (default: documents)
            LOCAL_STORAGE_PATH_PREFIX: Optional path prefix
            LOCAL_STORAGE_BASE_URL: Base URL for file access (optional, for URL generation)
        """
        # Get from settings with fallbacks
        self._base_path = Path(getattr(settings.storage, 'LOCAL_STORAGE_PATH', None) or "./local_storage")
        self._container = getattr(settings.storage, 'LOCAL_STORAGE_CONTAINER', None) or "documents"
        self._path_prefix = (getattr(settings.storage, 'LOCAL_STORAGE_PATH_PREFIX', None) or "").strip("/")
        self._base_url = getattr(settings.storage, 'LOCAL_STORAGE_BASE_URL', None) or ""

        # Create base directory structure
        self._container_path = self._base_path / self._container
        self._container_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Local storage adapter initialized",
            base_path=str(self._base_path),
            container=self._container,
            prefix=self._path_prefix,
        )

    @property
    def provider_name(self) -> str:
        return "local"

    @property
    def container_name(self) -> str:
        return self._container

    def _build_file_path(self, filename: str) -> Path:
        """Build full file path with prefix"""
        if self._path_prefix:
            relative_path = f"{self._path_prefix}/{filename}"
        else:
            relative_path = filename

        # Normalize path to prevent directory traversal
        full_path = (self._container_path / relative_path).resolve()

        # Ensure the path is within container directory (security check)
        if not str(full_path).startswith(str(self._container_path.resolve())):
            raise ValueError(f"Invalid filename: path traversal detected in {filename}")

        return full_path

    def _build_relative_path(self, filename: str) -> str:
        """Build relative path for URL generation"""
        if self._path_prefix:
            return f"{self._path_prefix}/{filename}"
        return filename

    async def upload(
        self, filename: str, data: bytes, content_type: Optional[str] = None
    ) -> BlobInfo:
        """Upload file to local storage"""
        if not filename or not filename.strip():
            raise ValueError("filename cannot be empty")

        file_path = self._build_file_path(filename.strip())

        try:
            # Create parent directories if they don't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            file_path.write_bytes(data)

            size_bytes = len(data)

            # Build URL
            relative_path = self._build_relative_path(filename.strip())
            if self._base_url:
                blob_url = f"{self._base_url.rstrip('/')}/{self._container}/{quote(relative_path)}"
            else:
                blob_url = f"file://{file_path.as_posix()}"

            logger.info(
                "File uploaded to local storage",
                file_path=str(file_path),
                size_bytes=size_bytes,
                content_type=content_type,
            )

            return BlobInfo(
                container=self._container,
                blob_name=relative_path,
                blob_url=blob_url,
                provider="local",
                size_bytes=size_bytes,
            )

        except Exception as e:
            logger.error(f"Failed to upload to local storage: {e}")
            raise

    async def generate_download_url(
        self, filename: str, expiry_minutes: Optional[int] = None
    ) -> str:
        """
        Generate download URL for local file.

        Note: Local storage doesn't support signed URLs with expiry.
        If LOCAL_STORAGE_BASE_URL is set, returns HTTP URL.
        Otherwise, returns file:// URL.
        """
        file_path = self._build_file_path(filename.strip())
        relative_path = self._build_relative_path(filename.strip())

        if self._base_url:
            # HTTP URL (if web server is configured to serve local_storage/)
            return f"{self._base_url.rstrip('/')}/{self._container}/{quote(relative_path)}"
        else:
            # file:// URL (for local access only)
            return f"file://{file_path.as_posix()}"

    async def blob_exists(self, filename: str) -> bool:
        """Check if file exists in local storage"""
        try:
            file_path = self._build_file_path(filename.strip())
            return file_path.exists() and file_path.is_file()
        except Exception as e:
            logger.warning(f"Error checking file existence: {e}")
            return False

    async def delete(self, filename: str) -> bool:
        """Delete file from local storage"""
        try:
            file_path = self._build_file_path(filename.strip())
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted local file: {file_path}")
                return True
            else:
                logger.warning(f"File not found for deletion: {file_path}")
                return False
        except Exception as e:
            logger.error(f"Failed to delete local file: {e}")
            return False

    async def download(self, filename: str) -> bytes:
        """Download file content from local storage"""
        try:
            file_path = self._build_file_path(filename.strip())
            if not file_path.exists():
                raise FileNotFoundError(f"Local file not found: {file_path}")

            content = file_path.read_bytes()
            logger.info(f"Downloaded local file: {file_path}, size: {len(content)} bytes")
            return content
        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to download local file: {e}")
            raise
