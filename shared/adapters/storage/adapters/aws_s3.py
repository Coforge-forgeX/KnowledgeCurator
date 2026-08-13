"""AWS S3 Storage adapter"""

from typing import Optional

from core.exceptions import ConfigurationException
from core.logging import get_logger

from ..models import BlobInfo
from ..protocols import StorageAdapter

logger = get_logger(__name__)


class AWSS3StorageAdapter(StorageAdapter):
    """AWS S3 Storage implementation"""

    def __init__(self) -> None:
        """
        Initialize AWS S3 adapter.

        Required settings:
            AWS_ACCESS_KEY_ID: AWS access key
            AWS_SECRET_ACCESS_KEY: AWS secret key
            AWS_REGION: AWS region (e.g., us-east-1)
            S3_BUCKET_NAME: S3 bucket name
            S3_PATH_PREFIX: Optional path prefix
        """
        try:
            import boto3
            from botocore.config import Config
        except ImportError:
            raise ConfigurationException(
                "boto3 not installed. Install with: pip install boto3",
                config_key="boto3",
            )

        from core.config import settings

        access_key = settings.storage.AWS_ACCESS_KEY_ID
        secret_key = settings.storage.AWS_SECRET_ACCESS_KEY
        region = getattr(settings.storage, 'AWS_REGION', None) or "us-east-1"

        if not access_key or not secret_key:
            raise ConfigurationException(
                "AWS credentials not configured. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY",
                config_key="AWS_ACCESS_KEY_ID",
            )

        self._bucket = getattr(settings.storage, 'S3_BUCKET_NAME', None)
        if not self._bucket:
            raise ConfigurationException(
                "S3_BUCKET_NAME not configured",
                config_key="S3_BUCKET_NAME",
            )

        self._path_prefix = (getattr(settings.storage, 'S3_PATH_PREFIX', None) or "").strip("/")
        self._expiry_minutes = int(getattr(settings.storage, 'S3_URL_EXPIRY_MINUTES', None) or 60)
        self._region = region

        # Create S3 client
        config = Config(signature_version="s3v4", region_name=region)
        self._s3_client = boto3.client(
            "s3",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=config,
        )

        logger.info(
            "AWS S3 adapter initialized",
            bucket=self._bucket,
            region=region,
            prefix=self._path_prefix,
        )

    @property
    def provider_name(self) -> str:
        return "aws"

    @property
    def container_name(self) -> str:
        return self._bucket

    def _build_key(self, filename: str) -> str:
        """Build full S3 key with prefix"""
        if not self._path_prefix:
            return filename
        return f"{self._path_prefix}/{filename}"

    async def upload(
        self, filename: str, data: bytes, content_type: Optional[str] = None
    ) -> BlobInfo:
        """Upload file to AWS S3"""
        if not filename or not filename.strip():
            raise ValueError("filename cannot be empty")

        key = self._build_key(filename.strip())

        try:
            import asyncio

            # Upload to S3 (using asyncio.to_thread for sync SDK)
            await asyncio.to_thread(
                self._s3_client.put_object,
                Bucket=self._bucket,
                Key=key,
                Body=data,
                ContentType=content_type or "application/octet-stream",
            )

            size_bytes = len(data)

            # Generate URL
            url = f"https://{self._bucket}.s3.{self._region}.amazonaws.com/{key}"

            logger.info(
                "File uploaded to S3",
                key=key,
                size_bytes=size_bytes,
                content_type=content_type,
            )

            return BlobInfo(
                container=self._bucket,
                blob_name=key,
                blob_url=url,
                provider="aws",
                size_bytes=size_bytes,
            )

        except Exception as e:
            logger.error(f"Failed to upload to S3: {e}")
            raise

    async def generate_download_url(
        self, filename: str, expiry_minutes: Optional[int] = None
    ) -> str:
        """Generate presigned URL for S3 object"""
        import asyncio

        key = self._build_key(filename.strip())
        expiry_seconds = (expiry_minutes or self._expiry_minutes) * 60

        try:
            url = await asyncio.to_thread(
                self._s3_client.generate_presigned_url,
                "get_object",
                Params={"Bucket": self._bucket, "Key": key},
                ExpiresIn=expiry_seconds,
            )
            return url
        except Exception as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            raise

    async def blob_exists(self, filename: str) -> bool:
        """Check if object exists in S3"""
        import asyncio

        key = self._build_key(filename.strip())
        try:
            await asyncio.to_thread(
                self._s3_client.head_object, Bucket=self._bucket, Key=key
            )
            return True
        except Exception:
            return False

    async def delete(self, filename: str) -> bool:
        """Delete object from S3"""
        import asyncio

        key = self._build_key(filename.strip())
        try:
            await asyncio.to_thread(
                self._s3_client.delete_object, Bucket=self._bucket, Key=key
            )
            logger.info(f"Deleted S3 object: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete S3 object {key}: {e}")
            return False

    async def download(self, filename: str) -> bytes:
        """Download object content from S3"""
        import asyncio

        key = self._build_key(filename.strip())
        try:
            response = await asyncio.to_thread(
                self._s3_client.get_object, Bucket=self._bucket, Key=key
            )
            content = response["Body"].read()
            logger.info(f"Downloaded S3 object: {key}, size: {len(content)} bytes")
            return content
        except self._s3_client.exceptions.NoSuchKey:
            raise FileNotFoundError(f"S3 object not found: {key}")
        except Exception as e:
            logger.error(f"Failed to download S3 object {key}: {e}")
            raise

    async def list_files(self, prefix: Optional[str] = None) -> list[str]:
        """List all objects in S3 bucket with optional prefix"""
        import asyncio

        try:
            # Build the full prefix including path_prefix
            if prefix:
                full_prefix = self._build_key(prefix.strip())
            elif self._path_prefix:
                full_prefix = self._path_prefix.rstrip("/") + "/"
            else:
                full_prefix = ""

            file_paths: list[str] = []

            # List objects with pagination
            def _list_objects():
                paths = []
                paginator = self._s3_client.get_paginator('list_objects_v2')
                page_iterator = paginator.paginate(
                    Bucket=self._bucket,
                    Prefix=full_prefix
                )

                for page in page_iterator:
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            key = obj['Key']
                            # Skip directory markers (keys ending with /)
                            if not key.endswith('/'):
                                # Remove path_prefix if present to return relative paths
                                if self._path_prefix and key.startswith(self._path_prefix):
                                    relative_key = key[len(self._path_prefix):].lstrip('/')
                                    paths.append(relative_key)
                                else:
                                    paths.append(key)
                return paths

            file_paths = await asyncio.to_thread(_list_objects)

            logger.info(
                f"Listed {len(file_paths)} files from S3",
                prefix=prefix or "(all)",
            )

            return file_paths

        except Exception as e:
            logger.error(f"Failed to list files from S3: {e}", exc_info=True)
            raise

    @property
    def provider_name(self) -> str:
        return "aws"

    @property
    def container_name(self) -> str:
        return self._bucket
