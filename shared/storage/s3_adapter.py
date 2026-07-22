"""
AWS S3 Storage Adapter

Implementation of StorageAdapter for AWS S3.
"""
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import aioboto3
from botocore.exceptions import ClientError

from core.exceptions import StorageException
from core.logging import get_logger

from .base import BlobContent, BlobMetadata, StorageAdapter

logger = get_logger(__name__)


class S3StorageAdapter(StorageAdapter):
    """
    AWS S3 storage adapter implementation.

    Provides async operations for AWS S3 using aioboto3.
    """

    def __init__(
        self,
        container_name: str,  # bucket_name in S3 terminology
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: Optional[str] = None,
        **config
    ):
        """
        Initialize AWS S3 storage adapter.

        Args:
            container_name: S3 bucket name
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
            region_name: AWS region
            **config: Additional configuration
        """
        super().__init__(container_name, **config)

        from core.config import settings

        self.aws_access_key_id = aws_access_key_id or settings.storage.AWS_ACCESS_KEY_ID
        self.aws_secret_access_key = (
            aws_secret_access_key or settings.storage.AWS_SECRET_ACCESS_KEY
        )
        self.region_name = region_name or settings.storage.AWS_REGION

        if not self.aws_access_key_id or not self.aws_secret_access_key:
            raise StorageException(
                message="AWS credentials not provided",
                operation="initialize",
            )

        self.session = aioboto3.Session(
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.region_name,
        )

        logger.info("AWS S3 storage adapter initialized", bucket_name=container_name)

    async def upload_file(
        self,
        file_path: str,
        content: bytes,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        overwrite: bool = True,
    ) -> BlobMetadata:
        """Upload file to AWS S3"""
        try:
            async with self.session.client("s3") as s3_client:
                extra_args = {}

                if content_type:
                    extra_args["ContentType"] = content_type

                if metadata:
                    extra_args["Metadata"] = metadata

                # Upload object
                await s3_client.put_object(
                    Bucket=self.container_name,
                    Key=file_path,
                    Body=content,
                    **extra_args,
                )

                # Get object metadata
                response = await s3_client.head_object(
                    Bucket=self.container_name,
                    Key=file_path,
                )

                logger.info(
                    "File uploaded to AWS S3",
                    file_path=file_path,
                    size=len(content),
                )

                return BlobMetadata(
                    name=os.path.basename(file_path),
                    path=file_path,
                    size=response.get("ContentLength", len(content)),
                    content_type=response.get("ContentType", ""),
                    created_at=response.get("LastModified"),
                    updated_at=response.get("LastModified"),
                    metadata=response.get("Metadata"),
                    etag=response.get("ETag"),
                )

        except Exception as e:
            logger.error("Failed to upload file to AWS S3", error=e)
            raise StorageException(
                message=f"Failed to upload file: {str(e)}",
                operation="upload_file",
            )

    async def download_file(self, file_path: str) -> BlobContent:
        """Download file from AWS S3"""
        try:
            async with self.session.client("s3") as s3_client:
                response = await s3_client.get_object(
                    Bucket=self.container_name,
                    Key=file_path,
                )

                content = await response["Body"].read()

                logger.info(
                    "File downloaded from AWS S3",
                    file_path=file_path,
                    size=len(content),
                )

                metadata = BlobMetadata(
                    name=os.path.basename(file_path),
                    path=file_path,
                    size=response.get("ContentLength", len(content)),
                    content_type=response.get("ContentType", ""),
                    created_at=response.get("LastModified"),
                    updated_at=response.get("LastModified"),
                    metadata=response.get("Metadata"),
                    etag=response.get("ETag"),
                )

                return BlobContent(data=content, metadata=metadata)

        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise StorageException(
                    message=f"File not found: {file_path}",
                    operation="download_file",
                )
            raise StorageException(
                message=f"Failed to download file: {str(e)}",
                operation="download_file",
            )
        except Exception as e:
            logger.error("Failed to download file from AWS S3", error=e)
            raise StorageException(
                message=f"Failed to download file: {str(e)}",
                operation="download_file",
            )

    async def delete_file(self, file_path: str) -> bool:
        """Delete file from AWS S3"""
        try:
            async with self.session.client("s3") as s3_client:
                await s3_client.delete_object(
                    Bucket=self.container_name,
                    Key=file_path,
                )

                logger.info("File deleted from AWS S3", file_path=file_path)
                return True

        except Exception as e:
            logger.error("Failed to delete file from AWS S3", error=e)
            raise StorageException(
                message=f"Failed to delete file: {str(e)}",
                operation="delete_file",
            )

    async def list_files(
        self,
        prefix: Optional[str] = None,
        max_results: Optional[int] = None,
    ) -> List[BlobMetadata]:
        """List files in AWS S3"""
        try:
            async with self.session.client("s3") as s3_client:
                kwargs = {"Bucket": self.container_name}

                if prefix:
                    kwargs["Prefix"] = prefix

                if max_results:
                    kwargs["MaxKeys"] = max_results

                response = await s3_client.list_objects_v2(**kwargs)

                blobs = []
                for obj in response.get("Contents", []):
                    blobs.append(
                        BlobMetadata(
                            name=obj["Key"].split("/")[-1],
                            path=obj["Key"],
                            size=obj["Size"],
                            content_type="",  # S3 doesn't return content type in list
                            created_at=obj.get("LastModified"),
                            updated_at=obj.get("LastModified"),
                            etag=obj.get("ETag"),
                        )
                    )

                logger.info(
                    "Listed files from AWS S3",
                    count=len(blobs),
                    prefix=prefix,
                )

                return blobs

        except Exception as e:
            logger.error("Failed to list files from AWS S3", error=e)
            raise StorageException(
                message=f"Failed to list files: {str(e)}",
                operation="list_files",
            )

    async def file_exists(self, file_path: str) -> bool:
        """Check if file exists in AWS S3"""
        try:
            async with self.session.client("s3") as s3_client:
                try:
                    await s3_client.head_object(
                        Bucket=self.container_name,
                        Key=file_path,
                    )
                    return True
                except ClientError as e:
                    if e.response["Error"]["Code"] == "404":
                        return False
                    raise

        except Exception as e:
            logger.error("Failed to check file existence in AWS S3", error=e)
            return False

    async def get_file_url(
        self,
        file_path: str,
        expiry_seconds: int = 3600,
    ) -> str:
        """Generate presigned URL for AWS S3 file"""
        try:
            async with self.session.client("s3") as s3_client:
                url = await s3_client.generate_presigned_url(
                    "get_object",
                    Params={
                        "Bucket": self.container_name,
                        "Key": file_path,
                    },
                    ExpiresIn=expiry_seconds,
                )

                logger.info("Generated presigned URL for S3 file", file_path=file_path)
                return url

        except Exception as e:
            logger.error("Failed to generate presigned URL", error=e)
            raise StorageException(
                message=f"Failed to generate presigned URL: {str(e)}",
                operation="get_file_url",
            )

    async def copy_file(
        self,
        source_path: str,
        destination_path: str,
    ) -> BlobMetadata:
        """Copy file within AWS S3"""
        try:
            async with self.session.client("s3") as s3_client:
                copy_source = {"Bucket": self.container_name, "Key": source_path}

                await s3_client.copy_object(
                    CopySource=copy_source,
                    Bucket=self.container_name,
                    Key=destination_path,
                )

                # Get metadata of copied file
                response = await s3_client.head_object(
                    Bucket=self.container_name,
                    Key=destination_path,
                )

                logger.info(
                    "File copied in AWS S3",
                    source=source_path,
                    destination=destination_path,
                )

                return BlobMetadata(
                    name=os.path.basename(destination_path),
                    path=destination_path,
                    size=response.get("ContentLength", 0),
                    content_type=response.get("ContentType", ""),
                    created_at=response.get("LastModified"),
                    updated_at=response.get("LastModified"),
                    metadata=response.get("Metadata"),
                    etag=response.get("ETag"),
                )

        except Exception as e:
            logger.error("Failed to copy file in AWS S3", error=e)
            raise StorageException(
                message=f"Failed to copy file: {str(e)}",
                operation="copy_file",
            )

    async def get_file_metadata(self, file_path: str) -> BlobMetadata:
        """Get file metadata from AWS S3"""
        try:
            async with self.session.client("s3") as s3_client:
                response = await s3_client.head_object(
                    Bucket=self.container_name,
                    Key=file_path,
                )

                return BlobMetadata(
                    name=os.path.basename(file_path),
                    path=file_path,
                    size=response.get("ContentLength", 0),
                    content_type=response.get("ContentType", ""),
                    created_at=response.get("LastModified"),
                    updated_at=response.get("LastModified"),
                    metadata=response.get("Metadata"),
                    etag=response.get("ETag"),
                )

        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                raise StorageException(
                    message=f"File not found: {file_path}",
                    operation="get_file_metadata",
                )
            raise StorageException(
                message=f"Failed to get file metadata: {str(e)}",
                operation="get_file_metadata",
            )
        except Exception as e:
            logger.error("Failed to get file metadata from AWS S3", error=e)
            raise StorageException(
                message=f"Failed to get file metadata: {str(e)}",
                operation="get_file_metadata",
            )
