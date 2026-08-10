"""Generate short-lived download URL for query_rag source references."""
import json

from src.core.abstractions import AbstractContext, AbstractRequest, AbstractResponse
from src.core.auth import get_user_id, require_auth
from src.core.config import settings
from src.core.exceptions import AuthorizationException, ValidationException
from src.core.logging import get_logger
from src.core.redis import redis_manager
from src.helpers.file_token import decode_signed_file_id
from src.services.workspace_service import get_workspace_service
from src.shared import create_error_response, create_success_response
from shared.adapters.storage import get_storage_adapter

logger = get_logger(__name__)

DOWNLOAD_URL_TTL_MINUTES = 5
FILE_KEY_PREFIX = "query_file:"


def _extract_file_id(path: str) -> str:
    """Extract file_id from /api/v2/files/{file_id}/download path."""
    marker = "/api/v2/files/"
    idx = path.find(marker)
    if idx < 0:
        return ""
    tail = path[idx + len(marker):]
    if not tail:
        return ""
    if "/" not in tail:
        return ""
    file_id, suffix = tail.split("/", 1)
    if suffix != "download":
        return ""
    return file_id.strip()


@require_auth()
async def main(req: AbstractRequest, context: AbstractContext) -> AbstractResponse:
    """Create a 5-minute download URL for a file returned in query_rag source."""
    correlation_id = context.correlation_id
    user_id = get_user_id(req)

    try:
        file_id = _extract_file_id(req.path)
        if not file_id:
            raise ValidationException(message="Invalid file download path")

        cached_mapping = decode_signed_file_id(file_id)
        if not cached_mapping:
            cached_mapping_str = redis_manager.get(f"{FILE_KEY_PREFIX}{file_id}")
            if not cached_mapping_str:
                raise ValidationException(
                    message="File reference expired. Please run query again."
                )

            try:
                cached_mapping = json.loads(cached_mapping_str)
            except json.JSONDecodeError as exc:
                raise ValidationException(message="Invalid file reference payload") from exc

        workspace_id = int(cached_mapping.get("workspace_id"))
        container_name = str(cached_mapping.get("container_name") or "").strip()
        blob_path = str(cached_mapping.get("blob_path") or "").strip()
        provider = str(cached_mapping.get("provider") or settings.storage.STORAGE_PROVIDER or "azure")
        file_name = str(cached_mapping.get("file_name") or "").strip()

        if not container_name or not blob_path:
            raise ValidationException(message="Invalid file reference data")

        workspace_service = get_workspace_service()
        is_authorized, _ = await workspace_service.validate_user_workspace_access(
            user_id=user_id,
            workspace_id=workspace_id,
        )
        if not is_authorized:
            raise AuthorizationException(
                message=f"You are not authorized to access workspace {workspace_id}"
            )

        storage = get_storage_adapter(
            provider=provider,
            connection_string=settings.storage.AZURE_BLOB_STORAGE_CONNECTION_STRING,
            container_name=container_name,
        )

        exists = await storage.blob_exists(blob_path)
        if not exists:
            raise ValidationException(message="Requested source file not found")

        download_url = await storage.generate_download_url(
            blob_path,
            expiry_minutes=DOWNLOAD_URL_TTL_MINUTES,
        )

        return create_success_response(
            message="Download URL generated successfully",
            data={
                "file_id": file_id,
                "file_name": file_name,
                "download_url": download_url,
                "expires_in_seconds": DOWNLOAD_URL_TTL_MINUTES * 60,
            },
            correlation_id=correlation_id,
        )

    except ValidationException as e:
        return create_error_response(
            message=e.message,
            error_code="VALIDATION_ERROR",
            status_code=400,
            correlation_id=correlation_id,
        )

    except AuthorizationException as e:
        return create_error_response(
            message=e.message,
            error_code="AUTHORIZATION_ERROR",
            status_code=403,
            correlation_id=correlation_id,
        )

    except Exception as e:
        logger.error("Failed to generate source download url", error=e, correlation_id=correlation_id)
        return create_error_response(
            message="Failed to generate source download URL",
            error_code="INTERNAL_ERROR",
            details={"error": str(e)},
            status_code=500,
            correlation_id=correlation_id,
        )
