"""Generate compressed ZIP archive containing all workspace documents and return signed download URL."""

import asyncio
from datetime import datetime, timezone
import io
import os
from typing import Dict, List, Tuple
import zipfile

from sqlalchemy import or_, select

from src.common import create_error_response, create_internal_error_response, create_success_response, parse_request
from src.core.abstractions import AbstractContext, AbstractRequest, AbstractResponse
from src.core.auth import get_user_id, require_auth
from src.core.config import settings
from src.core.database import DocumentMetadata, FileTask, WorkspaceIndustryIntentMap, get_async_session
from src.core.exceptions import AuthorizationException, ValidationException
from src.core.logging import get_logger
from src.helpers.workspace_helpers import get_workspace_storage_paths
from src.services.workspace_service import get_workspace_service
from src.storage import get_storage_adapter

from .payloads import WorkspaceDownloadZipRequest, WorkspaceDownloadZipResponse

logger = get_logger(__name__)


def _deduplicate_filename(filename: str, seen_names: Dict[str, int]) -> str:
    """Disambiguate duplicate filenames in ZIP archive (e.g., doc.pdf -> doc_1.pdf)."""
    base_name, ext = os.path.splitext(filename)
    if filename not in seen_names:
        seen_names[filename] = 1
        return filename

    seen_names[filename] += 1
    count = seen_names[filename] - 1
    return f"{base_name}_{count}{ext}"


async def _fetch_workspace_files(
    workspace_id: int,
    limit: int,
    user_id_filter: int | None = None,
    include_kb_files: bool = False,
) -> List[Tuple[str, str, str]]:
    """
    Fetch (blob_path, display_name, container_name) for workspace files from DB.
    Deduplicates files by file_path.
    If include_kb_files is True, also includes files linked to workspace Knowledge Bases / Knowledge Graph.
    Falls back to storage list_files if DB contains no tasks.
    """
    workspace_paths = await get_workspace_storage_paths(workspace_id)
    default_container = str((workspace_paths or {}).get("container") or settings.storage.STORAGE_CONTAINER_NAME)

    files: List[Tuple[str, str, str]] = []
    seen_paths: set[str] = set()

    async with get_async_session() as session:
        # 1. Fetch from FileTask
        stmt = (
            select(FileTask.file_path, FileTask.container_name)
            .where(
                FileTask.workspace_id == workspace_id,
                FileTask.status != "deleted",
            )
            .order_by(FileTask.created_at.desc())
        )
        result = await session.execute(stmt)
        rows = result.all()

        for file_path, container_name in rows:
            if not file_path:
                continue
            norm_path = file_path.strip().lower()
            if norm_path in seen_paths:
                continue
            display_name = os.path.basename(file_path)
            if display_name:
                seen_paths.add(norm_path)
                c_name = str(container_name or default_container)
                files.append((file_path, display_name, c_name))
                if len(files) >= limit:
                    return files

        # 2. If include_kb_files is True, fetch files from Knowledge Base / Knowledge Graph metadata
        if include_kb_files and len(files) < limit:
            map_stmt = (
                select(WorkspaceIndustryIntentMap.kb_id)
                .where(
                    WorkspaceIndustryIntentMap.workspace_id == workspace_id,
                    WorkspaceIndustryIntentMap.is_active.is_(True),
                    WorkspaceIndustryIntentMap.kb_id.isnot(None),
                )
                .distinct()
            )
            kb_ids = [int(row[0]) for row in (await session.execute(map_stmt)).fetchall() if row[0] is not None]

            if kb_ids:
                meta_stmt = (
                    select(
                        DocumentMetadata.file_path,
                        DocumentMetadata.doc_metadata,
                        FileTask.container_name,
                    )
                    .outerjoin(
                        FileTask,
                        or_(
                            DocumentMetadata.file_task_id == FileTask.id,
                            DocumentMetadata.file_path == FileTask.file_path,
                        ),
                    )
                    .where(
                        or_(
                            DocumentMetadata.workspace_id == workspace_id,
                            DocumentMetadata.kb_id.in_(kb_ids),
                        )
                    )
                )
                meta_rows = (await session.execute(meta_stmt)).all()
                for file_path, doc_meta, ft_container_name in meta_rows:
                    if not file_path:
                        continue
                    norm_path = file_path.strip().lower()
                    if norm_path in seen_paths:
                        continue
                    display_name = os.path.basename(file_path)
                    if display_name:
                        seen_paths.add(norm_path)
                        meta_dict = doc_meta if isinstance(doc_meta, dict) else {}
                        c_name = str(
                            ft_container_name
                            or meta_dict.get("container_name")
                            or default_container
                        )
                        files.append((file_path, display_name, c_name))
                        if len(files) >= limit:
                            return files
 

    if not files:
        if workspace_paths and workspace_paths.get("upload_path"):
            upload_prefix = str(workspace_paths.get("upload_path")).strip("/")
            storage = get_storage_adapter(container_override=default_container)
            blob_paths = await storage.list_files(prefix=f"{upload_prefix}/")
            for path in blob_paths[:limit]:
                norm_path = path.strip().lower()
                if norm_path in seen_paths:
                    continue
                name = os.path.basename(path)
                if name:
                    seen_paths.add(norm_path)
                    files.append((path, name, default_container))

    return files


@require_auth()
async def main(req: AbstractRequest, context: AbstractContext) -> AbstractResponse:
    """
    Generate ZIP archive of all workspace documents and return signed download URL.

    POST /api/v2/workspaces/download-zip
    """
    payload, error_response = parse_request(req, WorkspaceDownloadZipRequest)
    if error_response:
        return error_response

    workspace_id = payload.workspace_id
    user_id = get_user_id(req)
    expiry_minutes = payload.expiry_minutes if payload.expiry_minutes is not None else 30

    try:
        # Authorize user access to workspace
        workspace_service = get_workspace_service()
        is_authorized, _ = await workspace_service.validate_user_workspace_access(
            user_id=user_id,
            workspace_id=workspace_id,
        )
        if not is_authorized:
            raise AuthorizationException(f"You are not authorized to access workspace {workspace_id}")

        # Fetch files to include in ZIP
        file_entries = await _fetch_workspace_files(
            workspace_id=workspace_id,
            limit=payload.limit or 1000,
            user_id_filter=payload.user_id_filter,
            include_kb_files=bool(payload.include_kb_files),
        )

        if not file_entries:
            return create_error_response(
                message=f"No files found in workspace {workspace_id} to download",
                error_code="NO_FILES_FOUND",
                status_code=404,
                correlation_id=context.correlation_id,
            )

        workspace_paths = await get_workspace_storage_paths(workspace_id)
        default_container = str((workspace_paths or {}).get("container") or settings.storage.STORAGE_CONTAINER_NAME)
        export_storage = get_storage_adapter(container_override=default_container)
        semaphore = asyncio.Semaphore(10)

        zipped_count = 0
        failed_files: List[str] = []
        skipped_files: List[str] = []
        seen_filenames: Dict[str, int] = {}

        zip_buffer = io.BytesIO()

        async def _download_and_write(zip_file: zipfile.ZipFile, blob_path: str, orig_name: str, container_name: str):
            nonlocal zipped_count
            async with semaphore:
                try:
                    storage = get_storage_adapter(container_override=container_name if container_name else None)
                    exists = await storage.blob_exists(blob_path)
                    if not exists:
                        failed_files.append(f"{orig_name}: file missing from storage")
                        return

                    content_bytes = await storage.download(blob_path)
                    if not content_bytes:
                        failed_files.append(f"{orig_name}: 0 bytes retrieved")
                        return

                    unique_name = _deduplicate_filename(orig_name, seen_filenames)
                    zip_file.writestr(unique_name, content_bytes)
                    zipped_count += 1
                except Exception as ex:
                    logger.warning(f"Failed to fetch {orig_name} ({blob_path}) for zip: {ex}")
                    failed_files.append(f"{orig_name}: {str(ex)}")

        with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            tasks = [_download_and_write(zf, path, name, c_name) for path, name, c_name in file_entries]
            await asyncio.gather(*tasks)

        if zipped_count == 0:
            return create_error_response(
                message="Failed to bundle any workspace files into ZIP archive",
                error_code="ZIP_CREATION_FAILED",
                details={"failed_files": failed_files},
                status_code=500,
                correlation_id=context.correlation_id,
            )

        zip_bytes = zip_buffer.getvalue()
        total_zip_size = len(zip_bytes)

        # Upload zip archive to cloud storage
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        zip_filename = f"workspace_{workspace_id}_export_{timestamp}.zip"
        zip_blob_path = f"workspace_zips/{workspace_id}/{zip_filename}"

        await export_storage.upload(
            filename=zip_blob_path,
            data=zip_bytes,
            content_type="application/zip",
        )

        # Generate presigned download URL for ZIP archive
        download_url = await export_storage.generate_download_url(
            filename=zip_blob_path,
            expiry_minutes=expiry_minutes,
        )

        response_data = WorkspaceDownloadZipResponse(
            success=True,
            message=f"Successfully generated ZIP containing {zipped_count} document(s)",
            workspace_id=workspace_id,
            total_files_zipped=zipped_count,
            total_size_bytes=total_zip_size,
            zip_file_name=zip_filename,
            download_url=download_url,
            expires_in_seconds=expiry_minutes * 60,
            failed_files=failed_files,
            skipped_files=skipped_files,
        )

        return create_success_response(
            message=response_data.message,
            data=response_data.model_dump(),
            status_code=200,
            correlation_id=context.correlation_id,
        )

    except ValidationException as e:
        return create_error_response(
            message=e.message,
            error_code="VALIDATION_ERROR",
            status_code=400,
            correlation_id=context.correlation_id,
        )
    except AuthorizationException as e:
        return create_error_response(
            message=e.message,
            error_code="AUTHORIZATION_ERROR",
            status_code=403,
            correlation_id=context.correlation_id,
        )
    except Exception as e:
        logger.error(f"Failed to generate workspace ZIP download: {e}", exc_info=True)
        return create_internal_error_response(
            message="Failed to generate workspace documents ZIP archive",
            error=e,
            error_code="WORKSPACE_ZIP_ERROR",
            correlation_id=context.correlation_id,
        )
 