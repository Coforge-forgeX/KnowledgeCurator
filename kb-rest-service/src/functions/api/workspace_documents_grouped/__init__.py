"""Workspace documents API with backward-compatible data sources."""
import os
import json
from typing import Any, Dict, List, Optional, Tuple, cast

from sqlalchemy import or_, select

from src.core.abstractions import AbstractContext, AbstractRequest, AbstractResponse
from src.core.auth import get_user_id, get_workspace_ids, require_auth
from src.core.config import settings
from src.core.database import (
    DocumentMetadata,
    FileTask,
    KnowledgeBase,
    UserMap,
    Workspace,
    WorkspaceIndustryIntentMap,
    get_async_session,
)
from src.core.exceptions import AuthorizationException
from src.core.logging import get_logger
from src.core.redis import redis_manager
from src.helpers.file_token import create_signed_file_id
from src.common import ErrorMessages, create_error_response, create_success_response, parse_request

from .payloads import WorkspaceDocumentsGroupedRequest

logger = get_logger(__name__)

QUERY_EVIDENCE_TTL_SECONDS = 30 * 60
FILE_KEY_PREFIX = "query_file:"


def _extract_file_name(file_path: Optional[str]) -> str:
    if not file_path:
        return ""
    return os.path.basename(file_path)


def _doc_key(file_name: str, file_path: str) -> Tuple[str, str]:
    return (file_name.strip().lower(), file_path.strip().lower())


def _format_file_size(size_bytes: Optional[int]) -> Optional[str]:
    if size_bytes is None:
        return None
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


@require_auth()
async def main(req: AbstractRequest, context: AbstractContext) -> AbstractResponse:
    """Return workspace documents, grouped in response by workspace/KB."""
    user_id = get_user_id(req)
    user_workspaces = set(get_workspace_ids(req))

    payload_data = req.get_query_params() if req.method == "GET" else req.get_json()
    try:
        payload = WorkspaceDocumentsGroupedRequest.model_validate(payload_data)
        error_response = None
    except Exception:
        payload, error_response = parse_request(req, WorkspaceDocumentsGroupedRequest)

    if error_response:
        return error_response

    workspace_id = int(payload.workspace_id)

    try:
        async with get_async_session() as session:
            if workspace_id not in user_workspaces:
                # Token roles can be stale; verify membership in DB before denying access.
                membership_stmt = select(UserMap.workspace_id).where(
                    UserMap.user_id == user_id,
                    UserMap.workspace_id == workspace_id,
                    UserMap.is_active.is_(True),
                )
                membership = (await session.execute(membership_stmt)).scalar_one_or_none()
                if membership is None:
                    raise AuthorizationException(message=ErrorMessages.UNAUTHORIZED_ACCESS)

            workspace_stmt = select(Workspace).where(Workspace.workspace_id == workspace_id)
            workspace_row = (await session.execute(workspace_stmt)).scalar_one_or_none()
            workspace_name = workspace_row.workspace_name if workspace_row else f"workspace_{workspace_id}"

            map_stmt = (
                select(WorkspaceIndustryIntentMap.kb_id)
                .where(
                    WorkspaceIndustryIntentMap.workspace_id == workspace_id,
                    WorkspaceIndustryIntentMap.is_active.is_(True),
                    WorkspaceIndustryIntentMap.kb_id.isnot(None),
                )
                .distinct()
            )
            kb_ids: List[int] = [
                int(row[0]) for row in (await session.execute(map_stmt)).fetchall() if row[0] is not None
            ]

            kb_titles: Dict[int, str] = {}
            if kb_ids:
                kb_stmt = select(KnowledgeBase.id, KnowledgeBase.title).where(KnowledgeBase.id.in_(kb_ids))
                kb_rows = (await session.execute(kb_stmt)).fetchall()
                kb_titles = {int(row[0]): str(row[1] or f"kb_{row[0]}") for row in kb_rows}

            if kb_ids:
                metadata_stmt = select(DocumentMetadata).where(
                    or_(
                        DocumentMetadata.workspace_id == workspace_id,
                        DocumentMetadata.kb_id.in_(kb_ids),
                    )
                )
            else:
                metadata_stmt = select(DocumentMetadata).where(DocumentMetadata.workspace_id == workspace_id)
            metadata_rows: List[DocumentMetadata] = list((await session.execute(metadata_stmt)).scalars().all())

            file_task_ids: List[int] = [
                int(meta.file_task_id)
                for meta in metadata_rows
                if meta.file_task_id is not None
            ]

            if file_task_ids:
                file_tasks_stmt = select(FileTask).where(
                    or_(
                        FileTask.workspace_id == workspace_id,
                        FileTask.id.in_(file_task_ids),
                    )
                )
            else:
                file_tasks_stmt = select(FileTask).where(FileTask.workspace_id == workspace_id)
            file_task_rows: List[FileTask] = list((await session.execute(file_tasks_stmt)).scalars().all())

        file_tasks_by_id: Dict[int, FileTask] = {
            int(task.id): task for task in file_task_rows
        }
        file_tasks_by_path: Dict[Tuple[str, str], FileTask] = {}
        for task in file_task_rows:
            t_path = task.file_path or ""
            t_name = _extract_file_name(t_path)
            if t_name or t_path:
                file_tasks_by_path[_doc_key(t_name, t_path)] = task

        # Build response buckets grouped by workspace and KB.
        buckets: Dict[str, Dict[str, Any]] = {}

        def ensure_bucket(group_key: str, label: str, group_workspace_id: Optional[int], kb_id: Optional[int]) -> Dict[str, Any]:
            if group_key not in buckets:
                doc_index: Dict[Tuple[str, str], int] = {}
                documents: List[Dict[str, Any]] = []
                buckets[group_key] = {
                    "group_key": group_key,
                    "group_label": label,
                    "workspace_id": group_workspace_id,
                    "kb_id": kb_id,
                    "documents": documents,
                    "_doc_index": doc_index,
                }
            return buckets[group_key]

        current_bucket = ensure_bucket(
            group_key=f"workspace:{workspace_id}",
            label=f"{workspace_name}",
            group_workspace_id=workspace_id,
            kb_id=None,
        )

        for kb_id in kb_ids:
            title = kb_titles.get(kb_id, f"kb_{kb_id}")
            ensure_bucket(
                group_key=f"kb:{kb_id}",
                label=f"KB {title}",
                group_workspace_id=workspace_id,
                kb_id=kb_id,
            )

        # Seed current workspace docs from file_tasks for backward compatibility.
        current_doc_index: Dict[Tuple[str, str], int] = current_bucket["_doc_index"]
        current_documents: List[Dict[str, Any]] = current_bucket["documents"]
        for task in file_task_rows:
            if int(task.workspace_id) != workspace_id:
                continue

            file_path = task.file_path or ""
            file_name = _extract_file_name(file_path)
            if not file_name and not file_path:
                continue

            key = _doc_key(file_name, file_path)
            if key in current_doc_index:
                continue

            container_name = task.container_name or str(settings.storage.STORAGE_CONTAINER_NAME or "")
            doc: Dict[str, Any] = {
                "file_name": file_name,
                "file": file_path,
                "domain": task.domain,
                "kb_name": task.kb_name,
                "file_size": task.file_size,
                "uploaded_by": task.uploaded_by,
                "container_name": container_name,
                "kb_id": None,
                "status": task.status,
                "file_task_id": int(task.id),
                "indexed_at": None,
                "updated_at": str(task.updated_at) if task.updated_at else None,
            }
            current_documents.append(doc)
            current_doc_index[key] = len(current_documents) - 1

        # Merge document_metadata with higher preference in conflict cases.
        for meta in metadata_rows:
            file_name = (meta.file_name or "").strip()
            file_path = (meta.file_path or "").strip()
            if not file_name and file_path:
                file_name = _extract_file_name(file_path)

            if not file_name and not file_path:
                continue

            target_keys: List[str] = []
            is_current_workspace_doc = int(meta.workspace_id) == workspace_id

            # Current workspace docs should only appear in the workspace group.
            if is_current_workspace_doc:
                target_keys.append(f"workspace:{workspace_id}")
            elif meta.kb_id is not None and int(meta.kb_id) in kb_ids:
                target_keys.append(f"kb:{int(meta.kb_id)}")

            if not target_keys:
                continue

            ft: Optional[FileTask] = None
            if meta.file_task_id is not None:
                ft = file_tasks_by_id.get(int(meta.file_task_id))
            if not ft:
                ft = file_tasks_by_path.get(_doc_key(file_name, file_path))

            doc_meta: Dict[str, Any] = cast(Dict[str, Any], meta.doc_metadata) if isinstance(meta.doc_metadata, dict) else {}
            domain = ft.domain if ft else doc_meta.get("domain")

            kb_id_int = int(meta.kb_id) if meta.kb_id is not None else None
            kb_name = ft.kb_name if (ft and ft.kb_name) else (kb_titles.get(kb_id_int) if kb_id_int is not None else None)

            file_size = ft.file_size if (ft and ft.file_size) else _format_file_size(meta.file_size_bytes)
            uploaded_by = ft.uploaded_by if ft else doc_meta.get("uploaded_by")
            container_name = (ft.container_name if (ft and ft.container_name) else None) or str(settings.storage.STORAGE_CONTAINER_NAME or "")

            for target_key in target_keys:
                bucket = buckets.get(target_key)
                if not bucket:
                    continue

                key = _doc_key(file_name, file_path)
                bucket_doc_index: Dict[Tuple[str, str], int] = bucket["_doc_index"]
                bucket_documents: List[Dict[str, Any]] = bucket["documents"]
                replacement: Dict[str, Any] = {
                    "file_name": file_name,
                    "file": file_path,
                    "domain": domain,
                    "kb_name": kb_name,
                    "file_size": file_size,
                    "uploaded_by": uploaded_by,
                    "container_name": container_name,
                    "kb_id": kb_id_int,
                    "status": "indexed",
                    "file_task_id": int(meta.file_task_id) if meta.file_task_id is not None else None,
                    "indexed_at": str(meta.indexed_at) if meta.indexed_at else None,
                    "updated_at": str(meta.updated_at) if meta.updated_at else None,
                }

                if key in bucket_doc_index:
                    bucket_documents[bucket_doc_index[key]] = replacement
                else:
                    bucket_documents.append(replacement)
                    bucket_doc_index[key] = len(bucket_documents) - 1

        # Always attach signed file_id; Redis cache is optional optimization.
        for bucket in buckets.values():
            docs: List[Dict[str, Any]] = bucket["documents"]
            for doc in docs:
                file_name = str(doc.get("file_name") or "").strip()
                file_path = str(doc.get("file") or "").strip()
                if not file_path:
                    continue

                container_name = str(doc.get("container_name") or settings.storage.STORAGE_CONTAINER_NAME or "")

                file_id = create_signed_file_id(
                    workspace_id=workspace_id,
                    container_name=container_name,
                    blob_path=file_path,
                    provider=str(settings.storage.STORAGE_PROVIDER or "azure"),
                    file_name=file_name,
                )

                if redis_manager.is_available:
                    source_mapping = {
                        "file_id": file_id,
                        "workspace_id": workspace_id,
                        "container_name": container_name,
                        "blob_path": file_path,
                        "provider": str(settings.storage.STORAGE_PROVIDER or "azure"),
                        "file_name": file_name,
                        "citation": None,
                        "evidence_cache_key": None,
                    }
                    redis_manager.setex(
                        f"{FILE_KEY_PREFIX}{file_id}",
                        QUERY_EVIDENCE_TTL_SECONDS,
                        json.dumps(source_mapping),
                    )

                doc["file_id"] = file_id

        response_groups: List[Dict[str, Any]] = []
        for bucket in buckets.values():
            docs: List[Dict[str, Any]] = bucket["documents"]
            is_workspace_group = bucket["group_key"] == f"workspace:{workspace_id}"
            if not docs and not is_workspace_group:
                continue
            docs.sort(key=lambda d: str(d.get("file_name") or "").lower())
            response_groups.append(
                {
                    "key": bucket["group_key"],
                    "label": bucket["group_label"],
                    "kb_id": bucket["kb_id"],
                    "count": len(docs),
                    "total": len(docs),
                    "documents": docs,
                }
            )

        response_groups.sort(key=lambda g: str(g["key"]))
        overall_total = sum(int(group["total"]) for group in response_groups)

        return create_success_response(
            message="Workspace documents retrieved successfully",
            data={
                "workspace_id": workspace_id,
                "group_count": len(response_groups),
                "total": overall_total,
                "groups": response_groups,
            },
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
        logger.error("Failed to fetch workspace documents", error=e, workspace_id=workspace_id)
        return create_error_response(
            message=ErrorMessages.QUERY_EXECUTION_ERROR,
            error_code="WORKSPACE_DOCS_FAILED",
            status_code=500,
            correlation_id=context.correlation_id,
        )
 