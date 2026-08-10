"""Workspace documents API with backward-compatible data sources."""
import os
import json
from typing import Dict, List, Optional, Tuple

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
from src.shared import ErrorMessages, create_error_response, create_success_response, parse_request

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
                    UserMap.is_active == True,
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
                    WorkspaceIndustryIntentMap.is_active == True,
                    WorkspaceIndustryIntentMap.kb_id.isnot(None),
                )
                .distinct()
            )
            kb_ids = [row[0] for row in (await session.execute(map_stmt)).fetchall()]

            kb_titles: Dict[int, str] = {}
            if kb_ids:
                kb_stmt = select(KnowledgeBase.id, KnowledgeBase.title).where(KnowledgeBase.id.in_(kb_ids))
                kb_rows = (await session.execute(kb_stmt)).fetchall()
                kb_titles = {int(row[0]): (row[1] or f"kb_{row[0]}") for row in kb_rows}

            file_tasks_stmt = select(FileTask).where(FileTask.workspace_id == workspace_id)
            file_task_rows = (await session.execute(file_tasks_stmt)).scalars().all()

            if kb_ids:
                metadata_stmt = select(DocumentMetadata).where(
                    or_(
                        DocumentMetadata.workspace_id == workspace_id,
                        DocumentMetadata.kb_id.in_(kb_ids),
                    )
                )
            else:
                metadata_stmt = select(DocumentMetadata).where(DocumentMetadata.workspace_id == workspace_id)
            metadata_rows = (await session.execute(metadata_stmt)).scalars().all()

        # Build response buckets grouped by workspace and KB.
        buckets: Dict[str, Dict[str, object]] = {}

        def ensure_bucket(group_key: str, label: str, group_workspace_id: Optional[int], kb_id: Optional[int]) -> Dict[str, object]:
            if group_key not in buckets:
                buckets[group_key] = {
                    "group_key": group_key,
                    "group_label": label,
                    "workspace_id": group_workspace_id,
                    "kb_id": kb_id,
                    "documents": [],
                    "_doc_index": {},
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
        for task in file_task_rows:
            file_path = task.file_path or ""
            file_name = _extract_file_name(file_path)
            if not file_name and not file_path:
                continue

            key = _doc_key(file_name, file_path)
            doc_index = current_bucket["_doc_index"]
            if key in doc_index:
                continue

            doc = {
                "file_name": file_name,
                "file_path": file_path,
                # "workspace_id": int(task.workspace_id),
                "kb_id": None,
                # "source": "file_tasks",
                "status": task.status,
                "file_task_id": int(task.id),
                # "document_metadata_id": None,
                "indexed_at": None,
                "updated_at": str(task.updated_at) if task.updated_at else None,
            }
            current_bucket["documents"].append(doc)
            doc_index[key] = len(current_bucket["documents"]) - 1

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

            for target_key in target_keys:
                bucket = buckets.get(target_key)
                if not bucket:
                    continue

                key = _doc_key(file_name, file_path)
                doc_index = bucket["_doc_index"]
                replacement = {
                    "file_name": file_name,
                    "file_path": file_path,
                    # "workspace_id": int(meta.workspace_id),
                    "kb_id": int(meta.kb_id) if meta.kb_id is not None else None,
                    # "source": "document_metadata",
                    "status": "indexed",
                    "file_task_id": int(meta.file_task_id) if meta.file_task_id is not None else None,
                    # "document_metadata_id": int(meta.id),
                    "indexed_at": str(meta.indexed_at) if meta.indexed_at else None,
                    "updated_at": str(meta.updated_at) if meta.updated_at else None,
                }

                if key in doc_index:
                    bucket["documents"][doc_index[key]] = replacement
                else:
                    bucket["documents"].append(replacement)
                    doc_index[key] = len(bucket["documents"]) - 1

        # Always attach signed file_id; Redis cache is optional optimization.
        file_task_container_map = {
            int(task.id): str(task.container_name)
            for task in file_task_rows
            if task.container_name
        }
        for bucket in buckets.values():
            for doc in bucket["documents"]:
                file_name = str(doc.get("file_name") or "").strip()
                file_path = str(doc.get("file_path") or "").strip()
                if not file_path:
                    continue

                # Reuse container from task if present, else fallback to default storage container.
                task_id = doc.get("file_task_id")
                container_name = file_task_container_map.get(int(task_id)) if task_id is not None else None
                container_name = container_name or str(settings.storage.STORAGE_CONTAINER_NAME or "")

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

        response_groups = []
        for bucket in buckets.values():
            docs = bucket["documents"]
            is_workspace_group = bucket["group_key"] == f"workspace:{workspace_id}"
            if not docs and not is_workspace_group:
                continue
            docs.sort(key=lambda d: (d.get("file_name") or "").lower())
            response_groups.append(
                {
                    "key": bucket["group_key"],
                    "label": bucket["group_label"],
                    "kb_id": bucket["kb_id"],
                    "count": len(docs),
                    "documents": docs,
                }
            )

        response_groups.sort(key=lambda g: g["key"])

        return create_success_response(
            message="Workspace documents retrieved successfully",
            data={
                "workspace_id": workspace_id,
                "group_count": len(response_groups),
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
