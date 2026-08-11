"""File tasks status API with file_tasks_id-first semantics."""
import os
import json

from sqlalchemy import select

from src.core.abstractions import AbstractContext, AbstractRequest, AbstractResponse
from src.core.auth import get_user_id, get_workspace_ids, require_auth
from src.core.database import FileTask, UserMap, get_async_session
from src.core.exceptions import AuthorizationException, ValidationException
from src.core.logging import get_logger
from src.common import ErrorMessages, create_error_response, create_success_response, parse_request

from .payloads import FileTasksStatusRequest

logger = get_logger(__name__)


def _parse_int_list(value: str) -> list[int]:
    raw = str(value or "").strip()
    if not raw:
        return []

    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [int(item) for item in parsed]
            return []
        except Exception:
            return []

    result = []
    for chunk in raw.split(","):
        part = chunk.strip()
        if not part:
            continue
        result.append(int(part))
    return result


def _normalize_get_payload(req: AbstractRequest) -> dict:
    """Normalize GET query params to FileTasksStatusRequest payload shape."""
    task_ids_raw = req.get_query_param("file_tasks_id") or req.get_query_param("task_ids") or req.get_query_param("task_id")
    workspace_raw = req.get_query_param("workspace_id")

    payload: dict = {}
    if task_ids_raw:
        task_ids = _parse_int_list(task_ids_raw)
        if task_ids:
            payload["file_tasks_id"] = task_ids

    if workspace_raw:
        try:
            payload["workspace_id"] = int(str(workspace_raw).strip())
        except Exception:
            payload["workspace_id"] = workspace_raw

    return payload


def _to_status_row(task: FileTask) -> dict:
    file_name = os.path.basename(task.file_path) if task.file_path else None
    return {
        "file_task_id": task.id,
        "workspace_id": task.workspace_id,
        "file_name": file_name,
        "file_path": task.file_path,
        "status": task.status,
        "created_at": str(task.created_at) if task.created_at else None,
        "updated_at": str(task.updated_at) if task.updated_at else None,
    }


@require_auth()
async def main(req: AbstractRequest, context: AbstractContext) -> AbstractResponse:
    """Return status for specific file_tasks_id or all tasks in workspace."""
    user_id = get_user_id(req)
    user_workspaces = set(get_workspace_ids(req))

    try:
        if req.method.upper() == "GET":
            payload = FileTasksStatusRequest.model_validate(_normalize_get_payload(req))
        else:
            payload, error_response = parse_request(req, FileTasksStatusRequest)
            if error_response:
                return error_response

        rows = []
        mode = "file_tasks_id" if payload.file_tasks_id else "workspace"

        async with get_async_session() as session:
            # Token workspace claims can be stale; refresh from DB to avoid false 403s.
            membership_stmt = select(UserMap.workspace_id).where(
                UserMap.user_id == user_id,
                UserMap.is_active == True,
            )
            membership_rows = (await session.execute(membership_stmt)).scalars().all()
            effective_workspaces = set(user_workspaces)
            effective_workspaces.update(int(wid) for wid in membership_rows)

            if payload.file_tasks_id:
                stmt = (
                    select(FileTask)
                    .where(FileTask.id.in_(payload.file_tasks_id))
                    .order_by(FileTask.created_at.desc())
                )
                result = await session.execute(stmt)
                tasks = result.scalars().all()

                for task in tasks:
                    if int(task.workspace_id) not in effective_workspaces:
                        raise AuthorizationException(message=ErrorMessages.UNAUTHORIZED_ACCESS)
                    rows.append(_to_status_row(task))

            else:
                workspace_id = int(payload.workspace_id)
                if workspace_id not in effective_workspaces:
                    raise AuthorizationException(message=ErrorMessages.UNAUTHORIZED_ACCESS)

                stmt = (
                    select(FileTask)
                    .where(FileTask.workspace_id == workspace_id)
                    .order_by(FileTask.created_at.desc())
                )
                result = await session.execute(stmt)
                tasks = result.scalars().all()
                rows = [_to_status_row(task) for task in tasks]

        return create_success_response(
            message="File task status retrieved successfully",
            data={
                "mode": mode,
                "requested_file_tasks_id": payload.file_tasks_id or [],
                "workspace_id": payload.workspace_id,
                "count": len(rows),
                "statuses": rows,
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
    except ValidationException as e:
        return create_error_response(
            message=e.message,
            error_code="VALIDATION_ERROR",
            status_code=400,
            correlation_id=context.correlation_id,
        )
    except Exception as e:
        logger.error("Failed to fetch file task statuses", error=e, user_id=user_id)
        return create_error_response(
            message=ErrorMessages.QUERY_EXECUTION_ERROR,
            error_code="FILE_TASK_STATUS_FAILED",
            details={"error": str(e)},
            status_code=500,
            correlation_id=context.correlation_id,
        )
