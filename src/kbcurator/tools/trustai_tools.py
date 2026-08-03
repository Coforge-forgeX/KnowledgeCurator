import kbcurator.server.server as server
from kbcurator.server.server import mcp
import httpx
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from kbcurator.utils.auth import require_auth_async
# from kbcurator.utils.permission import is_admin, get_user_role_id

TRUSTAI_BASE_URL = "https://forgex-dev-trustai-qag.azurewebsites.net/trustai-api"


def _wrap_response(data: Any) -> Dict[str, Any]:
    """Wrap API response to ensure it's always a dict (FastMCP requirement)."""
    if isinstance(data, list):
        return {"response": data}
    if isinstance(data, dict):
        return data
    return {"response": data}


def _build_headers(
    config: Any,
    user_email: Optional[str] = None,
    user_id: Optional[str] = None,
    include_content_type: bool = False
) -> Dict[str, str]:
    headers = {
        "accept": "application/json",
        "x-app-id": config.x_app_id,
        "X-Api-Key": config.x_api_key
    }
    if include_content_type:
        headers["Content-Type"] = "application/json"
    if user_email:
        headers["X-User-ID"] = user_email

    return headers


@mcp.tool()
@require_auth_async
async def get_guardrail_config(workspace_id: str) -> Dict[str, Any]:
    """Get guardrail configuration for a workspace."""
    config = server.trustai_db_manager.get_workspace_config(workspace_id)
    if not config:
        return {"error": "Workspace configuration not found"}

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{TRUSTAI_BASE_URL}/guardrails/configuration",
            headers=_build_headers(config)
        )
        return _wrap_response(response.json())


@mcp.tool()
@require_auth_async
async def batch_update_guardrail_config(
    workspace_id: str,
    updates: List[Dict[str, Any]],
    user_email: str,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """Batch update guardrail configuration.

    Args:
        workspace_id: Workspace ID
        user_email: User email mapped to endpoint user_id
        updates: List of update objects with id, field, value
    """
    config = server.trustai_db_manager.get_workspace_config(workspace_id)
    if not config:
        return {"error": "Workspace configuration not found"}

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{TRUSTAI_BASE_URL}/guardrails/configuration/batch",
            headers=_build_headers(config, user_email=user_email, user_id=user_id, include_content_type=True),
            json={"updates": updates}
        )
        return _wrap_response(response.json())


@mcp.tool()
@require_auth_async
async def get_guardrail_logs(
    workspace_id: str,
    user_email: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """Get guardrail logs for a date range.

    Args:
        workspace_id: Workspace ID
        user_email: User email mapped to endpoint user_id
        start_date: Start date (YYYY-MM-DD, optional)
        end_date: End date (YYYY-MM-DD, optional)
        limit: Max records to return
        offset: Pagination offset
    """
    config = server.trustai_db_manager.get_workspace_config(workspace_id)
    if not config:
        return {"error": "Workspace configuration not found"}

    # Default: end_date = today, start_date = one month ago
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if not start_date:
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    params: Dict[str, Any] = {
        "start_date": start_date,
        "end_date": end_date,
        "limit": limit,
        "offset": offset
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{TRUSTAI_BASE_URL}/dashboard/guardrail-logs",
            params=params,
            headers=_build_headers(config, user_email=user_email, user_id=user_id, include_content_type=True)
        )
        return _wrap_response(response.json())


@mcp.tool()
@require_auth_async
async def get_pii_entities(
    workspace_id: str,
    user_email: str,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """Get PII entities list."""
    config = server.trustai_db_manager.get_workspace_config(workspace_id)
    if not config:
        return {"error": "Workspace configuration not found"}

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{TRUSTAI_BASE_URL}/guardrails/pii/entities",
            headers=_build_headers(config, user_email=user_email, user_id=user_id, include_content_type=True)
        )
        return _wrap_response(response.json())


@mcp.tool()
@require_auth_async
async def create_competitor(
    workspace_id: str,
    competitor_name: str,
    user_email: str,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """Create a new competitor."""
    config = server.trustai_db_manager.get_workspace_config(workspace_id)
    if not config:
        return {"error": "Workspace configuration not found"}

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{TRUSTAI_BASE_URL}/guardrails/competitors",
            headers=_build_headers(config, user_email=user_email, user_id=user_id, include_content_type=True),
            json={"competitor_name": competitor_name}
        )
        return _wrap_response(response.json())


@mcp.tool()
@require_auth_async
async def get_competitors(
    workspace_id: str,
    user_email: str,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """Get competitors list."""
    config = server.trustai_db_manager.get_workspace_config(workspace_id)
    if not config:
        return {"error": "Workspace configuration not found"}

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{TRUSTAI_BASE_URL}/guardrails/competitors",
            headers=_build_headers(config, user_email=user_email, user_id=user_id, include_content_type=True)
        )
        return _wrap_response(response.json())


@mcp.tool()
@require_auth_async
async def delete_competitor(
    workspace_id: str,
    competitor_id: int,
    user_email: str,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """Delete a competitor by ID."""
    config = server.trustai_db_manager.get_workspace_config(workspace_id)
    if not config:
        return {"error": "Workspace configuration not found"}

    async with httpx.AsyncClient() as client:
        response = await client.delete(
            f"{TRUSTAI_BASE_URL}/guardrails/competitors/{competitor_id}",
            headers=_build_headers(config, user_email=user_email, user_id=user_id, include_content_type=True)
        )
        return _wrap_response(response.json())


@mcp.tool()
@require_auth_async
async def batch_update_pii_entities(
    workspace_id: str,
    updates: List[Dict[str, Any]],
    user_email: str,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """Batch update PII entities."""
    config = server.trustai_db_manager.get_workspace_config(workspace_id)
    if not config:
        return {"error": "Workspace configuration not found"}

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{TRUSTAI_BASE_URL}/guardrails/pii/entities/batch",
            headers=_build_headers(config, user_email=user_email, user_id=user_id, include_content_type=True),
            json={"updates": updates}
        )
        return _wrap_response(response.json())


@mcp.tool()
@require_auth_async
async def get_regex_patterns(
    workspace_id: str,
    user_email: str,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """Get regex patterns list."""
    config = server.trustai_db_manager.get_workspace_config(workspace_id)
    if not config:
        return {"error": "Workspace configuration not found"}

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{TRUSTAI_BASE_URL}/guardrails/regex-patterns",
            headers=_build_headers(config, user_email=user_email, user_id=user_id)
        )
        return _wrap_response(response.json())


@mcp.tool()
@require_auth_async
async def update_regex_pattern_status(
    workspace_id: str,
    pattern_id: int,
    is_active: bool,
    user_email: str,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """Update regex pattern status."""
    config = server.trustai_db_manager.get_workspace_config(workspace_id)
    if not config:
        return {"error": "Workspace configuration not found"}

    async with httpx.AsyncClient() as client:
        response = await client.put(
            f"{TRUSTAI_BASE_URL}/guardrails/regex-patterns/{pattern_id}/status",
            headers=_build_headers(config, user_email=user_email, user_id=user_id, include_content_type=True),
            json={"is_active": is_active}
        )
        return _wrap_response(response.json())


@mcp.tool()
@require_auth_async
async def update_regex_pattern(
    workspace_id: str,
    pattern_id: int,
    name: str,
    pattern: str,
    action: str,
    is_active: bool,
    user_email: str,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """Update regex pattern."""
    config = server.trustai_db_manager.get_workspace_config(workspace_id)
    if not config:
        return {"error": "Workspace configuration not found"}

    async with httpx.AsyncClient() as client:
        response = await client.put(
            f"{TRUSTAI_BASE_URL}/guardrails/regex-patterns/{pattern_id}",
            headers=_build_headers(config, user_email=user_email, user_id=user_id, include_content_type=True),
            json={"name": name, "pattern": pattern, "action": action, "is_active": is_active}
        )
        return _wrap_response(response.json())


@mcp.tool()
@require_auth_async
async def delete_regex_pattern(
    workspace_id: str,
    pattern_id: int,
    user_email: str,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """Delete regex pattern."""
    config = server.trustai_db_manager.get_workspace_config(workspace_id)
    if not config:
        return {"error": "Workspace configuration not found"}

    async with httpx.AsyncClient() as client:
        response = await client.delete(
            f"{TRUSTAI_BASE_URL}/guardrails/regex-patterns/{pattern_id}",
            headers=_build_headers(config, user_email=user_email, user_id=user_id, include_content_type=True)
        )
        return _wrap_response(response.json())


@mcp.tool()
@require_auth_async
async def get_dashboard_overview(
    workspace_id: str,
    user_email: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """Get dashboard overview."""
    config = server.trustai_db_manager.get_workspace_config(workspace_id)
    if not config:
        return {"error": "Workspace configuration not found"}

    # Default: end_date = today, start_date = one month ago
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if not start_date:
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{TRUSTAI_BASE_URL}/dashboard/overview",
            params={"start_date": start_date, "end_date": end_date},
            headers=_build_headers(config, user_email=user_email, user_id=user_id)
        )
        return _wrap_response(response.json())


@mcp.tool()
@require_auth_async
async def list_api_keys(
    workspace_id: str,
    user_email: str,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """List API keys for app or workspace."""
    config = server.trustai_db_manager.get_workspace_config(workspace_id)
    if not config:
        return {"error": "Workspace configuration not found"}

    _ = user_id
    params: Dict[str, str] = {"user_id": user_email}

    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://forgex-dev-trustai-qag.azurewebsites.net/api/v1/api-keys/",
            params=params,
            headers={
                "accept": "application/json",
                "X-API-KEY": config.x_api_key
            }
        )
        return _wrap_response(response.json())


