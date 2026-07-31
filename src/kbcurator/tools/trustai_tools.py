import kbcurator.server.server as server
from kbcurator.server.server import mcp
import httpx
from typing import Optional, List, Dict, Any
# from kbcurator.utils.permission import is_admin, get_user_role_id

TRUSTAI_BASE_URL = "https://forgex-dev-trustai-qag.azurewebsites.net/trustai-api"


@mcp.tool()
async def get_guardrail_config(workspace_id: str) -> Dict[str, Any]:
    """Get guardrail configuration for a workspace."""
    config = server.trustai_db_manager.get_workspace_config(workspace_id)
    if not config:
        return {"error": "Workspace configuration not found"}

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{TRUSTAI_BASE_URL}/guardrails/configuration",
            headers={
                "accept": "application/json",
                "x-app-id": config.app_id,
                "X-Api-Key": config.app_key
            }
        )
        return response.json()


@mcp.tool()
async def batch_update_guardrail_config(
    workspace_id: str,
    user_id: str,
    updates: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Batch update guardrail configuration.

    Args:
        workspace_id: Workspace ID
        user_id: User email
        updates: List of update objects with id, field, value
    """
    config = server.trustai_db_manager.get_workspace_config(workspace_id)
    if not config:
        return {"error": "Workspace configuration not found"}

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{TRUSTAI_BASE_URL}/guardrails/configuration/batch",
            headers={
                "accept": "application/json",
                "x-app-id": config.app_id,
                "Content-Type": "application/json",
                "X-Api-Key": config.app_key,
                "X-User-ID": user_id
            },
            json={"updates": updates}
        )
        return response.json()


@mcp.tool()
async def get_guardrail_logs(
    workspace_id: str,
    user_id: str,
    start_date: str,
    end_date: str,
    limit: int = 20,
    offset: int = 0
) -> Dict[str, Any]:
    """Get guardrail logs for a date range.

    Args:
        workspace_id: Workspace ID
        user_id: User email
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        limit: Max records to return
        offset: Pagination offset
    """
    config = server.trustai_db_manager.get_workspace_config(workspace_id)
    if not config:
        return {"error": "Workspace configuration not found"}

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{TRUSTAI_BASE_URL}/dashboard/guardrail-logs",
            params={
                "start_date": start_date,
                "end_date": end_date,
                "limit": limit,
                "offset": offset
            },
            headers={
                "accept": "application/json",
                "x-app-id": config.app_id,
                "Content-Type": "application/json",
                "X-User-ID": user_id,
                "X-Api-Key": config.app_key
            }
        )
        return response.json()


@mcp.tool()
async def get_pii_entities(workspace_id: str, user_id: str) -> Dict[str, Any]:
    """Get PII entities list."""
    config = server.trustai_db_manager.get_workspace_config(workspace_id)
    if not config:
        return {"error": "Workspace configuration not found"}

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{TRUSTAI_BASE_URL}/guardrails/pii/entities",
            headers={
                "accept": "application/json",
                "x-app-id": config.app_id,
                "Content-Type": "application/json",
                "X-User-ID": user_id,
                "X-Api-Key": config.app_key
            }
        )
        return response.json()


@mcp.tool()
async def create_competitor(
    workspace_id: str,
    user_id: str,
    competitor_name: str
) -> Dict[str, Any]:
    """Create a new competitor."""
    config = server.trustai_db_manager.get_workspace_config(workspace_id)
    if not config:
        return {"error": "Workspace configuration not found"}

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{TRUSTAI_BASE_URL}/guardrails/competitors",
            headers={
                "accept": "application/json",
                "x-app-id": config.app_id,
                "Content-Type": "application/json",
                "X-User-ID": user_id,
                "X-Api-Key": config.app_key
            },
            json={"competitor_name": competitor_name}
        )
        return response.json()


@mcp.tool()
async def get_competitors(workspace_id: str, user_id: str) -> Dict[str, Any]:
    """Get competitors list."""
    config = server.trustai_db_manager.get_workspace_config(workspace_id)
    if not config:
        return {"error": "Workspace configuration not found"}

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{TRUSTAI_BASE_URL}/guardrails/competitors",
            headers={
                "accept": "application/json",
                "x-app-id": config.app_id,
                "Content-Type": "application/json",
                "X-User-ID": user_id,
                "X-Api-Key": config.app_key
            }
        )
        return response.json()


@mcp.tool()
async def delete_competitor(
    workspace_id: str,
    user_id: str,
    competitor_id: int
) -> Dict[str, Any]:
    """Delete a competitor by ID."""
    config = server.trustai_db_manager.get_workspace_config(workspace_id)
    if not config:
        return {"error": "Workspace configuration not found"}

    async with httpx.AsyncClient() as client:
        response = await client.delete(
            f"{TRUSTAI_BASE_URL}/guardrails/competitors/{competitor_id}",
            headers={
                "accept": "application/json",
                "x-app-id": config.app_id,
                "Content-Type": "application/json",
                "X-User-ID": user_id,
                "X-Api-Key": config.app_key
            }
        )
        return response.json()


@mcp.tool()
async def batch_update_pii_entities(
    workspace_id: str,
    user_id: str,
    updates: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Batch update PII entities."""
    config = server.trustai_db_manager.get_workspace_config(workspace_id)
    if not config:
        return {"error": "Workspace configuration not found"}

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{TRUSTAI_BASE_URL}/guardrails/pii/entities/batch",
            headers={
                "accept": "application/json",
                "x-app-id": config.app_id,
                "Content-Type": "application/json",
                "X-User-ID": user_id,
                "X-Api-Key": config.app_key
            },
            json={"updates": updates}
        )
        return response.json()


@mcp.tool()
async def get_regex_patterns(workspace_id: str, user_id: str) -> Dict[str, Any]:
    """Get regex patterns list."""
    config = server.trustai_db_manager.get_workspace_config(workspace_id)
    if not config:
        return {"error": "Workspace configuration not found"}

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{TRUSTAI_BASE_URL}/guardrails/regex-patterns",
            headers={
                "accept": "application/json",
                "x-app-id": config.app_id,
                "X-User-ID": user_id,
                "X-Api-Key": config.app_key
            }
        )
        return response.json()


@mcp.tool()
async def update_regex_pattern_status(
    workspace_id: str,
    user_id: str,
    pattern_id: int,
    is_active: bool
) -> Dict[str, Any]:
    """Update regex pattern status."""
    config = server.trustai_db_manager.get_workspace_config(workspace_id)
    if not config:
        return {"error": "Workspace configuration not found"}

    async with httpx.AsyncClient() as client:
        response = await client.put(
            f"{TRUSTAI_BASE_URL}/guardrails/regex-patterns/{pattern_id}/status",
            headers={
                "accept": "application/json",
                "x-app-id": config.app_id,
                "Content-Type": "application/json",
                "X-User-ID": user_id,
                "X-Api-Key": config.app_key
            },
            json={"is_active": is_active}
        )
        return response.json()


@mcp.tool()
async def update_regex_pattern(
    workspace_id: str,
    user_id: str,
    pattern_id: int,
    name: str,
    pattern: str,
    action: str,
    is_active: bool
) -> Dict[str, Any]:
    """Update regex pattern."""
    config = server.trustai_db_manager.get_workspace_config(workspace_id)
    if not config:
        return {"error": "Workspace configuration not found"}

    async with httpx.AsyncClient() as client:
        response = await client.put(
            f"{TRUSTAI_BASE_URL}/guardrails/regex-patterns/{pattern_id}",
            headers={
                "accept": "application/json",
                "x-app-id": config.app_id,
                "Content-Type": "application/json",
                "X-User-ID": user_id,
                "X-Api-Key": config.app_key
            },
            json={"name": name, "pattern": pattern, "action": action, "is_active": is_active}
        )
        return response.json()


@mcp.tool()
async def delete_regex_pattern(
    workspace_id: str,
    user_id: str,
    pattern_id: int
) -> Dict[str, Any]:
    """Delete regex pattern."""
    config = server.trustai_db_manager.get_workspace_config(workspace_id)
    if not config:
        return {"error": "Workspace configuration not found"}

    async with httpx.AsyncClient() as client:
        response = await client.delete(
            f"{TRUSTAI_BASE_URL}/guardrails/regex-patterns/{pattern_id}",
            headers={
                "accept": "application/json",
                "x-app-id": config.app_id,
                "Content-Type": "application/json",
                "X-User-ID": user_id,
                "X-Api-Key": config.app_key
            }
        )
        return response.json()


@mcp.tool()
async def get_dashboard_overview(
    workspace_id: str,
    user_id: str,
    start_date: str,
    end_date: str
) -> Dict[str, Any]:
    """Get dashboard overview."""
    config = server.trustai_db_manager.get_workspace_config(workspace_id)
    if not config:
        return {"error": "Workspace configuration not found"}

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{TRUSTAI_BASE_URL}/dashboard/overview",
            params={"start_date": start_date, "end_date": end_date},
            headers={
                "accept": "application/json",
                "x-app-id": config.app_id,
                "X-User-ID": user_id,
                "X-Api-Key": config.app_key
            }
        )
        return response.json()


@mcp.tool()
async def list_api_keys(workspace_id: str, user_id: str) -> Dict[str, Any]:
    """List API keys for app or workspace."""
    config = server.trustai_db_manager.get_workspace_config(workspace_id)
    if not config:
        return {"error": "Workspace configuration not found"}

    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://forgex-dev-trustai-qag.azurewebsites.net/api/v1/api-keys/",
            params={"user_id": user_id},
            headers={
                "accept": "application/json",
                "X-API-KEY": config.app_key
            }
        )
        return response.json()

