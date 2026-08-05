import kbcurator.server.server as server
from kbcurator.server.server import mcp
import httpx
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from kbcurator.utils.auth import require_auth_async
from os import getenv

# from kbcurator.utils.permission import is_admin, get_user_role_id

TRUSTAI_BASE_URL = getenv("TRUSTAI_BASE_URL", "https://forgex-dev-trustai-qag.azurewebsites.net")

_GUARDRAIL_SECTION_MAP: Dict[str, Dict[str, str]] = {
    "PROMPT_INJECTION": {
        "id": "security",
        "title": "SECURITY",
        "description": "Detects attempts to override system instructions"
    },
    "BSI_DETECTION": {
        "id": "security",
        "title": "SECURITY",
        "description": "Detects business-sensitive information"
    },
    "TOXIC": {
        "id": "content_safety",
        "title": "CONTENT SAFETY",
        "description": "Hateful, harassing or unsafe language"
    },
    "BIAS_DETECTION": {
        "id": "content_safety",
        "title": "CONTENT SAFETY",
        "description": "Discriminatory or unfair output"
    },
    "PII": {
        "id": "data_protection",
        "title": "DATA PROTECTION",
        "description": "Personal & health identifiers"
    },
    "FACTUAL_ACCURACY": {
        "id": "quality",
        "title": "QUALITY",
        "description": "Groundedness against provided context"
    },
    "CODE_HALLUCINATION": {
        "id": "quality",
        "title": "QUALITY",
        "description": "Fabricated APIs, packages or functions"
    },
    "COMPETITOR_CHECK": {
        "id": "brand_business",
        "title": "BRAND / BUSINESS",
        "description": "Fed by the Restricted Terms denylist"
    },
    "TOKEN_QUOTA": {
        "id": "usage",
        "title": "USAGE",
        "description": "Managed in Token & Quota"
    },
    "REGEX_CHECK": {
        "id": "quality",
        "title": "QUALITY",
        "description": "Custom regex pattern check"
    }
}


# def _scope_from_guardrail_type(guardrail_type: Optional[str]) -> str:
#     if not guardrail_type:
#         return "Input"
#     normalized = str(guardrail_type).strip().upper()
#     if normalized == "I":
#         return "Input"
#     if normalized == "O":
#         return "Output"
#     return "In/Out"


# def _policy_action_from_code(action_code: Optional[str]) -> str:
#     normalized = (action_code or "").strip().upper()
#     if normalized == "A":
#         return "Anonymize"
#     if normalized == "W":
#         return "Warn"
#     return "Block"


# def _pattern_action_from_code(action_code: Optional[str]) -> str:
#     normalized = (action_code or "").strip().upper()
#     if normalized in {"M", "MASK"}:
#         return "mask"
#     if normalized in {"W", "WARN"}:
#         return "warn"
#     if normalized in {"B", "BLOCK"}:
#         return "block"
#     return "mask"


# def _pii_action_from_code(action_code: Optional[str]) -> str:
#     normalized = (action_code or "").strip().upper()
#     if normalized in {"A", "ANONYMIZE"}:
#         return "anonymize"
#     if normalized in {"B", "BLOCK"}:
#         return "block"
#     return "mask"


# def _safe_float(value: Any, default: float = 0.0) -> float:
#     try:
#         return float(value)
#     except (TypeError, ValueError):
#         return default


# def _humanize_guardrail_name(name: Optional[str]) -> str:
#     if not name:
#         return ""
#     return str(name).replace("_", " ").title()


def _guardrail_policies_response(workspace_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    rows = payload.get("response", []) if isinstance(payload, dict) else []
    sections_by_id: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        if not isinstance(row, dict):
            continue
        guardrail_name = str(row.get("guardrail_name") or "").strip().upper()

        section_meta = _GUARDRAIL_SECTION_MAP.get(
            guardrail_name,
            {"id": "general", "title": "GENERAL", "description": "Guardrail policy"}
        )
        section_id = section_meta["id"]

        if section_id not in sections_by_id:
            sections_by_id[section_id] = {
                "id": section_meta["id"],
                "title": section_meta["title"],
                "description": section_meta["description"],
                "items": []
            }

        # Keep TrustAI keys and values exactly as returned; only group by class sections.
        sections_by_id[section_id]["items"].append(dict(row))

    return {
        "workspaceId": workspace_id,
        "sections": list(sections_by_id.values())
    }


def _data_protection_response(workspace_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    rows = payload.get("response", []) if isinstance(payload, dict) else []
    records: List[Any] = [dict(row) if isinstance(row, dict) else row for row in rows]

    return {
        "workspaceId": workspace_id,
        "records": records
    }


def _restricted_terms_response(workspace_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    rows = payload.get("response", []) if isinstance(payload, dict) else []
    terms: List[Any] = [dict(row) if isinstance(row, dict) else row for row in rows]

    return {
        "workspaceId": workspace_id,
        "terms": terms
    }


def _custom_patterns_response(workspace_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    rows = payload.get("response", []) if isinstance(payload, dict) else []
    patterns: List[Any] = [dict(row) if isinstance(row, dict) else row for row in rows]

    return {
        "workspaceId": workspace_id,
        "patterns": patterns
    }


def _guardrail_logs_response(workspace_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {"workspaceId": workspace_id, "metadata": {}, "data": []}
    return {
        "workspaceId": workspace_id,
        "metadata": payload.get("metadata", {}),
        "data": payload.get("data", [])
    }


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
            f"{TRUSTAI_BASE_URL}/trustai-api/guardrails/configuration",
            headers=_build_headers(config)
        )
        raw_payload = _wrap_response(response.json())
        return _guardrail_policies_response(workspace_id, raw_payload)


@mcp.tool()
@require_auth_async
async def batch_update_guardrail_config(
    workspace_id: str,
    updates: List[Dict[str, Any]],
    user_email: Optional[str] = None,
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
            f"{TRUSTAI_BASE_URL}/trustai-api/guardrails/configuration/batch",
            headers=_build_headers(config, user_email=user_email, user_id=user_id, include_content_type=True),
            json={"updates": updates}
        )
        return _wrap_response(response.json())


@mcp.tool()
@require_auth_async
async def get_guardrail_logs(
    workspace_id: str,
    user_email: Optional[str] = None,
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
            f"{TRUSTAI_BASE_URL}/trustai-api/dashboard/guardrail-logs",
            params=params,
            headers=_build_headers(config, user_email=user_email, user_id=user_id, include_content_type=True)
        )
        raw_payload = _wrap_response(response.json())
        return _guardrail_logs_response(workspace_id, raw_payload)


@mcp.tool()
@require_auth_async
async def get_pii_entities(
    workspace_id: str,
    user_email: Optional[str] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """Get PII entities list."""
    config = server.trustai_db_manager.get_workspace_config(workspace_id)
    if not config:
        return {"error": "Workspace configuration not found"}

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{TRUSTAI_BASE_URL}/trustai-api/guardrails/pii/entities",
            headers=_build_headers(config, user_email=user_email, user_id=user_id, include_content_type=True)
        )
        raw_payload = _wrap_response(response.json())
        return _data_protection_response(workspace_id, raw_payload)


@mcp.tool()
@require_auth_async
async def create_competitor(
    workspace_id: str,
    competitor_name: str,
    user_email: Optional[str] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """Create a new competitor."""
    config = server.trustai_db_manager.get_workspace_config(workspace_id)
    if not config:
        return {"error": "Workspace configuration not found"}

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{TRUSTAI_BASE_URL}/trustai-api/guardrails/competitors",
            headers=_build_headers(config, user_email=user_email, user_id=user_id, include_content_type=True),
            json={"competitor_name": competitor_name}
        )
        return _wrap_response(response.json())


@mcp.tool()
@require_auth_async
async def get_competitors(
    workspace_id: str,
    user_email: Optional[str] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """Get competitors list."""
    config = server.trustai_db_manager.get_workspace_config(workspace_id)
    if not config:
        return {"error": "Workspace configuration not found"}

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{TRUSTAI_BASE_URL}/trustai-api/guardrails/competitors",
            headers=_build_headers(config, user_email=user_email, user_id=user_id, include_content_type=True)
        )
        raw_payload = _wrap_response(response.json())
        return _restricted_terms_response(workspace_id, raw_payload)


@mcp.tool()
@require_auth_async
async def delete_competitor(
    workspace_id: str,
    competitor_id: int,
    user_email: Optional[str] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """Delete a competitor by ID."""
    config = server.trustai_db_manager.get_workspace_config(workspace_id)
    if not config:
        return {"error": "Workspace configuration not found"}

    async with httpx.AsyncClient() as client:
        response = await client.delete(
            f"{TRUSTAI_BASE_URL}/trustai-api/guardrails/competitors/{competitor_id}",
            headers=_build_headers(config, user_email=user_email, user_id=user_id, include_content_type=True)
        )
        return _wrap_response(response.json())


@mcp.tool()
@require_auth_async
async def batch_update_pii_entities(
    workspace_id: str,
    updates: List[Dict[str, Any]],
    user_email: Optional[str] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """Batch update PII entities."""
    config = server.trustai_db_manager.get_workspace_config(workspace_id)
    if not config:
        return {"error": "Workspace configuration not found"}

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{TRUSTAI_BASE_URL}/trustai-api/guardrails/pii/entities/batch",
            headers=_build_headers(config, user_email=user_email, user_id=user_id, include_content_type=True),
            json={"updates": updates}
        )
        return _wrap_response(response.json())


@mcp.tool()
@require_auth_async
async def get_regex_patterns(
    workspace_id: str,
    user_email: Optional[str] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """Get regex patterns list."""
    config = server.trustai_db_manager.get_workspace_config(workspace_id)
    if not config:
        return {"error": "Workspace configuration not found"}

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{TRUSTAI_BASE_URL}/trustai-api/guardrails/regex-patterns",
            headers=_build_headers(config, user_email=user_email, user_id=user_id)
        )
        raw_payload = _wrap_response(response.json())
        return _custom_patterns_response(workspace_id, raw_payload)


@mcp.tool()
@require_auth_async
async def create_regex_pattern(
    workspace_id: str,
    config_dict: Dict[str, Any],
    user_email: Optional[str] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """Create regex pattern."""
    config = server.trustai_db_manager.get_workspace_config(workspace_id)
    if not config:
        return {"error": "Workspace configuration not found"}

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{TRUSTAI_BASE_URL}/trustai-api/guardrails/regex-patterns",
            headers=_build_headers(config, user_email=user_email, user_id=user_id, include_content_type=True),
            json=config_dict
        )
        return _wrap_response(response.json())


@mcp.tool()
@require_auth_async
async def update_regex_pattern_status(
    workspace_id: str,
    pattern_id: int,
    is_active: bool,
    user_email: Optional[str] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """Update regex pattern status."""
    config = server.trustai_db_manager.get_workspace_config(workspace_id)
    if not config:
        return {"error": "Workspace configuration not found"}

    async with httpx.AsyncClient() as client:
        response = await client.put(
            f"{TRUSTAI_BASE_URL}/trustai-api/guardrails/regex-patterns/{pattern_id}/status",
            headers=_build_headers(config, user_email=user_email, user_id=user_id, include_content_type=True),
            json={"is_active": is_active}
        )
        return _wrap_response(response.json())


@mcp.tool()
@require_auth_async
async def update_regex_pattern(
    workspace_id: str,
    pattern_id: int,
    config_dict: Dict[str, Any],
    user_email: Optional[str] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """Update regex pattern."""
    config = server.trustai_db_manager.get_workspace_config(workspace_id)
    if not config:
        return {"error": "Workspace configuration not found"}

    async with httpx.AsyncClient() as client:
        response = await client.put(
            f"{TRUSTAI_BASE_URL}/trustai-api/guardrails/regex-patterns/{pattern_id}",
            headers=_build_headers(config, user_email=user_email, user_id=user_id, include_content_type=True),
            json=config_dict
        )
        return _wrap_response(response.json())


@mcp.tool()
@require_auth_async
async def delete_regex_pattern(
    workspace_id: str,
    pattern_id: int,
    user_email: Optional[str] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """Delete regex pattern."""
    config = server.trustai_db_manager.get_workspace_config(workspace_id)
    if not config:
        return {"error": "Workspace configuration not found"}

    async with httpx.AsyncClient() as client:
        response = await client.delete(
            f"{TRUSTAI_BASE_URL}/trustai-api/guardrails/regex-patterns/{pattern_id}",
            headers=_build_headers(config, user_email=user_email, user_id=user_id, include_content_type=True)
        )
        return _wrap_response(response.json())


@mcp.tool()
@require_auth_async
async def get_dashboard_overview(
    workspace_id: str,
    user_email: Optional[str] = None,
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
            f"{TRUSTAI_BASE_URL}/trustai-api/dashboard/overview",
            params={"start_date": start_date, "end_date": end_date},
            headers=_build_headers(config, user_email=user_email, user_id=user_id)
        )
        return _wrap_response(response.json())


@mcp.tool()
@require_auth_async
async def list_api_keys(
    workspace_id: str,
    user_email: Optional[str] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """List API keys for app or workspace."""
    config = server.trustai_db_manager.get_workspace_config(workspace_id)
    if not config:
        return {"error": "Workspace configuration not found"}

    _ = user_id
    params: Dict[str, str] = {}
    if user_email:
        params["user_id"] = user_email

    async with httpx.AsyncClient() as client:
        response = await client.get(
            F"{TRUSTAI_BASE_URL}/api/v1/api-keys/",
            params=params,
            headers={
                "accept": "application/json",
                "X-API-KEY": config.x_api_key
            }
        )
        return _wrap_response(response.json())


@mcp.tool()
@require_auth_async
async def get_ai_gateway_system_config(
    workspace_id: str,
    user_email: Optional[str] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """Get AI gateway system configuration for a workspace."""
    config = server.trustai_db_manager.get_workspace_config(workspace_id)
    if not config:
        return {"error": "Workspace configuration not found"}

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{TRUSTAI_BASE_URL}/trustai-api/ai-gateway/system-config",
            headers=_build_headers(config, user_email=user_email, user_id=user_id, include_content_type=True)
        )
        return _wrap_response(response.json())


@mcp.tool()
@require_auth_async
async def update_ai_gateway_system_config(
    workspace_id: str,
    config_dict: Dict[str, Any],
    user_email: Optional[str] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """Update AI gateway system configuration for a workspace."""
    config = server.trustai_db_manager.get_workspace_config(workspace_id)
    if not config:
        return {"error": "Workspace configuration not found"}

    async with httpx.AsyncClient() as client:
        response = await client.put(
            f"{TRUSTAI_BASE_URL}/trustai-api/ai-gateway/system-config",
            headers=_build_headers(config, user_email=user_email, user_id=user_id, include_content_type=True),
            json=config_dict
        )
        return _wrap_response(response.json())


