from os import getenv

from kbcurator.server.server import mcp
from kbcurator.utils.auth import _fetch_user_by_id, _fetch_user_workspaces
import logging

logger = logging.getLogger(__name__)


def _get_dummy_workspace_id() -> int:
    raw_id = getenv("DUMMY_WORKSPACE_ID", "1").strip().strip('"').strip("'")
    try:
        return int(raw_id)
    except ValueError:
        return 1


@mcp.tool()
async def check_account_status(user_id: int) -> dict:
    """
    Check the account status of a user at any point after SSO authentication.

    Useful for verifying whether a user has been assigned a real workspace
    by an admin, even after the initial SSO login has already occurred.

    Args:
        user_id: The user's ID to check.

    Returns on success:
        {
            "success": True,
            "account_status": "active" | "restricted"
        }
        account_status values:
            "active"     — user has at least one real (non-demo) workspace assigned
            "restricted" — user's only workspace is the demo workspace;
                           access is limited until an admin assigns a real workspace
    Returns on failure:
        { "success": False, "error": "<reason>", "code": <http_status> }
    """
    try:
        user = _fetch_user_by_id(user_id)
    except Exception as exc:
        logger.error("DB lookup failed for user_id=%s: %s", user_id, exc)
        return {"success": False, "error": "Database error.", "code": 500}

    if not user:
        return {"success": False, "error": "User not found.", "code": 404}

    try:
        workspaces = _fetch_user_workspaces(user_id)
    except Exception as exc:
        logger.error("Workspace lookup failed for user_id=%s: %s", user_id, exc)
        return {"success": False, "error": "Database error fetching workspaces.", "code": 500}

    dummy_workspace_id = _get_dummy_workspace_id()
    real_workspaces = [ws for ws in workspaces if ws.get("workspace_id") != dummy_workspace_id]
    account_status = "active" if real_workspaces else "restricted"

    logger.info("check_account_status user_id=%s account_status=%s", user_id, account_status)
    return {"success": True, "response": {"account_status": account_status}}
