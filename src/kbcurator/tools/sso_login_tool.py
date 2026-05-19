from os import getenv

from kbcurator.utils.azure_sso import verify_microsoft_token
from kbcurator.utils.auth import (
    _assign_user_to_workspace,
    _create_user_with_workspace,
    _fetch_user_by_email,
    _fetch_user_roles,
    _fetch_user_workspaces,
    _issue_backend_jwt,
    _serialize_user,
    ACCESS_TOKEN_EXPIRY,
)
from kbcurator.server.server import mcp
import logging

logger = logging.getLogger(__name__)


def _dummy_workspace_from_env() -> dict:
    """Build the fallback workspace dict from .env values."""
    raw_id = getenv("DUMMY_WORKSPACE_ID", "1").strip().strip('"').strip("'")
    try:
        ws_id = int(raw_id)
    except ValueError:
        ws_id = 1
    return {
        "workspace_id": ws_id,
        "workspace_name": getenv("DUMMY_WORKSPACE_NAME", "Demo Workspace").strip().strip('"').strip("'"),
        "workspace_desc": getenv("DUMMY_WORKSPACE_DESC", "Demo workspace for new users.").strip().strip('"').strip("'"),
    }


@mcp.tool()
async def sso_login_user(access_token: str, email: str) -> dict:
    """
    Authenticate a user via Microsoft Azure AD SSO.

    - Existing Coforge users with real workspaces: returns their full workspace/role data.
    - Existing Coforge users with only the demo workspace: returns account_status "restricted"
      until an administrator assigns them to a real workspace.
    - New Coforge users (not yet in DB): creates an account, assigns them to
      the configured demo workspace (DUMMY_WORKSPACE_ID), and returns account_status "new".

    Args:
        access_token: Microsoft Azure AD access token received after OAuth redirect.
        email:        User's email — must match the token identity (prevents substitution).

    Returns on success:
        {
            "success": True,
            "token": "<b64url-encoded backend JWT>",
            "token_transport": "b64url",
            "expires_in": <seconds>,
            "account_status": "active" | "restricted",
            "user_details": { user_id, email_id, is_admin, ... },
            "roles": [...],
            "workspaces": [...],
            "message": "..."
        }
        account_status values:
            "restricted" — user's only workspace is the demo workspace (includes first-ever logins);
                           access is limited until an admin assigns a real workspace
            "active"     — user has at least one real (non-demo) workspace assigned
    Returns on failure:
        { "success": False, "error": "<reason>", "code": <http_status> }
    """
    # ── 1. Verify the Microsoft token ────────────────────────────────────────
    try:
        claims = verify_microsoft_token(access_token)
    except PermissionError as exc:
        return {"success": False, "error": str(exc), "code": 403}
    except ValueError as exc:
        return {"success": False, "error": str(exc), "code": 401}
    except Exception as exc:
        logger.error("Unexpected token validation error: %s", exc)
        return {"success": False, "error": "Token validation failed.", "code": 500}

    # ── 2. Extract and validate email from token ──────────────────────────────
    token_email = (
        claims.get("preferred_username")
        or claims.get("upn")
        or claims.get("email")
        or ""
    ).strip().lower()

    if not token_email:
        return {"success": False, "error": "Token has no email/UPN claim.", "code": 401}

    if token_email != email.strip().lower():
        return {
            "success": False,
            "error": "Email does not match token identity. Possible substitution attempt.",
            "code": 403,
        }

    # ── 3. Look up user in database ───────────────────────────────────────────
    try:
        user = _fetch_user_by_email(token_email)
    except Exception as exc:
        logger.error("DB user lookup failed for %s: %s", token_email, exc)
        user = None

    # ── 4a. NEW USER — create account and assign to demo workspace ────────────
    if not user:
        logger.info("SSO login: new user %s — provisioning account", token_email)

        dummy_ws = _dummy_workspace_from_env()
        dummy_workspace_id = dummy_ws["workspace_id"]

        try:
            user = _create_user_with_workspace(token_email, dummy_workspace_id)
        except Exception as exc:
            logger.error("Failed to provision new user %s: %s", token_email, exc)
            return {"success": False, "error": "Failed to create user account.", "code": 500}

        # Fetch roles (empty for brand-new users) and workspaces
        try:
            roles = _fetch_user_roles(user["user_id"])
            workspaces = _fetch_user_workspaces(user["user_id"])
        except Exception as exc:
            logger.warning("Could not fetch roles/workspaces for new user %s: %s", token_email, exc)
            roles = []
            workspaces = []

        # Guarantee at least the demo workspace appears in the response
        if not workspaces:
            workspaces = [dummy_ws]

        try:
            backend_token = _issue_backend_jwt(user)
        except Exception as exc:
            logger.error("JWT issuance failed for new user %s: %s", token_email, exc)
            return {"success": False, "error": "Failed to issue session token.", "code": 500}

        logger.info("SSO login: new user provisioned user_id=%s email=%s", user["user_id"], token_email)
        return {
            "success": True,
            "account_status": "restricted",
            "token": backend_token,
            "token_transport": "b64url",
            "expires_in": ACCESS_TOKEN_EXPIRY,
            "user_details": _serialize_user(user),
            "roles": roles,
            "workspaces": workspaces,
            "message": "Account created. You have been added to the demo workspace with full tool access.",
        }

    # ── 4b. EXISTING USER — fetch their data and issue token ─────────────────
    user_id = user["user_id"]

    try:
        roles = _fetch_user_roles(user_id)
        workspaces = _fetch_user_workspaces(user_id)
    except Exception as exc:
        logger.error("DB role/workspace lookup failed for user %s: %s", user_id, exc)
        return {"success": False, "error": "Database error fetching roles/workspaces.", "code": 500}

    # Existing user with no workspace — assign to dummy workspace so they
    # can access the platform the same way a new user would.
    dummy_ws = _dummy_workspace_from_env()
    dummy_workspace_id = dummy_ws["workspace_id"]

    if not workspaces:
        try:
            _assign_user_to_workspace(user_id, dummy_workspace_id)
            workspaces = _fetch_user_workspaces(user_id)
        except Exception as exc:
            logger.warning("Could not assign existing user %s to dummy workspace: %s", user_id, exc)
        if not workspaces:
            workspaces = [dummy_ws]
        logger.info("Existing user %s had no workspaces — assigned to dummy workspace %s", user_id, dummy_workspace_id)

    # A user is still "restricted" if the only workspace they belong to is the
    # demo workspace — meaning no admin has assigned them to a real workspace yet.
    real_workspaces = [ws for ws in workspaces if ws.get("workspace_id") != dummy_workspace_id]
    is_restricted = len(real_workspaces) == 0

    try:
        backend_token = _issue_backend_jwt(user)
    except Exception as exc:
        logger.error("JWT issuance failed for user %s: %s", user_id, exc)
        return {"success": False, "error": "Failed to issue session token.", "code": 500}

    account_status = "restricted" if is_restricted else "active"
    message = (
        "Your account is pending workspace assignment by an administrator."
        if is_restricted
        else "Logged in"
    )

    logger.info("SSO login success user_id=%s email=%s account_status=%s", user_id, token_email, account_status)
    return {
        "success": True,
        "account_status": account_status,
        "token": backend_token,
        "token_transport": "b64url",
        "expires_in": ACCESS_TOKEN_EXPIRY,
        "user_details": _serialize_user(user),
        "roles": roles,
        "workspaces": workspaces,
        "message": message,
    }
