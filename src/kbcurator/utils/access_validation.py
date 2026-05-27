"""
Helper for validating user and workspace access based on JWT claims.
"""
from threading import RLock
from time import monotonic

from .db import db
from .request_context import request_var

_WORKSPACE_SCOPE_CACHE_TTL_SECONDS = 900
_workspace_scope_cache = {}
_workspace_scope_cache_lock = RLock()


def _normalize_text(value):
    if value is None:
        return ""
    return str(value).strip().lower()


def _normalize_kb_input(knowledge_bases):
    if knowledge_bases is None:
        return []
    if not isinstance(knowledge_bases, list):
        return None
    normalized = []
    for item in knowledge_bases:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def _load_workspace_scope_snapshot(workspace_id):
    session = db.Session()
    try:
        rows = (
            session.query(
                db.WorkspaceIndustrySubIndustryMap.industry_id,
                db.WorkspaceIndustrySubIndustryMap.subindustry_id,
                db.WorkspaceIndustrySubIndustryMap.kb_id,
                db.Industry.industry_name,
                db.SubIndustry.subindustry_name,
                db.KnowledgeBase.title,
            )
            .outerjoin(
                db.Industry,
                (db.Industry.industry_id == db.WorkspaceIndustrySubIndustryMap.industry_id)
                & (db.Industry.is_active == True),
            )
            .outerjoin(
                db.SubIndustry,
                (db.SubIndustry.subindustry_id == db.WorkspaceIndustrySubIndustryMap.subindustry_id)
                & (db.SubIndustry.is_active == True),
            )
            .outerjoin(
                db.KnowledgeBase,
                (db.KnowledgeBase.id == db.WorkspaceIndustrySubIndustryMap.kb_id)
                & (db.KnowledgeBase.is_active == True),
            )
            .filter(
                db.WorkspaceIndustrySubIndustryMap.workspace_id == workspace_id,
                db.WorkspaceIndustrySubIndustryMap.is_active == True,
            )
            .all()
        )

        if not rows:
            return {
                "industry_name": None,
                "subindustry_name": None,
                "knowledge_bases": set(),
                "knowledge_base_ids": set(),
            }

        industry_name = next((row[3] for row in rows if row[3]), None)
        subindustry_name = next((row[4] for row in rows if row[4]), None)

        kb_ids = set()
        kb_titles = set()
        for row in rows:
            kb_id = row[2]
            if kb_id is not None:
                try:
                    kb_ids.add(int(kb_id))
                except (TypeError, ValueError):
                    continue

            kb_title = row[5]
            if kb_title:
                kb_titles.add(_normalize_text(kb_title))

        return {
            "industry_name": _normalize_text(industry_name),
            "subindustry_name": _normalize_text(subindustry_name),
            "knowledge_bases": kb_titles,
            "knowledge_base_ids": kb_ids,
        }
    finally:
        session.close()


def _get_workspace_scope_snapshot(workspace_id):
    now = monotonic()
    cache_key = str(workspace_id)

    with _workspace_scope_cache_lock:
        cached = _workspace_scope_cache.get(cache_key)
        if cached and cached.get("expires_at", 0) > now:
            return cached.get("value")

    snapshot = _load_workspace_scope_snapshot(workspace_id)
    with _workspace_scope_cache_lock:
        _workspace_scope_cache[cache_key] = {
            "value": snapshot,
            "expires_at": now + _WORKSPACE_SCOPE_CACHE_TTL_SECONDS,
        }
    return snapshot


def validate_chatbot_request_scope(user_id, workspace_id, role_id, industry, sub_industry, knowledge_bases):
    """
    Validate that chatbot payload values match the caller's workspace mapping.

    Checks:
    - user_id is actively mapped to workspace_id
    - role_id matches mapped role_id
    - industry/sub_industry match workspace mapping (when mapping exists)
    - provided knowledge_bases are within mapped workspace KBs (if provided)
    """
    if user_id is None:
        return False, "user_id cannot be null"
    if not workspace_id:
        return False, "workspace_id is required for authentication."

    try:
        requested_role_id = int(role_id)
    except (TypeError, ValueError):
        return False, "Invalid role_id."

    normalized_kbs = _normalize_kb_input(knowledge_bases)
    if normalized_kbs is None:
        return False, "knowledge_bases must be a list when provided."

    session = db.Session()
    try:
        user_map = (
            session.query(db.UserMap)
            .filter_by(workspace_id=workspace_id, user_id=user_id, is_active=True)
            .first()
        )
        if not user_map:
            return False, "You are not authorized to access this workspace."

        mapped_role_id = getattr(user_map, "role_id", None)
        try:
            mapped_role_id_int = int(mapped_role_id)
        except (TypeError, ValueError):
            return False, "Invalid role mapping for the user in this workspace."

        if requested_role_id != mapped_role_id_int:
            return False, "Provided role_id is not valid for this user in the selected workspace."
    except Exception as e:
        return False, str(e)
    finally:
        session.close()

    scope_snapshot = _get_workspace_scope_snapshot(workspace_id)
    if scope_snapshot:
        expected_industry = scope_snapshot.get("industry_name")
        expected_subindustry = scope_snapshot.get("subindustry_name")

        if expected_industry and _normalize_text(industry) != expected_industry:
            return False, "Invalid industry for this workspace."

        if expected_subindustry and _normalize_text(sub_industry) != expected_subindustry:
            return False, "Invalid sub_industry for this workspace."

        if normalized_kbs:
            allowed_kb_titles = scope_snapshot.get("knowledge_bases", set())
            allowed_kb_ids = scope_snapshot.get("knowledge_base_ids", set())

            for kb in normalized_kbs:
                kb_norm = _normalize_text(kb)
                if kb_norm in allowed_kb_titles:
                    continue

                if kb.strip().isdigit() and int(kb.strip()) in allowed_kb_ids:
                    continue

                return False, "One or more knowledge_bases are invalid for this workspace."

    return True, None

def validate_user_workspace_access(user_id=None, workspace_id=None):
    """
    Validates that the user_id and/or workspace_id in the request payload matches the JWT claims.
    Also, if both user_id and workspace_id are provided, checks that the user is mapped to the workspace and is active.
    Args:
        user_id (int or str, optional): The user ID from the payload.
        workspace_id (int or str, optional): The workspace ID from the payload.
    Returns:
        (bool, str): (True, None) if valid, (False, error_message) if not.
    """
    request = request_var.get(None)
    if not request or not hasattr(request.state, "jwt_claims"):
        return False, "Unauthorized: JWT claims not found in request context"
    claims = request.state.jwt_claims
    jwt_user_id = claims.get("user_id") or claims.get("sub")
    jwt_workspace_id = claims.get("workspace_id") if "workspace_id" in claims else None

    # Validate user_id
    if user_id is not None and str(user_id) != str(jwt_user_id):
        return False, "The user_id in the request is not authorized. Only the authenticated user's data can be accessed."
    # Validate workspace_id if present in claims
    if workspace_id is not None and jwt_workspace_id is not None and str(workspace_id) != str(jwt_workspace_id):
        return False, "The workspace_id in the request is not authorized. Only the authenticated user's workspace can be accessed."

    # If all checks pass, return True, None
    return True, None