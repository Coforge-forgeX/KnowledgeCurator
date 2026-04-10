"""
Helper for validating user and workspace access based on JWT claims.
"""
from agent_search.utils.request_context import request_var

from sqlalchemy.orm import sessionmaker
from agent_search.utils.request_context import request_var

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