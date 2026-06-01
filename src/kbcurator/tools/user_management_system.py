from kbcurator.utils.access_validation import validate_user_workspace_access
from kbcurator.utils.permission import is_admin, get_user_role_id
from kbcurator.server.server import mcp
import psycopg2
from configparser import ConfigParser
from sqlalchemy import func
from kbcurator.utils.db import db
from os import getenv
import sys
from kbcurator.utils.auth import create_jwt_token, verify_jwt_token, create_refresh_token, verify_refresh_token
from kbcurator.utils.request_context import request_var
from sqlalchemy import select, func as sql_func
from datetime import datetime, timezone

# --- New Import for Password Hashing ---
from passlib.hash import argon2
from kbcurator.utils.auth import (
    JWT_TRANSPORT_ENCODE,
    JWT_RETURN_RAW_ACCESS,
    JWT_SET_ACCESS_COOKIE,
    extract_token_from_headers,
    _assign_user_to_workspace,
    revoke_token, 
    _fetch_user_by_email,
    encode_for_transport,
    require_auth,
    require_auth_async,
    get_current_user
)

from datetime import datetime, timezone
from kbcurator.utils.constants import DefaultValue, Role, WorkspaceType
from kbcurator.services.agent_llm_configuration_service import agent_llm_config_service
from kbcurator.services.workspace_provider_credentials_service import workspace_provider_credentials_service


@mcp.tool()
@require_auth_async
async def get_workspace_types_by_role():
    """
    Return workspace types allowed for the current platform role.

    Rules:
    - Platform Admin can create all workspace types, including KG.
    - Non-admin users can create only Demo, Trial, and Product workspace types.
    """
    claims, _ = get_current_user()
    jwt_role_id = claims.get("role_id")
    if jwt_role_id is None:
        return {"error": "Unauthorized: role_id not found in token claims"}

    try:
        normalized_role_id = int(jwt_role_id)
    except (TypeError, ValueError):
        return {"error": "Unauthorized: invalid role_id in token claims"}

    is_admin = normalized_role_id == Role.ADMIN.id

    if is_admin:
        allowed = [
            WorkspaceType.KG,
            WorkspaceType.DM,
            WorkspaceType.TR,
            WorkspaceType.PR,
        ]
    else:
        allowed = [WorkspaceType.DM, WorkspaceType.TR, WorkspaceType.PR]

    return {
        "is_admin": is_admin,
        "role_id": normalized_role_id,
        "workspace_types": [
            {"code": workspace_type.name, "name": workspace_type.value}
            for workspace_type in allowed
        ],
    }



@mcp.tool()
@require_auth
def fetch_user_workflow_stage(user_id: int, workspace_id: int):
    """
    Fetch workflow stage for a user in a workspace based on active role mapping.
    Args:
        user_id (int): User ID
        workspace_id (int): Workspace ID
    Returns:
        dict: { 'workflow_stage': 'ALL' or stage name, 'role_id': role_id }
    """
    _, jwt_user_id = get_current_user()
    if str(user_id) != str(jwt_user_id):
        return {"error": "You are not authorized to fetch this user's workflow stage."}

    session = db.Session()
    try:
        session.rollback()
        # Fetch active role mapping for user in workspace
        role_map = session.query(db.UserMap).filter(
                db.UserMap.user_id == user_id,
                db.UserMap.workspace_id == workspace_id,
                db.UserMap.is_active == True
        ).first()
        if not role_map:
            return {"workflow_stage": "ALL", "role_id": None, "message": "No active role mapping found."}

        role_id = getattr(role_map, "role_id", None)
        if not role_id:
            return {"workflow_stage": "ALL", "role_id": None, "message": "Role ID not found in mapping."}

        # Fetch workflow_stage from role_master table
        role = session.query(db.Role).filter(db.Role.role_id == role_id, db.Role.is_active == True).first()
        workflow_stage = getattr(role, "workflow_stage", None)
        if not workflow_stage:
            return {"workflow_stage": "ALL", "role_id": role_id, "message": "No workflow_stage set for role."}
        else:
            return {"workflow_stage": workflow_stage, "role_id": role_id}
    except Exception as e:
        session.rollback()
        return {"error": f"Failed to fetch workflow stage: {str(e)}"}
    finally:
        session.close()


@mcp.tool()
@require_auth
def update_user_kb_toggle(user_id: int, workspace_id: int, can_curate_kb: bool):
    """
    Update the can_curate_kb column for a user in a workspace (workspace_users_mapping).
    Args:
        user_id (int): User ID to update.
        workspace_id (int): Workspace ID to update mapping for.
        can_curate_kb (bool): True to enable, False to disable.
    Returns:
        dict: Success or error message.
    """

    _, jwt_user_id = get_current_user()
    session = db.Session()
    try:
        session.rollback()
  
        user_role_id = get_user_role_id(jwt_user_id, workspace_id)
        if user_role_id != Role.WS_ADMIN.id and user_role_id != Role.WS_MANAGER.id:
            return {"error": "Only Workspace Admin or Manager can update can_curate_kb for users in this workspace."}

        user_map = session.query(db.UserMap).filter(
            db.UserMap.user_id == user_id,
            db.UserMap.workspace_id == workspace_id,
            db.UserMap.is_active == True
        ).first()
        if not user_map:
            return {"error": "User mapping not found or inactive for this workspace."}
        setattr(user_map, "can_curate_kb", can_curate_kb)
        session.commit()
        return {"success": True, "message": f"can_curate_kb updated to {can_curate_kb} for user_id {user_id} in workspace_id {workspace_id}"}
    except Exception as e:
        session.rollback()
        return {"success": False, "error": f"Failed to update can_curate_kb: {str(e)}"}
    finally:
        session.close()


@mcp.tool()
def login_user(email: str, password: str):
    """
    Authenticate user and issue JWT access token + refresh token (cookie).
    Returns:
        dict:
            success: bool
            token: str (Base64URL-encoded JWT by default; see JWT_TRANSPORT_ENCODE)
            token_transport: "b64url" | "raw"  (so FE knows how to handle)
            expires_in: int
            user_details: {...}
            roles: [...]
            workspaces: [...]
            message: str
    """
    session = db.Session()
    try:
        session.rollback()
        normalized_email = email.strip().lower()
        user = session.query(db.User).filter(func.lower(db.User.email_id) == normalized_email).first()

        if user and hasattr(user, 'password'):
            user_db_password = getattr(user, 'password', None)
            password_matches = False

            if user_db_password is None:
                if password == DefaultValue.PASSWORD.value:
                    password_matches = True
            elif user_db_password.startswith('$argon2'):
                try:
                    password_matches = argon2.verify(password, user_db_password)
                except Exception as verify_e:
                    print(f"Argon2 verification failed: {verify_e}")
                    password_matches = False
            else:
                if password == user_db_password:
                    password_matches = True

            if password_matches:
                # Build safe user_details (exclude sensitive fields)
                all_cols = user.__table__.columns.keys()
                user_details = {col: getattr(user, col) for col in all_cols}
                user_details.pop('password', None)
                user_details.pop('salt', None)

                # Single-query fetch for user mappings, roles, workspaces (active only)
                # NOTE: UserRoleMap is DEPRECATED - UserMap contains role_id
                user_workspace_data = (
                    session.query(db.UserMap, db.Workspace)
                    .join(db.Workspace, 
                        (db.UserMap.workspace_id == db.Workspace.workspace_id) &
                        (db.UserMap.is_active == True)
                    )
                    # DEPRECATED: UserRoleMap table is redundant
                    # .outerjoin(
                    #     db.UserRoleMap,
                    #     (db.UserRoleMap.user_id == db.UserMap.user_id) &
                    #     (db.UserRoleMap.workspace_id == db.UserMap.workspace_id) &
                    #     (db.UserRoleMap.is_active == True)
                    # )
                    .filter(db.UserMap.user_id == user.user_id)
                    .filter(db.Workspace.is_active == True)
                    .all()
                )

                user_roles = []
                workspaces = []
                workspace_ids_with_roles = set()
                seen_workspaces = set()

                for user_map, workspace in user_workspace_data:
                    workspace_id = workspace.workspace_id

                    if workspace_id not in seen_workspaces:
                        seen_workspaces.add(workspace_id)
                        workspaces.append({
                            'workspace_id': workspace.workspace_id,
                            'workspace_name': workspace.workspace_name,
                            'workspace_desc': workspace.workspace_desc
                        })

                    role_id = None
                    workflow_stage = "All"
                    # Get role_id from UserMap (not UserRoleMap - it's deprecated)
                    if user_map and hasattr(user_map, 'role_id'):
                        role_id = user_map.role_id
                        if role_id is not None:
                            role_entry = session.query(db.Role).filter(
                                db.Role.role_id == role_id,
                                db.Role.is_active == True
                            ).first()
                            if role_entry and role_entry.workflow_stage:
                                workflow_stage = role_entry.workflow_stage
                        user_roles.append({
                            'workspace_id': workspace_id,
                            'role_id': role_id,
                            'workflow_stage': workflow_stage
                        })
                        workspace_ids_with_roles.add(workspace_id)

                # Admin: ensure "All" stage for workspaces without explicit role
                if getattr(user, 'role_id', Role.USER.id) == Role.ADMIN.id:
                    for ws in workspaces:
                        if ws['workspace_id'] not in workspace_ids_with_roles:
                            user_roles.append({
                                'workspace_id': ws['workspace_id'],
                                'role_id': None,
                                'workflow_stage': "All"
                            })

                role_id = getattr(user, 'role_id', None)
                # Prepare JWT claims
                claims = {
                    'sub': getattr(user, 'user_id', None),
                    'user_id': getattr(user, 'user_id', None),
                    'email': getattr(user, 'email_id', None),
                    'name': getattr(user, 'user_name', None) or getattr(user, 'name', None) or getattr(user, 'first_name', None),
                    'is_admin': True if (role_id == Role.ADMIN.id) else False,
                    'role_id': role_id,
                    'roles': user_roles,
                }

                # Create tokens
                access_token, access_ttl = create_jwt_token(claims)
                refresh_token, refresh_ttl = create_refresh_token(user.user_id)

                # Ask the middleware to set cookies (refresh already supported).
                # We also set access cookie if enabled.
                request = request_var.get(None)
                if request:
                    # Always set refresh token cookie via middleware
                    request.state.refresh_token = refresh_token
                    request.state.refresh_token_expires = refresh_ttl

                    # Optionally set access token cookie via middleware
                    if JWT_SET_ACCESS_COOKIE:
                        request.state.access_token = access_token
                        request.state.access_token_expires = access_ttl

                # Encode token for transport if configured
                if JWT_TRANSPORT_ENCODE and not JWT_RETURN_RAW_ACCESS:
                    token_out = encode_for_transport(access_token)
                    token_transport = "b64url"
                elif JWT_RETURN_RAW_ACCESS:
                    token_out = access_token
                    token_transport = "raw"
                else:
                    # Not returning raw token explicitly for safety, but honoring config
                    token_out = encode_for_transport(access_token)
                    token_transport = "b64url"

                return {
                    'success': True,
                    'token': token_out,
                    'token_transport': token_transport,  # so FE knows how to handle
                    'expires_in': access_ttl,
                    'user_details': user_details,
                    'roles': user_roles,
                    'workspaces': workspaces,
                    'message': 'Logged in'
                }

        # Invalid user or password
        return {
            'success': False,
            'message': 'Invalid credentials'
        }

    except Exception as e:
        session.rollback()
        print(f"Error during login: {e}")
        return {
            'success': False,
            'message': 'An error occurred during login. Please try again later.'
        }
    finally:
        session.close()

@mcp.tool()
def refresh_jwt_token(refresh_token: str = None):
    """
    Issue a new access token using a valid refresh token.
    Args:
        refresh_token (str, optional): The refresh token. If not provided, will attempt to read from request cookies.
    Returns:
        dict: {
            'success': True,
            'token': str,  # New access token
            'expires_in': int,  # New access token expiry
            'message': 'Token refreshed'
        }
        or { 'success': False, 'message': 'Invalid or expired refresh token' }
    """
    # Try to get refresh token from cookies if not provided
    if not refresh_token:
        request = request_var.get(None)
        if request:
            refresh_token = request.cookies.get('refresh_token')
    
    if not refresh_token:
        return {'success': False, 'message': 'Refresh token not provided'}
    
    session = db.Session()
    try:
        # Verify the refresh token
        payload = verify_refresh_token(refresh_token)
        user_id = payload.get('user_id')
        
        if not user_id:
            return {'success': False, 'message': 'Invalid refresh token: user_id missing'}
        
        # Fetch user data to rebuild access token claims
        user = session.query(db.User).filter(db.User.user_id == user_id, db.User.is_active == True).first()
        if not user:
            return {'success': False, 'message': 'User not found or inactive'}
        
        # Rebuild user roles and workspaces (same as login)
        # NOTE: UserRoleMap is DEPRECATED - UserMap contains role_id
        user_roles = []
        workspaces = []
        
        user_workspace_data = (
            session.query(db.UserMap, db.Workspace)
            .join(db.Workspace, db.UserMap.workspace_id == db.Workspace.workspace_id)
            .filter(db.UserMap.user_id == user.user_id)
            .filter(db.UserMap.is_active == True)
            .filter(db.Workspace.is_active == True)
            .all()
        )
        
        seen_workspaces = set()
        for user_map, workspace in user_workspace_data:
            workspace_id = workspace.workspace_id
            
            if workspace_id not in seen_workspaces:
                seen_workspaces.add(workspace_id)
                workspaces.append({
                    'workspace_id': workspace.workspace_id,
                    'workspace_name': workspace.workspace_name,
                    'workspace_desc': workspace.workspace_desc
                })
            
            # Get role_id from UserMap (not UserRoleMap - it's deprecated)
            if user_map and hasattr(user_map, 'role_id'):
                role_id = user_map.role_id
                if role_id is not None:
                    user_roles.append({
                        'workspace_id': workspace_id,
                        'role_id': role_id
                    })
        
        role_id = getattr(user, 'role_id', None)
        # Create new access token with fresh claims
        claims = {
            'sub': user.user_id,
            'user_id': user.user_id,
            'email': getattr(user, 'email_id', None),
            'name': getattr(user, 'user_name', None) or getattr(user, 'name', None) or getattr(user, 'first_name', None),
            'is_admin': True if (role_id == Role.ADMIN.id) else False,
            'role_id': role_id,
            'roles': user_roles,
        }
        
        access_token, access_ttl = create_jwt_token(claims)
        
        return {
            'success': True,
            'token': access_token,
            'expires_in': access_ttl,
            'message': 'Token refreshed'
        }
        
    except Exception as e:
        return {'success': False, 'message': f'Invalid or expired refresh token: {str(e)}'}
    finally:
        session.close()
            

@mcp.tool()
def fetch_knowledge_base(
    industry_id,
    sub_industry_id,
    workspace_id=None
    ):
    """
    Fetch knowledge bases mapped to a given industry_id and sub_industry_id.
    Args:
        industry_id (int): The industry ID to filter knowledge bases.
        sub_industry_id (int): The subindustry ID to filter knowledge bases.
        workspace_id (optional): The workspace ID to further filter knowledge bases.
    Returns:
        dict: List of knowledge bases with knowledge_id and knowledge_name.
    """
    session = db.Session()
    try:
        session.rollback()
        if workspace_id:
            kb_query = session.query(db.KnowledgeBase).filter(
                db.KnowledgeBase.industry_id == industry_id,
                db.KnowledgeBase.sub_industry_id == sub_industry_id,
                # db.KnowledgeBase.workspace_id == workspace_id,
                db.KnowledgeBase.is_active == True
            )
        else:
            kb_query = session.query(db.KnowledgeBase).filter(
                db.KnowledgeBase.industry_id == industry_id,
                db.KnowledgeBase.sub_industry_id == sub_industry_id,
                db.KnowledgeBase.is_active == True
            )
        kb_list = [
            {
                'id': getattr(kb, 'id', None),
                'title': getattr(kb, 'title', None),
                'description': getattr(kb, 'description', None)
            }
            for kb in kb_query.all()
        ]
        return {'response': kb_list}
    except Exception as e:
        session.rollback()
        print(f"Error in fetch_knowledge_base: {e}")
        return {'error': 'An error occurred while fetching knowledge base.'}
    finally:
        session.close()

# @mcp.tool()
# def fetch_workspaces_list(user_id):
#         """
#         Returns a summary of all workspaces for the authenticated user, including workspace_id, workspace_name, workspace_desc,
#         and counts of agents, tools, and users in each workspace.
#         Args:
#             user_id (int or str): The user ID to fetch workspaces for (must match JWT, otherwise ignored).
#         Returns:
#             dict: { 'response': [ { 'workspace_id', 'workspace_name', 'workspace_desc', 'agent_count', 'tool_count', 'user_count' }, ... ] }
#         """
#         if user_id is None:
#             return {"status": "error", "error": "user_id cannot be null"}
#         session = db.Session()
#         try:
#             session.rollback()
#             # Use JWT claims directly for authentication (faster, as in login_user)
#             request = request_var.get(None)
#             if not request or not hasattr(request.state, "jwt_claims"):
#                 return {"error": "Unauthorized: JWT claims not found in request context"}
#             claims = request.state.jwt_claims
#             jwt_user_id = claims.get("user_id") or claims.get("sub")
#             if not jwt_user_id:
#                 return {"error": "Unauthorized: user_id not found in token claims"}
#             # If user_id is provided and does not match JWT, return error
#             if user_id is not None and str(user_id) != str(jwt_user_id):
#                 return {"error": "The user_id in the request is not authorized. Only the authenticated user's workspaces can be accessed."}

#             # OPTIMIZED: Single query with subqueries for counts

#             agent_count_subq = (
#                 select(
#                     AgentMap.workspace_id,
#                     sql_func.count(AgentMap.agent_id).label('agent_count')
#                 )
#                 .where(AgentMap.is_active == True)
#                 .group_by(AgentMap.workspace_id)
#                 .subquery()
#             )
#             tool_count_subq = (
#                 select(
#                     ToolMap.workspace_id,
#                     sql_func.count(ToolMap.tool_id).label('tool_count')
#                 )
#                 .where(ToolMap.is_active == True)
#                 .group_by(ToolMap.workspace_id)
#                 .subquery()
#             )
#             user_count_subq = (
#                 select(
#                     UserMap.workspace_id,
#                     sql_func.count(UserMap.user_id).label('user_count')
#                 )
#                 .where(UserMap.is_active == True)
#                 .group_by(UserMap.workspace_id)
#                 .subquery()
#             )
#             workspaces_with_counts = (
#                 session.query(
#                     Workspace.workspace_id,
#                     Workspace.workspace_name,
#                     Workspace.workspace_desc,
#                     sql_func.coalesce(agent_count_subq.c.agent_count, 0).label('agent_count'),
#                     sql_func.coalesce(tool_count_subq.c.tool_count, 0).label('tool_count'),
#                     sql_func.coalesce(user_count_subq.c.user_count, 0).label('user_count')
#                 )
#                 .join(UserMap, UserMap.workspace_id == Workspace.workspace_id)
#                 .outerjoin(agent_count_subq, agent_count_subq.c.workspace_id == Workspace.workspace_id)
#                 .outerjoin(tool_count_subq, tool_count_subq.c.workspace_id == Workspace.workspace_id)
#                 .outerjoin(user_count_subq, user_count_subq.c.workspace_id == Workspace.workspace_id)
#                 .filter(UserMap.user_id == jwt_user_id, UserMap.is_active == True, Workspace.is_active == True)
#                 .all()
#             )
#             results = [
#                 {
#                     'workspace_id': ws.workspace_id,
#                     'workspace_name': ws.workspace_name,
#                     'workspace_desc': ws.workspace_desc,
#                     'agent_count': ws.agent_count,
#                     'tool_count': ws.tool_count,
#                     'user_count': ws.user_count
#                 }
#                 for ws in workspaces_with_counts
#             ]
#             return {'response': results}
#         except Exception as e:
#             session.rollback()
#             print(f"Fetch workspaces failed with error: {e}")
#             return {'error': 'An error occurred while fetching workspaces.'}
#         finally:
#             session.close()
@mcp.tool()
@require_auth
def fetch_workspaces_list(user_id):
    """
    Returns a summary of all workspaces for the authenticated user, including workspace_id, workspace_name, workspace_desc,
    and counts of agents, tools, and users in each workspace.

    Args:
        user_id (int or str): The user ID to fetch workspaces for (must match JWT, otherwise ignored).

    Returns:
        dict: { 'response': [ { 'workspace_id', 'workspace_name', 'workspace_desc', 'agent_count', 'tool_count', 'user_count' }, ... ] }
    """
    if user_id is None:
        return {"status": "error", "error": "user_id cannot be null"}
    session = db.Session()
    try:
        session.rollback()

        # Use get_current_user() for authentication
        claims, jwt_user_id = get_current_user()

        # --- DUMMY USER HANDLING ---
        # If the user is a dummy (not in DB), return dummy workspace from .env
        # Try to fetch user by email if available in claims
        email = claims.get("email") or claims.get("preferred_username") or claims.get("upn")
        user = None
        if email:
            try:
                user = _fetch_user_by_email(email)
            except Exception:
                user = None
        if not user:
            # For dummy users, ignore user_id mismatch and return dummy workspace
            dummy_workspace_id = getenv("DUMMY_WORKSPACE_ID", "1")
            dummy_workspace_name = getenv("DUMMY_WORKSPACE_NAME", "Demo Workspace")
            dummy_workspace_desc = getenv("DUMMY_WORKSPACE_DESC", "This is a demo workspace for new users.")
            try:
                dummy_workspace_id = int(dummy_workspace_id)
            except Exception:
                dummy_workspace_id = 1
            dummy_workspace = {
                'workspace_id': dummy_workspace_id,
                'workspace_name': dummy_workspace_name,
                'workspace_desc': dummy_workspace_desc,
                'agent_count': 0,
                'tool_count': 0,
                'user_count': 1
            }
            return {'response': [dummy_workspace]}

        # --- NORMAL USER HANDLING ---
        # OPTIMIZED: Single query with subqueries for counts
        agent_count_subq = (
            select(
                db.AgentMap.workspace_id,
                sql_func.count(db.AgentMap.agent_id).label('agent_count')
            )
            .where(db.AgentMap.is_active == True)
            .group_by(db.AgentMap.workspace_id)
            .subquery()
        )
        tool_count_subq = (
            select(
                db.ToolMap.workspace_id,
                sql_func.count(db.ToolMap.tool_id).label('tool_count')
            )
            .where(db.ToolMap.is_active == True)
            .group_by(db.ToolMap.workspace_id)
            .subquery()
        )
        user_count_subq = (
            select(
                db.UserMap.workspace_id,
                sql_func.count(db.UserMap.user_id).label('user_count')
            )
            .where(db.UserMap.is_active == True)
            .group_by(db.UserMap.workspace_id)
            .subquery()
        )

        from sqlalchemy import desc
        from sqlalchemy import case

        base_query = session.query(
            db.Workspace.workspace_id,
            db.Workspace.workspace_name,
            db.Workspace.workspace_desc,
            sql_func.coalesce(agent_count_subq.c.agent_count, 0).label('agent_count'),
            sql_func.coalesce(tool_count_subq.c.tool_count, 0).label('tool_count'),
            sql_func.coalesce(user_count_subq.c.user_count, 0).label('user_count')
        ).join(db.UserMap, db.UserMap.workspace_id == db.Workspace.workspace_id)
        base_query = base_query.outerjoin(agent_count_subq, agent_count_subq.c.workspace_id == db.Workspace.workspace_id)
        base_query = base_query.outerjoin(tool_count_subq, tool_count_subq.c.workspace_id == db.Workspace.workspace_id)
        base_query = base_query.outerjoin(user_count_subq, user_count_subq.c.workspace_id == db.Workspace.workspace_id)
        base_query = base_query.filter(
            db.UserMap.user_id == jwt_user_id,
            db.UserMap.is_active == True,
            db.Workspace.is_active == True
        )
        
        workspaces_with_counts = base_query.order_by(
            case((db.Workspace.last_updated == None, 1), else_=0),
            desc(db.Workspace.last_updated)
        ).all()

        results = [
            {
                'workspace_id': ws.workspace_id,
                'workspace_name': ws.workspace_name,
                'workspace_desc': ws.workspace_desc,
                'agent_count': ws.agent_count,
                'tool_count': ws.tool_count,
                'user_count': ws.user_count
            }
            for ws in workspaces_with_counts
        ]

        # User exists in DB but has no active workspace — assign to dummy workspace
        # so the frontend always has something to render.
        if not results:
            raw_id = getenv("DUMMY_WORKSPACE_ID", "1").strip().strip('"').strip("'")
            try:
                dummy_ws_id = int(raw_id)
            except ValueError:
                dummy_ws_id = 1
            dummy_ws_name = getenv("DUMMY_WORKSPACE_NAME", "Demo Workspace").strip().strip('"').strip("'")
            dummy_ws_desc = getenv("DUMMY_WORKSPACE_DESC", "Demo workspace for new users.").strip().strip('"').strip("'")
            try:
                _assign_user_to_workspace(int(jwt_user_id), dummy_ws_id)
            except Exception as assign_exc:
                print(f"Could not assign user {jwt_user_id} to dummy workspace: {assign_exc}")
            results = [{
                'workspace_id': dummy_ws_id,
                'workspace_name': dummy_ws_name,
                'workspace_desc': dummy_ws_desc,
                'agent_count': 0,
                'tool_count': 0,
                'user_count': 1,
            }]

        return {'response': results}
    except Exception as e:
        session.rollback()
        print(f"Fetch workspaces failed with error: {e}")
        return {'error': 'An error occurred while fetching workspaces.'}
    finally:
        session.close()


        
def _extract_workspace_payload(payload: dict) -> dict:
    return {
        'user_id': payload.get('user_id'),
        'workspace_name': payload.get('workspaceName'),
        'namespace': payload.get('namespace'),
        'workspace_desc': payload.get('description'),
        'intent': payload.get('intent'),
        'industry': payload.get('industry'),
        'sub_industry': payload.get('subIndustry'),
        'keywords': payload.get('keywords'),
        'agent_ids': payload.get('agent_ids', []),
        'tool_ids': payload.get('tool_ids', []),
        'kb_ids': payload.get('kb_ids', []),
        'kb_title': payload.get('kb_title', None),
        'kb_description': payload.get('kb_description', None),
    }


def _validate_workspace_type_and_kbs(session, claims: dict, fields: dict):
    keywords = fields.get('keywords')
    kb_ids = fields.get('kb_ids')

    if not isinstance(keywords, list) or not keywords:
        return None, None, {
            'error': (
                "'keywords' is required and must contain exactly one workspace type "
                f"from: {[wt.name for wt in WorkspaceType]}."
            )
        }

    if len(keywords) != 1:
        return None, None, {
            'error': (
                "Exactly one workspace type keyword is required in 'keywords'. "
                f"Allowed values: {[wt.name for wt in WorkspaceType]}."
            )
        }

    raw_keyword = keywords[0]
    if not isinstance(raw_keyword, str) or not raw_keyword.strip():
        return None, None, {'error': "Workspace type keyword must be a non-empty string."}

    keyword_text = raw_keyword.strip().upper()
    selected_workspace_type = WorkspaceType.__members__.get(keyword_text)
    if not selected_workspace_type:
        return None, None, {
            'error': (
                f"Invalid workspace type keyword '{raw_keyword}'. "
                f"Allowed values: {[wt.name for wt in WorkspaceType]}."
            )
        }

    jwt_role_id = claims.get('role_id')
    try:
        normalized_role_id = int(jwt_role_id)
    except (TypeError, ValueError):
        return None, None, {'error': 'Unauthorized: invalid role_id in token claims.'}

    if selected_workspace_type == WorkspaceType.KG and normalized_role_id != Role.ADMIN.id:
        return None, None, {
            'error': (
                "Workspace type 'KG' can only be created by Admin "
                f"({Role.ADMIN.name})."
            )
        }

    # if normalized_role_id != Role.ADMIN.id and selected_workspace_type not in {
    #     WorkspaceType.DM,
    #     WorkspaceType.TR,
    #     WorkspaceType.PR,
    # }:
    #     return None, None, {
    #         'error': (
    #             "Non-admin users can only create workspace types: "
    #             f"{[WorkspaceType.DM.name, WorkspaceType.TR.name, WorkspaceType.PR.name]}."
    #         )
    #     }

    if kb_ids is None:
        kb_ids = []
    if not isinstance(kb_ids, list):
        return None, None, {'error': "KB Ids must be a list"}

    if selected_workspace_type == WorkspaceType.KG and len(kb_ids) != 1:
        return None, None, {
            'error': "Please select only one knowledge base when the workspace type is KG"
        }

    if kb_ids:
        try:
            kb_ids_int = [int(kb_id) for kb_id in kb_ids]
        except (TypeError, ValueError):
            return None, None, {'error': "Invalid KB Ids"}

        # valid_kb_rows 
        valid_count = (
            session.query(db.KnowledgeBase.id)
            .filter(db.KnowledgeBase.id.in_(kb_ids_int), db.KnowledgeBase.is_active.is_(True))
            # .all()
            .count()
        )
        # valid_kb_ids = {row[0] for row in valid_kb_rows}
        # invalid_kb_ids = [kb_id for kb_id in kb_ids_int if kb_id not in valid_kb_ids]
        # if invalid_kb_ids:
        if valid_count != len(kb_ids_int):
            return None, None, {
                'error': (
                    "Invalid kb_ids."
                    # "The following IDs do not exist in active "
                    # f"knowledge_base_master records: {invalid_kb_ids}."
                )
            }
        kb_ids = kb_ids_int

    return selected_workspace_type.name, kb_ids, None


def _workspace_name_exists(session, workspace_name: str) -> bool:
    if not workspace_name or not workspace_name.strip():
        return False
    existing_ws = (
        session.query(db.Workspace)
        .filter(db.Workspace.workspace_name == workspace_name, db.Workspace.is_active == True)
        .first()
    )
    return existing_ws is not None


def _add_creator_workspace_admin(session, creator_id: int, workspace_id: int, namespace: str):
    """
    Add creator as Workspace Admin to the workspace.
    NOTE: UserRoleMap table is DEPRECATED/REDUNDANT - all role info is in UserMap.
    """
    admin_role_id = Role.WS_ADMIN.id
    user_map = session.query(db.UserMap).filter_by(user_id=creator_id, workspace_id=workspace_id).first()
    if not user_map:
        # Create new UserMap entry with admin role
        session.add(
            db.UserMap(
                user_id=creator_id,
                workspace_id=workspace_id,
                is_active=True,
                can_curate_kb=True,
                role_id=admin_role_id,
                namespace=namespace,
                created_date=datetime.now(timezone.utc),
                last_updated=datetime.now(timezone.utc),
            )
        )
    else:
        # Update existing UserMap with admin role
        user_map.role_id = admin_role_id
        user_map.is_active = True
        user_map.can_curate_kb = True
        user_map.last_updated = datetime.now(timezone.utc)
    
    # DEPRECATED: UserRoleMap table is redundant - UserMap already has role_id
    # user_role_map = session.query(db.UserRoleMap).filter_by(user_id=creator_id, workspace_id=workspace_id).first()
    # if user_role_map:
    #     user_role_map.role_id = admin_role_id
    #     user_role_map.is_active = True
    # else:
    #     session.add(db.UserRoleMap(...))


def _add_workspace_mappings(session, workspace_id: int, fields: dict, kb_ids: list[int]):
    industry = fields.get('industry')
    sub_industry = fields.get('sub_industry')
    intent = fields.get('intent')
    agent_ids = fields.get('agent_ids') or []
    tool_ids = fields.get('tool_ids') or []

    if db.WorkspaceIndustrySubIndustryMap and industry and sub_industry and intent and kb_ids:
        for kb_id in kb_ids:
            session.add(
                db.WorkspaceIndustrySubIndustryMap(
                    workspace_id=workspace_id,
                    industry_id=industry,
                    subindustry_id=sub_industry,
                    intent_id=intent,
                    kb_id=kb_id,
                    is_active=True,
                )
            )

    # for agent_id in agent_ids:
    #     session.add(db.AgentMap(workspace_id=workspace_id, agent_id=agent_id, is_active=True))
    
    session.bulk_save_objects([
        db.AgentMap(workspace_id=workspace_id, agent_id=aid, is_active=True)
        for aid in agent_ids
    ])


    # for tool_id in tool_ids:
    #     session.add(db.ToolMap(workspace_id=workspace_id, tool_id=tool_id, is_active=True))
    
    session.bulk_save_objects([
        db.ToolMap(workspace_id=workspace_id, tool_id=tid, is_active=True)
        for tid in tool_ids
    ])


@mcp.tool()
@require_auth
def create_workspace(payload):
    """
    Create a new workspace and map agents/tools/users as per the payload from frontend.
    Args:
        payload (dict): Workspace creation payload from frontend.
    Returns:
        dict: {'response': 'workspace created'}
    """
    claims, creator_id = get_current_user()

    session = db.Session()
    try:
        session.rollback()
        fields = _extract_workspace_payload(payload)
        workspace_name = fields['workspace_name']
        namespace = fields['namespace']
        workspace_desc = fields['workspace_desc']

        normalized_keyword, kb_ids, validation_error = _validate_workspace_type_and_kbs(session, claims, fields)
        if validation_error:
            return validation_error

        # Check for duplicate workspace name globally (active only)
        if _workspace_name_exists(session, workspace_name):
            return {'error': f"Workspace name '{workspace_name}' already exists. Please choose a different name."}

        new_workspace = db.Workspace(
            workspace_name=workspace_name,
            namespace=namespace,
            workspace_desc=workspace_desc,
            keywords=normalized_keyword,
            is_active=True,
            created_date=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc)
        )
        session.add(new_workspace)
        session.flush()
        session.refresh(new_workspace)
        workspace_id = new_workspace.workspace_id
        print(f"Created workspace with ID: {workspace_id}")

        try:
            _add_creator_workspace_admin(session, creator_id, workspace_id, namespace)
            print(f"[Transaction] Added creator as Workspace Admin to workspace {workspace_id}")
        except Exception as user_map_error:
            print(f"[Transaction] Failed to add creator as admin: {user_map_error}")
            raise Exception(f"Failed to map creator to workspace: {str(user_map_error)}")

        try:
            _add_workspace_mappings(session, workspace_id, fields, kb_ids)
            print(f"[Transaction] Added workspace mappings for workspace {workspace_id}")
        except Exception as mapping_error:
            print(f"[Transaction] Failed to add workspace mappings: {mapping_error}")
            raise Exception(f"Failed to add workspace mappings: {str(mapping_error)}")

        session.commit()

        # Seed workspace-level Azure credentials from environment if present.
        # This ensures switch_llm_provider can succeed for default Azure setups.
        try:
            azure_api_key = getenv("AZURE_OPENAI_LLM_MODEL_API_KEY")
            azure_endpoint = getenv("AZURE_OPENAI_LLM_MODEL_API_BASE")
            azure_model = getenv("AZURE_OPENAI_LLM_MODEL_LLM_MODEL")
            azure_api_version = getenv("AZURE_OPENAI_LLM_MODEL_API_VERSION")

            if azure_api_key and azure_endpoint and azure_model:
                workspace_provider_credentials_service.upsert_provider_credentials(
                    workspace_id=workspace_id,
                    provider_name='azure',
                    api_key=azure_api_key,
                    endpoint=azure_endpoint,
                    model=azure_model,
                    api_version=azure_api_version,
                    deployment_name=azure_model,
                    user_id=creator_id,
                )
                print(f"[Post-commit] Seeded workspace Azure credentials for workspace {workspace_id}")
            else:
                print(
                    "[Post-commit] Azure credentials not seeded for workspace "
                    f"{workspace_id}: one or more AZURE_OPENAI_LLM_MODEL_* env vars missing"
                )
        except Exception as workspace_azure_seed_error:
            print(f"[Post-commit] Failed to seed workspace Azure credentials: {workspace_azure_seed_error}")
            # Don't fail workspace creation if credential seeding fails, just log it
        
        # Always create a workspace-level default Azure configuration (after commit).
        # This guarantees every new workspace has a usable default provider selection.
        try:
            agent_llm_config_service.create_or_update_configuration(
                workspace_id=workspace_id,
                agent_id=None,
                configured_providers=['azure'],
                current_provider='azure',
                user_id=creator_id,
            )
            print(f"[Post-commit] Created workspace default Azure LLM configuration for workspace {workspace_id}")
        except Exception as workspace_llm_config_error:
            print(f"[Post-commit] Failed to create workspace default LLM configuration: {workspace_llm_config_error}")
            # Don't fail workspace creation if LLM config fails, just log it

        # Create LLM configurations for all selected agents (after commit)
        agent_ids = fields.get('agent_ids') or []
        if agent_ids:
            try:
                created_configs = agent_llm_config_service.bulk_create_agent_configurations(
                    workspace_id=workspace_id,
                    agent_ids=agent_ids,
                    configured_providers=['azure'],
                    current_provider='azure',
                    user_id=creator_id
                )
                print(f"[Post-commit] Created LLM configurations for {len(created_configs)} agents in workspace {workspace_id}")
            except Exception as agent_llm_config_error:
                print(f"[Post-commit] Failed to create agent LLM configurations: {agent_llm_config_error}")
                # Don't fail workspace creation if LLM config fails, just log it
            
        return {'response': 'Workspace Created'}
    except Exception as e:
        session.rollback()
        print(f"Error in create_workspace: {e}")
        return {'error': f'An error occurred while creating workspace: {str(e)}'}
    finally:
        session.close()

@mcp.tool()
def list_intent():
    """
    Return a list of all active intents from the intent_master table.
    Returns:
        dict: { 'response': [ { 'intent_id': ..., 'intent_name': ... }, ... ] }
    """
    session = db.Session()
    try:
        session.rollback()
        intents = session.query(db.Intent).filter(db.Intent.is_active == True).all()
        result = [
            {
                'intent_id': getattr(intent, 'intent_id', None),
                'intent_name': getattr(intent, 'intent_name', None)
            }
            for intent in intents
        ]
        return {'response': result}
    except Exception as e:
        session.rollback()
        print(f"Error in list_intent: {e}")
        return {'error': 'An error occurred while fetching intents.'}
    finally:
        session.close()

@mcp.tool()
@require_auth
def fetch_tools_info(user_id=None,intent=None):
    """
    Fetch all tool details from the tools_details table.
    Args:
        user_id (optional): The user ID to check for favourite tools.
    Returns:
        list of dicts: Each dict contains tool details and 'favourite' tag if user_id is provided.
    """
    if user_id is None:
        return {"status": "error", "error": "user_id cannot be null"}
    session = db.Session()
    try:
        session.rollback()
        # Use get_current_user() for authentication
        claims, jwt_user_id = get_current_user()

        # --- DUMMY USER HANDLING ---
        email = claims.get("email") or claims.get("preferred_username") or claims.get("upn")
        user = None
        if email:
            try:
                user = _fetch_user_by_email(email)
            except Exception:
                user = None
        if not user:
            # User not yet in DB (edge case — SSO normally inserts them on first login).
            # Return ALL active tools so they can explore the full platform.
            tools = session.query(db.Tool).filter(db.Tool.is_active == True).all()
            tool_list = []
            for t in tools:
                tool_dict = {col: getattr(t, col) for col in t.__table__.columns.keys()}
                tool_dict['favourite'] = False
                tool_list.append(tool_dict)
            return {'response': tool_list}

        # If intent is provided, filter tools by intent
        if intent:
            intent_tool_ids = session.query(db.ToolIntentMap.tool_id).filter(db.ToolIntentMap.intent_id == intent, db.ToolIntentMap.is_active == True).all()
            intent_tool_ids = [row.tool_id for row in intent_tool_ids]
            tools = session.query(db.Tool).filter(
                db.Tool.tool_id.in_(intent_tool_ids),
                db.Tool.is_active == True
            ).all()
        else:
            tools = session.query(db.Tool).filter(
                db.Tool.is_active == True
            ).all()

        favorite_tool_ids = set()
        if jwt_user_id is not None:
            tool_ids = [t.tool_id for t in tools]
            if tool_ids:
                favorites = session.query(db.FavouriteMappingTool.tool_id).filter(
                    db.FavouriteMappingTool.user_id == jwt_user_id,
                    db.FavouriteMappingTool.tool_id.in_(tool_ids),
                    db.FavouriteMappingTool.is_active == True
                ).all()
                favorite_tool_ids = {fav.tool_id for fav in favorites}

        tool_list = []
        for t in tools:
            tool_dict = {col: getattr(t, col) for col in t.__table__.columns.keys()}
            tool_dict['favourite'] = t.tool_id in favorite_tool_ids
            tool_list.append(tool_dict)
        return {'response': tool_list}
    except Exception as e:
        print(f"Error in fetch_tools_info: {e}")
        return {'error': 'An error occurred while fetching tools.'}
    finally:
        session.close()

@mcp.tool()
@require_auth
def fetch_intent_tools_info(user_id=None, intent=None):
    """
    Fetch all tool details from the tools_details table for a given intent/user.
    Does NOT include favourite mapping logic.
    Returns:
        list of dicts: Each dict contains tool details.
    """
    if user_id is None:
        return {"status": "error", "error": "user_id cannot be null"}
    session = db.Session()
    try:
        session.rollback()

        # If intent is provided, filter tools by intent
        if intent:
            intent_tool_ids = session.query(db.ToolIntentMap.tool_id).filter(db.ToolIntentMap.intent_id == intent, db.ToolIntentMap.is_active == True).all()
            intent_tool_ids = [row.tool_id for row in intent_tool_ids]
            #tools = session.query(Tool).filter(Tool.tool_id.in_(intent_tool_ids)).all()
            tools = session.query(db.Tool).filter(
                db.Tool.tool_id.in_(intent_tool_ids),
                db.Tool.is_active == True
            ).all()
        else:
            #tools = session.query(db.Tool).all()
            tools = session.query(db.Tool).filter(
                db.Tool.is_active == True
            ).all()

        tool_list = []
        for t in tools:
            tool_dict = {col: getattr(t, col) for col in t.__table__.columns.keys()}
            tool_list.append(tool_dict)
        return {'response': tool_list}
    except Exception as e:
        return {'error': 'An error occurred while fetching tools.'}
    finally:
        session.close()


@mcp.tool()
@require_auth
def fetch_agents_info(user_id=None, intent=None):
    """
    Fetch all agent details from the agents_details table.
    Returns:
        list of dicts: Each dict contains agent details.
    """
    if user_id is None:
        return {"status": "error", "error": "user_id cannot be null"}
    session = db.Session()
    try:
        session.rollback()
        # Use get_current_user() for authentication
        claims, jwt_user_id = get_current_user()

        # --- DUMMY USER HANDLING ---
        email = claims.get("email") or claims.get("preferred_username") or claims.get("upn")
        user = None
        if email:
            try:
                user = _fetch_user_by_email(email)
            except Exception:
                user = None
        if not user:
            # Dummy user: return all agents mapped to demo workspace
            dummy_workspace_id = getenv("DUMMY_WORKSPACE_ID", "1")
            try:
                dummy_workspace_id = int(dummy_workspace_id)
            except Exception:
                dummy_workspace_id = 1
            # Get all agent_ids mapped to the demo workspace
            mapped_agent_ids = session.query(db.AgentMap.agent_id).filter(
                db.AgentMap.workspace_id == dummy_workspace_id,
                db.AgentMap.is_active == True
            ).all()
            mapped_agent_ids = [row.agent_id for row in mapped_agent_ids]
            if not mapped_agent_ids:
                return {'response': []}
            agents = session.query(db.Agent).filter(
                db.Agent.agent_id.in_(mapped_agent_ids),
                db.Agent.is_active == True
            ).all()
            agent_list = []
            for a in agents:
                agent_dict = {col: getattr(a, col) for col in a.__table__.columns.keys()}
                agent_dict['favourite'] = False
                agent_list.append(agent_dict)
            return {'response': agent_list}

        # If intent is provided, filter agents by intent
        if intent:
            intent_agent_ids = session.query(db.AgentIntentMap.agent_id).filter(db.AgentIntentMap.intent_id == intent, db.AgentIntentMap.is_active == True).all()
            intent_agent_ids = [row.agent_id for row in intent_agent_ids]
            agents = session.query(db.Agent).filter(
                db.Agent.agent_id.in_(intent_agent_ids),
                db.Agent.is_active == True
            ).all()
        else:
            agents = session.query(db.Agent).filter(
                db.Agent.is_active == True
            ).all()

        favorite_agent_ids = set()
        if jwt_user_id is not None:
            agent_ids = [a.agent_id for a in agents]
            if agent_ids:
                favorites = session.query(db.FavouriteMappingAgent.agent_id).filter(
                    db.FavouriteMappingAgent.user_id == jwt_user_id,
                    db.FavouriteMappingAgent.agent_id.in_(agent_ids),
                    db.FavouriteMappingAgent.is_active == True
                ).all()
                favorite_agent_ids = {fav.agent_id for fav in favorites}

        agent_list = []
        for a in agents:
            agent_dict = {col: getattr(a, col) for col in a.__table__.columns.keys()}
            agent_dict['favourite'] = a.agent_id in favorite_agent_ids
            agent_list.append(agent_dict)
        return {'response': agent_list}
    except Exception as e:
        print(f"Error in fetch_agents_info: {e}")
        return {'error': 'An error occurred while fetching agents.'}
    finally:
        session.close()

@mcp.tool()
@require_auth
def fetch_intent_agents_info(user_id=None, intent=None):
    """
    Fetch all agent details from the agents_details table for a given intent/user.
    Does NOT include favourite mapping logic.
    Returns:
        list of dicts: Each dict contains agent details.
    """
    if user_id is None:
        return {"status": "error", "error": "user_id cannot be null"}
    session = db.Session()
    try:
        session.rollback()

        # If intent is provided, filter agents by intent
        if intent:
            intent_agent_ids = session.query(db.AgentIntentMap.agent_id).filter(db.AgentIntentMap.intent_id == intent, db.AgentIntentMap.is_active == True).all()
            intent_agent_ids = [row.agent_id for row in intent_agent_ids]
            #agents = session.query(Agent).filter(Agent.agent_id.in_(intent_agent_ids)).all()
            agents = session.query(db.Agent).filter(
                db.Agent.agent_id.in_(intent_agent_ids),
                db.Agent.is_active == True
            ).all()
        else:
            #agents = session.query(db.Agent).all()
            agents = session.query(db.Agent).filter(
                db.Agent.is_active == True
            ).all()

        agent_list = []
        for a in agents:
            agent_dict = {col: getattr(a, col) for col in a.__table__.columns.keys()}
            agent_list.append(agent_dict)
        return {'response': agent_list}
    except Exception as e:
        return {'error': 'An error occurred while fetching agents.'}
    finally:
        session.close()


@mcp.tool()
@require_auth
def update_workspace(payload):
    """
    Update workspace details (name, description, tools, agents, KBs, etc.) using a payload dict.
    Only Workspace Admin can update workspaces.
    Args:
        payload (dict): Should contain 'workspace_id', and optionally 'workspaceName', 'description', 'tool_ids', 'agent_ids', 'keywords', 'kb_ids', etc.
    Returns:
        dict: Response or error message.
    """
    claims, jwt_user_id = get_current_user()
    session = db.Session()
    try:
        session.rollback()
        workspace_id = payload.get('workspace_id')
        
        if workspace_id is None:
            return {'error': 'Missing Workspace id'}
        
        workspace_id = int(workspace_id)
        
        # RBAC: Only Workspace Admin can update
        if not is_admin(jwt_user_id, workspace_id):
            return {"error": "You are not authorized to update this workspace. Only Workspace Admin can update workspaces."}

        ws = session.query(db.Workspace).filter(
            db.Workspace.workspace_id == workspace_id,
            db.Workspace.is_active == True
        ).first()
        if not ws:
            return {"error": "Workspace not found or inactive"}

        # Extract payload
        fields = _extract_workspace_payload(payload)

        # Workspace type is immutable after creation.
        if 'keywords' in payload and payload.get('keywords') is not None:
            requested_keywords = payload.get('keywords')
            if isinstance(requested_keywords, list) and len(requested_keywords) == 1:
                requested_workspace_type = str(requested_keywords[0]).strip().upper()
                existing_workspace_type = (getattr(ws, 'keywords', None) or '').strip().upper()
                if requested_workspace_type != existing_workspace_type:
                    return {"error": "Workspace type cannot be updated."}
            else:
                return {"error": "Workspace type cannot be updated."}

        # Get current mapping context for KB updates.
        current_wiim = None
        if db.WorkspaceIndustrySubIndustryMap:
            current_wiim = session.query(db.WorkspaceIndustrySubIndustryMap).filter(
                db.WorkspaceIndustrySubIndustryMap.workspace_id == workspace_id,
                db.WorkspaceIndustrySubIndustryMap.is_active == True
            ).first()

        existing_workspace_type = (getattr(ws, 'keywords', None) or '').strip().upper()

        
        # Update WIIM mappings for KBs using existing industry/sub-industry context.
        effective_intent = getattr(current_wiim, 'intent_id', None) if current_wiim else None
        effective_industry = getattr(current_wiim, 'industry_id', None) if current_wiim else None
        effective_sub_industry = getattr(current_wiim, 'subindustry_id', None) if current_wiim else None

        # Industry, sub-industry, and intent are immutable after workspace creation.
        # Ignore these fields from update payload and always rely on existing mapping context.
        
        should_update_kb_mappings = existing_workspace_type != WorkspaceType.KG.name and 'kb_ids' in payload
        kb_ids = None
        if should_update_kb_mappings:
            kb_ids = fields.get('kb_ids')
            if kb_ids is None:
                kb_ids = []
            if kb_ids and not isinstance(kb_ids, list):
                return {'error': "KB Ids must be a list"}
            if kb_ids:
                try:
                    kb_ids = [int(kb_id) for kb_id in kb_ids]
                except (TypeError, ValueError):
                    return {'error': "Invalid KB Ids"}

                valid_count = (
                    session.query(db.KnowledgeBase.id)
                    .filter(db.KnowledgeBase.id.in_(kb_ids), db.KnowledgeBase.is_active.is_(True))
                    .count()
                )
                if valid_count != len(kb_ids):
                    return {'error': "Invalid KB selected."}

        # Update workspace master fields
        if fields.get('workspace_name'):
            ws.workspace_name = fields['workspace_name']
        if fields.get('workspace_desc'):
            ws.workspace_desc = fields['workspace_desc']
        if fields.get('namespace'):
            ws.namespace = fields['namespace']
        ws.last_updated = datetime.now(timezone.utc)

        if db.WorkspaceIndustrySubIndustryMap and effective_industry and effective_sub_industry and effective_intent and should_update_kb_mappings:
            session.query(db.WorkspaceIndustrySubIndustryMap).filter(
                db.WorkspaceIndustrySubIndustryMap.workspace_id == workspace_id,
                db.WorkspaceIndustrySubIndustryMap.industry_id == effective_industry,
                db.WorkspaceIndustrySubIndustryMap.subindustry_id == effective_sub_industry,
                db.WorkspaceIndustrySubIndustryMap.intent_id == effective_intent
            ).delete(synchronize_session=False)
            for kb_id in kb_ids:
                session.add(db.WorkspaceIndustrySubIndustryMap(
                    workspace_id=workspace_id,
                    industry_id=effective_industry,
                    subindustry_id=effective_sub_industry,
                    intent_id=effective_intent,
                    kb_id=kb_id,
                    is_active=True
                ))

        # Update agent mappings: mark all inactive, then activate/create new ones
        agent_ids = fields.get('agent_ids') or []
        if agent_ids is not None:
            session.query(db.AgentMap).filter(
                db.AgentMap.workspace_id == workspace_id
            ).update({db.AgentMap.is_active: False}, synchronize_session=False)
            
            for aid in agent_ids:
                existing = session.query(db.AgentMap).filter_by(
                    workspace_id=workspace_id,
                    agent_id=aid
                ).first()
                if existing:
                    existing.is_active = True
                else:
                    session.add(db.AgentMap(
                        workspace_id=workspace_id,
                        agent_id=aid,
                        is_active=True
                    ))

        # Update tool mappings: mark all inactive, then activate/create new ones
        tool_ids = fields.get('tool_ids') or []
        if tool_ids is not None:
            session.query(db.ToolMap).filter(
                db.ToolMap.workspace_id == workspace_id
            ).update({db.ToolMap.is_active: False}, synchronize_session=False)
            
            for tid in tool_ids:
                existing = session.query(db.ToolMap).filter_by(
                    workspace_id=workspace_id,
                    tool_id=tid
                ).first()
                if existing:
                    existing.is_active = True
                else:
                    session.add(db.ToolMap(
                        workspace_id=workspace_id,
                        tool_id=tid,
                        is_active=True
                    ))

        session.commit()
        return {"response": "Workspace updated"}
    except Exception as e:
        session.rollback()
        print(f"Error in update_workspace: {e}")
        return {"error": f"An error occurred while updating workspace: {str(e)}"}
    finally:
        session.close()

@mcp.tool()
@require_auth
def delete_workspace(workspace_id):
    '''
    Delete workspace: set is_active to False
    Only Workspace Admin can delete workspaces.
    '''
    claims, jwt_user_id = get_current_user()
    session = db.Session()
    try:
        if workspace_id is None:
            return {'error': 'Missing Workspace id'}
        
        workspace_id = int(workspace_id)
        session.rollback()
        # Check if user has Workspace Admin role for this workspace
        if not is_admin(jwt_user_id, workspace_id):
            return {"error": "You are not authorized to delete a workspace. Only Workspace Admin can delete workspaces."}

        ws = session.query(db.Workspace).filter(db.Workspace.workspace_id==workspace_id, db.Workspace.is_active==True).first()
        if not ws:
            return {"error": "Workspace not found or already inactive"}
        ws.is_active = False
        session.commit()
        
        # Delete all LLM configurations for this workspace (after commit)
        try:
            deleted_count = agent_llm_config_service.delete_workspace_configurations(
                workspace_id=workspace_id,
                user_id=jwt_user_id
            )
            print(f"[Post-commit] Deleted {deleted_count} LLM configurations for workspace {workspace_id}")
        except Exception as llm_config_error:
            print(f"[Post-commit] Failed to delete LLM configurations: {llm_config_error}")
            # Don't fail workspace deletion if LLM config cleanup fails, just log it
        
        return {"response": "Workspace deleted (set inactive)"}
    except Exception as e:
        session.rollback()
        print(f"Error in delete_workspace: {e}")
        return {"error": "An error occurred while deleting workspace."}
    finally:
        session.close()

@mcp.tool()
@require_auth
def fetch_workspace_details(workspace_id):
    '''
    Fetch all information about a workspace, including master table, mappings, and all tool/agent/user details.
    Args:
        workspace_id (int): ID of the workspace to fetch.
    Returns:
        dict: Workspace info, mappings, tools, agents, users, and related attributes.
    '''
    session = db.Session()
    try:
        session.rollback()
        # Use get_current_user() for authentication
        claims, jwt_user_id = get_current_user()

        # --- DUMMY USER HANDLING ---
        email = claims.get("email") or claims.get("preferred_username") or claims.get("upn")
        user = None
        if email:
            try:
                user = _fetch_user_by_email(email)
            except Exception:
                user = None
        is_dummy_user = not user
        # If dummy user, only allow access to demo workspace
        dummy_workspace_id = getenv("DUMMY_WORKSPACE_ID", "1")
        try:
            dummy_workspace_id = int(dummy_workspace_id)
        except Exception:
            dummy_workspace_id = 1
        if is_dummy_user:
            if int(workspace_id) != int(dummy_workspace_id):
                return {"error": "You are not authorized to access this workspace."}
            # else: skip user_map check and proceed
        else:
            # Check if user is mapped to this workspace
            user_map = session.query(db.UserMap).filter_by(workspace_id=workspace_id, user_id=jwt_user_id, is_active=True).first()
            if not user_map:
                return {"error": "You are not authorized to access this workspace."}
        ws = session.query(db.Workspace).filter(db.Workspace.workspace_id==workspace_id, db.Workspace.is_active==True).first()
        if not ws:
            # Workspace row missing or inactive — if this is the dummy workspace,
            # build a synthetic response with ALL active tools and agents so that
            # any user (new or existing) assigned here can explore the platform.
            if int(workspace_id) == int(dummy_workspace_id):
                dummy_workspace_name = getenv("DUMMY_WORKSPACE_NAME", "Demo Workspace").strip().strip('"').strip("'")
                dummy_workspace_desc = getenv("DUMMY_WORKSPACE_DESC", "Demo workspace for new users.").strip().strip('"').strip("'")
                ws_info = {
                    'workspace_id': dummy_workspace_id,
                    'workspace_name': dummy_workspace_name,
                    'workspace_desc': dummy_workspace_desc,
                    'is_active': True
                }
                # Category mapping (needed for enriching tool/agent dicts)
                categories = session.query(db.Category).filter(db.Category.is_active == True).all()
                cat_map = {str(c.category_id): c.category_name for c in categories}

                # Return ALL active tools (not just workspace-mapped — workspace 73 may have none)
                tools = []
                tool_query = session.query(db.Tool)
                if hasattr(db.Tool, 'is_active'):
                    tool_query = tool_query.filter(db.Tool.is_active == True)
                for t in tool_query.all():
                    tool_dict = {col: getattr(t, col) for col in t.__table__.columns.keys()}
                    tool_dict['last_updated'] = None
                    tool_dict['last_used'] = None
                    cat_ids = str(tool_dict.get('tool_category', '') or '').split(',')
                    tool_dict['tool_category'] = [cat_map.get(cid.strip()) for cid in cat_ids if cid.strip() in cat_map]
                    tools.append(tool_dict)

                # Return ALL active agents
                agents = []
                agent_query = session.query(db.Agent)
                if hasattr(db.Agent, 'is_active'):
                    agent_query = agent_query.filter(db.Agent.is_active == True)
                for a in agent_query.all():
                    agent_dict = {col: getattr(a, col) for col in a.__table__.columns.keys()}
                    agent_dict['last_updated'] = None
                    agent_dict['last_used'] = None
                    cat_ids = str(agent_dict.get('agent_category', '') or '').split(',')
                    agent_dict['agent_category'] = [cat_map.get(cid.strip()) for cid in cat_ids if cid.strip() in cat_map]
                    agent_dict['type'] = 'agent'
                    agents.append(agent_dict)

                return {
                    "workspace": ws_info,
                    "industry": None,
                    "industry_name": None,
                    "subindustry": None,
                    "subindustry_name": None,
                    "intent": None,
                    "tools": tools,
                    "agents": agents,
                    "users": [],
                    "knowledge_bases": []
                }
            else:
                return {"error": "Workspace not found or inactive"}

        # Master table info
        ws_info = {col: getattr(ws, col) for col in ws.__table__.columns.keys()}

        # Category mapping
        categories = session.query(db.Category).filter(db.Category.is_active == True).all()
        cat_map = {str(c.category_id): c.category_name for c in categories}

        # Mappings (region, intent, industry, subindustry, keywords) - only is_active if present
        def active_query(model, **kwargs):
            q = session.query(model).filter_by(**kwargs)
            if hasattr(model, 'is_active'):
                q = q.filter(model.is_active == True)
            return q

        # Fetch industry/subindustry mapping and names
        ws_ind_map = session.query(db.WorkspaceIndustrySubIndustryMap).filter(db.WorkspaceIndustrySubIndustryMap.workspace_id==workspace_id, db.WorkspaceIndustrySubIndustryMap.is_active==True).first()
        industry_id = subindustry_id = intent_id = industry_name = subindustry_name = None
        if ws_ind_map:
            industry_id = getattr(ws_ind_map, 'industry_id', None)
            subindustry_id = getattr(ws_ind_map, 'subindustry_id', None)
            intent_id = getattr(ws_ind_map, 'intent_id', None)
            if industry_id:
                industry_obj = session.query(db.Industry).filter(db.Industry.industry_id==industry_id,db.Industry.is_active==True).first()
                if industry_obj:
                    industry_name = getattr(industry_obj, 'industry_name', None)
            if subindustry_id:
                subindustry_obj = session.query(db.SubIndustry).filter(db.SubIndustry.subindustry_id==subindustry_id,db.SubIndustry.is_active==True).first()
                if subindustry_obj:
                    subindustry_name = getattr(subindustry_obj, 'subindustry_name', None)

        # Tools in workspace (only active)
        tool_maps = session.query(db.ToolMap).filter(db.ToolMap.workspace_id==workspace_id, db.ToolMap.is_active==True).all()
        tool_map_dict = {tm.tool_id: tm for tm in tool_maps}
        tool_ids = list(tool_map_dict.keys())
        tools = []
        if tool_ids:
            tool_query = session.query(db.Tool).filter(db.Tool.tool_id.in_(tool_ids))
            if hasattr(db.Tool, 'is_active'):
                tool_query = tool_query.filter(db.Tool.is_active == True)
            for t in tool_query.all():
                tool_dict = {col: getattr(t, col) for col in t.__table__.columns.keys()}
                # Add last_updated from ToolMap
                tm = tool_map_dict.get(t.tool_id)
                last_updated_val = getattr(tm, 'last_updated', None) if tm else None
                tool_dict['last_updated'] = last_updated_val
                tool_dict['last_used'] = last_updated_val
                # Replace tool_category IDs with names
                cat_ids = str(tool_dict.get('tool_category', '') or '').split(',')
                tool_dict['tool_category'] = [cat_map.get(cid.strip()) for cid in cat_ids if cid.strip() in cat_map]
                tools.append(tool_dict)

        # Agents in workspace (only active)
        agent_maps = session.query(db.AgentMap).filter(db.AgentMap.workspace_id==workspace_id, db.AgentMap.is_active==True).all()
        agent_map_dict = {am.agent_id: am for am in agent_maps}
        agent_ids = list(agent_map_dict.keys())
        agents = []
        if agent_ids:
            agent_query = session.query(db.Agent).filter(db.Agent.agent_id.in_(agent_ids))
            if hasattr(db.Agent, 'is_active'):
                agent_query = agent_query.filter(db.Agent.is_active == True)
            for a in agent_query.all():
                agent_dict = {col: getattr(a, col) for col in a.__table__.columns.keys()}
                # Add last_updated from AgentMap
                am = agent_map_dict.get(a.agent_id)
                last_updated_val = getattr(am, 'last_updated', None) if am else None
                agent_dict['last_updated'] = last_updated_val
                agent_dict['last_used'] = last_updated_val
                # Replace agent_category IDs with names
                cat_ids = str(agent_dict.get('agent_category', '') or '').split(',')
                agent_dict['agent_category'] = [cat_map.get(cid.strip()) for cid in cat_ids if cid.strip() in cat_map]
                agent_dict['type'] = 'agent'
                agents.append(agent_dict)

        # Users in workspace (with role/permissions, only active)
        # OPTIMIZED: Single JOIN query - UserMap contains role_id, no need for UserRoleMap
        # NOTE: UserRoleMap table is DEPRECATED - all role info is in UserMap
        users = []
        user_data_query = (
            session.query(db.User, db.Role, db.UserMap)
            .join(db.UserMap, db.UserMap.user_id == db.User.user_id)
            .outerjoin(db.Role, (db.UserMap.role_id == db.Role.role_id) & (db.Role.is_active == True))
            .filter(db.UserMap.workspace_id == workspace_id, db.UserMap.is_active == True)
        )
        if hasattr(db.User, 'is_active'):
            user_data_query = user_data_query.filter(db.User.is_active == True)

        for user, role, user_map in user_data_query.all():
            user_dict = {col: getattr(user, col) for col in user.__table__.columns.keys()}
            # Get role info from UserMap (which has role_id) joined with Role table
            user_dict['role'] = getattr(role, 'role_name', None) if role else None
            user_dict['role_id'] = getattr(user_map, 'role_id', None)  # role_id is in UserMap
            user_dict['permissions'] = getattr(user_map, 'permissions', None)  # permissions is in UserMap
            user_dict['can_curate_kb'] = getattr(user_map, 'can_curate_kb', None)
            users.append(user_dict)

        # Fetch knowledge bases for this workspace's industry and subindustry
        knowledge_bases = []
        if industry_name and subindustry_name:
            # Get industry_id and subindustry_id
            industry_obj = session.query(db.Industry).filter(func.lower(db.Industry.industry_name) == industry_name.strip().lower(), db.Industry.is_active == True).first()
            subindustry_obj = session.query(db.SubIndustry).filter(func.lower(db.SubIndustry.subindustry_name) == subindustry_name.strip().lower(), db.SubIndustry.is_active == True).first()
            if industry_obj and subindustry_obj:
                kb_query = session.query(db.WorkspaceIndustrySubIndustryMap).filter(
                    db.WorkspaceIndustrySubIndustryMap.industry_id == industry_obj.industry_id,
                    db.WorkspaceIndustrySubIndustryMap.subindustry_id == subindustry_obj.subindustry_id,
                    db.WorkspaceIndustrySubIndustryMap.workspace_id == workspace_id,
                    db.WorkspaceIndustrySubIndustryMap.is_active == True
                )
                for kb_id in [row.kb_id for row in kb_query.all() if row.kb_id]:
                    kb_obj = session.query(db.KnowledgeBase).filter(db.KnowledgeBase.id == kb_id, db.KnowledgeBase.is_active == True).first()
                    if kb_obj:
                        knowledge_bases.append({
                            'id': getattr(kb_obj, 'id', None),
                            'title': getattr(kb_obj, 'title', None),
                            'description': getattr(kb_obj, 'description', None)})

                print(f"Fetched knowledge bases for industry '{industry_name}' and subindustry '{subindustry_name}': {knowledge_bases}")

                # knowledge_bases = [
                #     {
                #         'id': getattr(kb, 'id', None),
                #         'title': getattr(kb, 'title', None),
                #         'description': getattr(kb, 'description', None)
                #     }
                #     for kb in kb_query.all()
                # ]

        # Dummy workspace exists in DB but has no tool/agent mappings yet —
        # return ALL active tools and agents so new users can explore the platform.
        if int(workspace_id) == int(dummy_workspace_id):
            if not tools:
                tool_query = session.query(db.Tool)
                if hasattr(db.Tool, 'is_active'):
                    tool_query = tool_query.filter(db.Tool.is_active == True)
                for t in tool_query.all():
                    tool_dict = {col: getattr(t, col) for col in t.__table__.columns.keys()}
                    tool_dict['last_updated'] = None
                    tool_dict['last_used'] = None
                    cat_ids = str(tool_dict.get('tool_category', '') or '').split(',')
                    tool_dict['tool_category'] = [cat_map.get(cid.strip()) for cid in cat_ids if cid.strip() in cat_map]
                    tools.append(tool_dict)

            if not agents:
                agent_query = session.query(db.Agent)
                if hasattr(db.Agent, 'is_active'):
                    agent_query = agent_query.filter(db.Agent.is_active == True)
                for a in agent_query.all():
                    agent_dict = {col: getattr(a, col) for col in a.__table__.columns.keys()}
                    agent_dict['last_updated'] = None
                    agent_dict['last_used'] = None
                    cat_ids = str(agent_dict.get('agent_category', '') or '').split(',')
                    agent_dict['agent_category'] = [cat_map.get(cid.strip()) for cid in cat_ids if cid.strip() in cat_map]
                    agent_dict['type'] = 'agent'
                    agents.append(agent_dict)

        return {
            "workspace": ws_info,
            "industry": industry_id,
            "industry_name": industry_name,
            "subindustry": subindustry_id,
            "subindustry_name": subindustry_name,
            "intent": intent_id,
            "tools": tools,
            "agents": agents,
            "users": users,
            "knowledge_bases": knowledge_bases
        }
    except Exception as e:
        print(f"Error in fetch_workspace_details: {e}")
        return {"error": "An error occurred while fetching workspace details."}
    finally:
        session.close()


@mcp.tool()
@require_auth
def fetch_agents_tools_by_ids(workspace_id):
    """
    Fetch all tools and agents for a given workspace_id, tagging each as 'tool' or 'agent'.
    Only returns mappings and entities where is_active == 'true'.
    Replaces agent_category/tool_category IDs with category names.
    """
    claims, jwt_user_id = get_current_user()
    session = db.Session()
    try:
        session.rollback()

        # Check if user is mapped to this workspace
        user_map = session.query(db.UserMap).filter_by(
            workspace_id=workspace_id, user_id=jwt_user_id, is_active=True
        ).first()
        if not user_map:
            return {"error": "You are not authorized to access this workspace."}

        # 1. Get category_id -> category_name mapping
        categories = session.query(db.Category).filter(db.Category.is_active != False).all()
        cat_map = {str(c.category_id): c.category_name for c in categories}

        results = []

        # 2. Fetch tools
        tool_maps_q = session.query(db.ToolMap).filter_by(workspace_id=workspace_id, is_active=True)
        tool_ids = [tm.tool_id for tm in tool_maps_q.all()]
        if tool_ids:
            tool_query = session.query(db.Tool).filter(db.Tool.tool_id.in_(tool_ids))
            if hasattr(db.Tool, 'is_active'):
                tool_query = tool_query.filter(db.Tool.is_active != 'false')
            for t in tool_query.all():
                tool_dict = {col: getattr(t, col) for col in t.__table__.columns.keys()}
                # Replace tool_category IDs with names
                cat_ids = str(tool_dict.get('tool_category', '') or '').split(',')
                tool_dict['tool_category'] = [cat_map.get(cid.strip()) for cid in cat_ids if cid.strip() in cat_map]
                tool_dict['type'] = 'tool'
                results.append(tool_dict)

        # 3. Fetch agents
        agent_maps_q = session.query(db.AgentMap).filter_by(workspace_id=workspace_id, is_active=True)
        agent_ids = [am.agent_id for am in agent_maps_q.all()]
        if agent_ids:
            agent_query = session.query(db.Agent).filter(db.Agent.agent_id.in_(agent_ids))
            if hasattr(db.Agent, 'is_active'):
                agent_query = agent_query.filter(db.Agent.is_active != 'false')
            for a in agent_query.all():
                agent_dict = {col: getattr(a, col) for col in a.__table__.columns.keys()}
                # Replace agent_category IDs with names
                cat_ids = str(agent_dict.get('agent_category', '') or '').split(',')
                agent_dict['agent_category'] = [cat_map.get(cid.strip()) for cid in cat_ids if cid.strip() in cat_map]
                agent_dict['type'] = 'agent'
                results.append(agent_dict)

        return {'response': results}
    except Exception as e:
        print(f"Error in fetch_agents_tools_by_ids: {e}")
        return {'error': 'An error occurred while fetching agents and tools.'}
    finally:
        session.close()

@mcp.tool()
@require_auth
def add_agent_tool_to_workspace(payload):
    """
    Add an agent or tool to a workspace.
    Args:
        payload (dict): {"user_id": ..., "workspace_id": ..., "type": "Agent" or "Tool", "id": ...}
    Returns:
        dict: {"response": "Successfully Added to workspace"}
    """
    _, jwt_user_id = get_current_user()

    session = db.Session()
    try:
        session.rollback()

        workspace_id = payload.get("workspace_id")
        entity_type = payload.get("type")
        entity_id = payload.get("id")

        # Authorization: JWT user must be mapped to this workspace
        user_map = session.query(db.UserMap).filter_by(
            workspace_id=workspace_id, user_id=jwt_user_id, is_active=True
        ).first()
        if not user_map:
            return {"error": "You are not authorized to modify this workspace."}

        if entity_type == "Agent":
            existing = session.query(db.AgentMap).filter_by(workspace_id=workspace_id, agent_id=entity_id).first()
            if existing:
                if not existing.is_active:
                    existing.is_active = True
                else:
                    return {"response": "Agent already mapped and active in workspace"}
            else:
                session.add(db.AgentMap(workspace_id=workspace_id, agent_id=entity_id, is_active=True))
        elif entity_type == "Tool":
            existing = session.query(db.ToolMap).filter_by(workspace_id=workspace_id, tool_id=entity_id).first()
            if existing:
                if not existing.is_active:
                    existing.is_active = True
                else:
                    return {"response": "Tool already mapped and active in workspace"}
            else:
                session.add(db.ToolMap(workspace_id=workspace_id, tool_id=entity_id, is_active=True))
        else:
            return {"error": "Invalid type. Must be 'Agent' or 'Tool'."}

        session.commit()
        return {"response": "Successfully Added to workspace"}
    except Exception as e:
        session.rollback()
        print(f"Error in add_agent_tool_to_workspace: {e}")
        return {"error": "An error occurred while adding agent/tool to workspace."}
    finally:
        session.close()

@mcp.tool()
@require_auth
def remove_workspace_agent_tool_mapping(workspace_id, agent_id=None, tool_id=None):
    """
    Remove mapping between workspace and agent/tool by setting is_active to 'false'.
    Args:
        workspace_id (int): Workspace ID.
        agent_id (int, optional): Agent ID to remove mapping for.
        tool_id (int, optional): Tool ID to remove mapping for.
    Returns:
        dict: Success or error message.
    """
    _, jwt_user_id = get_current_user()

    session = db.Session()
    try:
        session.rollback()
        # Check if user is mapped to this workspace
        user_map = session.query(db.UserMap).filter_by(
            workspace_id=workspace_id, user_id=jwt_user_id, is_active=True
        ).first()
        if not user_map:
            return {"error": "You are not authorized to modify this workspace."}

        updated = False
        if agent_id is not None:
            mapping = session.query(db.AgentMap).filter_by(workspace_id=workspace_id, agent_id=agent_id).first()
            if mapping and hasattr(mapping, 'is_active'):
                mapping.is_active = False
                updated = True
        if tool_id is not None:
            mapping = session.query(db.ToolMap).filter_by(workspace_id=workspace_id, tool_id=tool_id).first()
            if mapping and hasattr(mapping, 'is_active'):
                mapping.is_active = False
                updated = True

        if updated:
            session.commit()
            return {'response': 'Mapping removed (set inactive)'}
        else:
            return {'error': 'Mapping not found'}
    except Exception as e:
        session.rollback()
        print(f"Error in remove_workspace_agent_tool_mapping: {e}")
        return {'error': 'An error occurred while removing mapping.'}
    finally:
        session.close()

@mcp.tool()
@require_auth_async
async def update_fav_agent(user_id, agent_id, workspace_id=0) -> dict:
    if user_id is None:
        return {"status": "error", "error": "user_id cannot be null"}
    _, jwt_user_id = get_current_user()
    if str(user_id) != str(jwt_user_id):
        return {"status": "error", "error": "Unauthorized: user_id in request does not match user in token"}

    valid, err = validate_user_workspace_access(user_id=user_id, workspace_id=workspace_id)
    if not valid:
        return {"status": "error", "error": err}

    session = db.Session()
    try:
        session.rollback()
        user_agent_fav = session.query(db.FavouriteMappingAgent).filter(
            db.FavouriteMappingAgent.user_id == user_id,
            db.FavouriteMappingAgent.agent_id == agent_id,
            db.FavouriteMappingAgent.workspace_id == workspace_id
        ).first()
        if not user_agent_fav:
            new_fav = db.FavouriteMappingAgent(
                user_id=user_id,
                agent_id=agent_id,
                workspace_id=workspace_id,
                is_active=True
            )
            session.add(new_fav)
            user_agent_fav = new_fav
            print("Added new favourite mapping; is_active set to:", new_fav.is_active)
        else:
            user_agent_fav.is_active = not user_agent_fav.is_active
            print("Toggled favourite mapping is_active to:", user_agent_fav.is_active)
        session.commit()
        fav_flag = "favourites" if user_agent_fav.is_active else "not favourites"
        print(f"Agent_id {agent_id} updated to {fav_flag}")
        return {
            "status": "success", 
            "response": f"Agent_id {agent_id} updated to {fav_flag}",
            "favourite": user_agent_fav.is_active
            }
    except Exception as e:
        session.rollback()
        print(f"Error in update_fav_agent: {e}")
        return {"status": "error", "error": "An error occurred while updating favourite agent."}
    finally:
        session.close()

@mcp.tool()
@require_auth_async
async def update_fav_tool(user_id, tool_id, workspace_id=0) -> dict:
    if user_id is None:
        return {"status": "error", "error": "user_id cannot be null"}
    _, jwt_user_id = get_current_user()
    if str(user_id) != str(jwt_user_id):
        return {"status": "error", "error": "Unauthorized: user_id in request does not match user in token"}

    valid, err = validate_user_workspace_access(user_id=user_id, workspace_id=workspace_id)
    if not valid:
        return {"status": "error", "error": err}

    session = db.Session()
    try:
        session.rollback()
        user_tool_fav = session.query(db.FavouriteMappingTool).filter(
            db.FavouriteMappingTool.user_id == user_id,
            db.FavouriteMappingTool.tool_id == tool_id,
            db.FavouriteMappingTool.workspace_id == workspace_id
        ).first()
        if not user_tool_fav:
            new_fav = db.FavouriteMappingTool(
                user_id=user_id,
                tool_id=tool_id,
                workspace_id=workspace_id,
                is_active=True
            )
            session.add(new_fav)
            user_tool_fav = new_fav
            print("Added new tool favourite mapping; is_active set to:", new_fav.is_active)
        else:
            user_tool_fav.is_active = not user_tool_fav.is_active
            print("Toggled tool favourite mapping is_active to:", user_tool_fav.is_active)
        session.commit()
        fav_flag = "favourites" if user_tool_fav.is_active else "not favourites"
        print(f"Tool_id {tool_id} updated to {fav_flag}")
        return {"status": "success", "response": f"Tool_id {tool_id} updated to {fav_flag}"
            }
    except Exception as e:
        session.rollback()
        print(f"Error in update_fav_tool: {e}")
        return {
            "status": "error", 
            "error": "An error occurred while updating favourite tool.",
            "favourite": True if user_tool_fav.is_active else False
            }
    finally:
        session.close()

# @mcp.tool()
# def list_integrations_for_entity(id, type):
#     """
#     List all integrations for a specific agent or tool.
#     Args:
#         id (int or str): The ID of the agent or tool.
#         type (str): 'agent' or 'tool' (case-insensitive)
#     Returns:
#         dict: List of integrations with name, logo, and is_active flag.
#     """
#     # Enforce JWT presence; this is a generic listing but should require an authenticated context
#     request = request_var.get(None)
#     if not request or not hasattr(request.state, "jwt_claims"):
#         return {"error": "Unauthorized: JWT claims not found in request context"}
#     claims = request.state.jwt_claims
#     jwt_user_id = claims.get("user_id") or claims.get("sub")
#     if not jwt_user_id:
#         return {"error": "Unauthorized: user_id not found in token claims"}

#     session = db.Session()
#     try:
#         session.rollback()
#         results = []
#         if str(type).lower() == 'agent':
#             print("Fetching integrations for agent_id:", id)
#             # Join AgentsCMS, AgentCMSIntegrationMap, Integrations
#             query = (
#                 session.query(
#                     db.Integrations.integration_name,
#                     db.Integrations.integration_logo_url,
#                     db.Integrations.is_active
#                 )
#                 .join(AgentCMSIntegrationMap, AgentCMSIntegrationMap.integration_id == db.Integrations.integration_id)
#                 .join(AgentsCMS, AgentsCMS.agent_cms_id == AgentCMSIntegrationMap.agent_cms_id)
#                 .filter(AgentsCMS.agent_id == id)
#             )
#         elif str(type).lower() == 'tool':
#             print("Fetching integrations for tool_id:", id)
#             # Join db.ToolsCMS, ToolCMSIntegrationMap, Integrations
#             query = (
#                 session.query(
#                     db.Integrations.integration_name,
#                     db.Integrations.integration_logo_url,
#                     db.Integrations.is_active
#                 )
#                 .join(db.ToolCMSIntegrationMap, ToolCMSIntegrationMap.integration_id == db.Integrations.integration_id)
#                 .join(db.ToolsCMS, db.ToolsCMS.tool_cms_id == ToolCMSIntegrationMap.tool_cms_id)
#                 .filter(db.ToolsCMS.tool_id == id)
#             )
#         else:
#             return {'error': "Invalid type. Must be 'agent' or 'tool'."}

#         for row in query.all():
#             results.append({
#                 'integration_name': row.integration_name,
#                 'integration_logo': row.integration_logo_url,
#                 'is_active': bool(row.is_active) if row.is_active is not None else False
#             })
#         return {'response': results}
#     except Exception as e:
#         session.rollback()
#         print(f"Error in list_integrations_for_entity: {e}")
#         return {'error': 'An error occurred while fetching integrations.'}
#     finally:
#         session.close()
@mcp.tool()
@require_auth
def list_integrations_for_entity_prev(id, type):
    """
    List all integrations for a specific agent or tool.

    Args:
        id (int or str): The ID of the agent or tool.
        type (str): 'agent' or 'tool' (case-insensitive). If invalid, returns the fixed catalog.

    Returns:
        dict: List with exact frontend fields:
            - id (int)
            - name (str)
            - desc (str)  # exact copies for known integrations
            - connected (bool)
            - logo (str)
    """
    # Fixed catalog per frontend requirement (exact descriptions & logos)
    FIXED_CATALOG = [
        {
            "id": 1,
            "name": "Jira",
            "desc": "Connect your JIRA project to seamlessly sync issues and track progress in real-time. This integration allows your team to stay updated on task status...",
            "connected": False,
            "logo": "./images/insights/jira_core.png",
        },
        {
            "id": 2,
            "name": "Confluence",
            "desc": "Connect to your Confluence workspace to centralize documentation and collaborate effortlessly, ensuring you get real-time update in your agent.",
            "connected": False,
            "logo": "./images/branch-icon.png",
        },
        {
            "id": 3,
            "name": "SharePoint",
            "desc": "Link your SharePoint environment to streamline document management and enable secure, synchronized access in your agent.",
            "connected": False,
            "logo": "./images/sharepoint-logotype.png",
        },
    ]

    # For known names, force exact description and default logos if DB doesn't provide one
    DESC_MAP = {item["name"]: item["desc"] for item in FIXED_CATALOG}
    LOGO_FALLBACK = {item["name"]: item["logo"] for item in FIXED_CATALOG}

    def as_catalog_response(items):
        """Ensure list is returned under 'response' key."""
        return {"response": items}

    # Normalize type
    type_norm = (str(type).strip().lower() if type is not None else "")
    valid_type = type_norm in {"agent", "tool"}

    # If type is invalid (e.g., "199"), return fixed catalog immediately
    if not valid_type:
        print(f"[list_integrations_for_entity] Invalid type '{type}'. Returning fixed catalog.")
        return as_catalog_response(FIXED_CATALOG)

    session = db.Session()
    try:
        session.rollback()

        # Build query based on type
        if type_norm == "agent":
            print("Fetching integrations for agent_id:", id)
            query = (
                session.query(
                    db.Integrations.integration_id,        # id
                    db.Integrations.integration_name,      # name + maps for desc/logo
                    db.Integrations.integration_logo_url,  # logo
                    db.Integrations.is_active              # connected
                )
                .join(db.AgentCMSIntegrationMap, db.AgentCMSIntegrationMap.integration_id == db.Integrations.integration_id)
                .join(db.AgentsCMS, db.AgentsCMS.agent_cms_id == db.AgentCMSIntegrationMap.agent_cms_id)
                .filter(db.AgentsCMS.agent_id == id)
            )
        else:  # type_norm == "tool"
            print("Fetching integrations for tool_id:", id)
            query = (
                session.query(
                    db.Integrations.integration_id,
                    db.Integrations.integration_name,
                    db.Integrations.integration_logo_url,
                    db.Integrations.is_active
                )
                .join(db.ToolCMSIntegrationMap, db.ToolCMSIntegrationMap.integration_id == db.Integrations.integration_id)
                .join(db.ToolsCMS, db.ToolsCMS.tool_cms_id == db.ToolCMSIntegrationMap.tool_cms_id)
                .filter(db.ToolsCMS.tool_id == id)
            )

        rows = query.all()

        # If no rows, return the fixed catalog (frontend-safe)
        if not rows:
            print("[list_integrations_for_entity] No DB rows found. Returning fixed catalog.")
            return as_catalog_response(FIXED_CATALOG)

        results = []
        for idx, row in enumerate(rows, start=1):
            # Row could be a SQLAlchemy row or a tuple—handle both
            try:
                integration_id = getattr(row, "integration_id")
                name_val = getattr(row, "integration_name")
                logo_val = getattr(row, "integration_logo_url")
                is_active_val = getattr(row, "is_active")
            except Exception:
                # tuple fallback
                integration_id = row[0] if len(row) > 0 else None
                name_val = row[1] if len(row) > 1 else "Unknown"
                logo_val = row[2] if len(row) > 2 else ""
                is_active_val = row[3] if len(row) > 3 else False

            # Normalize common name variants for description & logo fallback
            name_norm = (name_val or "").strip()
            if name_norm.upper() == "JIRA":
                name_norm = "Jira"
            elif name_norm.lower() == "confluence":
                name_norm = "Confluence"
            elif name_norm.lower() == "sharepoint":
                name_norm = "SharePoint"

            # --- ONLY CHANGE: strict normalization for connected ---
            # True only for explicit truthy values; everything else -> False
            try:
                if isinstance(is_active_val, bool):
                    connected_status = is_active_val
                else:
                    connected_status = str(is_active_val).strip().lower() in ("true", "1", "yes", "y")
            except Exception:
                connected_status = False
            # -------------------------------------------------------

            # Prepare response item with exact keys
            item = {
                "id": int(integration_id) if integration_id is not None else idx,
                "name": name_norm or "Unknown",
                "desc": DESC_MAP.get(name_norm, ""),  # exact description for known names, else empty
                "connected": connected_status,
                "logo": (logo_val or "").strip() or LOGO_FALLBACK.get(name_norm, ""),
            }
            results.append(item)

        return as_catalog_response(results)

    except Exception as e:
        session.rollback()
        print(f"Error in list_integrations_for_entity: {e}")
        # As a resilience measure, still return the fixed catalog rather than an error shape,
        # so the frontend gets expected keys.
        return as_catalog_response(FIXED_CATALOG)
    finally:
        session.close()

@mcp.tool()
@require_auth
def list_integrations_for_entity(id: int, type: str, workspace_id: int = None, user_id: str = None):
    """
    List all integrations for a specific agent or tool, user-specific and workspace-specific.
    Args:
        id (int or str): The ID of the agent or tool.
        type (str): 'agent' or 'tool' (case-insensitive)
        workspace_id (int, optional): Workspace ID for context
        user_id (int, optional): User ID for context
    Returns:
        dict: List of integrations with name, logo, and is_active flag.
    """
    _, jwt_user_id = get_current_user()

    # Use provided user_id/workspace_id if given, else fallback to JWT
    user_id = user_id or jwt_user_id

    session = db.Session()
    try:
        session.rollback()
        results = []
        if str(type).lower() == 'agent':
            print("Fetching integrations for agent_id (user/workspace specific):", id)
            if db.AMUIntegrationMapping and user_id and workspace_id:
                # Get all integrations for this agent (from Integrations table)
                all_integrations = (
                    session.query(
                        db.Integrations.integration_id,
                        db.Integrations.integration_name,
                        db.Integrations.integration_logo_url,
                        db.Integrations.integration_desc
                    )
                    .join(db.AgentCMSIntegrationMap, db.AgentCMSIntegrationMap.integration_id == db.Integrations.integration_id)
                    .join(db.AgentsCMS, db.AgentsCMS.agent_cms_id == db.AgentCMSIntegrationMap.agent_cms_id)
                    .filter(db.AgentsCMS.agent_id == id)
                ).all()
                # For each integration, check if mapping exists for user/workspace/agent/integration
                for integ in all_integrations:
                    mapping = session.query(db.AMUIntegrationMapping).filter(
                        db.AMUIntegrationMapping.agent_id == id,
                        db.AMUIntegrationMapping.user_id == user_id,
                        db.AMUIntegrationMapping.workspace_id == workspace_id,
                        db.AMUIntegrationMapping.integration_id == integ.integration_id
                    ).first()
                    connected = bool(mapping.connected) if mapping and hasattr(mapping, 'connected') else False
                    results.append({
                        'id' : integ.integration_id,
                        'name': integ.integration_name,
                        'logo': integ.integration_logo_url,
                        'desc': integ.integration_desc,
                        'connected': connected
                    })
            else:
                query = (
                    session.query(
                        db.Integrations.integration_name,
                        db.Integrations.integration_logo_url,
                        db.Integrations.integration_desc
                    )
                    .join(db.AgentCMSIntegrationMap, db.AgentCMSIntegrationMap.integration_id == db.Integrations.integration_id)
                    .join(db.AgentsCMS, db.AgentsCMS.agent_cms_id == db.AgentCMSIntegrationMap.agent_cms_id)
                    .filter(db.AgentsCMS.agent_id == id)
                )
                for row in query.all():
                    results.append({
                        'id' : row.integration_id,
                        'name': row.integration_name,
                        'logo': row.integration_logo_url,
                        'desc': row.integration_desc,
                        'connected': False
                    })
        elif str(type).lower() == 'tool':
            print("Fetching integrations for tool_id (user/workspace specific):", id)
            if db.TMUIntegrationMapping and user_id and workspace_id:
                all_integrations = (
                    session.query(
                        db.Integrations.integration_id,
                        db.Integrations.integration_name,
                        db.Integrations.integration_logo_url,
                        db.Integrations.integration_desc
                    )
                    .join(db.ToolCMSIntegrationMap, db.ToolCMSIntegrationMap.integration_id == db.Integrations.integration_id)
                    .join(db.ToolsCMS, db.ToolsCMS.tool_cms_id == db.ToolCMSIntegrationMap.tool_cms_id)
                    .filter(db.ToolsCMS.tool_id == id)
                ).all()
                for integ in all_integrations:
                    mapping = session.query(db.TMUIntegrationMapping).filter(
                        db.TMUIntegrationMapping.tool_id == id,
                        db.TMUIntegrationMapping.user_id == user_id,
                        db.TMUIntegrationMapping.workspace_id == workspace_id,
                        db.TMUIntegrationMapping.integration_id == integ.integration_id
                    ).first()
                    connected = bool(mapping.connected) if mapping and hasattr(mapping, 'connected') else False
                    results.append({
                        'id' : integ.integration_id,
                        'name': integ.integration_name,
                        'logo': integ.integration_logo_url,
                        'desc': integ.integration_desc,
                        'connected': connected
                    })
            else:
                query = (
                    session.query(
                        db.Integrations.integration_name,
                        db.Integrations.integration_logo_url,
                        db.Integrations.integration_desc
                    )
                    .join(db.ToolCMSIntegrationMap, db.ToolCMSIntegrationMap.integration_id == db.Integrations.integration_id)
                    .join(db.ToolsCMS, db.ToolsCMS.tool_cms_id == db.ToolCMSIntegrationMap.tool_cms_id)
                    .filter(db.ToolsCMS.tool_id == id)
                )
                for row in query.all():
                    results.append({
                        'id' : row.integration_id,
                        'name': row.integration_name,
                        'logo': row.integration_logo_url,
                        'desc': row.integration_desc,
                        'connected': False
                    })
        else:
            return {'error': "Invalid type. Must be 'agent' or 'tool'."}

        return {'response': results}
    except Exception as e:
        session.rollback()
        print(f"Error in list_integrations_for_entity: {e}")
        return {'error': 'An error occurred while fetching integrations.'}
    finally:
        session.close()

@mcp.tool()
@require_auth_async
async def toggle_integration_connection(user_id, workspace_id ,integration_id, id , type) -> dict:
    if user_id is None:
        return {"status": "error", "error": "user_id cannot be null"}
    _, jwt_user_id = get_current_user()
    if str(user_id) != str(jwt_user_id):
        return {"status": "error", "error": "Unauthorized: user_id in request does not match user in token"}

    valid, err = validate_user_workspace_access(user_id=user_id, workspace_id=workspace_id)
    if not valid:
        return {"status": "error", "error": err}

    session = db.Session()
    try:
        session.rollback()
        
        if type == "agent":
            user_connection = session.query(db.AMUIntegrationMapping).filter(
                db.AMUIntegrationMapping.user_id == user_id,
                db.AMUIntegrationMapping.agent_id == id,
                db.AMUIntegrationMapping.workspace_id == workspace_id,
                db.AMUIntegrationMapping.integration_id == integration_id
            ).first()
            if not user_connection:
                new_connection = db.AMUIntegrationMapping(
                    user_id=user_id,
                    agent_id=id,
                    workspace_id=workspace_id,
                    integration_id = integration_id,
                    connected=True
                )
                session.add(new_connection)
                user_agent_fav = new_connection
                print("Added new connection mapping; connection set to:", new_connection.connected)
            else:
                user_connection.connected = not user_connection.connected
                print("Toggled connection mapping connected to:", user_connection.connected)
            session.commit()
        else : # type == tool
            user_connection = session.query(db.TMUIntegrationMapping).filter(
                db.TMUIntegrationMapping.user_id == user_id,
                db.TMUIntegrationMapping.agent_id == id,
                db.TMUIntegrationMapping.workspace_id == workspace_id,
                db.TMUIntegrationMapping.integration_id == integration_id
            ).first()
            if not user_connection:
                new_connection = db.TMUIntegrationMapping(
                    user_id=user_id,
                    agent_id=id,
                    workspace_id=workspace_id,
                    integration_id = integration_id,
                    connected=True
                )
                session.add(new_connection)
                user_agent_fav = new_connection
                print("Added new connection mapping; connection set to:", new_connection.connected)
            else:
                user_connection.connected = not user_connection.connected
                print("Toggled connection mapping connected to:", user_connection.connected)
            session.commit()
        if not user_connection:
            user_connection = new_connection
        connection_flag = "connected" if user_connection.connected else "disconnected"
        print(f"Integration_id {integration_id} updated to {connection_flag}")
        return {
            "status": "success", 
            "response": f"Integration_id {integration_id} updated to {connection_flag}",
            "connected": user_connection.connected
            }
    except Exception as e:
        session.rollback()
        print(f"Error in updating connection status: {e}")
        return {"status": "error", "error": "An error occurred while updating connection status."}
    finally:
        session.close()

@mcp.tool()
@require_auth_async
async def fetch_specific_agent_info(user_id, agent_id, workspace_id=0) -> dict:
    """
    Fetches detailed information about a specific agent for a given user.
    """
    if user_id is None:
        return {"status": "error", "error": "user_id cannot be null"}
    _, jwt_user_id = get_current_user()
    if str(user_id) != str(jwt_user_id):
        return {"status": "error", "error": "Unauthorized: user_id in request does not match user in token"}

    valid, err = validate_user_workspace_access(user_id=user_id, workspace_id=workspace_id)
    if not valid:
        return {"status": "error", "error": err}

    session = db.Session()
    try:
        agent_values = session.query(
            db.Agent.agent_name,
            db.Agent.agent_id,
            db.Agent.agent_desc,
            db.AgentsCMS.agent_owner,
            db.AgentsCMS.agent_contact,
            db.AgentsCMS.agent_feature,
            db.AgentsCMS.faqs,
            db.AgentsCMS.last_updated,
            db.Integrations.integration_id,
            db.Integrations.integration_name,
            db.Integrations.integration_logo_url,
            db.FavouriteMappingAgent.favourite_id,
            db.FavouriteMappingAgent.is_active
        ).filter(
            db.Agent.agent_id == agent_id
        ).outerjoin(
            db.AgentsCMS, db.AgentsCMS.agent_id == db.Agent.agent_id
        ).outerjoin(
            db.AgentCMSIntegrationMap, db.AgentCMSIntegrationMap.agent_cms_id == db.AgentsCMS.agent_cms_id
        ).outerjoin(
            db.Integrations, db.Integrations.integration_id == db.AgentCMSIntegrationMap.integration_id
        ).outerjoin(
            db.FavouriteMappingAgent,
            (db.FavouriteMappingAgent.agent_id == db.Agent.agent_id) &
            (db.FavouriteMappingAgent.user_id == user_id) &
            (db.FavouriteMappingAgent.workspace_id == workspace_id)
        ).all()

        if not agent_values:
            return {"status": "success", "message": f"No agent found with agent_id {agent_id}"}

        # Extract favourite status - need to check all rows since joins create multiple rows
        # Find the first row that has a favourite_id (not None)
        favourite_status = False
        for row in agent_values:
            if row.favourite_id is not None:
                favourite_status = bool(row.is_active)
                break

        # Fetch workspaces where this agent exists
        workspace_maps = session.query(db.AgentMap).filter(
            db.AgentMap.agent_id == agent_id, db.AgentMap.is_active == True
        ).all()
        workspace_ids = [wm.workspace_id for wm in workspace_maps]
        workspaces = []
        if workspace_ids:
            ws_query = session.query(db.Workspace).filter(
                db.Workspace.workspace_id.in_(workspace_ids), db.Workspace.is_active == True
            )
            for ws in ws_query:
                workspaces.append({
                    "workspace_id": ws.workspace_id,
                    "workspace_name": ws.workspace_name
                })

        agent_info = {
            "status": "success",
            "agent_id": agent_id,
            "agent_name": agent_values[0].agent_name if agent_values else None,
            "description": agent_values[0].agent_desc if agent_values else None,
            "favourite": favourite_status,
            "type": 'Agent',
            "cms_info": {
                "contact": agent_values[0].agent_contact if agent_values else None,
                "agentOwner": agent_values[0].agent_owner if agent_values else None,
                "lastUpdated": agent_values[0].last_updated if agent_values else None,
                "faqs": agent_values[0].faqs if agent_values else None,
                "features": agent_values[0].agent_feature if agent_values else None,
            },
            "Integrations": [
                {
                    "id": value.integration_id,
                    "name": value.integration_name,
                    "icon": value.integration_logo_url
                } for value in agent_values if value.integration_name and value.integration_logo_url and value.integration_id
            ],
            "related_tools": [],
            "workspaces": workspaces
        }
        return agent_info
    except Exception as e:
        session.rollback()
        print(f"Error in fetch_specific_agent_info: {e}")
        return {'error': 'An error occurred while fetching agent info.'}
    finally:
        session.close()

@mcp.tool()
@require_auth_async
async def fetch_specific_tool_info(user_id, tool_id, workspace_id=0) -> dict:
    """
    Fetches detailed information about a specific tool for a given user.
    """
    if user_id is None:
        return {"status": "error", "error": "user_id cannot be null"}
    _, jwt_user_id = get_current_user()
    if str(user_id) != str(jwt_user_id):
        return {"status": "error", "error": "Unauthorized: user_id in request does not match user in token"}

    valid, err = validate_user_workspace_access(user_id=user_id, workspace_id=workspace_id)
    if not valid:
        return {"status": "error", "error": err}

    session = db.Session()
    try:
        tool_values = session.query(
            db.Tool.tool_name,
            db.Tool.tool_id,
            db.Tool.tool_desc,
            db.ToolsCMS.tool_owner,
            db.ToolsCMS.tool_contact,
            db.ToolsCMS.tool_feature,
            db.ToolsCMS.faqs,
            db.ToolsCMS.last_updated,
            db.Integrations.integration_id,
            db.Integrations.integration_name,
            db.Integrations.integration_logo_url,
            db.FavouriteMappingTool.favourite_id
        ).filter(
            db.Tool.tool_id == tool_id
        ).outerjoin(
            db.ToolsCMS, db.ToolsCMS.tool_id == db.Tool.tool_id
        ).outerjoin(
            db.ToolCMSIntegrationMap, db.ToolCMSIntegrationMap.tool_cms_id == db.ToolsCMS.tool_cms_id
        ).outerjoin(
            db.Integrations, db.Integrations.integration_id == db.ToolCMSIntegrationMap.integration_id
        ).outerjoin(
            db.FavouriteMappingTool,
            (db.FavouriteMappingTool.user_id == user_id) &
            (db.FavouriteMappingTool.tool_id == tool_id) &
            (db.FavouriteMappingTool.workspace_id == workspace_id)
        ).all()

        workspace_maps = session.query(db.ToolMap).filter(
            db.ToolMap.tool_id == tool_id, db.ToolMap.is_active == True
        ).all()
        workspace_ids = [wm.workspace_id for wm in workspace_maps]
        workspaces = []
        if workspace_ids:
            ws_query = session.query(db.Workspace).filter(
                db.Workspace.workspace_id.in_(workspace_ids), db.Workspace.is_active == True
            )
            for ws in ws_query:
                workspaces.append({
                    "workspace_id": ws.workspace_id,
                    "workspace_name": ws.workspace_name
                })

        if not tool_values:
            return {"status": "success", "message": f"No tool found with tool_id {tool_id}"}

        tool_info = {
            "status": "success",
            "tool_id": tool_id,
            "tool_name": tool_values[0].tool_name if tool_values else None,  # fixed
            "description": tool_values[0].tool_desc if tool_values else None,
            "favourite": bool(tool_values[0].favourite_id) if tool_values and tool_values[0].favourite_id is not None else False,
            "type": 'Tool',
            "cms_info": {
                "contact": tool_values[0].tool_contact if tool_values else None,
                "agentOwner": tool_values[0].tool_owner if tool_values else None,
                "lastUpdated": tool_values[0].last_updated if tool_values else None,
                "faqs": tool_values[0].faqs if tool_values else None,
                "features": tool_values[0].tool_feature if tool_values else None,
            },
            "Integrations": [
                {
                    "id": value.integration_id,
                    "name": value.integration_name,
                    "icon": value.integration_logo_url
                } for value in tool_values if value.integration_name and value.integration_logo_url and value.integration_id
            ],
            "related_tools": [],
            "workspaces": workspaces
        }

        return tool_info
    except Exception as e:
        session.rollback()
        print(f"Error in fetch_specific_tool_info: {e}")
        return {'error': 'An error occurred while fetching tool info.'}
    finally:
        session.close()

def _get_sdlc_role_ids() -> list[int]:
    excluded_ids = {Role.ADMIN.id, Role.USER.id, Role.WS_ADMIN.id, Role.WS_MANAGER.id}
    return [role.id for role in Role if role.id not in excluded_ids]


def _get_active_workspace_role_id(session, user_id: int, workspace_id: int):
    role_map = (
        session.query(db.UserMap)
        .filter(
            db.UserMap.user_id == user_id,
            db.UserMap.workspace_id == workspace_id,
            db.UserMap.is_active == True,
        )
        .first()
    )
    return getattr(role_map, 'role_id', None) if role_map else None


def _get_assignable_role_ids(session, user_id: int, workspace_id: int):
    sdlc_role_ids = set(_get_sdlc_role_ids())

    caller_role_id = _get_active_workspace_role_id(session, user_id, workspace_id)
    if caller_role_id == Role.WS_ADMIN.id:
        return {Role.WS_ADMIN.id, Role.WS_MANAGER.id, *sdlc_role_ids}, caller_role_id
    if caller_role_id == Role.WS_MANAGER.id:
        return {Role.WS_MANAGER.id, *sdlc_role_ids}, caller_role_id
    return set(), caller_role_id


@mcp.tool()
@require_auth
def fetch_addable_roles_by_workspace(workspace_id: int):
    """
    Return roles the current user is allowed to add in the given workspace.

    Rules:
    - Workspace Admin can add Workspace Admin, Workspace Manager, and SDLC roles.
    - Workspace Manager can add Workspace Manager and SDLC roles.
    - SDLC roles cannot add users.
    - Platform Admin can add Workspace Admin, Workspace Manager, and SDLC roles.
    """
    _, jwt_user_id = get_current_user()

    session = db.Session()
    try:
        session.rollback()
        assignable_role_ids, caller_workspace_role_id = _get_assignable_role_ids(
            session=session,
            user_id=jwt_user_id,
            workspace_id=workspace_id,
        )

        if not assignable_role_ids:
            return {
                "error": (
                    "You are not authorized to add users in this workspace. "
                    "Only Workspace Admin or Workspace Manager can add users."
                )
            }

        # Use Role enum to build the result
        result = [
            {"role_id": role.id, "role_name": role.name}
            for role in Role
            if role.id in assignable_role_ids
        ]

        return {
            "workspace_id": workspace_id,
            "caller_workspace_role_id": caller_workspace_role_id,
            "response": sorted(result, key=lambda r: (str(r["role_name"] or "").lower(), r["role_id"] or 0)),
        }
    except Exception as e:
        session.rollback()
        print(f"Error in fetch_addable_roles_by_workspace: {e}")
        return {"error": "An error occurred while fetching addable roles."}
    finally:
        session.close()


# TODO: Will Deprecate it soon. Alternate tool added (fetch_addable_roles_by_workspace)
@mcp.tool()
@require_auth
def fetch_roles_list():
    """
    Fetch a list of roles (ids and name) from the role_master table.
    Only returns SDLC roles - filters out system/admin roles that should not be assignable to workspace users.
    Returns:
        dict: { 'response': [ { 'role_id': ..., 'role_name': ... }, ... ] }
    """
    session = db.Session()
    try:
        session.rollback()
        roles = session.query(db.Role).filter(db.Role.is_active == True).all()
        
        # Define restricted roles that should NEVER be selectable when adding users to workspace
        # Only SDLC roles (Product Owner, Scrum Master, Developer, QA, etc.) should be visible
        restricted_roles = {
            "forge-x admin"
        }
        
        result = []
        for role in roles:
            role_name = getattr(role, 'role_name', '').strip().lower()
            original_role_name = getattr(role, 'role_name', None)
            # Filter out all restricted roles - only show SDLC roles
            if role_name not in restricted_roles:
                result.append({
                    'role_id': getattr(role, 'role_id', None),
                    'role_name': original_role_name
                })
                print(f"[DEBUG] Including role: {original_role_name} (normalized: {role_name})")
            else:
                print(f"[DEBUG] Filtering out restricted role: {original_role_name} (normalized: {role_name})")
        print(f"[DEBUG] Total roles returned: {len(result)} out of {len(roles)} total roles")
        return {'response': result}
    except Exception as e:
        session.rollback()
        print(f"Error in fetch_roles_list: {e}")
        return {'error': 'An error occurred while fetching roles.'}
    finally:
        session.close()


@mcp.tool()
@require_auth
def add_user_to_workspace(workspace_id: int, user_email: str, role_id: int, first_name: str = None, last_name: str = None):
    """
    Add a user to a workspace by email and assign a role by role_id.
    If user does not exist, create new user. Only allow @coforge.com emails.
    Only Workspace Admin or Workspace Manager can add users.
    Cannot assign restricted roles (Forge-X admin, Workspace admin, SME, Knowledge Curator, DoD).
    Args:
        workspace_id (int): Workspace ID
        user_email (str): User's email address
        role_id (int): Role ID to assign (must be an SDLC role)
        first_name (str): First name (for new user)
        last_name (str): Last name (for new user)
    Returns:
        dict: Success or error message and notification
    """
    _, jwt_user_id = get_current_user()
    session = db.Session()
    try:
        session.rollback()
        try:
            role_id = int(role_id)
        except (TypeError, ValueError):
            return {"error": "role_id must be a valid integer."}

        # Email validation
        email = user_email.strip().lower()
        if not email.endswith(DefaultValue.EMAIL_ENDS_WITH_COFORGE.value):
            return {"error": f"Only {DefaultValue.EMAIL_ENDS_WITH_COFORGE.value} email addresses are allowed."}

        assignable_role_ids, _ = _get_assignable_role_ids(
            session=session,
            user_id=jwt_user_id,
            workspace_id=workspace_id,
        )
        if not assignable_role_ids:
            return {
                "error": (
                    "You are not authorized to add users to this workspace. "
                    "Only Workspace Admin or Workspace Manager can add users."
                )
            }

        if role_id not in assignable_role_ids:
            return {
                "error": (
                    f"You cannot assign '{Role.get_by_id(role_id)}' role in this workspace. "
                )
            }

        # # Validate target role exists and is active
        # role = session.query(db.Role).filter(db.Role.role_id == role_id, db.Role.is_active == True).first()
        # if not role:
        #     return {"error": f"Role with id '{role_id}' not found"}

        user = session.query(db.User).filter(func.lower(db.User.email_id) == email).first()
        print(f"[DEBUG] Checking if user exists for email: {email} -> Found: {user is not None}")
        
        notification = ""
        if user:
            print(f"[DEBUG] User already exists: user_id={user.user_id}, email={user.email_id}")
            user_id = user.user_id
            notification = f"User already exists and added to workspace."
        else:
            # Create new user
            if not first_name or not last_name:
                return {"error": "First name and last name required for new user."}
            print(f"[DEBUG] Creating new user with email: {email}")
            new_user = db.User(
                namespace="default",  # Set default namespace
                email_id=email,
                first_name=first_name,
                last_name=last_name,
                is_active=True,
                role_id=1
            )
            session.add(new_user)
            session.flush()  # Get new user_id
            user_id = new_user.user_id
            notification = f"New user created and added to workspace."

        # Add to workspace_users_mapping
        user_map = session.query(db.UserMap).filter(db.UserMap.user_id==user_id, db.UserMap.workspace_id==workspace_id).first()
        workspace = session.query(db.Workspace).filter(db.Workspace.workspace_id == workspace_id).first()
        if user_map and (not user_map.is_active):
            user_map.is_active = True
            user_map.role_id = role_id
        elif not user_map:
            session.add(
                db.UserMap(
                    user_id=user_id,
                    workspace_id=workspace_id,
                    is_active=True,
                    role_id=role_id,
                    namespace=workspace.namespace,
                    created_date=datetime.now(timezone.utc),
                    last_updated=datetime.now(timezone.utc),
                )
            )

        # DEPRECATED: UserRoleMap table is redundant - UserMap already contains role_id and all related fields
        # add/update user_role_mapping
        # user_role_map = session.query(db.UserRoleMap).filter(db.UserRoleMap.user_id==user_id, db.UserRoleMap.workspace_id==workspace_id).first()
        # if user_role_map and (not user_role_map.is_active):
        #     user_role_map.role_id = role_id
        #     user_role_map.is_active = True
        # elif not user_role_map:
        #     session.add(db.UserRoleMap(user_id=user_id, workspace_id=workspace_id, role_id=role_id, is_active=True, created_date = datetime.now(timezone.utc), last_updated=datetime.now(timezone.utc), namespace=workspace.namespace))

        session.commit()
        return {"response": notification, "user_id": user_id, "email": email, "workspace_id": workspace_id, "role_id": role_id}
    except Exception as e:
        session.rollback()
        print(f"Error in add_user_to_workspace: {e}")
        return {"error": "An error occurred while adding user to workspace."}
    finally:
        session.close()

@mcp.tool()
@require_auth
def list_workspace_users(workspace_id: int):
    session = db.Session()
    try:
        session.rollback()
        # Single optimized query: join UserMap, User, Role (NOTE: UserRoleMap is DEPRECATED/redundant)
        users = (
            session.query(
                db.User.user_id,
                db.User.first_name,
                db.User.last_name,
                db.User.email_id,
                db.Role.role_name,
                db.Role.role_id,
                db.UserMap.can_curate_kb
            )
            .join(db.UserMap, 
                (db.UserMap.user_id == db.User.user_id) & 
                (db.UserMap.workspace_id == workspace_id) & 
                (db.UserMap.is_active == True)
            )
            .outerjoin(db.Role, 
                (db.UserMap.role_id == db.Role.role_id) & 
                (db.Role.is_active == True)
            )
            .filter(db.User.is_active == True)
        ).all()
        result = [
            {
                'user_id': user_id,
                'name': (first_name or '') + ' ' + (last_name or ''),
                'email_id': email_id,
                'role': role_name,
                'role_id': role_id,
                'can_curate_kb': can_curate_kb if can_curate_kb is not None else False
            }
            for user_id, first_name, last_name, email_id, role_name, role_id, can_curate_kb in users
        ]
        return {'response': result}
    except Exception as e:
        session.rollback()
        print(f"Error in list_workspace_users: {e}")
        return {'error': 'An error occurred while fetching workspace users.'}
    finally:
        session.close()

@mcp.tool()
@require_auth
def remove_user_from_workspace(user_id: int, workspace_id: int):
    """
    Remove a user from a workspace (set is_active = False in mapping tables).
    Only Workspace Admin or Workspace Manager can remove users.
    Args:
        user_id (int): User ID
        workspace_id (int): Workspace ID
    Returns:
        dict: Success or error message
    """
    if user_id is None:
        return {"status": "error", "error": "user_id cannot be null"}
    
    _, jwt_user_id = get_current_user()
    session = db.Session()
    try:
        session.rollback()

        caller_role_id = _get_active_workspace_role_id(
            session=session,
            user_id=jwt_user_id,
            workspace_id=workspace_id,
        )
        if caller_role_id not in {Role.WS_ADMIN.id, Role.WS_MANAGER.id}:
            return {
                "error": (
                    "You are not authorized to remove users from this workspace. "
                    "You must be a Workspace Admin or Workspace Manager."
                )
            }

        user_map = session.query(db.UserMap).filter(
            db.UserMap.user_id == user_id,
            db.UserMap.workspace_id == workspace_id,
            db.UserMap.is_active == True,
        ).first()
        if not user_map:
            return {"error": "User does not exist in workspace."}

        target_role_id = getattr(user_map, "role_id", None)
        if target_role_id == Role.WS_ADMIN.id and caller_role_id == Role.WS_MANAGER.id:
            return {"error": "Workspace Manager cannot remove admin users from this workspace."}

        user_map.is_active = False
        session.commit()
        return {'response': 'User removed from the workspace'}
    except Exception as e:
        session.rollback()
        print(f"Error in remove_user_from_workspace: {e}")
        return {'error': 'An error occurred while removing user from workspace.'}
    finally:
        session.close()

@mcp.tool()
@require_auth
def update_workspace_user(user_id: int, workspace_id: int, role_id: int, first_name: str = None, last_name: str = None):
    """
    Update a user's role in a workspace by role_id.
    Only Workspace Admin or Workspace Manager can update user roles.
    Cannot assign roles that the caller is not authorized to assign.
    Args:
        user_id (int): User ID
        workspace_id (int): Workspace ID
        role_id (int): Role ID to assign
        first_name (str, optional): First name to update
        last_name (str, optional): Last name to update
    Returns:
        dict: Success or error message
    """
    if user_id is None:
        return {"status": "error", "error": "user_id cannot be null"}
    
    _, jwt_user_id = get_current_user()
    session = db.Session()
    try:
        session.rollback()
        
        # Validate role_id
        try:
            role_id = int(role_id)
        except (TypeError, ValueError):
            return {"error": "role_id must be a valid integer."}
        
        # Check authorization using helper function
        assignable_role_ids, caller_role_id = _get_assignable_role_ids(
            session=session,
            user_id=jwt_user_id,
            workspace_id=workspace_id,
        )
        if not assignable_role_ids:
            return {
                "error": (
                    "You are not authorized to update users in this workspace. "
                    "Only Workspace Admin or Workspace Manager can update users."
                )
            }

        if role_id not in assignable_role_ids:
            return {
                "error": (
                    f"You cannot assign '{Role.get_by_id(role_id)}' role in this workspace."
                )
            }

        # Validate target role exists and is active
        # role = session.query(db.Role).filter(db.Role.role_id == role_id, db.Role.is_active == True).first()
        # if not role:
        #     return {"error": f"Role with id '{role_id}' not found"}
        
        # Update or create user workspace mapping
        # NOTE: UserRoleMap table is DEPRECATED - all role info is in UserMap
        workspace_user_map = session.query(db.UserMap).filter_by(user_id=user_id, workspace_id=workspace_id).first()
        if workspace_user_map:
            workspace_user_map.role_id = role_id
            workspace_user_map.is_active = True
            workspace_user_map.last_updated = datetime.now(timezone.utc)
        else:
            workspace = session.query(db.Workspace).filter(db.Workspace.workspace_id == workspace_id).first()
            # DEPRECATED: No longer using UserRoleMap - UserMap contains role_id
            # session.add(db.UserRoleMap(...))
            session.add(db.UserMap(
                user_id=user_id,
                workspace_id=workspace_id,
                role_id=role_id,
                is_active=True,
                namespace=workspace.namespace if workspace else "default",
                created_date=datetime.now(timezone.utc),
                last_updated=datetime.now(timezone.utc)
            ))
        
        # Update can_curate_kb, first_name, last_name if provided
        import inspect
        frame = inspect.currentframe()
        _, _, _, values = inspect.getargvalues(frame)

        can_curate_kb = values.get('can_curate_kb', None)
        first_name = values.get('first_name', None)
        last_name = values.get('last_name', None)
        
        user_map = session.query(db.UserMap).filter_by(user_id=user_id, workspace_id=workspace_id).first()
        if caller_role_id == Role.WS_ADMIN.id and can_curate_kb is not None and user_map and hasattr(user_map, 'can_curate_kb'):
            user_map.can_curate_kb = can_curate_kb
        
        user_obj = session.query(db.User).filter_by(user_id=user_id).first()
        if first_name is not None and user_obj and hasattr(user_obj, 'first_name'):
            user_obj.first_name = first_name
        if last_name is not None and user_obj and hasattr(user_obj, 'last_name'):
            user_obj.last_name = last_name
        
        session.commit()
        print(f'User {user_id} role updated to role id {role_id} in workspace {workspace_id}')
        return {'response': 'User role updated successfully'}
    except Exception as e:
        session.rollback()
        print(f"Error in update_workspace_user: {e}")
        return {'error': 'An error occurred while updating user role.'}
    finally:
        session.close()

@mcp.tool()
@require_auth
def fetch_industry_info():
    """
    Fetch all industries and their subindustries.
    Returns:
        dict: {
            'response': [
                {
                    'industry_id': ...,
                    'industry_name': ...,
                    'subindustry': [
                        {'subindustry_id': ..., 'subindustry_name': ...},
                        ...
                    ]
                },
                ...
            ]
        }
    """
    session = db.Session()
    try:
        session.rollback()
        # industries = session.query(Industry).all()
        # subindustries = session.query(SubIndustry).all()
        industries = session.query(db.Industry).filter(db.Industry.is_active == True).all()
        subindustries = session.query(db.SubIndustry).filter(db.SubIndustry.is_active == True).all()

        # Build mapping: industry_id -> list of subindustries
        sub_map = {}
        for sub in subindustries:
            sid = getattr(sub, 'subindustry_id', None)
            sname = getattr(sub, 'subindustry_name', None)
            iid = getattr(sub, 'industry_id', None)
            if iid not in sub_map:
                sub_map[iid] = []
            sub_map[iid].append({'subindustry_id': sid, 'subindustry_name': sname})

        result = []
        for ind in industries:
            iid = getattr(ind, 'industry_id', None)
            iname = getattr(ind, 'industry_name', None)
            result.append({
                'industry_id': iid,
                'industry_name': iname,
                'subindustry': sub_map.get(iid, [])
            })
        return {'response': result}
    except Exception as e:
        session.rollback()
        print(f"Error in fetch_industry_info: {e}")
        return {'error': 'An error occurred while fetching industry info.'}
    finally:
        session.close()

@mcp.tool()
def logout_user(access_token: str | None = None, refresh_token: str | None = None):
    """
    Logout: revoke current access token and refresh token.
    Also signals middleware to clear the refresh token cookie.
    Args:
        access_token (str | None): Access token to revoke. If not provided, will try to extract from Authorization header.
        refresh_token (str | None): Refresh token to revoke. If not provided, will try to extract from cookies.
    Returns:
        {
          'success': True,
          'revoked': {'access': bool, 'refresh': bool},
          'message': '...'
        }
    """
    request = request_var.get(None)
    errors = []

    # Access token from parameter or Authorization header
    if not access_token:
        try:
            if request and hasattr(request, "headers"):
                access_token = extract_token_from_headers(getattr(request, "headers", {})) or None
        except Exception as e:
            print(f"[ERROR] Failed to extract access token from headers: {e}")
            access_token = None

    # Refresh token from parameter or cookies
    if not refresh_token:
        try:
            if request and hasattr(request, "cookies"):
                refresh_token = request.cookies.get("refresh_token")
        except Exception as e:
            print(f"[ERROR] Failed to extract refresh token from cookies: {e}")
            refresh_token = None

    # Check if we have any tokens to revoke
    if not access_token and not refresh_token:
        return {
            "success": True,
            "revoked": {"access": False, "refresh": False},
            "message": "No tokens found to revoke (already logged out)."
        }

    revoked_access = False
    revoked_refresh = False
    access_error = None
    refresh_error = None

    # Revoke access token if present
    if access_token:
        try:
            print(f"[DEBUG] Attempting to revoke access token...")
            ok, error_msg = revoke_token(access_token)
            revoked_access = bool(ok)
            if not ok:
                access_error = error_msg or "Unknown error"
                print(f"[ERROR] Failed to revoke access token: {access_error}")
                errors.append(f"Access token: {access_error}")
            else:
                print("[SUCCESS] Access token revoked successfully")
        except Exception as e:
            access_error = str(e)
            print(f"[ERROR] Exception while revoking access token: {e}")
            errors.append(f"Access token exception: {access_error}")
            revoked_access = False

    # Revoke refresh token if present
    if refresh_token:
        try:
            print(f"[DEBUG] Attempting to revoke refresh token...")
            ok, error_msg = revoke_token(refresh_token)
            revoked_refresh = bool(ok)
            if not ok:
                refresh_error = error_msg or "Unknown error"
                print(f"[ERROR] Failed to revoke refresh token: {refresh_error}")
                errors.append(f"Refresh token: {refresh_error}")
            else:
                print("[SUCCESS] Refresh token revoked successfully")
        except Exception as e:
            refresh_error = str(e)
            print(f"[ERROR] Exception while revoking refresh token: {e}")
            errors.append(f"Refresh token exception: {refresh_error}")
            revoked_refresh = False

    # Signal middleware to clear refresh cookie on response
    try:
        if request and hasattr(request, "state"):
            request.state.refresh_token = ""
            request.state.refresh_token_expires = 0
            request.state.clear_refresh_cookie = True
            print("[DEBUG] Middleware signaled to clear refresh cookie")
    except Exception as e:
        print(f"[ERROR] Failed to signal middleware to clear cookie: {e}")

    # Determine overall success
    attempted_access = access_token is not None
    attempted_refresh = refresh_token is not None

    # Success if all attempted revocations succeeded
    all_succeeded = (not attempted_access or revoked_access) and (not attempted_refresh or revoked_refresh)
    response = {
        "success": all_succeeded,
        "revoked": {"access": revoked_access, "refresh": revoked_refresh},
        "message": "Logged out successfully" if all_succeeded else "Logout completed with errors"
    }

    if errors:
        response["errors"] = errors

    return response

@mcp.tool()
@require_auth
def check_user_presence_by_email(user_email: str, workspace_id: int):
    """
    Check whether a user (by email) exists in the users table and whether they
    are already present (active mapping) in the given workspace.

    RBAC:
        - Allowed for Forge-X Admin OR Workspace Admin of the target workspace.

    Args:
        user_email (str): Email address to look up.
        workspace_id (int): Target workspace to check membership against.

    Returns:
        dict:
            On success:
                {
                    "response": "<message>",
                    "is_flag": <bool>,          # see below
                    "user_id": <int or None>,
                    "present_in_user_table": <bool>,
                    "present_in_workspace": <bool>
                }

            Messages / flags:
                - If user NOT found in users table:
                    response: "User does not exist in the user table."
                    is_flag: False

                - If user found in users table BUT not in workspace:
                    response: "User not present in the workspace but the user exist."
                    is_flag: False

                - If user found in users table AND present in workspace:
                    response: "The User is already present in User table and in this workspace."
                    is_flag: True
    """
    claims, jwt_user_id = get_current_user()

    session = db.Session()
    try:
        session.rollback()

        has_access = False

        _, caller_ws_id = _get_assignable_role_ids(session, jwt_user_id, workspace_id)
        if jwt_user_id == Role.WS_MANAGER.id or caller_ws_id == Role.WS_ADMIN.id:
            has_access = True

        if not has_access:
            return {"error": "You are not authorized to perform this check. Admin or Workspace Admin required."}

        # Normalize email and look up user
        email = (user_email or "").strip().lower()
        if not email:
            return {"error": "user_email is required"}

        user = session.query(db.User).filter(func.lower(db.User.email_id) == email).first()

        # Case 1: User doesn't exist in users table
        if not user:
            return {
                "response": "User does not exist in the user table.",
                "is_flag": False,
                "user_id": None,
                "present_in_user_table": False,
                "present_in_workspace": False
            }

        # Case 2/3: User exists; check workspace mapping (active only)
        mapping = session.query(db.UserMap).filter(
            db.UserMap.user_id == user.user_id,
            db.UserMap.workspace_id == workspace_id,
            db.UserMap.is_active == True
        ).first()

        present_in_workspace = bool(mapping)

        if present_in_workspace:
            # Present in user table AND in this workspace
            return {
                "response": "The User is already present in User table and in this workspace.",
                "is_flag": True,
                "user_id": user.user_id,
                "present_in_user_table": True,
                "present_in_workspace": True
            }
        else:
            # Present in user table BUT not in this workspace
            return {
                "response": "User not present in the workspace but the user exist.",
                "is_flag": False,
                "user_id": user.user_id,
                "present_in_user_table": True,
                "present_in_workspace": False
            }

    except Exception as e:
        session.rollback()
        print(f"Error in check_user_presence_by_email: {e}")
        return {"error": "An error occurred while checking user presence."}
    finally:
        session.close()

# Do NOT call this function for workflow launches.
# --- New Feature: Update last_updated for agent/tool usage ---
# def update_agent_tool_last_used(workspace_id, agent_id=None, tool_id=None):
#     """
#     Update last_updated for a specific agent or tool in workspace_agents_mapping2 table.
#     Only updates the entry for the agent/tool used, not all entries.
#     Args:
#         workspace_id (int): Workspace ID
#         agent_id (int, optional): Agent ID
#         tool_id (int, optional): Tool ID
#     Returns:
#         dict: Success or error message
#     """
#     session = db.Session()
#     try:
#         from datetime import datetime
#         if agent_id:
#             mapping = session.query(AgentMap).filter(
#                 AgentMap.workspace_id == workspace_id,
#                 AgentMap.agent_id == agent_id,
#                 AgentMap.is_active == True
#             ).first()
#             if mapping:
#                 mapping.last_updated = datetime.utcnow()
#                 session.commit()
#                 return {"success": True, "message": "Agent last_updated updated."}
#             else:
#                 return {"success": False, "message": "Agent mapping not found."}
#         elif tool_id:
#             mapping = session.query(ToolMap).filter(
#                 ToolMap.workspace_id == workspace_id,
#                 ToolMap.tool_id == tool_id,
#                 ToolMap.is_active == True
#             ).first()
#             if mapping:
#                 mapping.last_updated = datetime.utcnow()
#                 session.commit()
#                 return {"success": True, "message": "Tool last_updated updated."}
#             else:
#                 return {"success": False, "message": "Tool mapping not found."}
#         else:
#             return {"success": False, "message": "No agent_id or tool_id provided."}
#     except Exception as e:
#         session.rollback()
#         return {"success": False, "message": f"Error updating last_updated: {str(e)}"}
#     finally:
#         session.close()
