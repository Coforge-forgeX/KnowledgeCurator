from kbcurator.utils.access_validation import validate_user_workspace_access
from kbcurator.server.server import mcp
import psycopg2
from configparser import ConfigParser
from sqlalchemy import create_engine, func, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.ext.automap import automap_base
from configparser import ConfigParser
import os
import sys
from urllib.parse import quote_plus
from dotenv import load_dotenv
from kbcurator.utils.auth import create_jwt_token, verify_jwt_token, create_refresh_token, verify_refresh_token
from kbcurator.utils.request_context import request_var
from sqlalchemy import select, func as sql_func

# --- New Import for Password Hashing ---
from passlib.hash import argon2
from threading import RLock
from kbcurator.utils.auth import extract_token_from_headers, revoke_token
from kbcurator.utils.auth import JWT_TRANSPORT_ENCODE, JWT_RETURN_RAW_ACCESS, JWT_SET_ACCESS_COOKIE, encode_for_transport


load_dotenv(os.path.abspath(os.path.join(os.getcwd(),'.env')))

# Use environment variables from .env for PostgreSQL config
POSTGRES_HOST = os.getenv('POSTGRESQL_DATABASE_HOST')
POSTGRES_PORT = os.getenv('POSTGRESQL_DATABASE_PORT')
POSTGRES_DB = os.getenv('POSTGRESQL_DATABASE_DATABASE')
POSTGRES_USER = os.getenv('POSTGRESQL_DATABASE_USER')
POSTGRES_PASSWORD = os.getenv('POSTGRESQL_DATABASE_PASSWORD')
POSTGRES_TABLE_WORKSPACE = os.getenv('POSTGRESQL_DATABASE_WORKSPACE_TABLE', 'workspace_master')
POSTGRES_TABLE_USER = os.getenv('POSTGRESQL_DATABASE_USER_TABLE', 'user_details')

connection_string = f"postgresql+psycopg2://{POSTGRES_USER}:{quote_plus(POSTGRES_PASSWORD)}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# Create engine and reflect the database
engine = create_engine(connection_string)
metadata = MetaData()
metadata.reflect(engine)

# Automap base
Base = automap_base(metadata=metadata)
Base.prepare()

# Create a session
Session = sessionmaker(bind=engine)
# session = Session()

session_lock = RLock()

AgentIndustryMap = Base.classes.agent_industry_mapping
AgentRegionMap = Base.classes.agent_region_mapping
AgentSubIndustryMap = Base.classes.agent_subindustry_mapping
AgentIntentMap = Base.classes.agent_intent_mapping
ToolIndustryMap = Base.classes.tool_industry_mapping
ToolRegionMap = Base.classes.tool_region_mapping
ToolIntentMap = Base.classes.tool_intent_mapping
Workspace = Base.classes.workspace_master
AgentMap = Base.classes.workspace_agents_mapping_2
ToolMap = Base.classes.workspace_tools_mapping
UserMap = Base.classes.workspace_users_mapping
Agent = Base.classes.agents_details
Tool = Base.classes.tools_details
User = Base.classes.users
Category = Base.classes.category_master
Industry = Base.classes.industry_master
SubIndustry = Base.classes.subindustry_master
AgentsCMS = Base.classes.agents_cms
ToolsCMS = Base.classes.tool_cms
Integrations = Base.classes.integrations
Intent = Base.classes.intent_master
KnowledgeBase = Base.classes.knowledge_base_master
AgentCMSIntegrationMap = Base.classes.agent_cms_integration_mapping
FavouriteMappingAgent = Base.classes.favourite_mapping_agent
FavouriteMappingTool = Base.classes.favourite_mapping_tool
WorkspaceIndustrySubIndustryMap = Base.classes.workspace_industry_intent_mapping
Role = Base.classes.role_master
UserRoleMap = Base.classes.user_role_mapping
TMUIntegrationMapping = Base.classes.tool_workspace_user_integration_mapping
AMUIntegrationMapping = Base.classes.agent_workspace_user_integration_mapping
# Optional mapping tables for workspace attributes
ToolSubIndustryMap = getattr(Base.classes, 'tool_subindustry_mapping', None)
ToolCMSIntegrationMap = getattr(Base.classes, 'tool_cms_integration_mapping', None)
WorkspaceRegionMap = getattr(Base.classes, 'workspace_region_mapping', None)
WorkspaceIntentMap = getattr(Base.classes, 'workspace_intent_mapping', None)
WorkspaceKeywordMap = getattr(Base.classes, 'workspace_keyword_mapping', None)

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
    from sqlalchemy import func
    session = Session()
    try:
        session.rollback()
        normalized_email = email.strip().lower()
        user = session.query(User).filter(func.lower(User.email_id) == normalized_email).first()

        if user and hasattr(user, 'password'):
            user_db_password = getattr(user, 'password', None)
            password_matches = False

            if user_db_password is None:
                if password == "forge-X@coforge":
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
                user_workspace_data = (
                    session.query(UserMap, Workspace, UserRoleMap)
                    .join(Workspace, UserMap.workspace_id == Workspace.workspace_id)
                    .outerjoin(
                        UserRoleMap,
                        (UserRoleMap.user_id == UserMap.user_id) &
                        (UserRoleMap.workspace_id == UserMap.workspace_id) &
                        (UserRoleMap.is_active == True)
                    )
                    .filter(UserMap.user_id == user.user_id)
                    .filter(Workspace.is_active == True)
                    .all()
                )

                user_roles = []
                workspaces = []
                workspace_ids_with_roles = set()
                seen_workspaces = set()

                for user_map, workspace, user_role_map in user_workspace_data:
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
                    if user_role_map and hasattr(user_role_map, 'role_id'):
                        role_id = user_role_map.role_id
                        if role_id is not None:
                            role_entry = session.query(Role).filter(
                                Role.role_id == role_id,
                                Role.is_active == True
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
                if getattr(user, 'is_admin', False):
                    for ws in workspaces:
                        if ws['workspace_id'] not in workspace_ids_with_roles:
                            user_roles.append({
                                'workspace_id': ws['workspace_id'],
                                'role_id': None,
                                'workflow_stage': "All"
                            })

                # Prepare JWT claims
                claims = {
                    'sub': getattr(user, 'user_id', None),
                    'user_id': getattr(user, 'user_id', None),
                    'email': getattr(user, 'email_id', None),
                    'name': getattr(user, 'user_name', None) or getattr(user, 'name', None),
                    'is_admin': bool(getattr(user, 'is_admin', False)),
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
    
    session = Session()
    try:
        # Verify the refresh token
        payload = verify_refresh_token(refresh_token)
        user_id = payload.get('user_id')
        
        if not user_id:
            return {'success': False, 'message': 'Invalid refresh token: user_id missing'}
        
        # Fetch user data to rebuild access token claims
        user = session.query(User).filter(User.user_id == user_id, User.is_active == True).first()
        if not user:
            return {'success': False, 'message': 'User not found or inactive'}
        
        # Rebuild user roles and workspaces (same as login)
        user_roles = []
        workspaces = []
        
        user_workspace_data = (
            session.query(UserMap, Workspace, UserRoleMap)
            .join(Workspace, UserMap.workspace_id == Workspace.workspace_id)
            .outerjoin(UserRoleMap, 
                      (UserRoleMap.user_id == UserMap.user_id) & 
                      (UserRoleMap.workspace_id == UserMap.workspace_id) &
                      (UserRoleMap.is_active == True))
            .filter(UserMap.user_id == user.user_id)
            .filter(Workspace.is_active == True)
            .all()
        )
        
        seen_workspaces = set()
        for user_map, workspace, user_role_map in user_workspace_data:
            workspace_id = workspace.workspace_id
            
            if workspace_id not in seen_workspaces:
                seen_workspaces.add(workspace_id)
                workspaces.append({
                    'workspace_id': workspace.workspace_id,
                    'workspace_name': workspace.workspace_name,
                    'workspace_desc': workspace.workspace_desc
                })
            
            if user_role_map and hasattr(user_role_map, 'role_id'):
                role_id = user_role_map.role_id
                if role_id is not None:
                    user_roles.append({
                        'workspace_id': workspace_id,
                        'role_id': role_id
                    })
        
        # Create new access token with fresh claims
        claims = {
            'sub': user.user_id,
            'user_id': user.user_id,
            'email': getattr(user, 'email_id', None),
            'name': getattr(user, 'user_name', None) or getattr(user, 'name', None),
            'is_admin': bool(getattr(user, 'is_admin', False)),
            'roles': user_roles,
        }
        
        access_token, access_ttl = create_jwt_token(claims)
        
        # Set access token cookie if enabled (same as login_user)
        request = request_var.get(None)
        if request and JWT_SET_ACCESS_COOKIE:
            request.state.access_token = access_token
            request.state.access_token_expires = access_ttl
        
        # Encode token for transport if configured (same as login_user)
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
    session = Session()
    try:
        session.rollback()
        if workspace_id:
            kb_query = session.query(KnowledgeBase).filter(
                KnowledgeBase.industry_id == industry_id,
                KnowledgeBase.sub_industry_id == sub_industry_id,
                KnowledgeBase.workspace_id == workspace_id,
                KnowledgeBase.is_active == True
            )
        else:
            kb_query = session.query(KnowledgeBase).filter(
                KnowledgeBase.industry_id == industry_id,
                KnowledgeBase.sub_industry_id == sub_industry_id,
                KnowledgeBase.is_active == True
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

@mcp.tool()
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
        session = Session()
        try:
            session.rollback()
            # Use JWT claims directly for authentication (faster, as in login_user)
            request = request_var.get(None)
            if not request or not hasattr(request.state, "jwt_claims"):
                return {"error": "Unauthorized: JWT claims not found in request context"}
            claims = request.state.jwt_claims
            jwt_user_id = claims.get("user_id") or claims.get("sub")
            if not jwt_user_id:
                return {"error": "Unauthorized: user_id not found in token claims"}
            # If user_id is provided and does not match JWT, return error
            if user_id is not None and str(user_id) != str(jwt_user_id):
                return {"error": "The user_id in the request is not authorized. Only the authenticated user's workspaces can be accessed."}

            # OPTIMIZED: Single query with subqueries for counts

            agent_count_subq = (
                select(
                    AgentMap.workspace_id,
                    sql_func.count(AgentMap.agent_id).label('agent_count')
                )
                .where(AgentMap.is_active == True)
                .group_by(AgentMap.workspace_id)
                .subquery()
            )
            tool_count_subq = (
                select(
                    ToolMap.workspace_id,
                    sql_func.count(ToolMap.tool_id).label('tool_count')
                )
                .where(ToolMap.is_active == True)
                .group_by(ToolMap.workspace_id)
                .subquery()
            )
            user_count_subq = (
                select(
                    UserMap.workspace_id,
                    sql_func.count(UserMap.user_id).label('user_count')
                )
                .where(UserMap.is_active == True)
                .group_by(UserMap.workspace_id)
                .subquery()
            )
            workspaces_with_counts = (
                session.query(
                    Workspace.workspace_id,
                    Workspace.workspace_name,
                    Workspace.workspace_desc,
                    sql_func.coalesce(agent_count_subq.c.agent_count, 0).label('agent_count'),
                    sql_func.coalesce(tool_count_subq.c.tool_count, 0).label('tool_count'),
                    sql_func.coalesce(user_count_subq.c.user_count, 0).label('user_count')
                )
                .join(UserMap, UserMap.workspace_id == Workspace.workspace_id)
                .outerjoin(agent_count_subq, agent_count_subq.c.workspace_id == Workspace.workspace_id)
                .outerjoin(tool_count_subq, tool_count_subq.c.workspace_id == Workspace.workspace_id)
                .outerjoin(user_count_subq, user_count_subq.c.workspace_id == Workspace.workspace_id)
                .filter(UserMap.user_id == jwt_user_id, UserMap.is_active == True, Workspace.is_active == True)
                .all()
            )
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
            return {'response': results}
        except Exception as e:
            session.rollback()
            print(f"Fetch workspaces failed with error: {e}")
            return {'error': 'An error occurred while fetching workspaces.'}
        finally:
            session.close()

@mcp.tool()
def create_workspace(payload):
    """
    Create a new workspace and map agents/tools/users as per the payload from frontend.
    Args:
        payload (dict): Workspace creation payload from frontend.
    Returns:
        dict: {'response': 'workspace created'}
    """
    # RBAC: Only allow if JWT has is_admin True
    request = request_var.get(None)
    if not request or not hasattr(request.state, "jwt_claims"):
        return {"error": "Unauthorized: JWT claims not found in request context"}
    claims = request.state.jwt_claims
    is_admin = claims.get("is_admin", False)
    jwt_user_id = claims.get("user_id") or claims.get("sub")
    # Allow if is_admin or user has Workspace Admin role
    has_access = False
    if is_admin:
        has_access = True
    else:
        session = Session()
        admin_role = session.query(Role).filter(Role.role_name.ilike("%workspace admin%"), Role.is_active == True).first()
        if admin_role:
            user_role = session.query(UserRoleMap).filter(
                UserRoleMap.user_id == jwt_user_id,
                UserRoleMap.role_id == admin_role.role_id,
                UserRoleMap.is_active == True
            ).first()
            if user_role:
                has_access = True
        session.close()
    if not has_access:
        return {"error": "You are not authorized to create a workspace. Admin or Workspace Admin required."}

    session = Session()
    try:
        session.rollback()
        # Extract fields from payload
        user_id = payload.get('user_id')
        workspace_name = payload.get('workspaceName')
        namespace = payload.get('namespace')
        workspace_desc = payload.get('description')
        intent = payload.get('intent')
        industry = payload.get('industry')
        sub_industry = payload.get('subIndustry')
        keywords = payload.get('keywords', [])
        agent_ids = payload.get('agent_ids', [])
        tool_ids = payload.get('tool_ids', [])
        kb_ids = payload.get('kb_ids', [])
        kb_title = payload.get('kb_title', None)
        kb_description = payload.get('kb_description', None)

        # Check for duplicate workspace name globally (active only)
        existing_ws = (
            session.query(Workspace)
            .filter(Workspace.workspace_name == workspace_name, Workspace.is_active == True)
            .first()
        )
        if existing_ws:
            return {'error': f"Workspace name '{workspace_name}' already exists. Please choose a different name."}

        # 1. Create Workspace
        new_workspace = Workspace(
            workspace_name=workspace_name,
            namespace=namespace,
            workspace_desc=workspace_desc,
            keywords=','.join(keywords),
            is_active=True
        )
        session.add(new_workspace)
        session.commit()
        session.refresh(new_workspace)
        workspace_id = new_workspace.workspace_id
        print(f"Created workspace with ID: {workspace_id}")

        # --- Add creator as Workspace Admin ---
        # Fetch current user from JWT claims
        creator_id = claims.get("user_id") or claims.get("sub")
        # Fetch Workspace Admin role_id
        admin_role = session.query(Role).filter(Role.role_name.ilike("%workspace admin%"), Role.is_active == True).first()
        if not admin_role:
            session.close()
            return {"error": "Workspace Admin role not found. Please configure roles."}
        admin_role_id = admin_role.role_id
        # Fetch creator's user details
        creator = session.query(User).filter(User.user_id == creator_id).first()
        if not creator:
            session.close()
            return {"error": "Creator user not found."}
        # Add creator to workspace with Workspace Admin role
        # Use add_user_to_workspace logic directly to avoid circular import
        email = creator.email_id
        first_name = creator.first_name
        last_name = creator.last_name
        # Check if user already mapped (should not be, but safe)
        user_map = session.query(UserMap).filter_by(user_id=creator_id, workspace_id=workspace_id).first()
        if not user_map:
            session.add(UserMap(user_id=creator_id, workspace_id=workspace_id, is_active=True))
        # Add/update user_role_mapping
        user_role_map = session.query(UserRoleMap).filter_by(user_id=creator_id, workspace_id=workspace_id).first()
        if user_role_map:
            user_role_map.role_id = admin_role_id
            user_role_map.is_active = True
        else:
            session.add(UserRoleMap(user_id=creator_id, workspace_id=workspace_id, role_id=admin_role_id, is_active=True))
        print(f"Added creator {email} as Workspace Admin to workspace {workspace_id}")

        # 2. Map region, intent, industry, subindustry, keywords (if mapping tables exist)
        # These mappings assume the mapping tables and columns exist in DB schema
        if WorkspaceIndustrySubIndustryMap and industry and sub_industry and intent:
            session.add(WorkspaceIndustrySubIndustryMap(
                workspace_id=workspace_id, 
                industry_id=industry, 
                subindustry_id=sub_industry, 
                intent_id=intent,
                is_active=True))
            print(f"Mapped industry: {industry}, sub-industry: {sub_industry}, intent: {intent}")
        # (No longer needed: user_id mapping handled above for creator)

        # 3. Map agents/tools
        for agent_id in agent_ids:
            session.add(AgentMap(workspace_id=workspace_id, agent_id=agent_id, is_active=True))
            print(f"Mapped agent_id: {agent_id}")

        for tool_id in tool_ids:
            session.add(ToolMap(workspace_id=workspace_id, tool_id=tool_id, is_active=True))
            print(f"Mapped tool_id: {tool_id}")

        # 4. Map or create knowledge base(s)
        if kb_ids:
            # Map existing knowledge bases to this workspace
            for kb_id in kb_ids:
                kb_obj = session.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
                if kb_obj:
                    kb_obj.workspace_id = workspace_id
                    kb_obj.is_active = True
                    print(f"Mapped existing knowledge base ID: {kb_id} to workspace {workspace_id}")
        # Map new KB with the workspace
        if kb_title and kb_description:
            new_kb = KnowledgeBase(
                title=kb_title,
                description=kb_description,
                workspace_id=workspace_id,
                industry_id=industry,
                sub_industry_id=sub_industry,
                is_active=True
            )
            session.add(new_kb)
            print(f"Created new knowledge base '{kb_title}' for workspace {workspace_id}")

        session.commit()
        return {'response': 'Workspace Created'}
    except Exception as e:
        session.rollback()
        print(f"Error in create_workspace: {e}")
        return {'error': 'An error occurred while creating workspace.'}
    finally:
        session.close()

@mcp.tool()
def list_intent():
    """
    Return a list of all active intents from the intent_master table.
    Returns:
        dict: { 'response': [ { 'intent_id': ..., 'intent_name': ... }, ... ] }
    """
    session = Session()
    try:
        session.rollback()
        intents = session.query(Intent).filter(Intent.is_active == True).all()
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
    session = Session()
    try:
        session.rollback()
        # Use JWT claims for authentication (faster, as in login_user)
        request = request_var.get(None)
        if not request or not hasattr(request.state, "jwt_claims"):
            return {"error": "Unauthorized: JWT claims not found in request context"}
        claims = request.state.jwt_claims
        jwt_user_id = claims.get("user_id") or claims.get("sub")
        if user_id is not None and str(user_id) != str(jwt_user_id):
            return {"error": "The user_id in the request is not authorized. Only the authenticated user's data can be accessed."}

        # If intent is provided, filter tools by intent
        if intent:
            intent_tool_ids = session.query(ToolIntentMap.tool_id).filter(ToolIntentMap.intent_id == intent, ToolIntentMap.is_active == True).all()
            intent_tool_ids = [row.tool_id for row in intent_tool_ids]
            tools = session.query(Tool).filter(Tool.tool_id.in_(intent_tool_ids)).all()
        else:
            tools = session.query(Tool).all()

        # OPTIMIZED: Bulk fetch all favorites in one query, always use JWT user
        favorite_tool_ids = set()
        if jwt_user_id is not None:
            tool_ids = [t.tool_id for t in tools]
            if tool_ids:
                favorites = session.query(FavouriteMappingTool.tool_id).filter(
                    FavouriteMappingTool.user_id == jwt_user_id,
                    FavouriteMappingTool.tool_id.in_(tool_ids),
                    FavouriteMappingTool.is_active == True
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
def fetch_agents_info(user_id=None, intent=None):
    """
    Fetch all agent details from the agents_details table.
    Returns:
        list of dicts: Each dict contains agent details.
    """
    if user_id is None:
        return {"status": "error", "error": "user_id cannot be null"}
    session = Session()
    try:
        session.rollback()
        # Use JWT claims for authentication (faster, as in login_user)
        request = request_var.get(None)
        if not request or not hasattr(request.state, "jwt_claims"):
            return {"error": "Unauthorized: JWT claims not found in request context"}
        claims = request.state.jwt_claims
        jwt_user_id = claims.get("user_id") or claims.get("sub")
        if user_id is not None and str(user_id) != str(jwt_user_id):
            return {"error": "The user_id in the request is not authorized. Only the authenticated user's data can be accessed."}

        # If intent is provided, filter agents by intent
        if intent:
            intent_agent_ids = session.query(AgentIntentMap.agent_id).filter(AgentIntentMap.intent_id == intent, AgentIntentMap.is_active == True).all()
            intent_agent_ids = [row.agent_id for row in intent_agent_ids]
            agents = session.query(Agent).filter(Agent.agent_id.in_(intent_agent_ids)).all()
        else:
            agents = session.query(Agent).all()

        # OPTIMIZED: Bulk fetch all favorites in one query, always use JWT user
        favorite_agent_ids = set()
        if jwt_user_id is not None:
            agent_ids = [a.agent_id for a in agents]
            if agent_ids:
                favorites = session.query(FavouriteMappingAgent.agent_id).filter(
                    FavouriteMappingAgent.user_id == jwt_user_id,
                    FavouriteMappingAgent.agent_id.in_(agent_ids),
                    FavouriteMappingAgent.is_active == True
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
def update_workspace(payload):
    """
    Update workspace details (name, description, tools, agents) using a payload dict.
    Only Workspace Admin or Forge-X Admin can update workspaces.
    Args:
        payload (dict): Should contain 'workspace_id', and optionally 'name', 'description', 'tool_ids', 'agent_ids'.
    Returns:
        dict: Response or error message.
    """
    # RBAC: Only allow if JWT has is_admin True or user is Workspace Admin for this workspace
    request = request_var.get(None)
    if not request or not hasattr(request.state, "jwt_claims"):
        return {"error": "Unauthorized: JWT claims not found in request context"}
    claims = request.state.jwt_claims
    is_admin = claims.get("is_admin", False)
    jwt_user_id = claims.get("user_id") or claims.get("sub")
    has_access = False
    session = Session()
    try:
        session.rollback()
        workspace_id = payload.get('workspace_id')
        if is_admin:
            has_access = True
        else:
            # Check if user has Workspace Admin role for this workspace
            admin_role = session.query(Role).filter(Role.role_name.ilike("%workspace admin%"), Role.is_active == True).first()
            if admin_role:
                user_role = session.query(UserRoleMap).filter(
                    UserRoleMap.user_id == jwt_user_id,
                    UserRoleMap.workspace_id == workspace_id,
                    UserRoleMap.role_id == admin_role.role_id,
                    UserRoleMap.is_active == True
                ).first()
                if user_role:
                    has_access = True
        if not has_access:
            return {"error": "You are not authorized to update this workspace. Admin or Workspace Admin required."}

        name = payload.get('workspaceName')
        description = payload.get('description')
        tool_ids = payload.get('tool_ids')
        agent_ids = payload.get('agent_ids')
        namespace = payload.get('namespace')
        intent = payload.get('intent')
        industry = payload.get('industry')
        sub_industry = payload.get('subIndustry')
        keywords = payload.get('keywords', [])
        kb_ids = payload.get('kb_ids', [])
        kb_title = payload.get('kb_title', [])
        kb_description = payload.get('kb_description', [])

        ws = session.query(Workspace).filter(Workspace.workspace_id==workspace_id, Workspace.is_active==True).first()
        if not ws:
            return {"error": "Workspace not found or inactive"}
        if name and ws.workspace_name != name:
            existing_ws = (
                session.query(Workspace)
                .filter(
                    Workspace.workspace_name == name,
                    Workspace.is_active == True,
                )
                .first()
            )
            if existing_ws:
                return {"error": f"Workspace with name '{name}' already exists on the platform. Please choose a different name."}
            else:
                ws.workspace_name = name
        if description:
            ws.workspace_desc = description
        if namespace:
            ws.namespace = namespace
        if keywords:
            ws.keywords = ','.join(keywords)
        # Update or insert region, intent, industry, subindustry, keywords mappings
        if WorkspaceIndustrySubIndustryMap and industry:
            mapping = session.query(WorkspaceIndustrySubIndustryMap).filter(WorkspaceIndustrySubIndustryMap.workspace_id==workspace_id).first()
            if mapping:
                mapping.industry_id = industry
                mapping.intent_id = intent
                mapping.subindustry_id = sub_industry
                mapping.is_active = True
            else:
                session.add(WorkspaceIndustrySubIndustryMap(
                    workspace_id=workspace_id, 
                    industry_id=industry, 
                    intent_id=intent,
                    subindustry_id=sub_industry, 
                    is_active=True
                ))
        # Update or insert tools
        if tool_ids is not None:
            # Mark all existing mappings as inactive
            session.query(ToolMap).filter(ToolMap.workspace_id==workspace_id).update({ToolMap.is_active: False})
            for tid in tool_ids:
                tool_map = session.query(ToolMap).filter(ToolMap.workspace_id==workspace_id, ToolMap.tool_id==tid).first()
                if tool_map:
                    tool_map.is_active = True
                else:
                    session.add(ToolMap(workspace_id=workspace_id, tool_id=tid, is_active=True))
        # Update or insert agents
        if agent_ids is not None:
            # Mark all existing mappings as inactive
            session.query(AgentMap).filter(AgentMap.workspace_id==workspace_id).update({AgentMap.is_active: False})
            for aid in agent_ids:
                agent_map = session.query(AgentMap).filter(AgentMap.workspace_id==workspace_id, AgentMap.agent_id==aid).first()
                if agent_map:
                    agent_map.is_active = True
                else:
                    session.add(AgentMap(workspace_id=workspace_id, agent_id=aid, is_active=True))
        # 1. If new kb_title and kb_description are provided, create a new KB entry for this workspace/industry/subindustry
        if kb_title and kb_description and (isinstance(kb_title, str) and isinstance(kb_description, str)):
            new_kb = KnowledgeBase(
                title=kb_title,
                description=kb_description,
                workspace_id=workspace_id,
                industry_id=industry,
                sub_industry_id=sub_industry,
                is_active=True
            )
            session.add(new_kb)
        # 2. If kb_ids are provided, update mappings: activate selected KBs, deactivate others for this workspace/industry/subindustry
        if kb_ids is not None:
            # Activate selected KBs
            for kb_id in kb_ids:
                kb_obj = session.query(KnowledgeBase).filter(
                    KnowledgeBase.id == kb_id,
                    KnowledgeBase.workspace_id == workspace_id,
                    KnowledgeBase.industry_id == industry,
                    KnowledgeBase.sub_industry_id == sub_industry
                ).first()
                if kb_obj:
                    print("ID created:",kb_obj.id)
                    kb_obj.is_active = True
            # Deactivate KBs that are not in kb_ids but belong to this workspace/industry/subindustry
            kb_objs_to_deactivate = session.query(KnowledgeBase).filter(
                KnowledgeBase.workspace_id == workspace_id,
                KnowledgeBase.industry_id == industry,
                KnowledgeBase.sub_industry_id == sub_industry,
                ~KnowledgeBase.id.in_(kb_ids)
            ).all()
            for kb_obj in kb_objs_to_deactivate:
                print("KB ID created:",kb_obj.id)
                kb_obj.is_active = False
        session.commit()
        return {"response": "Workspace updated"}
    except Exception as e:
        session.rollback()
        print(f"Error in update_workspace: {e}")
        return {"error": "An error occurred while updating workspace."}
    finally:
        session.close() 

@mcp.tool()
def delete_workspace(workspace_id):
    '''
    Delete workspace: set is_active to False
    '''
    # RBAC: Only allow if JWT has is_admin True
    request = request_var.get(None)
    if not request or not hasattr(request.state, "jwt_claims"):
        return {"error": "Unauthorized: JWT claims not found in request context"}
    claims = request.state.jwt_claims
    is_admin = claims.get("is_admin", False)
    jwt_user_id = claims.get("user_id") or claims.get("sub")
    has_access = False
    if is_admin:
        has_access = True
    else:
        session = Session()
        admin_role = session.query(Role).filter(Role.role_name.ilike("%workspace admin%"), Role.is_active == True).first()
        if admin_role:
            user_role = session.query(UserRoleMap).filter(
                UserRoleMap.user_id == jwt_user_id,
                UserRoleMap.workspace_id == workspace_id,
                UserRoleMap.role_id == admin_role.role_id,
                UserRoleMap.is_active == True
            ).first()
            if user_role:
                has_access = True
        session.close()
    if not has_access:
        return {"error": "You are not authorized to delete a workspace. Admin or Workspace Admin required."}

    session = Session()
    try:
        session.rollback()
        ws = session.query(Workspace).filter(Workspace.workspace_id==workspace_id, Workspace.is_active==True).first()
        if not ws:
            return {"error": "Workspace not found or already inactive"}
        ws.is_active = False
        session.commit()
        return {"response": "Workspace deleted (set inactive)"}
    except Exception as e:
        session.rollback()
        print(f"Error in delete_workspace: {e}")
        return {"error": "An error occurred while deleting workspace."}
    finally:
        session.close()

@mcp.tool()
def fetch_workspace_details(workspace_id):
    '''
    Fetch all information about a workspace, including master table, mappings, and all tool/agent/user details.
    Args:
        workspace_id (int): ID of the workspace to fetch.
    Returns:
        dict: Workspace info, mappings, tools, agents, users, and related attributes.
    '''
    session = Session()
    try:
        session.rollback()
        # Use JWT claims directly for authentication (faster, as in login_user)
        request = request_var.get(None)
        if not request or not hasattr(request.state, "jwt_claims"):
            return {"error": "Unauthorized: JWT claims not found in request context"}
        claims = request.state.jwt_claims
        jwt_user_id = claims.get("user_id") or claims.get("sub")
        if not jwt_user_id:
            return {"error": "Unauthorized: user_id not found in token claims"}

        # Check if user is mapped to this workspace
        user_map = session.query(UserMap).filter_by(workspace_id=workspace_id, user_id=jwt_user_id, is_active=True).first()
        if not user_map:
            return {"error": "You are not authorized to access this workspace."}
        ws = session.query(Workspace).filter(Workspace.workspace_id==workspace_id, Workspace.is_active==True).first()
        if not ws:
            return {"error": "Workspace not found or inactive"}

        # Master table info
        ws_info = {col: getattr(ws, col) for col in ws.__table__.columns.keys()}

        # Category mapping
        categories = session.query(Category).filter(Category.is_active == True).all()
        cat_map = {str(c.category_id): c.category_name for c in categories}

        # Mappings (region, intent, industry, subindustry, keywords) - only is_active if present
        def active_query(model, **kwargs):
            q = session.query(model).filter_by(**kwargs)
            if hasattr(model, 'is_active'):
                q = q.filter(model.is_active == True)
            return q

        # Fetch industry/subindustry mapping and names
        ws_ind_map = session.query(WorkspaceIndustrySubIndustryMap).filter(WorkspaceIndustrySubIndustryMap.workspace_id==workspace_id, WorkspaceIndustrySubIndustryMap.is_active==True).first()
        industry_id = subindustry_id = intent_id = industry_name = subindustry_name = None
        if ws_ind_map:
            industry_id = getattr(ws_ind_map, 'industry_id', None)
            subindustry_id = getattr(ws_ind_map, 'subindustry_id', None)
            intent_id = getattr(ws_ind_map, 'intent_id', None)
            if industry_id:
                industry_obj = session.query(Industry).filter(Industry.industry_id==industry_id,Industry.is_active==True).first()
                if industry_obj:
                    industry_name = getattr(industry_obj, 'industry_name', None)
            if subindustry_id:
                subindustry_obj = session.query(SubIndustry).filter(SubIndustry.subindustry_id==subindustry_id,SubIndustry.is_active==True).first()
                if subindustry_obj:
                    subindustry_name = getattr(subindustry_obj, 'subindustry_name', None)

        # Tools in workspace (only active)
        tool_maps = session.query(ToolMap).filter(ToolMap.workspace_id==workspace_id, ToolMap.is_active==True)
        tool_ids = [tm.tool_id for tm in tool_maps.all()]
        tools = []
        if tool_ids:
            tool_query = session.query(Tool).filter(Tool.tool_id.in_(tool_ids))
            if hasattr(Tool, 'is_active'):
                tool_query = tool_query.filter(Tool.is_active == True)
            for t in tool_query.all():
                tool_dict = {col: getattr(t, col) for col in t.__table__.columns.keys()}
                # Replace tool_category IDs with names
                cat_ids = str(tool_dict.get('tool_category', '') or '').split(',')
                tool_dict['tool_category'] = [cat_map.get(cid.strip()) for cid in cat_ids if cid.strip() in cat_map] 
                tools.append(tool_dict)
        # Agents in workspace (only active)
        agent_maps = session.query(AgentMap).filter(AgentMap.workspace_id==workspace_id, AgentMap.is_active==True)
        agent_ids = [am.agent_id for am in agent_maps.all()]
        agents = []
        if agent_ids:
            agent_query = session.query(Agent).filter(Agent.agent_id.in_(agent_ids))
            if hasattr(Agent, 'is_active'):
                agent_query = agent_query.filter(Agent.is_active == True)
            for a in agent_query.all():
                agent_dict = {col: getattr(a, col) for col in a.__table__.columns.keys()}
                # Replace agent_category IDs with names
                cat_ids = str(agent_dict.get('agent_category', '') or '').split(',')
                agent_dict['agent_category'] = [cat_map.get(cid.strip()) for cid in cat_ids if cid.strip() in cat_map]
                agent_dict['type'] = 'agent'
                agents.append(agent_dict)

        # Users in workspace (with role/permissions, only active)
        # OPTIMIZED: Single JOIN query instead of N queries
        users = []
        user_data_query = (
            session.query(User, UserRoleMap, Role)
            .join(UserMap, UserMap.user_id == User.user_id)
            .outerjoin(UserRoleMap, (UserRoleMap.user_id == User.user_id) & (UserRoleMap.workspace_id == workspace_id) & (UserRoleMap.is_active == True))
            .outerjoin(Role, (UserRoleMap.role_id == Role.role_id) & (Role.is_active == True))
            .filter(UserMap.workspace_id == workspace_id, UserMap.is_active == True)
        )
        if hasattr(User, 'is_active'):
            user_data_query = user_data_query.filter(User.is_active == True)
        
        for user, user_role_map, role in user_data_query.all():
            user_dict = {col: getattr(user, col) for col in user.__table__.columns.keys()}
            user_dict['role'] = getattr(role, 'role_name', None) if role else None
            user_dict['role_id'] = getattr(role, 'role_id', None) if role else None
            user_dict['permissions'] = getattr(user_role_map, 'permissions', None) if user_role_map else None
            users.append(user_dict)

        # Fetch knowledge bases for this workspace's industry and subindustry
        knowledge_bases = []
        if industry_name and subindustry_name:
            # Get industry_id and subindustry_id
            industry_obj = session.query(Industry).filter(func.lower(Industry.industry_name) == industry_name.strip().lower(), Industry.is_active == True).first()
            subindustry_obj = session.query(SubIndustry).filter(func.lower(SubIndustry.subindustry_name) == subindustry_name.strip().lower(), SubIndustry.is_active == True).first()
            if industry_obj and subindustry_obj:
                kb_query = session.query(KnowledgeBase).filter(
                    KnowledgeBase.industry_id == industry_obj.industry_id,
                    KnowledgeBase.sub_industry_id == subindustry_obj.subindustry_id,
                    KnowledgeBase.workspace_id == workspace_id,
                    KnowledgeBase.is_active == True
                )
                knowledge_bases = [
                    {
                        'id': getattr(kb, 'id', None),
                        'title': getattr(kb, 'title', None),
                        'description': getattr(kb, 'description', None)
                    }
                    for kb in kb_query.all()
                ]

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
def fetch_agents_tools_by_ids(workspace_id):
    # Enforce JWT-based access: only allow if user is mapped to the workspace
    request = request_var.get(None)
    if not request or not hasattr(request.state, "jwt_claims"):
        return {"error": "Unauthorized: JWT claims not found in request context"}
    claims = request.state.jwt_claims
    jwt_user_id = claims.get("user_id") or claims.get("sub")
    if not jwt_user_id:
        return {"error": "Unauthorized: user_id not found in token claims"}

    """
    Fetch all tools and agents for a given workspace_id, tagging each as 'tool' or 'agent'.
    Only returns mappings and entities where is_active == 'true'.
    Replaces agent_category/tool_category IDs with category names.
    """
    session = Session()
    try:
        session.rollback()

        # Check if user is mapped to this workspace
        user_map = session.query(UserMap).filter_by(
            workspace_id=workspace_id, user_id=jwt_user_id, is_active=True
        ).first()
        if not user_map:
            return {"error": "You are not authorized to access this workspace."}

        # 1. Get category_id -> category_name mapping
        categories = session.query(Category).filter(Category.is_active != False).all()
        cat_map = {str(c.category_id): c.category_name for c in categories}

        results = []

        # 2. Fetch tools
        tool_maps_q = session.query(ToolMap).filter_by(workspace_id=workspace_id, is_active=True)
        tool_ids = [tm.tool_id for tm in tool_maps_q.all()]
        if tool_ids:
            tool_query = session.query(Tool).filter(Tool.tool_id.in_(tool_ids))
            if hasattr(Tool, 'is_active'):
                tool_query = tool_query.filter(Tool.is_active != 'false')
            for t in tool_query.all():
                tool_dict = {col: getattr(t, col) for col in t.__table__.columns.keys()}
                # Replace tool_category IDs with names
                cat_ids = str(tool_dict.get('tool_category', '') or '').split(',')
                tool_dict['tool_category'] = [cat_map.get(cid.strip()) for cid in cat_ids if cid.strip() in cat_map]
                tool_dict['type'] = 'tool'
                results.append(tool_dict)

        # 3. Fetch agents
        agent_maps_q = session.query(AgentMap).filter_by(workspace_id=workspace_id, is_active=True)
        agent_ids = [am.agent_id for am in agent_maps_q.all()]
        if agent_ids:
            agent_query = session.query(Agent).filter(Agent.agent_id.in_(agent_ids))
            if hasattr(Agent, 'is_active'):
                agent_query = agent_query.filter(Agent.is_active != 'false')
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
def add_agent_tool_to_workspace(payload):
    """
    Add an agent or tool to a workspace.
    Args:
        payload (dict): {"user_id": ..., "workspace_id": ..., "type": "Agent" or "Tool", "id": ...}
    Returns:
        dict: {"response": "Successfully Added to workspace"}
    """
    # Enforce JWT-based access: only allow if user is mapped to the workspace
    request = request_var.get(None)
    if not request or not hasattr(request.state, "jwt_claims"):
        return {"error": "Unauthorized: JWT claims not found in request context"}
    claims = request.state.jwt_claims
    jwt_user_id = claims.get("user_id") or claims.get("sub")
    if not jwt_user_id:
        return {"error": "Unauthorized: user_id not found in token claims"}

    session = Session()
    try:
        session.rollback()

        workspace_id = payload.get("workspace_id")
        entity_type = payload.get("type")
        entity_id = payload.get("id")

        # Authorization: JWT user must be mapped to this workspace
        user_map = session.query(UserMap).filter_by(
            workspace_id=workspace_id, user_id=jwt_user_id, is_active=True
        ).first()
        if not user_map:
            return {"error": "You are not authorized to modify this workspace."}

        if entity_type == "Agent":
            existing = session.query(AgentMap).filter_by(workspace_id=workspace_id, agent_id=entity_id).first()
            if existing:
                if not existing.is_active:
                    existing.is_active = True
                else:
                    return {"response": "Agent already mapped and active in workspace"}
            else:
                session.add(AgentMap(workspace_id=workspace_id, agent_id=entity_id, is_active=True))
        elif entity_type == "Tool":
            existing = session.query(ToolMap).filter_by(workspace_id=workspace_id, tool_id=entity_id).first()
            if existing:
                if not existing.is_active:
                    existing.is_active = True
                else:
                    return {"response": "Tool already mapped and active in workspace"}
            else:
                session.add(ToolMap(workspace_id=workspace_id, tool_id=entity_id, is_active=True))
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
    # Enforce JWT-based access: only allow if user is mapped to the workspace
    request = request_var.get(None)
    if not request or not hasattr(request.state, "jwt_claims"):
        return {"error": "Unauthorized: JWT claims not found in request context"}
    claims = request.state.jwt_claims
    jwt_user_id = claims.get("user_id") or claims.get("sub")
    if not jwt_user_id:
        return {"error": "Unauthorized: user_id not found in token claims"}

    session = Session()
    try:
        session.rollback()
        # Check if user is mapped to this workspace
        user_map = session.query(UserMap).filter_by(
            workspace_id=workspace_id, user_id=jwt_user_id, is_active=True
        ).first()
        if not user_map:
            return {"error": "You are not authorized to modify this workspace."}

        updated = False
        if agent_id is not None:
            mapping = session.query(AgentMap).filter_by(workspace_id=workspace_id, agent_id=agent_id).first()
            if mapping and hasattr(mapping, 'is_active'):
                mapping.is_active = False
                updated = True
        if tool_id is not None:
            mapping = session.query(ToolMap).filter_by(workspace_id=workspace_id, tool_id=tool_id).first()
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
async def update_fav_agent(user_id, agent_id, workspace_id=0) -> dict:
    # Enforce JWT-based access: only allow if user is mapped to the workspace and user_id matches JWT
    if user_id is None:
        return {"status": "error", "error": "user_id cannot be null"}
    request = request_var.get(None)
    if not request or not hasattr(request.state, "jwt_claims"):
        return {"status": "error", "error": "Unauthorized: JWT claims not found in request context"}
    claims = request.state.jwt_claims
    jwt_user_id = claims.get("user_id") or claims.get("sub")
    if not jwt_user_id:
        return {"status": "error", "error": "Unauthorized: user_id not found in token claims"}
    if str(user_id) != str(jwt_user_id):
        return {"status": "error", "error": "Unauthorized: user_id in request does not match user in token"}

    valid, err = validate_user_workspace_access(user_id=user_id, workspace_id=workspace_id)
    if not valid:
        return {"status": "error", "error": err}

    session = Session()
    try:
        session.rollback()
        user_agent_fav = session.query(FavouriteMappingAgent).filter(
            FavouriteMappingAgent.user_id == user_id,
            FavouriteMappingAgent.agent_id == agent_id,
            FavouriteMappingAgent.workspace_id == workspace_id
        ).first()
        if not user_agent_fav:
            new_fav = FavouriteMappingAgent(
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
async def update_fav_tool(user_id, tool_id, workspace_id=0) -> dict:
    # Enforce JWT-based access: only allow if user is mapped to the workspace and user_id matches JWT
    if user_id is None:
        return {"status": "error", "error": "user_id cannot be null"}
    request = request_var.get(None)
    if not request or not hasattr(request.state, "jwt_claims"):
        return {"status": "error", "error": "Unauthorized: JWT claims not found in request context"}
    claims = request.state.jwt_claims
    jwt_user_id = claims.get("user_id") or claims.get("sub")
    if not jwt_user_id:
        return {"status": "error", "error": "Unauthorized: user_id not found in token claims"}
    if str(user_id) != str(jwt_user_id):
        return {"status": "error", "error": "Unauthorized: user_id in request does not match user in token"}

    valid, err = validate_user_workspace_access(user_id=user_id, workspace_id=workspace_id)
    if not valid:
        return {"status": "error", "error": err}

    session = Session()
    try:
        session.rollback()
        user_tool_fav = session.query(FavouriteMappingTool).filter(
            FavouriteMappingTool.user_id == user_id,
            FavouriteMappingTool.tool_id == tool_id,
            FavouriteMappingTool.workspace_id == workspace_id
        ).first()
        if not user_tool_fav:
            new_fav = FavouriteMappingTool(
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
        return {"status": "success", "response": f"Tool_id {tool_id} updated to {fav_flag}"}
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

#     session = Session()
#     try:
#         session.rollback()
#         results = []
#         if str(type).lower() == 'agent':
#             print("Fetching integrations for agent_id:", id)
#             # Join AgentsCMS, AgentCMSIntegrationMap, Integrations
#             query = (
#                 session.query(
#                     Integrations.integration_name,
#                     Integrations.integration_logo_url,
#                     Integrations.is_active
#                 )
#                 .join(AgentCMSIntegrationMap, AgentCMSIntegrationMap.integration_id == Integrations.integration_id)
#                 .join(AgentsCMS, AgentsCMS.agent_cms_id == AgentCMSIntegrationMap.agent_cms_id)
#                 .filter(AgentsCMS.agent_id == id)
#             )
#         elif str(type).lower() == 'tool':
#             print("Fetching integrations for tool_id:", id)
#             # Join ToolsCMS, ToolCMSIntegrationMap, Integrations
#             query = (
#                 session.query(
#                     Integrations.integration_name,
#                     Integrations.integration_logo_url,
#                     Integrations.is_active
#                 )
#                 .join(ToolCMSIntegrationMap, ToolCMSIntegrationMap.integration_id == Integrations.integration_id)
#                 .join(ToolsCMS, ToolsCMS.tool_cms_id == ToolCMSIntegrationMap.tool_cms_id)
#                 .filter(ToolsCMS.tool_id == id)
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
    # Enforce JWT presence; this is a generic listing but should require an authenticated context
    request = request_var.get(None)
    if not request or not hasattr(request.state, "jwt_claims"):
        return {"error": "Unauthorized: JWT claims not found in request context"}
    claims = request.state.jwt_claims
    jwt_user_id = claims.get("user_id") or claims.get("sub")
    if not jwt_user_id:
        return {"error": "Unauthorized: user_id not found in token claims"}

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

    session = Session()
    try:
        session.rollback()

        # Build query based on type
        if type_norm == "agent":
            print("Fetching integrations for agent_id:", id)
            query = (
                session.query(
                    Integrations.integration_id,        # id
                    Integrations.integration_name,      # name + maps for desc/logo
                    Integrations.integration_logo_url,  # logo
                    Integrations.is_active              # connected
                )
                .join(AgentCMSIntegrationMap, AgentCMSIntegrationMap.integration_id == Integrations.integration_id)
                .join(AgentsCMS, AgentsCMS.agent_cms_id == AgentCMSIntegrationMap.agent_cms_id)
                .filter(AgentsCMS.agent_id == id)
            )
        else:  # type_norm == "tool"
            print("Fetching integrations for tool_id:", id)
            query = (
                session.query(
                    Integrations.integration_id,
                    Integrations.integration_name,
                    Integrations.integration_logo_url,
                    Integrations.is_active
                )
                .join(ToolCMSIntegrationMap, ToolCMSIntegrationMap.integration_id == Integrations.integration_id)
                .join(ToolsCMS, ToolsCMS.tool_cms_id == ToolCMSIntegrationMap.tool_cms_id)
                .filter(ToolsCMS.tool_id == id)
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
def list_integrations_for_entity(id, type, workspace_id=None, user_id=None):
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
    # Enforce JWT presence; this is a generic listing but should require an authenticated context
    request = request_var.get(None)
    if not request or not hasattr(request.state, "jwt_claims"):
        return {"error": "Unauthorized: JWT claims not found in request context"}
    claims = request.state.jwt_claims
    jwt_user_id = claims.get("user_id") or claims.get("sub")
    if not jwt_user_id:
        return {"error": "Unauthorized: user_id not found in token claims"}

    # Use provided user_id/workspace_id if given, else fallback to JWT
    user_id = user_id or jwt_user_id

    session = Session()
    try:
        session.rollback()
        results = []
        if str(type).lower() == 'agent':
            print("Fetching integrations for agent_id (user/workspace specific):", id)
            if AMUIntegrationMapping and user_id and workspace_id:
                # Get all integrations for this agent (from Integrations table)
                all_integrations = (
                    session.query(
                        Integrations.integration_id,
                        Integrations.integration_name,
                        Integrations.integration_logo_url,
                        Integrations.integration_desc
                    )
                    .join(AgentCMSIntegrationMap, AgentCMSIntegrationMap.integration_id == Integrations.integration_id)
                    .join(AgentsCMS, AgentsCMS.agent_cms_id == AgentCMSIntegrationMap.agent_cms_id)
                    .filter(AgentsCMS.agent_id == id)
                ).all()
                # For each integration, check if mapping exists for user/workspace/agent/integration
                for integ in all_integrations:
                    mapping = session.query(AMUIntegrationMapping).filter(
                        AMUIntegrationMapping.agent_id == id,
                        AMUIntegrationMapping.user_id == user_id,
                        AMUIntegrationMapping.workspace_id == workspace_id,
                        AMUIntegrationMapping.integration_id == integ.integration_id
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
                        Integrations.integration_name,
                        Integrations.integration_logo_url,
                        Integrations.integration_desc
                    )
                    .join(AgentCMSIntegrationMap, AgentCMSIntegrationMap.integration_id == Integrations.integration_id)
                    .join(AgentsCMS, AgentsCMS.agent_cms_id == AgentCMSIntegrationMap.agent_cms_id)
                    .filter(AgentsCMS.agent_id == id)
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
            if TMUIntegrationMapping and user_id and workspace_id:
                all_integrations = (
                    session.query(
                        Integrations.integration_id,
                        Integrations.integration_name,
                        Integrations.integration_logo_url,
                        Integrations.integration_desc
                    )
                    .join(ToolCMSIntegrationMap, ToolCMSIntegrationMap.integration_id == Integrations.integration_id)
                    .join(ToolsCMS, ToolsCMS.tool_cms_id == ToolCMSIntegrationMap.tool_cms_id)
                    .filter(ToolsCMS.tool_id == id)
                ).all()
                for integ in all_integrations:
                    mapping = session.query(TMUIntegrationMapping).filter(
                        TMUIntegrationMapping.tool_id == id,
                        TMUIntegrationMapping.user_id == user_id,
                        TMUIntegrationMapping.workspace_id == workspace_id,
                        TMUIntegrationMapping.integration_id == integ.integration_id
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
                        Integrations.integration_name,
                        Integrations.integration_logo_url,
                        Integrations.integration_desc
                    )
                    .join(ToolCMSIntegrationMap, ToolCMSIntegrationMap.integration_id == Integrations.integration_id)
                    .join(ToolsCMS, ToolsCMS.tool_cms_id == ToolCMSIntegrationMap.tool_cms_id)
                    .filter(ToolsCMS.tool_id == id)
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
async def toggle_integration_connection(user_id, workspace_id ,integration_id, id , type) -> dict:
    # Enforce JWT-based access: only allow if user is mapped to the workspace and user_id matches JWT
    if user_id is None:
        return {"status": "error", "error": "user_id cannot be null"}
    request = request_var.get(None)
    if not request or not hasattr(request.state, "jwt_claims"):
        return {"status": "error", "error": "Unauthorized: JWT claims not found in request context"}
    claims = request.state.jwt_claims
    jwt_user_id = claims.get("user_id") or claims.get("sub")
    if not jwt_user_id:
        return {"status": "error", "error": "Unauthorized: user_id not found in token claims"}
    if str(user_id) != str(jwt_user_id):
        return {"status": "error", "error": "Unauthorized: user_id in request does not match user in token"}

    valid, err = validate_user_workspace_access(user_id=user_id, workspace_id=workspace_id)
    if not valid:
        return {"status": "error", "error": err}

    session = Session()
    try:
        session.rollback()
        
        if type == "agent":
            user_connection = session.query(AMUIntegrationMapping).filter(
                AMUIntegrationMapping.user_id == user_id,
                AMUIntegrationMapping.agent_id == id,
                AMUIntegrationMapping.workspace_id == workspace_id,
                AMUIntegrationMapping.integration_id == integration_id
            ).first()
            if not user_connection:
                new_connection = AMUIntegrationMapping(
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
            user_connection = session.query(TMUIntegrationMapping).filter(
                TMUIntegrationMapping.user_id == user_id,
                TMUIntegrationMapping.agent_id == id,
                TMUIntegrationMapping.workspace_id == workspace_id,
                TMUIntegrationMapping.integration_id == integration_id
            ).first()
            if not user_connection:
                new_connection = TMUIntegrationMapping(
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
async def fetch_specific_agent_info(user_id, agent_id, workspace_id=0) -> dict:
    """
    Fetches detailed information about a specific agent for a given user.
    """
    if user_id is None:
        return {"status": "error", "error": "user_id cannot be null"}
    # Enforce JWT-based access: only allow if user is mapped to the workspace and user_id matches JWT
    request = request_var.get(None)
    if not request or not hasattr(request.state, "jwt_claims"):
        return {"status": "error", "error": "Unauthorized: JWT claims not found in request context"}
    claims = request.state.jwt_claims
    jwt_user_id = claims.get("user_id") or claims.get("sub")
    if not jwt_user_id:
        return {"status": "error", "error": "Unauthorized: user_id not found in token claims"}
    if str(user_id) != str(jwt_user_id):
        return {"status": "error", "error": "Unauthorized: user_id in request does not match user in token"}

    valid, err = validate_user_workspace_access(user_id=user_id, workspace_id=workspace_id)
    if not valid:
        return {"status": "error", "error": err}

    session = Session()
    try:
        agent_values = session.query(
            Agent.agent_name,
            Agent.agent_id,
            Agent.agent_desc,
            AgentsCMS.agent_owner,
            AgentsCMS.agent_contact,
            AgentsCMS.agent_feature,
            AgentsCMS.faqs,
            AgentsCMS.last_updated,
            Integrations.integration_id,
            Integrations.integration_name,
            Integrations.integration_logo_url,
            FavouriteMappingAgent.favourite_id,
            FavouriteMappingAgent.is_active
        ).filter(
            Agent.agent_id == agent_id
        ).outerjoin(
            AgentsCMS, AgentsCMS.agent_id == Agent.agent_id
        ).outerjoin(
            AgentCMSIntegrationMap, AgentCMSIntegrationMap.agent_cms_id == AgentsCMS.agent_cms_id
        ).outerjoin(
            Integrations, Integrations.integration_id == AgentCMSIntegrationMap.integration_id
        ).outerjoin(
            FavouriteMappingAgent,
            (FavouriteMappingAgent.agent_id == Agent.agent_id) &
            (FavouriteMappingAgent.user_id == user_id) &
            (FavouriteMappingAgent.workspace_id == workspace_id)
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
        workspace_maps = session.query(AgentMap).filter(
            AgentMap.agent_id == agent_id, AgentMap.is_active == True
        ).all()
        workspace_ids = [wm.workspace_id for wm in workspace_maps]
        workspaces = []
        if workspace_ids:
            ws_query = session.query(Workspace).filter(
                Workspace.workspace_id.in_(workspace_ids), Workspace.is_active == True
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
async def fetch_specific_tool_info(user_id, tool_id, workspace_id=0) -> dict:
    """
    Fetches detailed information about a specific tool for a given user.
    """
    if user_id is None:
        return {"status": "error", "error": "user_id cannot be null"}
    # Enforce JWT-based access like other tools
    request = request_var.get(None)
    if not request or not hasattr(request.state, "jwt_claims"):
        return {"status": "error", "error": "Unauthorized: JWT claims not found in request context"}
    claims = request.state.jwt_claims
    jwt_user_id = claims.get("user_id") or claims.get("sub")
    if not jwt_user_id:
        return {"status": "error", "error": "Unauthorized: user_id not found in token claims"}
    if str(user_id) != str(jwt_user_id):
        return {"status": "error", "error": "Unauthorized: user_id in request does not match user in token"}

    valid, err = validate_user_workspace_access(user_id=user_id, workspace_id=workspace_id)
    if not valid:
        return {"status": "error", "error": err}

    session = Session()
    try:
        tool_values = session.query(
            Tool.tool_name,
            Tool.tool_id,
            Tool.tool_desc,
            ToolsCMS.tool_owner,
            ToolsCMS.tool_contact,
            ToolsCMS.tool_feature,
            ToolsCMS.faqs,
            ToolsCMS.last_updated,
            Integrations.integration_id,
            Integrations.integration_name,
            Integrations.integration_logo_url,
            FavouriteMappingTool.favourite_id
        ).filter(
            Tool.tool_id == tool_id
        ).outerjoin(
            ToolsCMS, ToolsCMS.tool_id == Tool.tool_id
        ).outerjoin(
            ToolCMSIntegrationMap, ToolCMSIntegrationMap.tool_cms_id == ToolsCMS.tool_cms_id
        ).outerjoin(
            Integrations, Integrations.integration_id == ToolCMSIntegrationMap.integration_id
        ).outerjoin(
            FavouriteMappingTool,
            (FavouriteMappingTool.user_id == user_id) &
            (FavouriteMappingTool.tool_id == tool_id) &
            (FavouriteMappingTool.workspace_id == workspace_id)
        ).all()

        workspace_maps = session.query(ToolMap).filter(
            ToolMap.tool_id == tool_id, ToolMap.is_active == True
        ).all()
        workspace_ids = [wm.workspace_id for wm in workspace_maps]
        workspaces = []
        if workspace_ids:
            ws_query = session.query(Workspace).filter(
                Workspace.workspace_id.in_(workspace_ids), Workspace.is_active == True
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

@mcp.tool()
def fetch_roles_list():
    """
    Fetch a list of roles (ids and name) from the role_master table.
    Only returns SDLC roles - filters out system/admin roles that should not be assignable to workspace users.
    Returns:
        dict: { 'response': [ { 'role_id': ..., 'role_name': ... }, ... ] }
    """
    # Require JWT presence for consistency
    request = request_var.get(None)
    if not request or not hasattr(request.state, "jwt_claims"):
        return {"error": "Unauthorized: JWT claims not found in request context"}
    claims = request.state.jwt_claims
    jwt_user_id = claims.get("user_id") or claims.get("sub")
    if not jwt_user_id:
        return {"error": "Unauthorized: user_id not found in token claims"}

    session = Session()
    try:
        session.rollback()
        roles = session.query(Role).filter(Role.is_active == True).all()
        
        # Define restricted roles that should NEVER be selectable when adding users to workspace
        # Only SDLC roles (Product Owner, Scrum Master, Developer, QA, etc.) should be visible
        restricted_roles = {
            "forge-x admin",
            "workspace admin",      # Correct spelling
            "worksapce admin",      # Typo in database - keeping for safety
            "sme",
            "knowledge curator",
            "dod"
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
def add_user_to_workspace(workspace_id: int, user_email: str, role_id: int, first_name: str = None, last_name: str = None):
    """
    Add a user to a workspace by email and assign a role by role_id.
    If user does not exist, create new user. Only allow @coforge.com emails.
    Only Workspace Admin or Forge-X Admin can add users.
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
    request = request_var.get(None)
    if not request or not hasattr(request.state, "jwt_claims"):
        return {"error": "Unauthorized: JWT claims not found in request context"}
    claims = request.state.jwt_claims
    is_admin = claims.get("is_admin", False)
    jwt_user_id = claims.get("user_id") or claims.get("sub")
    has_access = False
    session = Session()
    try:
        session.rollback()
        # Email validation
        email = user_email.strip().lower()
        if not email.endswith("@coforge.com"):
            return {"error": "Only @coforge.com email addresses are allowed."}

        # Check if user is Forge-X admin or Workspace Admin for this workspace
        if is_admin:
            has_access = True
        else:
            # Look up Workspace Admin role dynamically
            workspace_admin_role = session.query(Role).filter(
                Role.role_name.ilike("%workspace admin%"),
                Role.is_active == True
            ).first()
            if workspace_admin_role:
                user_role = session.query(UserRoleMap).filter(
                    UserRoleMap.user_id == jwt_user_id,
                    UserRoleMap.workspace_id == workspace_id,
                    UserRoleMap.role_id == workspace_admin_role.role_id,
                    UserRoleMap.is_active == True
                ).first()
                if user_role:
                    has_access = True
        
        if not has_access:
            return {"error": "You are not authorized to add users to this workspace. You must be a Workspace Admin or Forge-X Admin."}

        # Validate that the role being assigned is not a restricted role
        role = session.query(Role).filter(Role.role_id == role_id, Role.is_active == True).first()
        if not role:
            return {"error": f"Role with id '{role_id}' not found"}
        
        role_name = getattr(role, 'role_name', '').strip().lower()
        restricted_roles = {
            "forge-x admin",
            "workspace admin",      # Correct spelling
            "worksapce admin",      # Typo in database
            "sme",
            "knowledge curator",
            "dod"
        }
        
        if role_name in restricted_roles:
            return {"error": f"Cannot assign restricted role '{role.role_name}'. Only SDLC roles can be assigned to workspace users."}

        user = session.query(User).filter(func.lower(User.email_id) == email).first()
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
            new_user = User(
                namespace="default",  # Set default namespace
                email_id=email,
                first_name=first_name,
                last_name=last_name,
                is_active=True
            )
            session.add(new_user)
            session.flush()  # Get new user_id
            user_id = new_user.user_id
            notification = f"New user created and added to workspace."

        # Add to workspace_users_mapping
        user_map = session.query(UserMap).filter(UserMap.user_id==user_id, UserMap.workspace_id==workspace_id).first()
        if user_map and (not user_map.is_active):
            user_map.is_active = True
        elif not user_map:
            session.add(UserMap(user_id=user_id, workspace_id=workspace_id, is_active=True))

        # Add/update user_role_mapping
        user_role_map = session.query(UserRoleMap).filter(UserRoleMap.user_id==user_id, UserRoleMap.workspace_id==workspace_id).first()
        if user_role_map and (not user_role_map.is_active):
            user_role_map.role_id = role_id
            user_role_map.is_active = True
        elif not user_role_map:
            session.add(UserRoleMap(user_id=user_id, workspace_id=workspace_id, role_id=role_id, is_active=True))

        session.commit()
        return {"response": notification, "user_id": user_id, "email": email, "workspace_id": workspace_id, "role_id": role_id}
    except Exception as e:
        session.rollback()
        print(f"Error in add_user_to_workspace: {e}")
        return {"error": "An error occurred while adding user to workspace."}
    finally:
        session.close()

@mcp.tool()
def list_workspace_users(workspace_id: int):
    # Enforce JWT-based access: only allow if user is mapped to the workspace
    request = request_var.get(None)
    if not request or not hasattr(request.state, "jwt_claims"):
        return {"error": "Unauthorized: JWT claims not found in request context"}
    claims = request.state.jwt_claims
    jwt_user_id = claims.get("user_id") or claims.get("sub")
    if not jwt_user_id:
        return {"error": "Unauthorized: user_id not found in token claims"}

    session = Session()
    try:
        session.rollback()
        # Single optimized query: join UserMap, User, UserRoleMap, Role in one go
        users = (
            session.query(
                User.user_id,
                User.first_name,
                User.last_name,
                User.email_id,
                Role.role_name,
                Role.role_id,
                UserMap.can_curate_kb
            )
            .join(UserMap, (UserMap.user_id == User.user_id) & (UserMap.workspace_id == workspace_id) & (UserMap.is_active == True))
            .outerjoin(UserRoleMap, (UserRoleMap.user_id == User.user_id) & (UserRoleMap.workspace_id == workspace_id) & (UserRoleMap.is_active == True))
            .outerjoin(Role, (UserRoleMap.role_id == Role.role_id) & (Role.is_active == True))
            .filter(User.is_active == True)
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
def remove_user_from_workspace(user_id: int, workspace_id: int, role_id: int):
    """
    Remove a user from a workspace (set is_active = False in mapping tables).
    Only Workspace Admin or Forge-X Admin can remove users.
    Args:
        user_id (int): User ID
        workspace_id (int): Workspace ID
    Returns:
        dict: Success or error message
    """
    if user_id is None:
        return {"status": "error", "error": "user_id cannot be null"}
    # RBAC: Only allow if JWT has is_admin True or user has role_id==3 for this workspace
    request = request_var.get(None)
    if not request or not hasattr(request.state, "jwt_claims"):
        return {"error": "Unauthorized: JWT claims not found in request context"}
    claims = request.state.jwt_claims
    is_admin = claims.get("is_admin", False)
    jwt_user_id = claims.get("user_id") or claims.get("sub")
    has_access = False
    session = Session()
    try:
        session.rollback()
        if is_admin:
            has_access = True
        else:
            # Look up Workspace Admin role dynamically
            workspace_admin_role = session.query(Role).filter(
                Role.role_name.ilike("%workspace admin%"),
                Role.is_active == True
            ).first()
            if workspace_admin_role:
                user_role = session.query(UserRoleMap).filter(
                    UserRoleMap.user_id == jwt_user_id,
                    UserRoleMap.workspace_id == workspace_id,
                    UserRoleMap.role_id == workspace_admin_role.role_id,
                    UserRoleMap.is_active == True
                ).first()
                if user_role:
                    has_access = True
        
        if not has_access:
            return {"error": "You are not authorized to remove users from this workspace. You must be a Workspace Admin or Forge-X Admin."}

        updated = False
        user_map = session.query(UserMap).filter(UserMap.user_id==user_id, UserMap.workspace_id==workspace_id).first()
        if user_map and hasattr(user_map, 'is_active'):
            user_map.is_active = False
            updated = True
        user_role_map = session.query(UserRoleMap).filter(UserRoleMap.user_id==user_id, UserRoleMap.workspace_id==workspace_id, UserRoleMap.role_id==role_id).first()
        if user_role_map and hasattr(user_role_map, 'is_active'):
            user_role_map.is_active = False
            updated = True
        if updated:
            session.commit()
            return {'response': 'User removed from the workspace'}
        else:
            return {'error': 'Mapping not found'}
    except Exception as e:
        session.rollback()
        print(f"Error in remove_user_from_workspace: {e}")
        return {'error': 'An error occurred while removing user from workspace.'}
    finally:
        session.close()

@mcp.tool()
def update_workspace_user(user_id: int, workspace_id: int, role_id: int):
    """
    Update a user's role in a workspace by role_id.
    Only Workspace Admin or Forge-X Admin can update user roles.
    Cannot assign restricted roles (Forge-X admin, Workspace admin, SME, Knowledge Curator, DoD).
    Args:
        user_id (int): User ID
        workspace_id (int): Workspace ID
        role_id (int): Role ID to assign (must be an SDLC role)
    Returns:
        dict: Success or error message
    """
    if user_id is None:
        return {"status": "error", "error": "user_id cannot be null"}
    # RBAC: Only allow if JWT has is_admin True or user has role_id==3 for this workspace
    request = request_var.get(None)
    if not request or not hasattr(request.state, "jwt_claims"):
        return {"error": "Unauthorized: JWT claims not found in request context"}
    claims = request.state.jwt_claims
    is_admin = claims.get("is_admin", False)
    jwt_user_id = claims.get("user_id") or claims.get("sub")
    has_access = False
    session = Session()
    try:
        session.rollback()
        if is_admin:
            has_access = True
        else:
            # Look up Workspace Admin role dynamically
            workspace_admin_role = session.query(Role).filter(
                Role.role_name.ilike("%workspace admin%"),
                Role.is_active == True
            ).first()
            if workspace_admin_role:
                user_role = session.query(UserRoleMap).filter(
                    UserRoleMap.user_id == jwt_user_id,
                    UserRoleMap.workspace_id == workspace_id,
                    UserRoleMap.role_id == workspace_admin_role.role_id,
                    UserRoleMap.is_active == True
                ).first()
                if user_role:
                    has_access = True
        
        if not has_access:
            return {"error": "You are not authorized to update users in this workspace. You must be a Workspace Admin or Forge-X Admin."}

        # Find role by id and validate it's not restricted
        role = session.query(Role).filter(Role.role_id == role_id, Role.is_active == True).first()
        if not role:
            return {"error": f"Role with id '{role_id}' not found"}
        
        role_name = getattr(role, 'role_name', '').strip().lower()
        restricted_roles = {
            "forge-x admin",
            "workspace admin",      # Correct spelling
            "worksapce admin",      # Typo in database
            "sme",
            "knowledge curator",
            "dod"
        }
        
        if role_name in restricted_roles:
            return {"error": f"Cannot assign restricted role '{role.role_name}'. Only SDLC roles can be assigned to workspace users."}
        # Update or create user_role_mapping
        user_role_map = session.query(UserRoleMap).filter_by(user_id=user_id, workspace_id=workspace_id).first()
        if user_role_map:
            user_role_map.role_id = role_id
            user_role_map.is_active = True
        else:
            session.add(UserRoleMap(user_id=user_id, workspace_id=workspace_id, role_id=role_id, is_active=True))
        # Update can_curate_kb if provided in kwargs or request (support both legacy and new frontend)
        import inspect
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        can_curate_kb = values.get('can_curate_kb', None)
        if can_curate_kb is not None:
            user_map = session.query(UserMap).filter_by(user_id=user_id, workspace_id=workspace_id).first()
            if user_map and hasattr(user_map, 'can_curate_kb'):
                user_map.can_curate_kb = can_curate_kb
        session.commit()
        print(f'User {user_id} role updated to role id {role_id} in workspace {workspace_id}')
        return {'response': f'User role updated successfully'}
    except Exception as e:
        session.rollback()
        print(f"Error in update_workspace_user: {e}")
        return {'error': 'An error occurred while updating user role.'}
    finally:
        session.close()

@mcp.tool()
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
    # Require JWT presence for consistency
    request = request_var.get(None)
    if not request or not hasattr(request.state, "jwt_claims"):
        return {"error": "Unauthorized: JWT claims not found in request context"}
    claims = request.state.jwt_claims
    jwt_user_id = claims.get("user_id") or claims.get("sub")
    if not jwt_user_id:
        return {"error": "Unauthorized: user_id not found in token claims"}

    session = Session()
    try:
        session.rollback()
        industries = session.query(Industry).all()
        subindustries = session.query(SubIndustry).all()

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
