"""
Authentication & Authorization Service

Centralized service for JWT validation and workspace access control.
Following Single Responsibility Principle.
"""

import os
from typing import Optional, Tuple
import psycopg2


class AuthService:
    """
    Handles authentication and authorization for workspace operations.

    Responsibilities:
    - JWT token validation
    - User-workspace mapping verification
    - Role-based access control
    - Curation permission checks
    """

    @staticmethod
    def validate_user_workspace_access(
        user_id: int,
        workspace_id: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that user has access to the specified workspace.

        Args:
            user_id: User identifier
            workspace_id: Workspace identifier

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            conn = psycopg2.connect(
                host=os.environ["POSTGRES_HOST"],
                user=os.environ["POSTGRES_USER"],
                password=os.environ["POSTGRES_PASSWORD"],
                dbname=os.environ.get("POSTGRESQL_DATABASE_DATABASE_2"),
            )

            with conn.cursor() as cur:
                # Check if user is mapped to workspace and is active
                cur.execute(
                    """
                    SELECT user_id, is_active
                    FROM user_mapping
                    WHERE user_id = %s AND workspace_id = %s AND is_active = TRUE
                    """,
                    (user_id, workspace_id),
                )
                result = cur.fetchone()

                if not result:
                    return False, f"User {user_id} not authorized to access workspace {workspace_id}"

                return True, None

        except Exception as e:
            return False, f"Database error during authorization: {str(e)}"
        finally:
            if conn:
                conn.close()

    @staticmethod
    def check_curation_permission(
        user_id: int,
        workspace_id: int,
        role_id: int
    ) -> bool:
        """
        Check if user can curate (upload, edit, delete) in the workspace.

        Args:
            user_id: User identifier
            role_id: User role identifier
            workspace_id: Workspace identifier

        Returns:
            bool: True if user can curate, False if read-only
        """
        try:
            # Role ID 34 = SME (Subject Matter Expert) - has full curation rights
            # Other roles = workspace users - may have restricted access
            if role_id == 34:
                return True

            # Check workspace-specific permissions
            conn = psycopg2.connect(
                host=os.environ["POSTGRES_HOST"],
                user=os.environ["POSTGRES_USER"],
                password=os.environ["POSTGRES_PASSWORD"],
                dbname=os.environ.get("POSTGRESQL_DATABASE_DATABASE_2"),
            )

            with conn.cursor() as cur:
                # Check if user has curation permissions in workspace
                cur.execute(
                    """
                    SELECT can_curate
                    FROM user_workspace_permissions
                    WHERE user_id = %s AND workspace_id = %s
                    """,
                    (user_id, workspace_id),
                )
                result = cur.fetchone()

                if result:
                    return bool(result[0])

                # Default: no curation rights for non-SME users
                return False

        except Exception as e:
            print(f"Error checking curation permission: {e}")
            # Fail closed: deny curation on error
            return False
        finally:
            if conn:
                conn.close()

    @staticmethod
    def validate_chatbot_request_scope(
        user_id: int,
        workspace_id: int,
        role_id: int,
        industry: str,
        sub_industry: str,
        knowledge_bases: Optional[list] = None,
    ) -> Tuple[bool, Optional[str], bool]:
        """
        Validate chatbot request scope and permissions.

        Args:
            user_id: User identifier
            workspace_id: Workspace identifier
            role_id: User role identifier
            industry: Domain name
            sub_industry: Knowledge base name
            knowledge_bases: Optional list of KB suffixes

        Returns:
            Tuple of (is_valid, error_message, can_curate_kb)
        """
        try:
            # Validate user has workspace access
            valid, err = AuthService.validate_user_workspace_access(user_id, workspace_id)
            if not valid:
                return False, err, False

            # Check curation permissions
            can_curate = AuthService.check_curation_permission(user_id, workspace_id, role_id)

            # Validate domain/KB matches workspace configuration
            conn = psycopg2.connect(
                host=os.environ["POSTGRES_HOST"],
                user=os.environ["POSTGRES_USER"],
                password=os.environ["POSTGRES_PASSWORD"],
                dbname=os.environ.get("POSTGRESQL_DATABASE_DATABASE_2"),
            )

            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT domain, kb_name
                    FROM workspaces
                    WHERE workspace_id = %s
                    """,
                    (workspace_id,),
                )
                workspace_config = cur.fetchone()

                if not workspace_config:
                    return False, f"Workspace {workspace_id} not found", False

                expected_domain, expected_kb = workspace_config

                if industry != expected_domain or sub_industry != expected_kb:
                    return (
                        False,
                        f"Domain/KB mismatch: expected {expected_domain}/{expected_kb}, got {industry}/{sub_industry}",
                        False,
                    )

            return True, None, can_curate

        except Exception as e:
            return False, f"Validation error: {str(e)}", False
        finally:
            if conn:
                conn.close()
