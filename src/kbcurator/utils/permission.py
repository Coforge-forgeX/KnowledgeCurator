
from typing import Optional
from sqlalchemy import text
from .constants import Role
from .db import db


def get_user_role_id(user_id: int, workspace_id: Optional[int] = None) -> Optional[int]:
	"""
	Return role_id for a user.

	- If workspace_id is provided, reads active role from public.user_role_mapping.
	- If workspace_id is not provided, reads role from public.users and falls back
	  to platform role ids based on is_admin when role_id is null.
	"""
	session = db.Session()
	try:
		# Normalize ids to integers so DB comparisons are type-safe.
		if workspace_id is not None:
			result = session.execute(
				text("""
				SELECT role_id
				FROM public.workspace_users_mapping
				WHERE user_id = :user_id
				  AND workspace_id = :workspace_id
				  AND is_active = TRUE
				ORDER BY last_updated DESC NULLS LAST, created_date DESC NULLS LAST
				LIMIT 1
				"""),
				{"user_id": user_id, "workspace_id": workspace_id}
			)
			row = result.fetchone()
			return row[0] if row else None

		result = session.execute(
			text("""
			SELECT role_id, is_admin
			FROM public.users
			WHERE user_id = :user_id
			  AND is_active = TRUE
			LIMIT 1
			"""),
			{"user_id": user_id}
		)
		row = result.fetchone()
		if not row:
			return None
		role_id, is_admin = row
		if role_id is not None:
			return role_id
		return Role.ADMIN.id if bool(is_admin) else Role.USER.id
	except Exception:
		session.rollback()
		return None
	finally:
		session.close()


def is_admin(user_id: int, workspace_id: Optional[int] = None) -> bool:
	"""
	Return whether user is admin.

	- If workspace_id is provided, checks workspace admin role from mapping.
	- If workspace_id is not provided, checks platform admin role.
	"""
	try:
		normalized_user_id = int(user_id)
		normalized_workspace_id = int(workspace_id)
	except (TypeError, ValueError):
		return False
	
	role_id = get_user_role_id(user_id=normalized_user_id, workspace_id=normalized_workspace_id)
	if workspace_id is not None:
		return role_id == Role.WS_ADMIN.id
	return role_id == Role.ADMIN.id


def is_workspace_manager(user_id: int, workspace_id: int) -> bool:
	"""Return whether user has Workspace Manager role in a workspace."""
	role_id = get_user_role_id(user_id=user_id, workspace_id=workspace_id)
	return role_id == Role.WS_MANAGER.id

