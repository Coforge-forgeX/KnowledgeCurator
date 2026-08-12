"""Domain models shared across the chat subsystem."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class AccessContext:
    """
    Result of a single validated user+workspace access check.

    Carries every downstream handler needs so no handler has to re-query
    UserMap/Workspace for the same request.
    """

    user_id: int
    workspace_id: int
    role_id: int
    can_curate_kb: bool
    domain: str
    kb_name: str
    all_kb_titles: List[str] = field(default_factory=list)
    is_kg: bool = False

    @property
    def additional_kbs(self) -> Optional[List[str]]:
        if len(self.all_kb_titles) > 1:
            return self.all_kb_titles[1:]
        return None


@dataclass(frozen=True)
class WorkspaceMembership:
    """
    Result of a membership-only access check.

    Session-management endpoints (start/history/load/rename/delete) need to know
    *that* the user belongs to the workspace, not the workspace's KB
    configuration. Resolving the full :class:`AccessContext` for them would make
    them fail whenever a workspace has no KB wired up yet, which has nothing to
    do with owning a conversation.
    """

    user_id: int
    workspace_id: int
    role_id: int
    can_curate_kb: bool = False


@dataclass
class HandlerResult:
    """Uniform result shape produced by every mode handler."""

    response: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    task_ids: List[int] = field(default_factory=list)
    intent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
