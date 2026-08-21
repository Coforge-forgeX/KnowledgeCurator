from kbcurator.utils.access_validation import validate_user_workspace_access
from kbcurator.utils.permission import is_admin, get_user_role_id
from kbcurator.server.server import mcp , get_postgres_connection_string
import psycopg2
from configparser import ConfigParser
from sqlalchemy import func
from kbcurator.utils.db import db
from os import getenv
import sys
import threading
import requests
from kbcurator.utils.auth import create_jwt_token, verify_jwt_token, create_refresh_token, verify_refresh_token
from kbcurator.utils.request_context import request_var
from sqlalchemy import select, func , case 
from datetime import datetime, timezone
import os
from urllib.parse import quote_plus
import kbcurator.server.server as server

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

from datetime import datetime, timezone , timedelta , time
from kbcurator.utils.constants import DefaultValue, Role, WorkspaceType
from fastmcp.tools.tool import ToolResult
from cachetools import TTLCache
from cachetools import cached

from kbcurator.trustai_analytics.model.db_model import (
    AnalyticsEventFact,
    GuardrailOutcomeFact,
)

from kbcurator.trustai_analytics.trustai_db import (
    analytics_db,
)

_access_scope_cache = TTLCache(
    maxsize=100,
    ttl=300,
)


@cached(_access_scope_cache)
def _get_access_scope_cached(
    current_user_id: str,
):
    kb_session = db.Session()

    try:

        workspace_ids = [
            row.workspace_id
            for row in (
                kb_session.query(
                    db.UserMap.workspace_id
                )
                .join(
                    db.Workspace,
                    db.Workspace.workspace_id == db.UserMap.workspace_id
                )
                .filter(
                    db.UserMap.user_id == current_user_id,
                    db.UserMap.is_active == True,
                    db.Workspace.is_active == True,
                )
                .distinct()
                .all()
            )
        ]

        user_ids = []
        agent_ids = []

        if workspace_ids:

            user_ids = [
                str(row.email_id)
                for row in (
                    kb_session.query(
                        db.User.email_id
                    )
                    .join(
                        db.UserMap,
                        db.User.user_id == db.UserMap.user_id
                    )
                    .filter(
                        db.UserMap.workspace_id.in_(workspace_ids),
                        db.UserMap.is_active == True,
                        db.User.is_active == True,
                    )
                    .distinct()
                    .all()
                )
            ]

            agent_ids = [
                str(row.agent_id)
                for row in (
                    kb_session.query(
                        db.AgentMap.agent_id
                    )
                    .join(
                        db.Agent,
                        db.Agent.agent_id == db.AgentMap.agent_id
                    )
                    .filter(
                        db.AgentMap.workspace_id.in_(workspace_ids),
                        db.AgentMap.is_active == True,
                        db.Agent.is_active == True,
                    )
                    .distinct()
                    .all()
                )
            ]

        return (
            tuple(workspace_ids),
            tuple(user_ids),
            tuple(agent_ids),
        )

    finally:
        kb_session.close()


def _resolve_effective_scope(
    current_user_id,
    workspace_ids=None,
    user_ids=None,
    agent_ids=None,
):
    (
        accessible_workspace_ids,
        accessible_user_ids,
        accessible_agent_ids,
    ) = _get_access_scope_cached(
        str(current_user_id)
    )

    accessible_workspace_ids = list(
        accessible_workspace_ids
    )

    accessible_user_ids = list(
        accessible_user_ids
    )

    accessible_agent_ids = list(
        accessible_agent_ids
    )

    # --------------------------------------------------
    # Effective Workspaces
    # --------------------------------------------------

    effective_workspace_ids = (
        workspace_ids
        if workspace_ids
        else accessible_workspace_ids
    )

    invalid_workspaces = (
        set(effective_workspace_ids)
        - set(accessible_workspace_ids)
    )

    if invalid_workspaces:
        raise ValueError(
            f"Unauthorized workspace access: "
            f"{sorted(invalid_workspaces)}"
        )

    # --------------------------------------------------
    # Effective Users
    # --------------------------------------------------

    if user_ids:

        invalid_users = (
            set(user_ids)
            - set(accessible_user_ids)
        )

        if invalid_users:
            raise ValueError(
                f"Unauthorized user access: "
                f"{sorted(invalid_users)}"
            )

        effective_user_ids = user_ids

    else:

        effective_user_ids = (
            accessible_user_ids
        )

    # --------------------------------------------------
    # Effective Agents
    # --------------------------------------------------

    if agent_ids:

        invalid_agents = (
            set(agent_ids)
            - set(accessible_agent_ids)
        )

        if invalid_agents:
            raise ValueError(
                f"Unauthorized agent access: "
                f"{sorted(invalid_agents)}"
            )

        effective_agent_ids = agent_ids

    else:

        effective_agent_ids = (
            accessible_agent_ids
        )

    print("workspace_ids =", effective_workspace_ids)
    print("user_ids count =", effective_user_ids)
    print("agent_ids count =", effective_agent_ids)
    
    return {
        "workspace_ids": effective_workspace_ids,
        "user_ids": effective_user_ids,
        "agent_ids": effective_agent_ids,
    }



def _build_fact_filters(
    workspace_ids: list[int],
    user_ids: list[str],
    agent_ids: list[str],
    start_date: str | None = None,
    end_date: str | None = None
):
    """
    Creates reusable SQLAlchemy filters for
    AnalyticsEventFact queries.

    Filters on:
        - Time range
        - Workspace scope
        - User scope
        - Agent scope
    """

    start_dt = (
        datetime.combine(
            datetime.fromisoformat(start_date).date(),
            time.min,
        )
        if start_date
        else datetime.combine(
            (datetime.utcnow() - timedelta(days=30)).date(),
            time.min,
        )
    )

    end_dt = (
        datetime.fromisoformat(end_date)
        if end_date
        else datetime.utcnow()
    )

    filters = [
        AnalyticsEventFact.created_on >= start_dt,
        AnalyticsEventFact.created_on <= end_dt,
    ]

    if workspace_ids:
        filters.append(
            AnalyticsEventFact.app_name.in_(
                [str(ws_id) for ws_id in workspace_ids]
            )
        )

    if user_ids:
        filters.append(
            AnalyticsEventFact.user_id.in_(
                [str(user_id) for user_id in user_ids]
            )
        )

    if agent_ids:
        filters.append(
            AnalyticsEventFact.agent_id.in_(
                [str(agent_id) for agent_id in agent_ids]
            )
        )

    return filters


def _get_analytics_context(
    current_user_id: str,
    workspace_ids: list[int] | None,
    user_ids: list[str] | None,
    agent_ids: list[str] | None,
    start_date: str | None = None,
    end_date: str | None = None
):
    trustai_session = analytics_db.Session()
    kb_session = db.Session()

    scope = _resolve_effective_scope(
        current_user_id=current_user_id,
        workspace_ids=workspace_ids,
        user_ids=user_ids,
        agent_ids=agent_ids,
    )

    filters = _build_fact_filters(
        workspace_ids=scope["workspace_ids"],
        user_ids=scope["user_ids"],
        agent_ids=scope["agent_ids"],
        start_date = start_date,
        end_date= end_date
    )

    return {
        "trustai_session": trustai_session,
        "kb_session": kb_session,
        "workspace_ids": scope["workspace_ids"],
        "user_ids": scope["user_ids"],
        "agent_ids": scope["agent_ids"],
        "filters": filters,
    }
    



@mcp.tool()
@require_auth
def get_workspace_agent_user_directory():
    """
    Returns:
    {
        "workspaces": [
            {
                "workspace_id": 1,
                "workspace_name": "Demo"
            }
        ],
        "agents": [
            {
                "agent_id": 10,
                "agent_name": "PO Agent"
            }
        ],
        "users": [
            {
                "user_id": 100,
                "user_email": "john@coforge.com",
                "full_name": "John Doe"
            }
        ]
    }
    """
    _, jwt_user_id = get_current_user()

    session = db.Session()
    try:
        session.rollback()

        # ------------------------------------------------------------------
        # 1. Get all workspaces accessible to current user
        # ------------------------------------------------------------------
        workspace_rows = (
            session.query(
                db.Workspace.workspace_id,
                db.Workspace.workspace_name
            )
            .join(
                db.UserMap,
                db.UserMap.workspace_id == db.Workspace.workspace_id
            )
            .filter(
                db.UserMap.user_id == jwt_user_id,
                db.UserMap.is_active == True,
                db.Workspace.is_active == True
            )
            .all()
        )

        workspaces = [
            {
                "workspace_id": row.workspace_id,
                "workspace_name": row.workspace_name,
            }
            for row in workspace_rows
        ]

        workspace_ids = [row.workspace_id for row in workspace_rows]

        if not workspace_ids:
            return {
                "workspaces": [],
                "agents": [],
                "users": []
            }

        # ------------------------------------------------------------------
        # 2. Get union of agents from all accessible workspaces
        # ------------------------------------------------------------------
        agent_rows = (
            session.query(
                db.Agent.agent_id,
                db.Agent.agent_name
            )
            .join(
                db.AgentMap,
                db.AgentMap.agent_id == db.Agent.agent_id
            )
            .filter(
                db.AgentMap.workspace_id.in_(workspace_ids),
                db.AgentMap.is_active == True,
                db.Agent.is_active == True
            )
            .distinct(db.Agent.agent_id)
            .all()
        )

        agents = [
            {
                "agent_id": row.agent_id,
                "agent_name": row.agent_name,
            }
            for row in agent_rows
        ]

        # ------------------------------------------------------------------
        # 3. Get union of users from all accessible workspaces
        # ------------------------------------------------------------------
        user_rows = (
            session.query(
                db.User.user_id,
                db.User.email_id,
                db.User.first_name,
                db.User.last_name
            )
            .join(
                db.UserMap,
                db.UserMap.user_id == db.User.user_id
            )
            .filter(
                db.UserMap.workspace_id.in_(workspace_ids),
                db.UserMap.is_active == True,
                db.User.is_active == True
            )
            .distinct(db.User.user_id)
            .all()
        )

        users = [
            {
                "user_id": row.user_id,
                "user_email": row.email_id,
                "full_name": f"{row.first_name or ''} {row.last_name or ''}".strip()
            }
            for row in user_rows
        ]

        return {
            "workspaces": workspaces,
            "agents": agents,
            "users": users
        }

    except Exception as e:
        session.rollback()
        print(f"Error in get_workspace_agent_user_directory: {e}")
        return {
            "error": str(e)
        }
    finally:
        session.close()
        

@mcp.tool()
@require_auth
def get_overall_metrics(
    workspace_ids: list[int] | None = None,
    user_ids: list[str] | None = None,
    agent_ids: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None
):
    """
    Returns overall analytics metrics for the
    requested scope.
    """

    _, current_user_id = get_current_user()

    ctx = _get_analytics_context(
        current_user_id=current_user_id,
        workspace_ids=workspace_ids,
        user_ids=user_ids,
        agent_ids=agent_ids,
        start_date = start_date,
        end_date= end_date
    )

    try:

        trustai_session = ctx["trustai_session"]

        overall = (
            trustai_session.query(
                func.count().label(
                    "total_requests"
                ),

                func.count(
                    func.distinct(
                        AnalyticsEventFact.user_id
                    )
                ).label(
                    "unique_active_users"
                ),

                func.count(
                    func.distinct(
                        AnalyticsEventFact.agent_id
                    )
                ).label(
                    "agents_used"
                ),

                func.sum(
                    func.coalesce(
                        AnalyticsEventFact.llm_total_tokens,
                        0
                    )
                    +
                    func.coalesce(
                        AnalyticsEventFact.ig_total_tokens,
                        0
                    )
                    +
                    func.coalesce(
                        AnalyticsEventFact.og_total_tokens,
                        0
                    )
                ).label(
                    "overall_token_consumed"
                ),

                func.sum(
                    case(
                        (
                            AnalyticsEventFact.outcome == "Block",
                            1,
                        ),
                        else_=0,
                    )
                ).label(
                    "blocked_requests"
                ),

                func.sum(
                    case(
                        (
                            AnalyticsEventFact.outcome.in_(
                                ["Pass", "Warn"]
                            ),
                            1,
                        ),
                        else_=0,
                    )
                ).label(
                    "passed_requests"
                ),
            )
            .filter(
                *ctx["filters"]
            )
            .one()
        )

        daily_active_users_rows = (
            trustai_session.query(
                func.date(
                    AnalyticsEventFact.created_on
                ).label(
                    "activity_date"
                ),

                func.count(
                    func.distinct(
                        AnalyticsEventFact.user_id
                    )
                ).label(
                    "active_users"
                ),
            )
            .filter(
                *ctx["filters"]
            )
            .group_by(
                func.date(
                    AnalyticsEventFact.created_on
                )
            )
            .all()
        )

        avg_daily_active_users = (
            round(
                sum(
                    row.active_users or 0
                    for row in daily_active_users_rows
                )
                / len(daily_active_users_rows),
                2,
            )
            if daily_active_users_rows
            else 0
        )

        total_requests = (
            overall.total_requests or 0
        )

        total_passed = (
            overall.passed_requests or 0
        )

        pass_percentage = (
            round(
                (
                    total_passed
                    * 100.0
                    / total_requests
                ),
                2,
            )
            if total_requests > 0
            else 0
        )

        result = {
            "overall_token_consumed":
                overall.overall_token_consumed or 0,

            "overall_total_request":
                total_requests,

            "total_request_passed":
                total_passed,

            "total_request_pass_percentage":
                pass_percentage,

            "total_requests_blocked":
                overall.blocked_requests or 0,

            "count_of_workspace_passed":
                len(
                    ctx["workspace_ids"]
                ),

            "union_count_of_agents_present":
                len(
                    ctx["agent_ids"]
                ),

            "count_of_agents_that_got_request":
                overall.agents_used or 0,

            "total_unique_active_users":
                overall.unique_active_users or 0,

            "avg_daily_active_users":
                avg_daily_active_users,
        }

        return result

    finally:
        ctx["trustai_session"].close()
        ctx["kb_session"].close()
        
        
@mcp.tool()
@require_auth
def get_workspace_metrics(
    workspace_ids: list[int] | None = None,
    user_ids: list[str] | None = None,
    agent_ids: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None
):
    """
    Returns workspace-level analytics metrics.
    """

    _, current_user_id = get_current_user()

    ctx = _get_analytics_context(
        current_user_id=current_user_id,
        workspace_ids=workspace_ids,
        user_ids=user_ids,
        agent_ids=agent_ids,
        start_date = start_date,
        end_date= end_date
    )

    try:

        trustai_session = ctx["trustai_session"]
        kb_session = ctx["kb_session"]

        # --------------------------------------------------
        # Workspace Names
        # --------------------------------------------------

        workspace_name_lookup = {
            str(row.workspace_id): row.workspace_name
            for row in (
                kb_session.query(
                    db.Workspace.workspace_id,
                    db.Workspace.workspace_name,
                )
                .filter(
                    db.Workspace.workspace_id.in_(
                        ctx["workspace_ids"]
                    )
                )
                .all()
            )
        }

        # --------------------------------------------------
        # Users In Workspace
        # --------------------------------------------------

        workspace_user_counts = {
            str(row.workspace_id): row.user_count
            for row in (
                kb_session.query(
                    db.UserMap.workspace_id,
                    func.count(
                        func.distinct(
                            db.UserMap.user_id
                        )
                    ).label(
                        "user_count"
                    ),
                )
                .filter(
                    db.UserMap.workspace_id.in_(
                        ctx["workspace_ids"]
                    )
                )
                .group_by(
                    db.UserMap.workspace_id
                )
                .all()
            )
        }

        # --------------------------------------------------
        # Agents Added To Workspace
        # --------------------------------------------------

        workspace_agent_counts = {
            str(row.workspace_id): row.agent_count
            for row in (
                kb_session.query(
                    db.AgentMap.workspace_id,
                    func.count(
                        func.distinct(
                            db.AgentMap.agent_id
                        )
                    ).label(
                        "agent_count"
                    ),
                )
                .filter(
                    db.AgentMap.workspace_id.in_(
                        ctx["workspace_ids"]
                    )
                )
                .group_by(
                    db.AgentMap.workspace_id
                )
                .all()
            )
        }

        # --------------------------------------------------
        # Analytics Aggregation
        # --------------------------------------------------

        rows = (
            trustai_session.query(
                AnalyticsEventFact.app_name,

                func.count().label(
                    "request_count"
                ),

                func.count(
                    func.distinct(
                        AnalyticsEventFact.user_id
                    )
                ).label(
                    "active_users"
                ),

                func.count(
                    func.distinct(
                        AnalyticsEventFact.agent_id
                    )
                ).label(
                    "agents_used"
                ),

                func.sum(
                    case(
                        (
                            AnalyticsEventFact.outcome == "Warn",
                            1,
                        ),
                        else_=0,
                    )
                ).label(
                    "warnings"
                ),

                func.sum(
                    case(
                        (
                            AnalyticsEventFact.outcome == "Block",
                            1,
                        ),
                        else_=0,
                    )
                ).label(
                    "blocks"
                ),

                func.sum(
                    func.coalesce(
                        AnalyticsEventFact.ig_input_tokens,
                        0,
                    )
                ).label(
                    "ig_input_tokens"
                ),

                func.sum(
                    func.coalesce(
                        AnalyticsEventFact.ig_output_tokens,
                        0,
                    )
                ).label(
                    "ig_output_tokens"
                ),

                func.sum(
                    func.coalesce(
                        AnalyticsEventFact.ig_total_tokens,
                        0,
                    )
                ).label(
                    "ig_total_tokens"
                ),

                func.sum(
                    func.coalesce(
                        AnalyticsEventFact.og_input_tokens,
                        0,
                    )
                ).label(
                    "og_input_tokens"
                ),

                func.sum(
                    func.coalesce(
                        AnalyticsEventFact.og_output_tokens,
                        0,
                    )
                ).label(
                    "og_output_tokens"
                ),

                func.sum(
                    func.coalesce(
                        AnalyticsEventFact.og_total_tokens,
                        0,
                    )
                ).label(
                    "og_total_tokens"
                ),

                func.sum(
                    func.coalesce(
                        AnalyticsEventFact.llm_input_tokens,
                        0,
                    )
                ).label(
                    "llm_input_tokens"
                ),

                func.sum(
                    func.coalesce(
                        AnalyticsEventFact.llm_output_tokens,
                        0,
                    )
                ).label(
                    "llm_output_tokens"
                ),

                func.sum(
                    func.coalesce(
                        AnalyticsEventFact.llm_total_tokens,
                        0,
                    )
                ).label(
                    "llm_total_tokens"
                ),

                func.avg(
                    AnalyticsEventFact.duration
                ).label(
                    "avg_response_time"
                ),
            )
            .filter(
                *ctx["filters"]
            )
            .group_by(
                AnalyticsEventFact.app_name
            )
            .all()
        )

        metrics = []

        for row in rows:

            workspace_id = str(
                row.app_name
            )

            warnings = row.warnings or 0
            blocks = row.blocks or 0

            metrics.append(
                {
                    "workspace_id": workspace_id,

                    "workspace_name":
                        workspace_name_lookup.get(
                            workspace_id
                        ),

                    "users_in_workspace":
                        workspace_user_counts.get(
                            workspace_id,
                            0,
                        ),

                    "active_users":
                        row.active_users or 0,

                    "agents_added_in_workspace":
                        workspace_agent_counts.get(
                            workspace_id,
                            0,
                        ),

                    "agents_used":
                        row.agents_used or 0,

                    "requests":
                        row.request_count or 0,

                    "violations":
                        warnings + blocks,

                    "warnings":
                        warnings,

                    "blocks":
                        blocks,

                    "input_guardrails": {
                        "input_tokens":
                            row.ig_input_tokens or 0,

                        "output_tokens":
                            row.ig_output_tokens or 0,

                        "total_tokens":
                            row.ig_total_tokens or 0,
                    },

                    "output_guardrails": {
                        "input_tokens":
                            row.og_input_tokens or 0,

                        "output_tokens":
                            row.og_output_tokens or 0,

                        "total_tokens":
                            row.og_total_tokens or 0,
                    },

                    "llm": {
                        "input_tokens":
                            row.llm_input_tokens or 0,

                        "output_tokens":
                            row.llm_output_tokens or 0,

                        "total_tokens":
                            row.llm_total_tokens or 0,
                    },

                    "avg_response_time":
                        round(
                            float(
                                row.avg_response_time or 0
                            ),
                            2,
                        ),
                }
            )

        return {"metrics": metrics}

    finally:

        ctx["trustai_session"].close()
        ctx["kb_session"].close()
                        

@mcp.tool()
@require_auth
def get_agent_metrics(
    workspace_ids: list[int] | None = None,
    user_ids: list[str] | None = None,
    agent_ids: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None
):
    """
    Returns agent-level analytics metrics.
    """

    _, current_user_id = get_current_user()

    ctx = _get_analytics_context(
        current_user_id=current_user_id,
        workspace_ids=workspace_ids,
        user_ids=user_ids,
        agent_ids=agent_ids,
        start_date = start_date,
        end_date= end_date
    )

    try:

        trustai_session = ctx["trustai_session"]
        kb_session = ctx["kb_session"]

        # --------------------------------------------------
        # Agent Name Lookup
        # --------------------------------------------------

        agent_name_lookup = {
            str(row.agent_id): row.agent_name
            for row in (
                kb_session.query(
                    db.Agent.agent_id,
                    db.Agent.agent_name,
                )
                .filter(
                    db.Agent.agent_id.in_(
                        ctx["agent_ids"]
                    )
                )
                .all()
            )
        }

        # --------------------------------------------------
        # Agent Metrics Aggregation
        # --------------------------------------------------

        rows = (
            trustai_session.query(
                AnalyticsEventFact.agent_id,

                func.count().label(
                    "request_count"
                ),

                func.count(
                    func.distinct(
                        AnalyticsEventFact.user_id
                    )
                ).label(
                    "unique_users"
                ),

                func.sum(
                    case(
                        (
                            AnalyticsEventFact.outcome == "Warn",
                            1,
                        ),
                        else_=0,
                    )
                ).label(
                    "warnings"
                ),

                func.sum(
                    case(
                        (
                            AnalyticsEventFact.outcome == "Block",
                            1,
                        ),
                        else_=0,
                    )
                ).label(
                    "blocks"
                ),

                func.sum(
                    func.coalesce(
                        AnalyticsEventFact.ig_input_tokens,
                        0,
                    )
                ).label(
                    "ig_input_tokens"
                ),

                func.sum(
                    func.coalesce(
                        AnalyticsEventFact.ig_output_tokens,
                        0,
                    )
                ).label(
                    "ig_output_tokens"
                ),

                func.sum(
                    func.coalesce(
                        AnalyticsEventFact.ig_total_tokens,
                        0,
                    )
                ).label(
                    "ig_total_tokens"
                ),

                func.sum(
                    func.coalesce(
                        AnalyticsEventFact.og_input_tokens,
                        0,
                    )
                ).label(
                    "og_input_tokens"
                ),

                func.sum(
                    func.coalesce(
                        AnalyticsEventFact.og_output_tokens,
                        0,
                    )
                ).label(
                    "og_output_tokens"
                ),

                func.sum(
                    func.coalesce(
                        AnalyticsEventFact.og_total_tokens,
                        0,
                    )
                ).label(
                    "og_total_tokens"
                ),

                func.sum(
                    func.coalesce(
                        AnalyticsEventFact.llm_input_tokens,
                        0,
                    )
                ).label(
                    "llm_input_tokens"
                ),

                func.sum(
                    func.coalesce(
                        AnalyticsEventFact.llm_output_tokens,
                        0,
                    )
                ).label(
                    "llm_output_tokens"
                ),

                func.sum(
                    func.coalesce(
                        AnalyticsEventFact.llm_total_tokens,
                        0,
                    )
                ).label(
                    "llm_total_tokens"
                ),

                func.avg(
                    AnalyticsEventFact.duration
                ).label(
                    "avg_response_time"
                ),
            )
            .filter(
                *ctx["filters"]
            )
            .filter(
                AnalyticsEventFact.agent_id.isnot(None)
            )
            .filter(
                AnalyticsEventFact.agent_id != ""
            )
            .group_by(
                AnalyticsEventFact.agent_id
            )
            .order_by(
                func.count().desc()
            )
            .all()
        )

        metrics = []

        for row in rows:

            warnings = row.warnings or 0
            blocks = row.blocks or 0

            metrics.append(
                {
                    "agent_id": row.agent_id,

                    "agent_name":
                        agent_name_lookup.get(
                            str(row.agent_id)
                        ),

                    "requests":
                        row.request_count or 0,

                    "violations":
                        warnings + blocks,

                    "warning_events":
                        warnings,

                    "blocked_requests":
                        blocks,

                    "unique_users":
                        row.unique_users or 0,

                    "input_guardrails": {
                        "input_tokens":
                            row.ig_input_tokens or 0,
                        "output_tokens":
                            row.ig_output_tokens or 0,
                        "total_tokens":
                            row.ig_total_tokens or 0,
                    },

                    "output_guardrails": {
                        "input_tokens":
                            row.og_input_tokens or 0,
                        "output_tokens":
                            row.og_output_tokens or 0,
                        "total_tokens":
                            row.og_total_tokens or 0,
                    },

                    "llm": {
                        "input_tokens":
                            row.llm_input_tokens or 0,
                        "output_tokens":
                            row.llm_output_tokens or 0,
                        "total_tokens":
                            row.llm_total_tokens or 0,
                    },

                    "avg_response_time":
                        round(
                            float(
                                row.avg_response_time or 0
                            ),
                            2,
                        ),
                }
            )

        return {"metrics":metrics}

    finally:
        ctx["trustai_session"].close()
        ctx["kb_session"].close()
                            
                            
@mcp.tool()
@require_auth
def get_user_metrics(
    workspace_ids: list[int] | None = None,
    user_ids: list[str] | None = None,
    agent_ids: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None
):
    """
    Returns user-level analytics metrics.
    """

    _, current_user_id = get_current_user()

    ctx = _get_analytics_context(
        current_user_id=current_user_id,
        workspace_ids=workspace_ids,
        user_ids=user_ids,
        agent_ids=agent_ids,
        start_date = start_date,
        end_date= end_date
    )

    try:

        trustai_session = ctx["trustai_session"]
        kb_session = ctx["kb_session"]

        # ------------------------------------------
        # User Name Lookup (Optional)
        # ------------------------------------------

        user_name_lookup = {
            str(row.email_id).lower(): {
                "user_name": (
                    f"{row.first_name or ''} "
                    f"{row.last_name or ''}"
                ).strip(),
                "email": row.email_id,
            }
            for row in (
                kb_session.query(db.User)
                .filter(
                    db.User.email_id.in_(ctx["user_ids"])
                )
                .all()
            )
        }

        # ------------------------------------------
        # User Aggregation
        # ------------------------------------------

        rows = (
            trustai_session.query(
                AnalyticsEventFact.user_id,

                func.count().label(
                    "request_count"
                ),

                func.sum(
                    case(
                        (
                            AnalyticsEventFact.outcome == "Warn",
                            1,
                        ),
                        else_=0,
                    )
                ).label(
                    "warnings"
                ),

                func.sum(
                    case(
                        (
                            AnalyticsEventFact.outcome == "Block",
                            1,
                        ),
                        else_=0,
                    )
                ).label(
                    "blocks"
                ),

                func.sum(
                    func.coalesce(
                        AnalyticsEventFact.ig_input_tokens,
                        0,
                    )
                ).label(
                    "ig_input_tokens"
                ),

                func.sum(
                    func.coalesce(
                        AnalyticsEventFact.ig_output_tokens,
                        0,
                    )
                ).label(
                    "ig_output_tokens"
                ),

                func.sum(
                    func.coalesce(
                        AnalyticsEventFact.ig_total_tokens,
                        0,
                    )
                ).label(
                    "ig_total_tokens"
                ),

                func.sum(
                    func.coalesce(
                        AnalyticsEventFact.og_input_tokens,
                        0,
                    )
                ).label(
                    "og_input_tokens"
                ),

                func.sum(
                    func.coalesce(
                        AnalyticsEventFact.og_output_tokens,
                        0,
                    )
                ).label(
                    "og_output_tokens"
                ),

                func.sum(
                    func.coalesce(
                        AnalyticsEventFact.og_total_tokens,
                        0,
                    )
                ).label(
                    "og_total_tokens"
                ),

                func.sum(
                    func.coalesce(
                        AnalyticsEventFact.llm_input_tokens,
                        0,
                    )
                ).label(
                    "llm_input_tokens"
                ),

                func.sum(
                    func.coalesce(
                        AnalyticsEventFact.llm_output_tokens,
                        0,
                    )
                ).label(
                    "llm_output_tokens"
                ),

                func.sum(
                    func.coalesce(
                        AnalyticsEventFact.llm_total_tokens,
                        0,
                    )
                ).label(
                    "llm_total_tokens"
                ),

                (
                    func.sum(
                        func.coalesce(
                            AnalyticsEventFact.llm_total_tokens,
                            0,
                        )
                    )
                    +
                    func.sum(
                        func.coalesce(
                            AnalyticsEventFact.ig_total_tokens,
                            0,
                        )
                    )
                    +
                    func.sum(
                        func.coalesce(
                            AnalyticsEventFact.og_total_tokens,
                            0,
                        )
                    )
                ).label(
                    "total_token_consumption"
                ),

                func.count(
                    func.distinct(
                        func.date(
                            AnalyticsEventFact.created_on
                        )
                    )
                ).label(
                    "active_days"
                ),
            )
            .filter(
                *ctx["filters"]
            )
            .group_by(
                AnalyticsEventFact.user_id
            )
            .order_by(
                func.count().desc()
            )
            .all()
        )

        metrics = []

        for row in rows:

            warnings = row.warnings or 0
            blocks = row.blocks or 0

            user_info = user_name_lookup.get(
                str(row.user_id),
                {},
            )

            metrics.append(
                {
                    "user_id": row.user_id,

                    "user_name":
                        user_info.get(
                            "user_name"
                        ),

                    "email":
                        user_info.get(
                            "email"
                        ),

                    "requests":
                        row.request_count or 0,

                    "violations":
                        warnings + blocks,

                    "warnings":
                        warnings,

                    "blocks":
                        blocks,

                    "input_guardrails": {
                        "input_tokens":
                            row.ig_input_tokens or 0,
                        "output_tokens":
                            row.ig_output_tokens or 0,
                        "total_tokens":
                            row.ig_total_tokens or 0,
                    },

                    "output_guardrails": {
                        "input_tokens":
                            row.og_input_tokens or 0,
                        "output_tokens":
                            row.og_output_tokens or 0,
                        "total_tokens":
                            row.og_total_tokens or 0,
                    },

                    "llm": {
                        "input_tokens":
                            row.llm_input_tokens or 0,
                        "output_tokens":
                            row.llm_output_tokens or 0,
                        "total_tokens":
                            row.llm_total_tokens or 0,
                    },

                    "total_token_consumption":
                        row.total_token_consumption or 0,

                    "active_days":
                        row.active_days or 0,
                }
            )

        return {"metrics": metrics}

    finally:

        ctx["trustai_session"].close()
        ctx["kb_session"].close()
        
@mcp.tool()
@require_auth
def get_block_warn_pass_metrics(
    workspace_ids: list[int] | None = None,
    user_ids: list[str] | None = None,
    agent_ids: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None
):
    """
    Returns date-wise Pass/Warn/Block trend.
    """

    _, current_user_id = get_current_user()

    ctx = _get_analytics_context(
        current_user_id=current_user_id,
        workspace_ids=workspace_ids,
        user_ids=user_ids,
        agent_ids=agent_ids,
        start_date = start_date,
        end_date= end_date
    )

    try:

        trustai_session = ctx["trustai_session"]

        rows = (
            trustai_session.query(
                func.date(
                    AnalyticsEventFact.created_on
                ).label(
                    "activity_date"
                ),

                func.sum(
                    case(
                        (
                            AnalyticsEventFact.outcome == "Pass",
                            1,
                        ),
                        else_=0,
                    )
                ).label(
                    "pass_count"
                ),

                func.sum(
                    case(
                        (
                            AnalyticsEventFact.outcome == "Warn",
                            1,
                        ),
                        else_=0,
                    )
                ).label(
                    "warn_count"
                ),

                func.sum(
                    case(
                        (
                            AnalyticsEventFact.outcome == "Block",
                            1,
                        ),
                        else_=0,
                    )
                ).label(
                    "block_count"
                ),
            )
            .filter(
                *ctx["filters"]
            )
            .group_by(
                func.date(
                    AnalyticsEventFact.created_on
                )
            )
            .order_by(
                func.date(
                    AnalyticsEventFact.created_on
                ).desc()
            )
            .all()
        )

        return {"metrics": [
                {
                    "date": row.activity_date.isoformat(),
                    "pass": row.pass_count or 0,
                    "warn": row.warn_count or 0,
                    "block": row.block_count or 0,
                }
                for row in rows
            ]
        }

    finally:

        ctx["trustai_session"].close()
        ctx["kb_session"].close()
        
        
@mcp.tool()
@require_auth
def get_llm_model_token_and_request_metrics(
    workspace_ids: list[int] | None = None,
    user_ids: list[str] | None = None,
    agent_ids: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None
):
    """
    Returns model-wise token consumption
    and request distribution.
    """

    _, current_user_id = get_current_user()

    ctx = _get_analytics_context(
        current_user_id=current_user_id,
        workspace_ids=workspace_ids,
        user_ids=user_ids,
        agent_ids=agent_ids,
        start_date = start_date,
        end_date= end_date
    )

    try:

        trustai_session = ctx["trustai_session"]

        rows = (
            trustai_session.query(
                AnalyticsEventFact.llm_type,

                func.sum(
                    func.coalesce(
                        AnalyticsEventFact.llm_input_tokens,
                        0,
                    )
                ).label(
                    "overall_input_tokens"
                ),

                func.sum(
                    func.coalesce(
                        AnalyticsEventFact.llm_output_tokens,
                        0,
                    )
                ).label(
                    "overall_output_tokens"
                ),

                func.sum(
                    func.coalesce(
                        AnalyticsEventFact.llm_total_tokens,
                        0,
                    )
                ).label(
                    "overall_total_tokens"
                ),

                func.count().label(
                    "total_request_count"
                ),
            )
            .filter(
                *ctx["filters"]
            )
            .filter(
                AnalyticsEventFact.llm_type.isnot(None)
            )
            .group_by(
                AnalyticsEventFact.llm_type
            )
            .order_by(
                func.count().desc()
            )
            .all()
        )

        overall_request_count = sum(
            row.total_request_count
            for row in rows
        )

        metrics = []

        for row in rows:

            percent_req = (
                round(
                    (
                        row.total_request_count
                        * 100.0
                    )
                    / overall_request_count,
                    2,
                )
                if overall_request_count > 0
                else 0
            )

            metrics.append(
                {
                    "llm_type":
                        row.llm_type,

                    "overall_input_tokens":
                        row.overall_input_tokens or 0,

                    "overall_output_tokens":
                        row.overall_output_tokens or 0,

                    "overall_total_tokens":
                        row.overall_total_tokens or 0,

                    "total_request_count":
                        row.total_request_count or 0,

                    "percent_req":
                        percent_req,
                }
            )

        return {"metrics": metrics}

    finally:

        ctx["trustai_session"].close()
        ctx["kb_session"].close()
        
        
@mcp.tool()
@require_auth
def get_guardrail_detection_metrics(
    workspace_ids: list[int] | None = None,
    user_ids: list[str] | None = None,
    agent_ids: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None
):
    """
    Returns guardrail detection metrics.

    Output:
    [
        {
            "guardrail_name": "...",
            "total_warn_count": ...,
            "total_block_count": ...,
            "total_detect_count": ...,
            "detect_percentage": ...
        }
    ]
    """

    _, current_user_id = get_current_user()

    ctx = _get_analytics_context(
        current_user_id=current_user_id,
        workspace_ids=workspace_ids,
        user_ids=user_ids,
        agent_ids=agent_ids,
        start_date = start_date,
        end_date= end_date
    )
    
    start_dt = (
        datetime.combine(
            datetime.fromisoformat(start_date).date(),
            time.min,
        )
        if start_date
        else datetime.combine(
            (datetime.utcnow() - timedelta(days=30)).date(),
            time.min,
        )
    )

    end_dt = (
        datetime.fromisoformat(end_date)
        if end_date
        else datetime.utcnow()
    )

    try:

        trustai_session = ctx["trustai_session"]

        query = (
            trustai_session.query(
                GuardrailOutcomeFact.eval_name,

                func.sum(
                    case(
                        (
                            GuardrailOutcomeFact.outcome
                            == "Warn",
                            1,
                        ),
                        else_=0,
                    )
                ).label(
                    "total_warn_count"
                ),

                func.sum(
                    case(
                        (
                            GuardrailOutcomeFact.outcome
                            == "Block",
                            1,
                        ),
                        else_=0,
                    )
                ).label(
                    "total_block_count"
                ),

                func.count().label(
                    "total_detect_count"
                ),
            )
            .filter(
                GuardrailOutcomeFact.created_on
                >= start_dt
            )
            .filter(
                GuardrailOutcomeFact.created_on
                <= end_dt
            )
            .filter(
                GuardrailOutcomeFact.app_name.in_(
                    [
                        str(ws_id)
                        for ws_id in ctx["workspace_ids"]
                    ]
                )
            )
        )

        if ctx["user_ids"]:
            query = query.filter(
                GuardrailOutcomeFact.user_id.in_(
                    ctx["user_ids"]
                )
            )

        if ctx["agent_ids"]:
            query = query.filter(
                GuardrailOutcomeFact.agent_id.in_(
                    ctx["agent_ids"]
                )
            )

        rows = (
            query.group_by(
                GuardrailOutcomeFact.eval_name
            )
            .order_by(
                func.count().desc()
            )
            .all()
        )

        overall_detect_count = sum(
            row.total_detect_count or 0
            for row in rows
        )

        return {"metrics": [
                {
                    "guardrail_name":
                        row.eval_name,

                    "total_warn_count":
                        int(
                            row.total_warn_count or 0
                        ),

                    "total_block_count":
                        int(
                            row.total_block_count or 0
                        ),

                    "total_detect_count":
                        int(
                            row.total_detect_count or 0
                        ),

                    "detect_percentage":
                        round(
                            (
                                float(
                                    row.total_detect_count or 0
                                )
                                * 100.0
                                / float(
                                    overall_detect_count
                                )
                            ),
                            2,
                        )
                        if overall_detect_count > 0
                        else 0,
                }
                for row in rows
            ]
        }

    finally:

        ctx["trustai_session"].close()
        ctx["kb_session"].close()