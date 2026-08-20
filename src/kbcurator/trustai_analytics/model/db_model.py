from sqlalchemy import (
    Column,
    String,
    BigInteger,
    DateTime,
    Text,
    Index,
    UniqueConstraint
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class WorkspaceAgentUserSummary(Base):
    __tablename__ = "workspace_agent_user_summary"

    id = Column(BigInteger, primary_key=True)

    app_name = Column(String(255), nullable=False)

    agent_id = Column(String(255), nullable=True)

    user_id = Column(String(255), nullable=False)

    bucket_start_timestamp = Column(
        DateTime,
        nullable=False,
    )

    request_count = Column(
        BigInteger,
        nullable=False,
        default=0,
    )

    pass_count = Column(
        BigInteger,
        nullable=False,
        default=0,
    )

    warn_count = Column(
        BigInteger,
        nullable=False,
        default=0,
    )

    block_count = Column(
        BigInteger,
        nullable=False,
        default=0,
    )

    llm_input_tokens = Column(BigInteger, nullable=False, default=0)
    llm_output_tokens = Column(BigInteger, nullable=False, default=0)
    llm_total_tokens = Column(BigInteger, nullable=False, default=0)

    ig_input_tokens = Column(BigInteger, nullable=False, default=0)
    ig_output_tokens = Column(BigInteger, nullable=False, default=0)
    ig_total_tokens = Column(BigInteger, nullable=False, default=0)

    og_input_tokens = Column(BigInteger, nullable=False, default=0)
    og_output_tokens = Column(BigInteger, nullable=False, default=0)
    og_total_tokens = Column(BigInteger, nullable=False, default=0)

    total_response_time_ms = Column(
        BigInteger,
        nullable=False,
        default=0,
    )

    min_response_time_ms = Column(
        BigInteger,
        nullable=False,
        default=0,
    )

    max_response_time_ms = Column(
        BigInteger,
        nullable=False,
        default=0,
    )

    __table_args__ = (
        UniqueConstraint(
            "app_name",
            "agent_id",
            "user_id",
            "bucket_start_timestamp",
            name="uq_ws_agent_user_bucket",
        ),
    )

class WorkspaceSummary(Base):
    __tablename__ = "workspace_summary"

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    app_name = Column(String(255), nullable=False)

    bucket_start_timestamp = Column(
        DateTime(timezone=True),
        nullable=False
    )

    request_count = Column(BigInteger, nullable=False, default=0)

    pass_count = Column(BigInteger, nullable=False, default=0)
    warn_count = Column(BigInteger, nullable=False, default=0)
    block_count = Column(BigInteger, nullable=False, default=0)

    ig_input_tokens = Column(BigInteger, nullable=False, default=0)
    ig_output_tokens = Column(BigInteger, nullable=False, default=0)
    ig_total_tokens = Column(BigInteger, nullable=False, default=0)

    og_input_tokens = Column(BigInteger, nullable=False, default=0)
    og_output_tokens = Column(BigInteger, nullable=False, default=0)
    og_total_tokens = Column(BigInteger, nullable=False, default=0)

    llm_input_tokens = Column(BigInteger, nullable=False, default=0)
    llm_output_tokens = Column(BigInteger, nullable=False, default=0)
    llm_total_tokens = Column(BigInteger, nullable=False, default=0)

    total_response_time_ms = Column(BigInteger, nullable=False, default=0)

    min_response_time_ms = Column(BigInteger)
    max_response_time_ms = Column(BigInteger)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "app_name",
            "bucket_start_timestamp",
            name="uq_workspace_summary"
        ),
        Index(
            "idx_workspace_bucket",
            "app_name",
            "bucket_start_timestamp"
        ),
    )


class AgentSummary(Base):
    __tablename__ = "agent_summary"

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    app_name = Column(String(255), nullable=False)
    agent_id = Column(String(255), nullable=False)

    bucket_start_timestamp = Column(
        DateTime(timezone=True),
        nullable=False
    )

    request_count = Column(BigInteger, nullable=False, default=0)

    pass_count = Column(BigInteger, nullable=False, default=0)
    warn_count = Column(BigInteger, nullable=False, default=0)
    block_count = Column(BigInteger, nullable=False, default=0)

    ig_input_tokens = Column(BigInteger, nullable=False, default=0)
    ig_output_tokens = Column(BigInteger, nullable=False, default=0)
    ig_total_tokens = Column(BigInteger, nullable=False, default=0)

    og_input_tokens = Column(BigInteger, nullable=False, default=0)
    og_output_tokens = Column(BigInteger, nullable=False, default=0)
    og_total_tokens = Column(BigInteger, nullable=False, default=0)

    llm_input_tokens = Column(BigInteger, nullable=False, default=0)
    llm_output_tokens = Column(BigInteger, nullable=False, default=0)
    llm_total_tokens = Column(BigInteger, nullable=False, default=0)

    total_response_time_ms = Column(BigInteger, nullable=False, default=0)

    min_response_time_ms = Column(BigInteger)
    max_response_time_ms = Column(BigInteger)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "app_name",
            "agent_id",
            "bucket_start_timestamp",
            name="uq_agent_summary"
        ),
        Index(
            "idx_agent_bucket",
            "app_name",
            "agent_id",
            "bucket_start_timestamp"
        ),
    )
    
    
class UserSummary(Base):
    __tablename__ = "user_summary"

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    app_name = Column(String(255), nullable=False)
    user_id = Column(String(512), nullable=False)

    bucket_start_timestamp = Column(
        DateTime(timezone=True),
        nullable=False
    )

    request_count = Column(BigInteger, nullable=False, default=0)

    pass_count = Column(BigInteger, nullable=False, default=0)
    warn_count = Column(BigInteger, nullable=False, default=0)
    block_count = Column(BigInteger, nullable=False, default=0)

    total_token_consumption = Column(
        BigInteger,
        nullable=False,
        default=0
    )

    ig_input_tokens = Column(BigInteger, nullable=False, default=0)
    ig_output_tokens = Column(BigInteger, nullable=False, default=0)
    ig_total_tokens = Column(BigInteger, nullable=False, default=0)

    og_input_tokens = Column(BigInteger, nullable=False, default=0)
    og_output_tokens = Column(BigInteger, nullable=False, default=0)
    og_total_tokens = Column(BigInteger, nullable=False, default=0)

    llm_input_tokens = Column(BigInteger, nullable=False, default=0)
    llm_output_tokens = Column(BigInteger, nullable=False, default=0)
    llm_total_tokens = Column(BigInteger, nullable=False, default=0)

    total_response_time_ms = Column(BigInteger, nullable=False, default=0)

    min_response_time_ms = Column(BigInteger)
    max_response_time_ms = Column(BigInteger)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "app_name",
            "user_id",
            "bucket_start_timestamp",
            name="uq_user_summary"
        ),
    )
    
class AnalyticsEventFact(Base):
    __tablename__ = "analytics_event_fact"

    event_id = Column(
        BigInteger,
        primary_key=True,
    )

    bucket_start_timestamp = Column(
        DateTime,
        nullable=False,
        index=True,
    )

    created_on = Column(
        DateTime,
        nullable=False,
        index=True,
    )

    app_name = Column(
        String(255),
        nullable=False,
        index=True,
    )

    user_id = Column(
        String(255),
        nullable=True,
        index=True,
    )

    agent_id = Column(
        String(255),
        nullable=True,
        index=True,
    )

    llm_type = Column(
        String(255),
        nullable=True,
        index=True,
    )

    duration = Column(
        BigInteger,
        nullable=False,
        default=0,
    )

    outcome = Column(
        String(20),
        nullable=False,
        index=True,
    )

    llm_input_tokens = Column(
        BigInteger,
        nullable=False,
        default=0,
    )

    llm_output_tokens = Column(
        BigInteger,
        nullable=False,
        default=0,
    )

    llm_total_tokens = Column(
        BigInteger,
        nullable=False,
        default=0,
    )

    ig_input_tokens = Column(
        BigInteger,
        nullable=False,
        default=0,
    )

    ig_output_tokens = Column(
        BigInteger,
        nullable=False,
        default=0,
    )

    ig_total_tokens = Column(
        BigInteger,
        nullable=False,
        default=0,
    )

    og_input_tokens = Column(
        BigInteger,
        nullable=False,
        default=0,
    )

    og_output_tokens = Column(
        BigInteger,
        nullable=False,
        default=0,
    )

    og_total_tokens = Column(
        BigInteger,
        nullable=False,
        default=0,
    )
    
    # __table_args__ = (
        
    # )


class BlockWarnPassSummary(Base):
    __tablename__ = "block_warn_pass_summary"

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    app_name = Column(String(255), nullable=False)

    agent_id = Column(String(255))
    user_id = Column(String(512))

    bucket_start_timestamp = Column(
        DateTime(timezone=True),
        nullable=False
    )

    pass_count = Column(BigInteger, nullable=False, default=0)
    warn_count = Column(BigInteger, nullable=False, default=0)
    block_count = Column(BigInteger, nullable=False, default=0)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "app_name",
            "agent_id",
            "user_id",
            "bucket_start_timestamp",
            name="uq_block_warn_pass"
        ),
    )
    
    
class GuardrailOutcomeSummary(Base):
    __tablename__ = "guardrail_outcome_summary"

    id = Column(
        BigInteger,
        primary_key=True,
        autoincrement=True
    )

    app_name = Column(
        String(255),
        nullable=False
    )

    user_id = Column(
        String(512),
        nullable=False
    )

    agent_id = Column(
        String(255),
        nullable=False
    )

    eval_name = Column(
        String(255),
        nullable=False
    )

    bucket_start_timestamp = Column(
        DateTime(timezone=True),
        nullable=False
    )

    warn_count = Column(
        BigInteger,
        nullable=False,
        default=0
    )

    block_count = Column(
        BigInteger,
        nullable=False,
        default=0
    )

    total_detect_count = Column(
        BigInteger,
        nullable=False,
        default=0
    )

    __table_args__ = (
        UniqueConstraint(
            "app_name",
            "user_id",
            "agent_id",
            "eval_name",
            "bucket_start_timestamp",
            name="uq_guardrail_outcome_summary"
        ),
    )
    

class UserActivitySummary(Base):
    __tablename__ = "user_activity_summary"

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    app_name = Column(
        String(255),
        nullable=False
    )

    user_id = Column(
        String(512),
        nullable=False
    )

    latest_request_timestamp = Column(
        DateTime(timezone=True),
        nullable=False
    )

    bucket_start_timestamp = Column(
        DateTime(timezone=True),
        nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "app_name",
            "user_id",
            "bucket_start_timestamp",
            name="uq_user_activity"
        ),
    )
    
class AgentActivitySummary(Base):
    __tablename__ = "agent_activity_summary"

    id = Column(
        BigInteger,
        primary_key=True,
        autoincrement=True
    )

    app_name = Column(
        String(255),
        nullable=False
    )

    agent_id = Column(
        String(255),
        nullable=False
    )

    latest_request_timestamp = Column(
        DateTime(timezone=True),
        nullable=False
    )

    bucket_start_timestamp = Column(
        DateTime(timezone=True),
        nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "app_name",
            "agent_id",
            "bucket_start_timestamp",
            name="uq_agent_activity"
        ),
    )
    

class ModelTokenSummary(Base):
    __tablename__ = "model_token_summary"

    id = Column(
        BigInteger,
        primary_key=True,
        autoincrement=True
    )

    app_name = Column(
        String(255),
        nullable=False
    )

    user_id = Column(
        String(512),
        nullable=False
    )

    agent_id = Column(
        String(255),
        nullable=False
    )

    llm_type = Column(
        String(255),
        nullable=False
    )

    bucket_start_timestamp = Column(
        DateTime(timezone=True),
        nullable=False
    )

    total_requests = Column(
        BigInteger,
        nullable=False,
        default=0
    )

    llm_total_tokens = Column(
        BigInteger,
        nullable=False,
        default=0
    )

    ig_total_tokens = Column(
        BigInteger,
        nullable=False,
        default=0
    )

    og_total_tokens = Column(
        BigInteger,
        nullable=False,
        default=0
    )

    total_tokens = Column(
        BigInteger,
        nullable=False,
        default=0
    )

    __table_args__ = (
        UniqueConstraint(
            "app_name",
            "user_id",
            "agent_id",
            "llm_type",
            "bucket_start_timestamp",
            name="uq_model_token_summary"
        ),
    )
    

    
class AnalyticsWorkerState(Base):
    __tablename__ = "analytics_worker_state"

    worker_name = Column(
        String(100),
        primary_key=True,
        nullable=False,
    )

    last_processed_event_id = Column(
        BigInteger,
        nullable=True,
    )

    last_processed_timestamp = Column(
        DateTime(timezone=True),
        nullable=True,
    )

    last_run_start_time = Column(
        DateTime(timezone=True),
        nullable=True,
    )

    last_run_end_time = Column(
        DateTime(timezone=True),
        nullable=True,
    )

    rows_processed = Column(
        BigInteger,
        nullable=False,
        default=0,
    )

    backlog_items = Column(
        BigInteger,
        nullable=False,
        default=0,
    )

    last_run_duration_ms = Column(
        BigInteger,
        nullable=False,
        default=0,
    )

    status = Column(
        String(50),
        nullable=False,
        default="SUCCESS",
    )

    error_message = Column(
        Text,
        nullable=True,
    )

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        Index(
            "idx_analytics_worker_state_status",
            "status",
        ),
        Index(
            "idx_analytics_worker_state_last_event",
            "last_processed_event_id",
        ),
    )
    
    
class AnalyticsWorkerExecutionHistory(Base):
    __tablename__ = "analytics_worker_execution_history"

    id = Column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    worker_name = Column(
        String(100),
        nullable=False,
    )

    run_started_at = Column(
        DateTime(timezone=True),
        nullable=False,
    )

    run_completed_at = Column(
        DateTime(timezone=True),
        nullable=True,
    )

    start_checkpoint = Column(
        BigInteger,
        nullable=True,
    )

    end_checkpoint = Column(
        BigInteger,
        nullable=True,
    )

    rows_processed = Column(
        BigInteger,
        nullable=False,
        default=0,
    )

    backlog_before = Column(
        BigInteger,
        nullable=False,
        default=0,
    )

    backlog_after = Column(
        BigInteger,
        nullable=False,
        default=0,
    )

    run_duration_ms = Column(
        BigInteger,
        nullable=False,
        default=0,
    )

    status = Column(
        String(50),
        nullable=False,
    )

    error_message = Column(
        Text,
        nullable=True,
    )

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # __table_args__ = (
    #     Index(
    #         "idx_worker_history_worker_name",
    #         "worker_name",
    #     ),
    #     Index(
    #         "idx_worker_history_started_at",
    #         "run_started_at",
    #     ),
    #     Index(
    #         "idx_worker_history_status",
    #         "status",
    #     ),
    #     Index(
    #         "idx_worker_history_worker_started",
    #         "worker_name",
    #         "run_started_at",
    #     ),
    # )
    
