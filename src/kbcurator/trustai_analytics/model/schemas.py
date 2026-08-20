from datetime import datetime
from pydantic import BaseModel, ConfigDict


class ORMBaseSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

class WorkspaceSummarySchema(ORMBaseSchema):
    app_name: str
    bucket_start_timestamp: datetime

    request_count: int

    pass_count: int
    warn_count: int
    block_count: int

    ig_input_tokens: int
    ig_output_tokens: int
    ig_total_tokens: int

    og_input_tokens: int
    og_output_tokens: int
    og_total_tokens: int

    llm_input_tokens: int
    llm_output_tokens: int
    llm_total_tokens: int

    total_response_time_ms: int

    min_response_time_ms: int | None = None
    max_response_time_ms: int | None = None
    
    
class AgentSummarySchema(ORMBaseSchema):
    app_name: str
    agent_id: str

    bucket_start_timestamp: datetime

    request_count: int

    pass_count: int
    warn_count: int
    block_count: int

    ig_input_tokens: int
    ig_output_tokens: int
    ig_total_tokens: int

    og_input_tokens: int
    og_output_tokens: int
    og_total_tokens: int

    llm_input_tokens: int
    llm_output_tokens: int
    llm_total_tokens: int

    total_response_time_ms: int

    min_response_time_ms: int | None = None
    max_response_time_ms: int | None = None
    
    
class UserSummarySchema(ORMBaseSchema):
    app_name: str
    user_id: str

    bucket_start_timestamp: datetime

    request_count: int

    pass_count: int
    warn_count: int
    block_count: int

    total_token_consumption: int

    ig_input_tokens: int
    ig_output_tokens: int
    ig_total_tokens: int

    og_input_tokens: int
    og_output_tokens: int
    og_total_tokens: int

    llm_input_tokens: int
    llm_output_tokens: int
    llm_total_tokens: int

    total_response_time_ms: int

    min_response_time_ms: int | None = None
    max_response_time_ms: int | None = None
    
class BlockWarnPassSummarySchema(ORMBaseSchema):
    app_name: str

    agent_id: str | None = None
    user_id: str | None = None

    bucket_start_timestamp: datetime

    pass_count: int
    warn_count: int
    block_count: int
    
class UserActivitySummarySchema(ORMBaseSchema):
    app_name: str

    user_id: str

    latest_request_timestamp: datetime

    bucket_start_timestamp: datetime
    
class AgentActivitySummarySchema(ORMBaseSchema):
    app_name: str

    agent_id: str
    
    latest_request_timestamp: datetime

    bucket_start_timestamp: datetime
    
    
class ModelTokenSummarySchema(ORMBaseSchema):
    app_name: str

    user_id: str

    agent_id: str

    llm_type: str

    bucket_start_timestamp: datetime

    total_requests: int

    llm_total_tokens: int
    ig_total_tokens: int
    og_total_tokens: int

    total_tokens: int
    
class GuardrailOutcomeSummarySchema(ORMBaseSchema):
    app_name: str

    user_id: str

    agent_id: str

    eval_name: str

    bucket_start_timestamp: datetime

    warn_count: int

    block_count: int

    total_detect_count: int
    


    
class AnalyticsWorkerStateSchema(ORMBaseSchema):
    worker_name: str

    last_processed_event_id: int | None = None

    last_processed_timestamp: datetime | None = None

    last_run_start_time: datetime | None = None
    last_run_end_time: datetime | None = None

    rows_processed: int = 0

    # NEW
    backlog_items: int = 0

    # NEW
    last_run_duration_ms: int = 0

    status: str = "SUCCESS"

    error_message: str | None = None

    created_at: datetime | None = None
    updated_at: datetime | None = None
    
    
class AnalyticsWorkerExecutionHistorySchema(
    ORMBaseSchema
):
    id: int

    worker_name: str

    run_started_at: datetime

    run_completed_at: datetime | None = None

    start_checkpoint: int | None = None

    end_checkpoint: int | None = None

    rows_processed: int = 0

    backlog_before: int = 0

    backlog_after: int = 0

    run_duration_ms: int = 0

    status: str

    error_message: str | None = None
    
class AggregationMetrics(BaseModel):
    request_count: int = 0

    pass_count: int = 0
    warn_count: int = 0
    block_count: int = 0

    ig_input_tokens: int = 0
    ig_output_tokens: int = 0
    ig_total_tokens: int = 0

    og_input_tokens: int = 0
    og_output_tokens: int = 0
    og_total_tokens: int = 0

    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_total_tokens: int = 0

    total_response_time_ms: int = 0

    min_response_time_ms: int | None = None
    max_response_time_ms: int | None = None