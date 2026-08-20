The task is to create a worker job that can use the following cte and create the aggregated tables for Workspace , Agent , User Summaries and others as shared below, i have below queries that can be used with some aggregations on them to creat the rows for the required aggregated tables. 

# QUERY 1:
```sql
WITH guardrail_agg as (
    SELECT gerl.event_id,gerl.guardrail_type ,
    SUM(CASE 
    WHEN gerl.results->'detail'->>'outcome' IN ('Success','Skipped') THEN 1
    ELSE 0
    END ) as pass,
    SUM(CASE 
    WHEN gerl.results->'detail'->>'outcome' IN ('Fail','Error') THEN 1
    ELSE 0
    END ) as block,
    SUM(CASE 
    WHEN gerl.results->'detail'->>'outcome' IN ('Warning') THEN 1
    ELSE 0
    END ) as warn,
    SUM(
        COALESCE((gerl.results->'usage'->>'input_tokens')::BIGINT,0)
    ) as guardrail_input_tokens,
    SUM(
        COALESCE((gerl.results->'usage'->>'input_tokens')::BIGINT,0)
    ) as guardrail_output_tokens,
    SUM(
        COALESCE((gerl.results->'usage'->>'input_tokens')::BIGINT,0)
    ) as guardrail_total_tokens
    FROM public.guardrail_event_result_log as gerl
    JOIN public.guardrail_event_log as gel
    ON gel.id = gerl.event_id
    WHERE created_on > '2026-08-19' AND gel.blocked_by IS NOT NULL
    GROUP BY gerl.event_id, gerl.guardrail_type
),
input_guardrail_agg as (
    SELECT *
    FROM guardrail_agg 
    WHERE guardrail_type = 'input'
),
output_guardrail_agg as (
    SELECT *
    FROM guardrail_agg
    WHERE guardrail_type = 'output'
)
SELECT gel.id,gel.llm_type,gel.created_on ,
(CASE 
WHEN length(gel.blocked_by) > 0 THEN 'Block'
WHEN input_guardrail_agg.warn + output_guardrail_agg.warn > 0 THEN 'Warn'
ELSE 'Pass'
END) AS outcome

, gel.duration , gel.user_id , a.app_name , gel.additional_info->>'X-Agent-Id' as agent_id , COALESCE(llm.input_tokens,0) as llm_input_tokens , COALESCE(llm.output_tokens,0) as llm_output_tokens , COALESCE(llm.total_tokens,0) as llm_total_tokens , COALESCE(input_guardrail_agg.guardrail_input_tokens,0) as ig_input_tokens ,COALESCE(input_guardrail_agg.guardrail_output_tokens,0) as ig_output_tokens , COALESCE(input_guardrail_agg.guardrail_total_tokens,0) as ig_total_tokens , COALESCE(output_guardrail_agg.guardrail_input_tokens,0) as og_input_tokens ,COALESCE(output_guardrail_agg.guardrail_output_tokens,0) as og_output_tokens , COALESCE(output_guardrail_agg.guardrail_total_tokens,0) as og_total_tokens  
FROM output_guardrail_agg
JOIN public.guardrail_event_log as gel 
ON gel.id = output_guardrail_agg.event_id
JOIN input_guardrail_agg
ON gel.id = input_guardrail_agg.event_id
JOIN public.applications as a
ON a.app_id = gel.app_id
JOIN public.llm_details as llm 
ON llm.event_id = gel.id
ORDER BY gel.id DESC;
```

From the above query i want to extract: 

class ORMBaseSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

class WorkspaceSummarySchema(ORMBaseSchema):
    app_name: str (group by app_name)
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
    agent_id: str (group by agent_id)

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
    user_id: str (group by one user_id)

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

    agent_id: str | None = None

    user_id: str

    latest_request_timestamp: datetime

    bucket_start_timestamp: datetime
    
class AgentActivitySummarySchema(ORMBaseSchema):
    app_name: str

    agent_id: str

    lastest_request_timestamp: datetime

    bucket_start_timestamp: datetime


And give schema for following extra tables that also needs to be extracted from the above table: 
1. modeltokensummary: 
workspace_id
user_id
agent_id
bucket_time_stamp
llm_type
total_requests
total_tokens = llm + ig + og total tokens


# QUERY 2:
```sql

SELECT gel.created_on , a.app_name , gel.user_id , gel.additional_info->>'X-Agent-Id' as agent_id, gerl.eval_name , 
(CASE
WHEN gerl.results->'detail'->>'outcome' IN ('Fail','Error') THEN 'Block'
WHEN gerl.results->'detail'->>'outcome' IN ('Warning') THEN 'Warn'
END) as outcome
FROM public.guardrail_event_result_log as gerl 
JOIN public.guardrail_event_log as gel 
ON gel.id = gerl.event_id
JOIN public.applications as a 
ON gel.app_id = a.app_id
WHERE gerl.results->'detail'->>'outcome' in ('Fail','Error','Warning') AND  gel.additional_info->>'X-Agent-Id' is not NULL
ORDER BY gerl.event_id DESC
limit 1000;

```

use the above query table and apply some more aggregation on it to create rows for the following aggregator table
GuardrailOutcomeSummary
app_name
user_id
agent_id
eval_name
warn_count
block_count
total_detect_count = warn + block counts


# Trustai_db class 

This class has the responsibility to maintin the following tables: 

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

the trustai_db current class is : 

from threading import RLock
from urllib.parse import quote_plus

from sqlalchemy import (
    MetaData,
    create_engine,
    text,
)
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import cast
from sqlalchemy import case , func , BigInteger
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import insert


from .config import settings

# Analytics ORM Models
from datetime import datetime
from .model.db_model import (
    Base,
    AnalyticsWorkerState,
    AnalyticsWorkerExecutionHistory
)

WORKER_LOCKS = {
    "analytics_worker": 1001,
}


class TrustaiAnalyticsDB:
    """
    Singleton database manager.

    Responsibilities:
    - Initialize analytics tables
    - Reflect source TrustAI tables
    - Expose SQLAlchemy session
    - Manage worker checkpoints
    - Provide paginated source data access
    """

    _instance = None
    _lock = RLock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_db()

        return cls._instance

    def _init_db(self):

        conn_str = (
            f"postgresql+psycopg2://"
            f"{settings.POSTGRES_USER}:"
            f"{quote_plus(settings.POSTGRES_PASSWORD)}"
            f"@{settings.POSTGRES_HOST}:"
            f"{settings.POSTGRES_PORT}/"
            f"{settings.TRUSTAI_DB}"
        )

        self.engine = create_engine(
            conn_str,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
            pool_recycle=3600,
            pool_timeout=30,
            echo=False,
        )

        self.Session = sessionmaker(
            bind=self.engine
        )

        # Create analytics tables
        self.initialize_trustai_summary_tables()

        # Reflect existing source tables
        self.metadata = MetaData()

        self.metadata.reflect(
            bind=self.engine,
            schema="public"
        )

        self.AutoBase = automap_base(
            metadata=self.metadata
        )

        self.AutoBase.prepare()

        self._load_automapped_tables()

    # ------------------------------------------------------------------
    # Analytics table initialization
    # ------------------------------------------------------------------

    def initialize_trustai_summary_tables(self):
        Base.metadata.create_all(
            bind=self.engine
        )

    # ------------------------------------------------------------------
    # Existing TrustAI source tables
    # ------------------------------------------------------------------

    def _load_automapped_tables(self):

        available = self.AutoBase.classes.keys()

        self.GuardrailEventLog = getattr(
            self.AutoBase.classes,
            "guardrail_event_log",
            None,
        )

        self.GuardrailEventResultLog = getattr(
            self.AutoBase.classes,
            "guardrail_event_result_log",
            None,
        )

        self.Applications = getattr(
            self.AutoBase.classes,
            "applications",
            None,
        )

        self.LLMDetails = getattr(
            self.AutoBase.classes,
            "llm_details",
            None,
        )

        self.available_tables = available

    # ------------------------------------------------------------------
    # Worker State Methods
    # ------------------------------------------------------------------

    def get_worker_state(
        self,
        worker_name: str,
    ):

        session = self.Session()

        try:

            return (
                session.query(
                    AnalyticsWorkerState
                )
                .filter(
                    AnalyticsWorkerState.worker_name
                    == worker_name
                )
                .first()
            )

        finally:
            session.close()

    def create_worker_state(
        self,
        worker_name: str,
    ):

        session = self.Session()

        try:

            state = AnalyticsWorkerState(
                worker_name=worker_name
            )

            session.add(state)

            session.commit()

            return state

        finally:
            session.close()

    def get_last_processed_event_id(
        self,
        worker_name: str,
    ) -> int:

        state = self.get_worker_state(
            worker_name
        )

        if state is None:

            self.create_worker_state(
                worker_name
            )

            return 0

        return (
            state.last_processed_event_id
            or 0
        )
    
     
	def get_checkpoint(
     	self,
		worker_name: str
	) -> int:
		
  		state = self.get_worker_state(worker_name)

		return (
			state.last_processed_event_id
			if state and state.last_processed_event_id is not None
			else 0
		)

	def release_worker_lock(
		self,
		session,
		worker_name: str,
	):

		lock_id = WORKER_LOCKS.get(worker_name)

		session.execute(
			text(
				"""
				SELECT pg_advisory_unlock(
					:lock_id
				)
				"""
			),
			{
				"lock_id": lock_id
			}
		)
 
	def acquire_worker_lock(
		self,
		session,
		worker_name: str,
	) -> bool:

		lock_id = WORKER_LOCKS.get(worker_name)

		if lock_id is None:
			raise ValueError(
				f"No lock configured for {worker_name}"
			)

		result = session.execute(
			text(
				"""
				SELECT pg_try_advisory_lock(
					:lock_id
				)
				"""
			),
			{
				"lock_id": lock_id
			}
		)

		return bool(result.scalar())


    def update_worker_checkpoint(
		self,
		session,
		worker_name: str,
		last_processed_event_id: int,
		last_processed_timestamp,
		rows_processed: int,
		backlog_items: int = 0,
		last_run_duration_ms: int = 0,
		status: str = "SUCCESS",
		error_message: str | None = None,
	):

		state = (
			session.query(
				AnalyticsWorkerState
			)
			.filter(
				AnalyticsWorkerState.worker_name
				== worker_name
			)
			.first()
		)

		if state is None:

			state = AnalyticsWorkerState(
				worker_name=worker_name
			)

			session.add(state)

		state.last_processed_event_id = (
			last_processed_event_id
		)

		state.last_processed_timestamp = (
			last_processed_timestamp
		)

		state.rows_processed = rows_processed

		state.backlog_items = backlog_items

		state.last_run_duration_ms = (
			last_run_duration_ms
		)

		state.status = status

		state.error_message = error_message
            
    def get_current_max_event_id(self) -> int:

		session = self.Session()

		try:

			result = (
				session.query(
					func.max(
						self.GuardrailEventLog.id
					)
				)
				.scalar()
			)

			return result or 0

		finally:

			session.close()
   
   	def bulk_upsert(
		self,
		session,
		model,
		rows: list[dict],
		conflict_columns: list[str],
	):

		if not rows:
			return

		stmt = insert(model).values(rows)

		update_values = {}

		for column in model.__table__.columns:

			if column.name in conflict_columns:
				continue

			if column.name in {"id", "created_at"}:
				continue

			update_values[column.name] = getattr(
				stmt.excluded,
				column.name
			)

		stmt = stmt.on_conflict_do_update(
			index_elements=conflict_columns,
			set_=update_values,
		)

		session.execute(stmt)
	
	def bulk_aggregate_upsert(
		self,
		session,
		model,
		rows: list[dict],
		conflict_columns: list[str],
	):

		if not rows:
			return

		stmt = insert(model).values(rows)

		updates = {}

		additive_columns = {
			"request_count",

			"pass_count",
			"warn_count",
			"block_count",

			"ig_input_tokens",
			"ig_output_tokens",
			"ig_total_tokens",

			"og_input_tokens",
			"og_output_tokens",
			"og_total_tokens",

			"llm_input_tokens",
			"llm_output_tokens",
			"llm_total_tokens",

			"input_tokens",
			"output_tokens",
			"total_tokens",

			"total_token_consumption",

			"total_response_time_ms",
		}

		for column_name in additive_columns:

			if hasattr(model, column_name):

				updates[column_name] = (
					getattr(model, column_name)
					+ getattr(stmt.excluded, column_name)
				)

		if hasattr(model, "min_response_time_ms"):

			updates["min_response_time_ms"] = func.least(
				model.min_response_time_ms,
				stmt.excluded.min_response_time_ms,
			)

		if hasattr(model, "max_response_time_ms"):

			updates["max_response_time_ms"] = func.greatest(
				model.max_response_time_ms,
				stmt.excluded.max_response_time_ms,
			)

		if hasattr(model, "updated_at"):

			updates["updated_at"] = func.now()

		stmt = stmt.on_conflict_do_update(
			index_elements=conflict_columns,
			set_=updates,
		)

		session.execute(stmt)

	
	def bulk_insert_ignore_duplicates(
			self,
			session,
			model,
			rows,
			conflict_columns
	):
		if not rows:
			return


		try:

			stmt = insert(model).values(rows)

			stmt = stmt.on_conflict_do_nothing(
				index_elements=conflict_columns
			)

			session.execute(stmt)

		except Exception:
			raise
 
 
	def create_worker_state_if_missing(
		self,
		worker_name: str,
	):

		state = self.get_worker_state(
			worker_name
		)

		if state:
			return state

		return self.create_worker_state(
			worker_name
		)
		
		
	def insert_worker_execution_history(
		self,
		session,
		worker_name: str,
		run_started_at: datetime,
		run_completed_at: datetime,
		start_checkpoint: int,
		end_checkpoint: int,
		rows_processed: int,
		backlog_before: int,
		backlog_after: int,
		run_duration_ms: int,
		status: str,
		error_message: str | None = None,
	):

		history = AnalyticsWorkerExecutionHistory(
			worker_name=worker_name,
			run_started_at=run_started_at,
			run_completed_at=run_completed_at,
			start_checkpoint=start_checkpoint,
			end_checkpoint=end_checkpoint,
			rows_processed=rows_processed,
			backlog_before=backlog_before,
			backlog_after=backlog_after,
			run_duration_ms=run_duration_ms,
			status=status,
			error_message=error_message,
		)

		session.add(history)

    # ------------------------------------------------------------------
    # Source Event Query
    # ------------------------------------------------------------------

	def get_current_events_batch(
		self,
		last_processed_event_id: int,
		batch_size: int = 1000,
	):

		session = self.Session()

		try:

			query = (
				session.query(
					self.GuardrailEventLog.id.label("id"),

					self.GuardrailEventLog.duration.label(
						"duration"
					),

					self.GuardrailEventLog.llm_type.label(
						"llm_type"
					),

					self.GuardrailEventLog.created_on.label(
						"created_on"
					),

					self.GuardrailEventLog.user_id.label(
						"user_id"
					),

					self.GuardrailEventLog.created_at.label(
						"created_at"
					),

					self.GuardrailEventLog.blocked_by.label(
						"blocked_by"
					),

					func.coalesce(
						self.GuardrailEventLog.additional_info[
							"X-Agent-Id"
						].astext,
						None
					).label("agent_id"),

					self.Applications.app_name.label(
						"app_name"
					),

					func.coalesce(
						self.LLMDetails.input_tokens,
						0
					).label("input_tokens"),

					func.coalesce(
						self.LLMDetails.output_tokens,
						0
					).label("output_tokens"),

					func.coalesce(
						self.LLMDetails.total_tokens,
						0
					).label("total_tokens"),
				)
				.outerjoin(
					self.Applications,
					self.GuardrailEventLog.app_id
					== self.Applications.app_id
				)
				.outerjoin(
					self.LLMDetails,
					self.GuardrailEventLog.id
					== self.LLMDetails.event_id
				)
				.filter(
					self.GuardrailEventLog.id >
					last_processed_event_id
				)
				.order_by(
					self.GuardrailEventLog.id
				)
				.limit(batch_size)
			)

			return query.all()

		finally:

			session.close()

    # ------------------------------------------------------------------
    # Guardrail Result Query
    # ------------------------------------------------------------------
	def get_token_usage_and_eval_outcomes_for_event_range(
		self,
		start_event_id: int,
		end_event_id: int,
	):

		session = self.Session()

		try:

			outcome_case = case(
				(
					self.GuardrailEventResultLog.results[
						"detail"
					]["outcome"]
					.astext
					.in_(["Success", "Skipped"]),
					"Pass",
				),
				(
					self.GuardrailEventResultLog.results[
						"detail"
					]["outcome"]
					.astext
					.in_(["Fail", "Error"]),
					"Block",
				),
				else_="Warn",
			)

			query = (
				session.query(

					self.GuardrailEventLog.id.label(
						"event_id"
					),

					self.Applications.app_name.label(
						"app_name"
					),

					self.GuardrailEventLog.user_id.label(
						"user_id"
					),

					self.GuardrailEventLog.created_at.label(
						"created_at"
					),

					func.coalesce(
						self.GuardrailEventLog.additional_info[
							"X-Agent-Id"
						].astext,
						None
					).label(
						"agent_id"
					),

					self.GuardrailEventResultLog.eval_name.label(
						"eval_name"
					),

					self.GuardrailEventResultLog.guardrail_type.label(
						"guardrail_type"
					),

					outcome_case.label(
						"outcome"
					),

					func.coalesce(
						self.GuardrailEventResultLog.results[
							"token_usage"
						]["input_tokens"]
						.astext
						.cast(BigInteger),
						0,
					).label(
						"input_tokens"
					),

					func.coalesce(
						self.GuardrailEventResultLog.results[
							"token_usage"
						]["output_tokens"]
						.astext
						.cast(BigInteger),
						0,
					).label(
						"output_tokens"
					),

					func.coalesce(
						self.GuardrailEventResultLog.results[
							"token_usage"
						]["total_tokens"]
						.astext
						.cast(BigInteger),
						0,
					).label(
						"total_tokens"
					),
				)
				.join(
					self.GuardrailEventLog,
					self.GuardrailEventResultLog.event_id
					== self.GuardrailEventLog.id
				)
				.join(
					self.Applications,
					self.GuardrailEventLog.app_id
					== self.Applications.app_id
				)
				.filter(
					self.GuardrailEventLog.id >=
					start_event_id
				)
				.filter(
					self.GuardrailEventLog.id <=
					end_event_id
				)
				.order_by(
					self.GuardrailEventLog.id
				)
			)

			return query.all()

		finally:

			session.close()
        
    
	def get_token_usage_and_eval_outcomes(
		self,
		last_processed_event_id: int,
		batch_size: int = 1000,
	):

		session = self.Session()

		try:

			outcome_case = case(
				(
					self.GuardrailEventResultLog.results[
						"detail"
					]["outcome"]
					.astext
					.in_(["Success", "Skipped"]),
					"Pass",
				),
				(
					self.GuardrailEventResultLog.results[
						"detail"
					]["outcome"]
					.astext
					.in_(["Fail", "Error"]),
					"Block",
				),
				else_="Warn",
			)

			query = (
				session.query(

					self.GuardrailEventLog.id.label("id"),

					self.Applications.app_name.label(
						"app_name"
					),

					self.GuardrailEventLog.user_id.label(
						"user_id"
					),

					self.GuardrailEventLog.created_on.label(
						"created_on"
					),

					func.coalesce(
						self.GuardrailEventLog.additional_info[
							"X-Agent-Id"
						].astext,
						None
					).label("agent_id"),

					self.GuardrailEventResultLog.eval_name.label(
						"eval_name"
					),

					self.GuardrailEventResultLog.guardrail_type.label(
						"guardrail_type"
					),

					outcome_case.label(
						"outcome"
					),

					func.coalesce(
						self.GuardrailEventResultLog.results[
							"token_usage"
						]["input_tokens"]
						.astext
						.cast(BigInteger),
						0,
					).label(
						"input_tokens"
					),

					func.coalesce(
						self.GuardrailEventResultLog.results[
							"token_usage"
						]["output_tokens"]
						.astext
						.cast(BigInteger),
						0,
					).label(
						"output_tokens"
					),

					func.coalesce(
						self.GuardrailEventResultLog.results[
							"token_usage"
						]["total_tokens"]
						.astext
						.cast(BigInteger),
						0,
					).label(
						"total_tokens"
					),
				)
				.join(
					self.GuardrailEventLog,
					self.GuardrailEventResultLog.event_id
					== self.GuardrailEventLog.id
				)
				.join(
					self.Applications,
					self.GuardrailEventLog.app_id
					== self.Applications.app_id
				)
				.filter(
					self.GuardrailEventLog.id >
					last_processed_event_id
				)
				.order_by(
					self.GuardrailEventLog.id
				)
				.limit(batch_size)
			)

			return query.all()

		finally:

			session.close()


analytics_db = TrustaiAnalyticsDB()



I want you to help me fix all the issues in this implementation, correct the table schemas if required, and 

i have the folder structure as 

trustai_analytics
    |- model/
                db_model.py (orm db tables for these schemas)
                schema.py (pydantic schemas)
    |- workders/ 
                analytics_worker.py
    |
    |- trustai_db.py


Let's create end to end solution.



# ALL THE TEST SQL QUERIES: 

-- SELECT id,
--        event_id,
--        results,
--        eval_name,
--        score,
--        eval_duration,
--        guardrail_type
-- FROM public.guardrail_event_result_log
-- WHERE eval_name != 'TOKEN_QUOTA'
-- ORDER BY event_id DESC
-- LIMIT 1000;

-- SELECT gel.id , gel.created_at, gel.duration, gel.llm_type ,gel.app_id,gel.user_id ,gel.guardrail_type,  gel.additional_info , gel.llm_router_api_duration, gel.llm_response_duration , gel.message_id , gerl.eval_name , gerl.guardrail_type , gerl.score , gerl.results
-- FROM public.guardrail_event_log as gel 
-- JOIN public.guardrail_event_result_log as gerl ON gerl.event_id = gel.id
-- WHERE gel.additional_info is not NULL
-- LIMIT 1000;

-- SELECT DISTINCT results->'detail'->>'outcome' AS outcome , rseults->'detail''->>
-- FROM public.guardrail_event_result_log
-- WHERE results IS NOT NULL
--   AND results->'action' IN ('B','W');


-- SELECT DISTINCT results AS outcome ,event_id , gel.blocked_by
-- FROM public.guardrail_event_result_log AS gerl
-- JOIN public.guardrail_event_log as gel 
-- ON gel.id = gerl.event_id
-- WHERE results IS NOT NULL AND results->'detail'->>'outcome' IN ('Fail')
-- ORDER BY event_id DESC
-- LIMIT 1000;

-- WITH guardrail_agg as (
--     SELECT gerl.event_id,gerl.guardrail_type ,
--     SUM(CASE 
--     WHEN gerl.results->'detail'->>'outcome' IN ('Success','Skipped') THEN 1
--     ELSE 0
--     END ) as pass,
--     SUM(CASE 
--     WHEN gerl.results->'detail'->>'outcome' IN ('Fail','Error') THEN 1
--     ELSE 0
--     END ) as block,
--     SUM(CASE 
--     WHEN gerl.results->'detail'->>'outcome' IN ('Warning') THEN 1
--     ELSE 0
--     END ) as warn,
--     SUM(
--         COALESCE((gerl.results->'usage'->>'input_tokens')::BIGINT,0)
--     ) as guardrail_input_tokens,
--     SUM(
--         COALESCE((gerl.results->'usage'->>'input_tokens')::BIGINT,0)
--     ) as guardrail_output_tokens,
--     SUM(
--         COALESCE((gerl.results->'usage'->>'input_tokens')::BIGINT,0)
--     ) as guardrail_total_tokens
--     FROM public.guardrail_event_result_log as gerl
--     JOIN public.guardrail_event_log as gel
--     ON gel.id = gerl.event_id
--     WHERE created_on > '2026-08-19' AND gel.blocked_by IS NOT NULL
--     GROUP BY gerl.event_id, gerl.guardrail_type
-- ),
-- input_guardrail_agg as (
--     SELECT *
--     FROM guardrail_agg 
--     WHERE guardrail_type = 'input'
-- ),
-- output_guardrail_agg as (
--     SELECT *
--     FROM guardrail_agg
--     WHERE guardrail_type = 'output'
-- )
-- SELECT gel.id,gel.llm_type,gel.created_on ,
-- (CASE 
-- WHEN length(gel.blocked_by) > 0 THEN 'Block'
-- WHEN input_guardrail_agg.warn + output_guardrail_agg.warn > 0 THEN 'Warn'
-- ELSE 'Pass'
-- END) AS outcome

-- , gel.duration , gel.user_id , a.app_name , gel.additional_info->>'X-Agent-Id' as agent_id , COALESCE(llm.input_tokens,0) as llm_input_tokens , COALESCE(llm.output_tokens,0) as llm_output_tokens , COALESCE(llm.total_tokens,0) as llm_total_tokens , COALESCE(input_guardrail_agg.guardrail_input_tokens,0) as ig_input_tokens ,COALESCE(input_guardrail_agg.guardrail_output_tokens,0) as ig_output_tokens , COALESCE(input_guardrail_agg.guardrail_total_tokens,0) as ig_total_tokens , COALESCE(output_guardrail_agg.guardrail_input_tokens,0) as og_input_tokens ,COALESCE(output_guardrail_agg.guardrail_output_tokens,0) as og_output_tokens , COALESCE(output_guardrail_agg.guardrail_total_tokens,0) as og_total_tokens  
-- FROM output_guardrail_agg
-- JOIN public.guardrail_event_log as gel 
-- ON gel.id = output_guardrail_agg.event_id
-- JOIN input_guardrail_agg
-- ON gel.id = input_guardrail_agg.event_id
-- JOIN public.applications as a
-- ON a.app_id = gel.app_id
-- JOIN public.llm_details as llm 
-- ON llm.event_id = gel.id
-- ORDER BY gel.id DESC;

-- SELECT gel.created_on , a.app_name , gel.user_id , gel.additional_info->>'X-Agent-Id' as agent_id, gerl.eval_name , 
-- (CASE
-- WHEN gerl.results->'detail'->>'outcome' IN ('Fail','Error') THEN 'Block'
-- WHEN gerl.results->'detail'->>'outcome' IN ('Warning') THEN 'Warn'
-- END) as outcome
-- FROM public.guardrail_event_result_log as gerl 
-- JOIN public.guardrail_event_log as gel 
-- ON gel.id = gerl.event_id
-- JOIN public.applications as a 
-- ON gel.app_id = a.app_id
-- WHERE gerl.results->'detail'->>'outcome' in ('Fail','Error','Warning') AND  gel.additional_info->>'X-Agent-Id' is not NULL
-- ORDER BY gerl.event_id DESC
-- limit 1000;



-- SELECT gel.id,gel.duration , gel.llm_type, gel.created_on,gel.user_id,gel.created_at,gel.blocked_by,gel.additional_info->>'X-Agent-Id' as agent_id , a.app_name , llm.input_tokens as llm_input_tokens , llm.output_tokens as llm_output_tokens , llm.total_tokens as llm_output_tokens
-- FROM public.guardrail_event_log AS gel
-- LEFT JOIN public.applications AS a
-- ON gel.app_id = a.app_id
-- LEFT JOIN llm_details as llm
-- ON gel.id = llm.event_id
-- ORDER BY id DESC
-- LIMIT 1000;

-- SELECT gel.id , a.app_name , gel.user_id, gel.created_on , gel.additional_info->>'X-Agent-Id' ,gerl.eval_name, gerl.results , gerl.results->>'token_usage' as usage , 

-- (CASE 
-- WHEN gerl.results->'detail'->>'outcome' IN ('Success','Skipped') THEN 'Pass'
-- WHEN gerl.results->'detail'->>'outcome' IN ('Fail','Error') THEN 'Block'
-- ELSE 'Warn'
-- END) as outcome
-- , gerl.guardrail_type

-- FROM public.guardrail_event_result_log as gerl 
-- JOIN public.guardrail_event_log as gel 
-- on gel.id = gerl.event_id
-- JOIN public.applications as a
-- ON gel.app_id = a.app_id
-- ORDER BY id DESC
-- limit 1000;

# SQL TO DROP ALL THE ABOVE TABLES

-- DROP TABLE IF EXISTS
--     workspace_agent_user_summary,
--     workspace_summary,
--     agent_summary,
--     user_summary,
--     block_warn_pass_summary,
--     guardrail_outcome_summary,
--     user_activity_summary,
--     agent_activity_summary,
--     model_token_summary,
--     analytics_worker_state,
--     analytics_worker_execution_history
-- CASCADE;