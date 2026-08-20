from threading import RLock
from urllib.parse import quote_plus
import os
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
from sqlalchemy import MetaData, Table, text


from kbcurator.utils.config import settings

# Analytics ORM Models
from datetime import datetime
from .model.db_model import (
    Base,
    AnalyticsWorkerState,
    AnalyticsWorkerExecutionHistory,
    
    AnalyticsEventFact,
    WorkspaceSummary,
    AgentSummary,
    UserSummary,
    BlockWarnPassSummary,
    UserActivitySummary,
    AgentActivitySummary,
    ModelTokenSummary,
    GuardrailOutcomeSummary,
)

WORKER_LOCKS = {
    "analytics_worker": 1001,
    "fact_analytics_worker": 1002
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
        self.initialize_trustai_fact_tables()

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

    def initialize_trustai_fact_tables(self):

        Base.metadata.create_all(
            bind=self.engine,
            tables=[
                AnalyticsEventFact.__table__,
                GuardrailOutcomeSummary.__table__,
                AnalyticsWorkerState.__table__,
                AnalyticsWorkerExecutionHistory.__table__,
            ],
        )
        
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
        session,
        worker_name: str,
    ):

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

    def create_worker_state(
        self,
        session,
        worker_name: str,
    ):
        state = AnalyticsWorkerState(
            worker_name=worker_name
        )

        session.add(state)

        session.commit()

        return state
    
     
    def get_checkpoint(
         self,
        session,
        worker_name: str
    ) -> int:
        
        state = self.get_worker_state(session,worker_name)

        return int(
            state.last_processed_event_id
            if state and state.last_processed_event_id is not None
            else (os.getenv("TRUSTAI_LAST_PROCESSED_EVENT_ID",0))
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
        
    def get_max_processible_event_id(
        self,
        session,
        processing_cutoff: datetime,
    ):
        return (
            session.query(
                func.max(
                    self.GuardrailEventLog.id
                )
            )
            .filter(
                self.GuardrailEventLog.created_on
                <= processing_cutoff
            )
            .scalar()
            or 0
        )
            
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
        session,
        worker_name: str,
    ):

        state = self.get_worker_state(
            session,
            worker_name
        )

        if state:
            return state

        return self.create_worker_state(
            session,
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

 
    def get_bucket_expr(self):

        return func.to_timestamp(
            (
                func.extract(
                    "epoch",
                    self.GuardrailEventLog.created_on
                ) / 900
            ).cast(BigInteger) * 900
        )
  
    def rows_to_dict(self, result):
        return [
            dict(row._mapping)
            for row in result
        ]
    
 
    def create_event_stage_table(
        self,
        session,
        start_event_id: int,
        end_event_id: int,
    ):
        session.execute(
            text(
                """
                CREATE TEMP TABLE analytics_event_stage
                ON COMMIT DROP
                AS

                WITH guardrail_agg AS (
                    SELECT
                        gerl.event_id,
                        gerl.guardrail_type,

                        SUM(
                            CASE
                                WHEN gerl.results->'detail'->>'outcome'
                                    IN ('Success', 'Skipped')
                                THEN 1
                                ELSE 0
                            END
                        ) AS pass_count,

                        SUM(
                            CASE
                                WHEN gerl.results->'detail'->>'outcome'
                                    IN ('Fail', 'Error')
                                THEN 1
                                ELSE 0
                            END
                        ) AS block_count,

                        SUM(
                            CASE
                                WHEN gerl.results->'detail'->>'outcome'
                                    = 'Warning'
                                THEN 1
                                ELSE 0
                            END
                        ) AS warn_count,

                        SUM(
                            COALESCE(
                                (
                                    gerl.results->'usage'
                                    ->>'input_tokens'
                                )::BIGINT,
                                0
                            )
                        ) AS guardrail_input_tokens,

                        SUM(
                            COALESCE(
                                (
                                    gerl.results->'usage'
                                    ->>'output_tokens'
                                )::BIGINT,
                                0
                            )
                        ) AS guardrail_output_tokens,

                        SUM(
                            COALESCE(
                                (
                                    gerl.results->'usage'
                                    ->>'total_tokens'
                                )::BIGINT,
                                0
                            )
                        ) AS guardrail_total_tokens

                    FROM guardrail_event_result_log gerl

                    WHERE gerl.event_id BETWEEN
                        :start_event_id
                        AND
                        :end_event_id

                    GROUP BY
                        gerl.event_id,
                        gerl.guardrail_type
                ),

                input_guardrail_agg AS (
                    SELECT *
                    FROM guardrail_agg
                    WHERE guardrail_type = 'input'
                ),

                output_guardrail_agg AS (
                    SELECT *
                    FROM guardrail_agg
                    WHERE guardrail_type = 'output'
                )

                SELECT
                    gel.id AS event_id,

                    (
                        date_trunc(
                            'hour',
                            gel.created_on
                        )
                        +
                        floor(
                            extract(
                                minute
                                FROM gel.created_on
                            ) / 15
                        ) * interval '15 minute'
                    ) AS bucket_start_timestamp,

                    gel.created_on,

                    a.app_name,

                    gel.user_id,

                    gel.additional_info->>'X-Agent-Id'
                        AS agent_id,

                    gel.llm_type,

                    gel.duration,

                    CASE
                        WHEN length(
                            COALESCE(
                                gel.blocked_by,
                                ''
                            )
                        ) > 0
                        THEN 'Block'

                        WHEN
                            COALESCE(
                                input_guardrail_agg.warn_count,
                                0
                            )
                            +
                            COALESCE(
                                output_guardrail_agg.warn_count,
                                0
                            ) > 0
                        THEN 'Warn'

                        ELSE 'Pass'
                    END AS outcome,

                    COALESCE(
                        llm.input_tokens,
                        0
                    ) AS llm_input_tokens,

                    COALESCE(
                        llm.output_tokens,
                        0
                    ) AS llm_output_tokens,

                    COALESCE(
                        llm.total_tokens,
                        0
                    ) AS llm_total_tokens,

                    COALESCE(
                        input_guardrail_agg.guardrail_input_tokens,
                        0
                    ) AS ig_input_tokens,

                    COALESCE(
                        input_guardrail_agg.guardrail_output_tokens,
                        0
                    ) AS ig_output_tokens,

                    COALESCE(
                        input_guardrail_agg.guardrail_total_tokens,
                        0
                    ) AS ig_total_tokens,

                    COALESCE(
                        output_guardrail_agg.guardrail_input_tokens,
                        0
                    ) AS og_input_tokens,

                    COALESCE(
                        output_guardrail_agg.guardrail_output_tokens,
                        0
                    ) AS og_output_tokens,

                    COALESCE(
                        output_guardrail_agg.guardrail_total_tokens,
                        0
                    ) AS og_total_tokens

                FROM guardrail_event_log gel

                JOIN applications a
                    ON a.app_id = gel.app_id

                LEFT JOIN llm_details llm
                    ON llm.event_id = gel.id

                LEFT JOIN input_guardrail_agg
                    ON input_guardrail_agg.event_id = gel.id

                LEFT JOIN output_guardrail_agg
                    ON output_guardrail_agg.event_id = gel.id

                WHERE gel.id BETWEEN
                    :start_event_id
                    AND
                    :end_event_id
                """
            ),
            {
                "start_event_id": start_event_id,
                "end_event_id": end_event_id,
            },
        )
        
        count = session.execute(
            text("""
                SELECT COUNT(*)
                FROM analytics_event_stage
            """)
        ).scalar()

        print(f"Stage Rows = {count}")

        metadata = MetaData()
        
        stage_table = Table(
            "analytics_event_stage",
            metadata,
            autoload_with=session.connection(),
        )

        return stage_table

    def get_event_fact_rows(
        self,
        session,
        stage_table,
    ):
        rows = (
            session.query(
                stage_table.c.event_id,
                stage_table.c.bucket_start_timestamp,
                stage_table.c.created_on,
                stage_table.c.app_name,
                stage_table.c.user_id,
                stage_table.c.agent_id,
                stage_table.c.llm_type,
                stage_table.c.duration,
                stage_table.c.outcome,
                stage_table.c.llm_input_tokens,
                stage_table.c.llm_output_tokens,
                stage_table.c.llm_total_tokens,
                stage_table.c.ig_input_tokens,
                stage_table.c.ig_output_tokens,
                stage_table.c.ig_total_tokens,
                stage_table.c.og_input_tokens,
                stage_table.c.og_output_tokens,
                stage_table.c.og_total_tokens,
            )
            .all()
        )

        return self.rows_to_dict(rows)


    def get_workspace_agent_user_summary_rows(
        self,
        session,
        stage_table,
    ):
        query = (
            session.query(
                stage_table.c.app_name,

                stage_table.c.agent_id,

                stage_table.c.user_id,

                stage_table.c.bucket_start_timestamp,

                func.count().label(
                    "request_count"
                ),

                func.sum(
                    case(
                        (
                            stage_table.c.outcome == "Pass",
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
                            stage_table.c.outcome == "Warn",
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
                            stage_table.c.outcome == "Block",
                            1,
                        ),
                        else_=0,
                    )
                ).label(
                    "block_count"
                ),

                func.sum(
                    stage_table.c.llm_input_tokens
                ).label(
                    "llm_input_tokens"
                ),

                func.sum(
                    stage_table.c.llm_output_tokens
                ).label(
                    "llm_output_tokens"
                ),

                func.sum(
                    stage_table.c.llm_total_tokens
                ).label(
                    "llm_total_tokens"
                ),

                func.sum(
                    stage_table.c.ig_input_tokens
                ).label(
                    "ig_input_tokens"
                ),

                func.sum(
                    stage_table.c.ig_output_tokens
                ).label(
                    "ig_output_tokens"
                ),

                func.sum(
                    stage_table.c.ig_total_tokens
                ).label(
                    "ig_total_tokens"
                ),

                func.sum(
                    stage_table.c.og_input_tokens
                ).label(
                    "og_input_tokens"
                ),

                func.sum(
                    stage_table.c.og_output_tokens
                ).label(
                    "og_output_tokens"
                ),

                func.sum(
                    stage_table.c.og_total_tokens
                ).label(
                    "og_total_tokens"
                ),

                func.sum(
                    stage_table.c.duration
                ).label(
                    "total_response_time_ms"
                ),

                func.min(
                    stage_table.c.duration
                ).label(
                    "min_response_time_ms"
                ),

                func.max(
                    stage_table.c.duration
                ).label(
                    "max_response_time_ms"
                ),
            )
            .group_by(
                stage_table.c.app_name,
                stage_table.c.agent_id,
                stage_table.c.user_id,
                stage_table.c.bucket_start_timestamp,
            )
        )

        return self.rows_to_dict(
            query.all()
        )

    def get_workspace_summary_rows(
        self,
        session,
        stage_table
    ):


        query = (
            session.query(
                stage_table.c.app_name,

                stage_table.c.bucket_start_timestamp,

                func.count().label(
                    "request_count"
                ),

                func.sum(
                    case(
                        (
                            stage_table.c.outcome == "Pass",
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
                            stage_table.c.outcome == "Warn",
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
                            stage_table.c.outcome == "Block",
                            1,
                        ),
                        else_=0,
                    )
                ).label(
                    "block_count"
                ),

                func.sum(
                    stage_table.c.ig_input_tokens
                ).label(
                    "ig_input_tokens"
                ),

                func.sum(
                    stage_table.c.ig_output_tokens
                ).label(
                    "ig_output_tokens"
                ),

                func.sum(
                    stage_table.c.ig_total_tokens
                ).label(
                    "ig_total_tokens"
                ),

                func.sum(
                    stage_table.c.og_input_tokens
                ).label(
                    "og_input_tokens"
                ),

                func.sum(
                    stage_table.c.og_output_tokens
                ).label(
                    "og_output_tokens"
                ),

                func.sum(
                    stage_table.c.og_total_tokens
                ).label(
                    "og_total_tokens"
                ),

                func.sum(
                    stage_table.c.llm_input_tokens
                ).label(
                    "llm_input_tokens"
                ),

                func.sum(
                    stage_table.c.llm_output_tokens
                ).label(
                    "llm_output_tokens"
                ),

                func.sum(
                    stage_table.c.llm_total_tokens
                ).label(
                    "llm_total_tokens"
                ),

                func.sum(
                    stage_table.c.duration
                ).label(
                    "total_response_time_ms"
                ),

                func.min(
                    stage_table.c.duration
                ).label(
                    "min_response_time_ms"
                ),

                func.max(
                    stage_table.c.duration
                ).label(
                    "max_response_time_ms"
                ),
            )
            .group_by(
                stage_table.c.app_name,
                stage_table.c.bucket_start_timestamp,
            )
        )

        result = query.all()

        return [
            dict(row._mapping)
            for row in result
        ]
   
    def get_agent_summary_rows(
        self,
        session,
        stage_table
    ):

        query = (
            session.query(
                stage_table.c.app_name,

                stage_table.c.agent_id,

                stage_table.c.bucket_start_timestamp,

                func.count().label(
                    "request_count"
                ),

                func.sum(
                    case(
                        (
                            stage_table.c.outcome == "Pass",
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
                            stage_table.c.outcome == "Warn",
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
                            stage_table.c.outcome == "Block",
                            1,
                        ),
                        else_=0,
                    )
                ).label(
                    "block_count"
                ),

                func.sum(
                    stage_table.c.ig_input_tokens
                ).label(
                    "ig_input_tokens"
                ),

                func.sum(
                    stage_table.c.ig_output_tokens
                ).label(
                    "ig_output_tokens"
                ),

                func.sum(
                    stage_table.c.ig_total_tokens
                ).label(
                    "ig_total_tokens"
                ),

                func.sum(
                    stage_table.c.og_input_tokens
                ).label(
                    "og_input_tokens"
                ),

                func.sum(
                    stage_table.c.og_output_tokens
                ).label(
                    "og_output_tokens"
                ),

                func.sum(
                    stage_table.c.og_total_tokens
                ).label(
                    "og_total_tokens"
                ),

                func.sum(
                    stage_table.c.llm_input_tokens
                ).label(
                    "llm_input_tokens"
                ),

                func.sum(
                    stage_table.c.llm_output_tokens
                ).label(
                    "llm_output_tokens"
                ),

                func.sum(
                    stage_table.c.llm_total_tokens
                ).label(
                    "llm_total_tokens"
                ),

                func.sum(
                    stage_table.c.duration
                ).label(
                    "total_response_time_ms"
                ),

                func.min(
                    stage_table.c.duration
                ).label(
                    "min_response_time_ms"
                ),

                func.max(
                    stage_table.c.duration
                ).label(
                    "max_response_time_ms"
                ),
            )
            .filter(
                stage_table.c.agent_id.isnot(None)
            )
            .group_by(
                stage_table.c.app_name,
                stage_table.c.agent_id,
                stage_table.c.bucket_start_timestamp,
            )
        )

        result = query.all()

        return [
            dict(row._mapping)
            for row in result
        ]

    def get_user_summary_rows(
        self,
        session,
        stage_table
    ):

        query = (
            session.query(
                stage_table.c.app_name,

                stage_table.c.user_id,

                stage_table.c.bucket_start_timestamp,

                func.count().label(
                    "request_count"
                ),

                func.sum(
                    case(
                        (
                            stage_table.c.outcome == "Pass",
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
                            stage_table.c.outcome == "Warn",
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
                            stage_table.c.outcome == "Block",
                            1,
                        ),
                        else_=0,
                    )
                ).label(
                    "block_count"
                ),

                (
                    func.sum(
                        stage_table.c.llm_total_tokens
                    )
                    +
                    func.sum(
                        stage_table.c.ig_total_tokens
                    )
                    +
                    func.sum(
                        stage_table.c.og_total_tokens
                    )
                ).label(
                    "total_token_consumption"
                ),

                func.sum(
                    stage_table.c.ig_input_tokens
                ).label(
                    "ig_input_tokens"
                ),

                func.sum(
                    stage_table.c.ig_output_tokens
                ).label(
                    "ig_output_tokens"
                ),

                func.sum(
                    stage_table.c.ig_total_tokens
                ).label(
                    "ig_total_tokens"
                ),

                func.sum(
                    stage_table.c.og_input_tokens
                ).label(
                    "og_input_tokens"
                ),

                func.sum(
                    stage_table.c.og_output_tokens
                ).label(
                    "og_output_tokens"
                ),

                func.sum(
                    stage_table.c.og_total_tokens
                ).label(
                    "og_total_tokens"
                ),

                func.sum(
                    stage_table.c.llm_input_tokens
                ).label(
                    "llm_input_tokens"
                ),

                func.sum(
                    stage_table.c.llm_output_tokens
                ).label(
                    "llm_output_tokens"
                ),

                func.sum(
                    stage_table.c.llm_total_tokens
                ).label(
                    "llm_total_tokens"
                ),

                func.sum(
                    stage_table.c.duration
                ).label(
                    "total_response_time_ms"
                ),

                func.min(
                    stage_table.c.duration
                ).label(
                    "min_response_time_ms"
                ),

                func.max(
                    stage_table.c.duration
                ).label(
                    "max_response_time_ms"
                ),
            )
            .group_by(
                stage_table.c.app_name,
                stage_table.c.user_id,
                stage_table.c.bucket_start_timestamp,
            )
        )

        result = query.all()

        return self.rows_to_dict(result)
   
    def get_block_warn_pass_rows(
        self,
        session,
        stage_table
    ):
        query = (
            session.query(
                stage_table.c.app_name,

                stage_table.c.agent_id,

                stage_table.c.user_id,

                stage_table.c.bucket_start_timestamp,

                func.sum(
                    case(
                        (
                            stage_table.c.outcome == "Pass",
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
                            stage_table.c.outcome == "Warn",
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
                            stage_table.c.outcome == "Block",
                            1,
                        ),
                        else_=0,
                    )
                ).label(
                    "block_count"
                ),
            )
            .group_by(
                stage_table.c.app_name,
                stage_table.c.agent_id,
                stage_table.c.user_id,
                stage_table.c.bucket_start_timestamp,
            )
        )

        result = query.all()

        return self.rows_to_dict(result)
   
    def get_user_activity_rows(
        self,
        session,
        stage_table
    ):

        query = (
            session.query(
                stage_table.c.app_name,

                stage_table.c.user_id,

                stage_table.c.bucket_start_timestamp,

                func.max(
                    stage_table.c.created_on
                ).label(
                    "latest_request_timestamp"
                ),
            )
            .group_by(
                stage_table.c.app_name,
                stage_table.c.user_id,
                stage_table.c.bucket_start_timestamp,
            )
        )

        result = query.all()

        return self.rows_to_dict(result)
   
    def get_agent_activity_rows(
        self,
        session,
        stage_table
    ):

        query = (
            session.query(
                stage_table.c.app_name,

                stage_table.c.agent_id,

                stage_table.c.bucket_start_timestamp,

                func.max(
                    stage_table.c.created_on
                ).label(
                    "latest_request_timestamp"
                ),
            )
            .filter(
                stage_table.c.agent_id.isnot(None)
            )
            .group_by(
                stage_table.c.app_name,
                stage_table.c.agent_id,
                stage_table.c.bucket_start_timestamp,
            )
        )

        result = query.all()

        return self.rows_to_dict(result)
   
    def get_model_token_summary_rows(
        self,
        session,
        stage_table
    ):

        query = (
            session.query(
                stage_table.c.app_name,

                stage_table.c.user_id,

                stage_table.c.agent_id,

                stage_table.c.llm_type,

                stage_table.c.bucket_start_timestamp,

                func.count().label(
                    "total_requests"
                ),

                func.sum(
                    stage_table.c.llm_total_tokens
                ).label(
                    "llm_total_tokens"
                ),

                func.sum(
                    stage_table.c.ig_total_tokens
                ).label(
                    "ig_total_tokens"
                ),

                func.sum(
                    stage_table.c.og_total_tokens
                ).label(
                    "og_total_tokens"
                ),

                (
                    func.sum(
                        stage_table.c.llm_total_tokens
                    )
                    +
                    func.sum(
                        stage_table.c.ig_total_tokens
                    )
                    +
                    func.sum(
                        stage_table.c.og_total_tokens
                    )
                ).label(
                    "total_tokens"
                ),
            )
            .filter(
                stage_table.c.agent_id.isnot(None)
            )
            .group_by(
                stage_table.c.app_name,
                stage_table.c.user_id,
                stage_table.c.agent_id,
                stage_table.c.llm_type,
                stage_table.c.bucket_start_timestamp,
            )
        )

        result = query.all()

        return self.rows_to_dict(result)
   
    def get_guardrail_outcome_summary_rows(
        self,
        session,
        start_event_id: int,
        end_event_id: int,
    ):

        bucket_expr = self.get_bucket_expr()
  
        agent_id_expr =(
            self.GuardrailEventLog.additional_info[
                "X-Agent-Id"
            ].astext
        )

        outcome_case = case(
            (
                self.GuardrailEventResultLog.results[
                    "detail"
                ]["outcome"].astext.in_(
                    ["Fail", "Error"]
                ),
                "Block",
            ),
            (
                self.GuardrailEventResultLog.results[
                    "detail"
                ]["outcome"].astext
                == "Warning",
                "Warn",
            ),
            else_=None,
        )

        query = (
            session.query(
                self.Applications.app_name.label(
                    "app_name"
                ),

                self.GuardrailEventLog.user_id.label(
                    "user_id"
                ),

                agent_id_expr.label("agent_id"),

                self.GuardrailEventResultLog.eval_name.label(
                    "eval_name"
                ),

                bucket_expr.label(
                    "bucket_start_timestamp"
                ),

                func.sum(
                    case(
                        (
                            outcome_case == "Warn",
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
                            outcome_case == "Block",
                            1,
                        ),
                        else_=0,
                    )
                ).label(
                    "block_count"
                ),

                func.sum(
                    case(
                        (
                            outcome_case.in_(
                                ["Warn", "Block"]
                            ),
                            1,
                        ),
                        else_=0,
                    )
                ).label(
                    "total_detect_count"
                ),
            )
            .join(
                self.GuardrailEventLog,
                self.GuardrailEventResultLog.event_id
                == self.GuardrailEventLog.id,
            )
            .join(
                self.Applications,
                self.GuardrailEventLog.app_id
                == self.Applications.app_id,
            )
            .filter(
                self.GuardrailEventLog.id >= start_event_id
            )
            .filter(
                self.GuardrailEventLog.id <= end_event_id
            )
            .filter(
                self.GuardrailEventResultLog.results[
                    "detail"
                ]["outcome"].astext.in_(
                    [
                        "Fail",
                        "Error",
                        "Warning",
                    ]
                )
            )
            .filter(
                agent_id_expr.isnot(None)
            )
            .group_by(
                self.Applications.app_name,
                self.GuardrailEventLog.user_id,
                agent_id_expr.label("agent_id"),
                self.GuardrailEventResultLog.eval_name,
                bucket_expr,
            )
        )

        result = query.all()

        return self.rows_to_dict(result)
   

analytics_db = TrustaiAnalyticsDB()