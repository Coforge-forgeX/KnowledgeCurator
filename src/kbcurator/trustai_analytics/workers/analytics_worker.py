from datetime import datetime
import time

from trustai_analytics.trustai_db import analytics_db

from trustai_analytics.model.db_model import (
    WorkspaceSummary,
    AgentSummary,
    UserSummary,
    BlockWarnPassSummary,
    UserActivitySummary,
    AgentActivitySummary,
    ModelTokenSummary,
    GuardrailOutcomeSummary,
)

import logging
from datetime import datetime
logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)
logger = logging.getLogger(__name__)

WORKER_NAME = "ANALYTICS_AGGREGATION_WORKER"

BATCH_SIZE = 10000


def run_analytics_worker():
    
    print("=" * 80)
    print("ANALYTICS WORKER STARTED")
    print("=" * 80)

    run_started_at = datetime.utcnow()

    logger.info(
        "Analytics worker started. worker=%s",
        WORKER_NAME,
    )

    session = analytics_db.Session()

    checkpoint = 0
    end_event_id = 0

    try:

        logger.info(
            "Attempting to acquire worker lock."
        )

        if not analytics_db.acquire_worker_lock(
            session,
            "analytics_worker",
        ):
            logger.info(
                "Worker lock already held. Exiting."
            )
            return

        logger.info(
            "Worker lock acquired."
        )

        analytics_db.create_worker_state_if_missing(
            session,
            WORKER_NAME
        )

        checkpoint = analytics_db.get_checkpoint(
            session,
            WORKER_NAME
        )

        max_event_id = (
            analytics_db.get_current_max_event_id()
        )

        logger.info(
            "Checkpoint=%s MaxEventId=%s",
            checkpoint,
            max_event_id,
        )

        if max_event_id <= checkpoint:
            logger.info(
                "No new events to process."
            )
            return

        start_event_id = checkpoint + 1

        end_event_id = min(
            checkpoint + BATCH_SIZE,
            max_event_id,
        )

        logger.info(
            "Processing event range [%s - %s]",
            start_event_id,
            end_event_id,
        )

        backlog_before = (
            max_event_id - checkpoint
        )

        logger.info(
            "Creating analytics staging table."
        )

        stage_table = (
            analytics_db.create_event_stage_table(
                session,
                start_event_id,
                end_event_id,
            )
        )

        logger.info(
            "Stage table created successfully."
        )

        #
        # Workspace Summary
        #

        workspace_rows = (
            analytics_db.get_workspace_summary_rows(
                session,
                stage_table,
            )
        )

        logger.info(
            "WorkspaceSummary rows=%s",
            len(workspace_rows),
        )

        analytics_db.bulk_aggregate_upsert(
            session,
            WorkspaceSummary,
            workspace_rows,
            [
                "app_name",
                "bucket_start_timestamp",
            ],
        )

        #
        # Agent Summary
        #

        agent_rows = (
            analytics_db.get_agent_summary_rows(
                session,
                stage_table,
            )
        )

        logger.info(
            "AgentSummary rows=%s",
            len(agent_rows),
        )

        analytics_db.bulk_aggregate_upsert(
            session,
            AgentSummary,
            agent_rows,
            [
                "app_name",
                "agent_id",
                "bucket_start_timestamp",
            ],
        )

        #
        # User Summary
        #

        user_rows = (
            analytics_db.get_user_summary_rows(
                session,
                stage_table,
            )
        )

        logger.info(
            "UserSummary rows=%s",
            len(user_rows),
        )

        analytics_db.bulk_aggregate_upsert(
            session,
            UserSummary,
            user_rows,
            [
                "app_name",
                "user_id",
                "bucket_start_timestamp",
            ],
        )

        #
        # BlockWarnPass Summary
        #

        block_warn_pass_rows = (
            analytics_db.get_block_warn_pass_rows(
                session,
                stage_table,
            )
        )

        logger.info(
            "BlockWarnPassSummary rows=%s",
            len(block_warn_pass_rows),
        )

        analytics_db.bulk_aggregate_upsert(
            session,
            BlockWarnPassSummary,
            block_warn_pass_rows,
            [
                "app_name",
                "agent_id",
                "user_id",
                "bucket_start_timestamp",
            ],
        )

        #
        # User Activity
        #

        user_activity_rows = (
            analytics_db.get_user_activity_rows(
                session,
                stage_table,
            )
        )

        logger.info(
            "UserActivitySummary rows=%s",
            len(user_activity_rows),
        )

        analytics_db.bulk_upsert(
            session,
            UserActivitySummary,
            user_activity_rows,
            [
                "app_name",
                "user_id",
                "bucket_start_timestamp",
            ],
        )

        #
        # Agent Activity
        #

        agent_activity_rows = (
            analytics_db.get_agent_activity_rows(
                session,
                stage_table,
            )
        )

        logger.info(
            "AgentActivitySummary rows=%s",
            len(agent_activity_rows),
        )

        analytics_db.bulk_upsert(
            session,
            AgentActivitySummary,
            agent_activity_rows,
            [
                "app_name",
                "agent_id",
                "bucket_start_timestamp",
            ],
        )

        #
        # Model Token Summary
        #

        model_token_rows = (
            analytics_db.get_model_token_summary_rows(
                session,
                stage_table,
            )
        )

        logger.info(
            "ModelTokenSummary rows=%s",
            len(model_token_rows),
        )

        analytics_db.bulk_aggregate_upsert(
            session,
            ModelTokenSummary,
            model_token_rows,
            [
                "app_name",
                "user_id",
                "agent_id",
                "llm_type",
                "bucket_start_timestamp",
            ],
        )

        #
        # Guardrail Outcome Summary
        #

        guardrail_rows = (
            analytics_db.get_guardrail_outcome_summary_rows(
                session,
                start_event_id,
                end_event_id,
            )
        )

        logger.info(
            "GuardrailOutcomeSummary rows=%s",
            len(guardrail_rows),
        )

        analytics_db.bulk_aggregate_upsert(
            session,
            GuardrailOutcomeSummary,
            guardrail_rows,
            [
                "app_name",
                "user_id",
                "agent_id",
                "eval_name",
                "bucket_start_timestamp",
            ],
        )

        run_completed_at = datetime.utcnow()

        run_duration_ms = int(
            (
                run_completed_at -
                run_started_at
            ).total_seconds() * 1000
        )

        rows_processed = (
            end_event_id - checkpoint
        )

        backlog_after = max(
            0,
            max_event_id - end_event_id,
        )

        logger.info(
            "Updating worker checkpoint. "
            "rows_processed=%s backlog_after=%s",
            rows_processed,
            backlog_after,
        )

        analytics_db.update_worker_checkpoint(
            session=session,
            worker_name=WORKER_NAME,
            last_processed_event_id=end_event_id,
            last_processed_timestamp=run_completed_at,
            rows_processed=rows_processed,
            backlog_items=backlog_after,
            last_run_duration_ms=run_duration_ms,
            status="SUCCESS",
            error_message=None,
        )

        analytics_db.insert_worker_execution_history(
            session=session,
            worker_name=WORKER_NAME,
            run_started_at=run_started_at,
            run_completed_at=run_completed_at,
            start_checkpoint=checkpoint,
            end_checkpoint=end_event_id,
            rows_processed=rows_processed,
            backlog_before=backlog_before,
            backlog_after=backlog_after,
            run_duration_ms=run_duration_ms,
            status="SUCCESS",
            error_message=None,
        )

        logger.info(
            "Committing transaction."
        )

        session.commit()

        logger.info(
            "Analytics worker completed successfully. "
            "Duration=%sms",
            run_duration_ms,
        )

    except Exception:

        logger.exception(
            "Analytics worker failed."
        )

        session.rollback()

        raise

    finally:

        try:
            analytics_db.release_worker_lock(
                session,
                "analytics_worker",
            )

            logger.info(
                "Worker lock released."
            )

        except Exception:
            logger.exception(
                "Failed to release worker lock."
            )

        session.close()

        logger.info(
            "Database session closed."
        )