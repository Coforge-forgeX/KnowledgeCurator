from datetime import datetime, timedelta
import logging

from kbcurator.trustai_analytics.trustai_db import analytics_db

from kbcurator.trustai_analytics.model.db_model import (
    GuardrailOutcomeFact,
    AnalyticsEventFact
)

logger = logging.getLogger(__name__)

WORKER_NAME = "ANALYTICS_FACT_WORKER"

BATCH_SIZE = 10000
LAG_MINUTES = 2
MAX_BATCHES_PER_RUN = 100


def run_analytics_fact_worker():

    print("=" * 80)
    print("ANALYTICS FACT WORKER STARTED")
    print("=" * 80)

    worker_run_time = datetime.utcnow()

    processing_cutoff = (
        worker_run_time
        - timedelta(minutes=LAG_MINUTES)
    )

    logger.info(
        "Analytics worker started. "
        "worker=%s cutoff=%s",
        WORKER_NAME,
        processing_cutoff,
    )

    session = analytics_db.Session()

    try:

        logger.info(
            "Attempting to acquire worker lock."
        )

        if not analytics_db.acquire_worker_lock(
            session,
            "fact_analytics_worker",
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
            WORKER_NAME,
        )

        max_event_id = (
            analytics_db.get_max_processible_event_id(
                session,
                processing_cutoff,
            )
        )

        logger.info(
            "Worker cutoff=%s max_event_id=%s",
            processing_cutoff,
            max_event_id,
        )

        batch_number = 0

        while True:

            if batch_number >= MAX_BATCHES_PER_RUN:

                logger.warning(
                    "Maximum batches reached. "
                    "batch_count=%s",
                    batch_number,
                )

                break

            checkpoint = (
                analytics_db.get_checkpoint(
                    session,
                    WORKER_NAME,
                )
            )

            logger.info(
                "Checkpoint=%s MaxEventId=%s",
                checkpoint,
                max_event_id,
            )

            if max_event_id <= checkpoint:

                logger.info(
                    "No more events to process."
                )

                break

            run_started_at = datetime.utcnow()

            start_event_id = checkpoint + 1

            end_event_id = min(
                checkpoint + BATCH_SIZE,
                max_event_id,
            )

            backlog_before = (
                max_event_id - checkpoint
            )

            logger.info(
                "Processing batch=%s range=[%s-%s]",
                batch_number + 1,
                start_event_id,
                end_event_id,
            )

            #
            # Create Stage Table
            #

            stage_table = (
                analytics_db.create_event_stage_table(
                    session,
                    start_event_id,
                    end_event_id,
                )
            )
            
            event_fact_rows = (
                analytics_db.get_event_fact_rows(
                    session,
                    stage_table,
                )
            )
            
            analytics_db.bulk_upsert(
                session,
                AnalyticsEventFact,
                event_fact_rows,
                [
                    "event_id",
                ],
            )


            #
            # Guardrail Outcome Summary
            #

            guardrail_rows = (
                analytics_db.get_guardrail_outcome_rows(
                    session,
                    start_event_id,
                    end_event_id,
                )
            )

            analytics_db.bulk_upsert(
                session,
                GuardrailOutcomeFact,
                guardrail_rows,
                [
                    "source_event_id",
                    "eval_name",
                ],
            )

            run_completed_at = datetime.utcnow()

            run_duration_ms = int(
                (
                    run_completed_at
                    - run_started_at
                ).total_seconds()
                * 1000
            )

            rows_processed = (
                end_event_id
                - start_event_id
                + 1
            )

            backlog_after = max(
                0,
                max_event_id
                - end_event_id,
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

            session.commit()

            logger.info(
                "Batch committed. "
                "batch=%s rows=%s backlog_after=%s",
                batch_number + 1,
                rows_processed,
                backlog_after,
            )

            batch_number += 1

            if end_event_id >= max_event_id:

                logger.info(
                    "Reached watermark cutoff."
                )

                break

        logger.info(
            "Analytics worker completed successfully."
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