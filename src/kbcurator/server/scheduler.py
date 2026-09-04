# scheduler.py

import logging
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime

from kbcurator.utils.config import settings

logger = logging.getLogger(__name__)

run_analytics_fact_worker = None

if settings.TRUSTAI_DB:
    try:
        from kbcurator.trustai_analytics.workers.analytics_fact_worker import run_analytics_fact_worker
    except Exception as exc:
        logger.warning(f"TrustAI analytics disabled: {exc}")

scheduler = BackgroundScheduler()

def start_scheduler():
    if run_analytics_fact_worker is None:
        logger.info("TrustAI scheduler disabled - TRUSTAI_DEV_DB not configured")
        return

    scheduler.add_job(
        run_analytics_fact_worker,
        trigger="interval",
        minutes=15,
        max_instances=1,
        coalesce=True,
        next_run_time=datetime.now(),
    )

    scheduler.start()
