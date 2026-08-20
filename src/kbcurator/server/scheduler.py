# scheduler.py

from apscheduler.schedulers.background import BackgroundScheduler
from trustai_analytics.workers.analytics_worker import run_analytics_worker
from datetime import datetime

scheduler = BackgroundScheduler()

def start_scheduler():
    scheduler.add_job(
        run_analytics_worker,
        trigger="interval",
        minutes=15,
        max_instances=1,
        coalesce=True,
        next_run_time=datetime.now(),
    )

    scheduler.start()