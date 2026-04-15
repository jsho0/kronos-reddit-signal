"""
APScheduler wrapper for the pipeline.

Runs PipelineRunner.run() on a cron schedule (default: 8am ET weekdays).
max_instances=1 prevents overlapping runs if one takes too long.
coalesce=True skips missed runs rather than queuing them up.

Job execution listener logs outcome and detects persistent failures.
"""
import logging

from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED
from apscheduler.schedulers.blocking import BlockingScheduler

import config
from meta.runner import MetaRunner
from pipeline.runner import PipelineRunner

logger = logging.getLogger(__name__)


def _make_listener(runner_ref: list):
    """Returns an APScheduler event listener that logs job outcomes."""
    def listener(event):
        if event.exception:
            logger.error("Pipeline job FAILED: %s", event.exception)
        else:
            health = event.retval
            if health is not None:
                logger.info(
                    "Pipeline job completed: %d/%d ok in %.1fs",
                    health.tickers_succeeded,
                    health.tickers_attempted,
                    health.duration_seconds,
                )
    return listener


def run_once(tickers: list[str] = None):
    """Run the pipeline once immediately (no scheduler). Useful for manual runs."""
    runner = PipelineRunner()
    return runner.run(tickers)


def start_scheduler(
    hour: int = 8,
    minute: int = 0,
    timezone: str = "America/New_York",
    tickers: list[str] = None,
):
    """
    Start the blocking scheduler. Runs pipeline daily at hour:minute in given timezone.
    Only fires on weekdays (day_of_week='mon-fri').

    This call blocks — run it as the main process or in a dedicated thread.
    """
    runner = PipelineRunner()
    meta = MetaRunner()
    scheduler = BlockingScheduler(timezone=timezone)

    scheduler.add_listener(
        _make_listener([runner]),
        EVENT_JOB_EXECUTED | EVENT_JOB_ERROR,
    )

    scheduler.add_job(
        func=runner.run,
        kwargs={"tickers": tickers or config.WATCHLIST},
        trigger="cron",
        day_of_week="mon-fri",
        hour=hour,
        minute=minute,
        max_instances=1,
        coalesce=True,
        id="daily_pipeline",
        name="Kronos+Reddit daily signal pipeline",
    )

    # Weekly meta runner: Sunday 11pm ET — proposes/activates/analyzes data source experiments
    scheduler.add_job(
        func=meta.run,
        trigger="cron",
        day_of_week="sun",
        hour=23,
        minute=0,
        max_instances=1,
        coalesce=True,
        id="weekly_meta",
        name="ASI-Evolve meta runner (data source evolution)",
    )

    logger.info(
        "Scheduler started: pipeline at %02d:%02d ET weekdays, meta runner Sundays 23:00 ET",
        hour, minute,
    )

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")
        scheduler.shutdown()
