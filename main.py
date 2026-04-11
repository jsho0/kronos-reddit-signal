"""
Entry point.

Usage:
    # Run pipeline once immediately
    venv\Scripts\python.exe main.py --once

    # Run pipeline once for specific tickers
    venv\Scripts\python.exe main.py --once --tickers AAPL TSLA NVDA

    # Start the daily scheduler (blocks, runs at 8am ET weekdays)
    venv\Scripts\python.exe main.py --schedule

    # Run scheduler at a custom time
    venv\Scripts\python.exe main.py --schedule --hour 9 --minute 30
"""
import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")


def main():
    parser = argparse.ArgumentParser(description="Kronos + Reddit signal pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--once", action="store_true", help="Run pipeline once and exit")
    group.add_argument("--schedule", action="store_true", help="Start daily scheduler")
    parser.add_argument("--tickers", nargs="+", help="Override watchlist (--once only)")
    parser.add_argument("--hour", type=int, default=8, help="Scheduler hour (ET, default 8)")
    parser.add_argument("--minute", type=int, default=0, help="Scheduler minute (default 0)")
    args = parser.parse_args()

    if args.once:
        from pipeline.runner import PipelineRunner
        runner = PipelineRunner()
        health = runner.run(tickers=args.tickers)
        print(f"\nDone: {health.tickers_succeeded}/{health.tickers_attempted} tickers ok "
              f"in {health.duration_seconds:.1f}s")
        if health.tickers_failed:
            print(f"Failures: {'; '.join(health.notes)}")
            sys.exit(1)

    elif args.schedule:
        if args.tickers:
            logger.warning("--tickers is ignored with --schedule (uses WATCHLIST from .env)")
        from pipeline.scheduler import start_scheduler
        start_scheduler(hour=args.hour, minute=args.minute)


if __name__ == "__main__":
    main()
