"""
Entry point.

Usage:
    # Run discovery then pipeline immediately
    venv\Scripts\python.exe main.py --once

    # Run only discovery (build/update dynamic watchlist)
    venv\Scripts\python.exe main.py --discover

    # Run pipeline only on already-discovered tickers
    venv\Scripts\python.exe main.py --pipeline

    # Debug: run pipeline on specific tickers (skips discovery)
    venv\Scripts\python.exe main.py --pipeline --tickers AAPL TSLA NVDA

    # Start the scheduler (blocks, discovery 6am + pipeline 7am ET weekdays)
    venv\Scripts\python.exe main.py --schedule
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
    parser = argparse.ArgumentParser(description="Kronos + Reddit discovery pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--once", action="store_true", help="Run discovery + pipeline and exit")
    group.add_argument("--discover", action="store_true", help="Run discovery only (update watchlist)")
    group.add_argument("--pipeline", action="store_true", help="Run pipeline only on discovered tickers")
    group.add_argument("--schedule", action="store_true", help="Start daily scheduler")
    parser.add_argument("--tickers", nargs="+", help="Debug: override with specific tickers (--pipeline only)")
    parser.add_argument("--hour", type=int, default=6, help="Discovery hour ET (default 6, pipeline runs at hour+1)")
    parser.add_argument("--minute", type=int, default=0, help="Scheduler minute (default 0)")
    args = parser.parse_args()

    if args.once:
        from pipeline.discovery_runner import DiscoveryRunner
        from pipeline.runner import PipelineRunner

        logger.info("Running discovery...")
        disc_summary = DiscoveryRunner().run()
        print(f"\nDiscovery: {disc_summary['qualified']} tickers qualified "
              f"({disc_summary['rejected']} rejected) in {disc_summary['duration_seconds']}s")
        print(f"Tickers: {', '.join(disc_summary['tickers']) or 'none'}")

        if not disc_summary["tickers"]:
            print("No tickers qualified — pipeline skipped.")
            return

        logger.info("Running pipeline on %d discovered tickers...", len(disc_summary["tickers"]))
        runner = PipelineRunner()
        health = runner.run()
        print(f"\nPipeline: {health.tickers_succeeded}/{health.tickers_attempted} tickers ok "
              f"in {health.duration_seconds:.1f}s")
        if health.tickers_failed:
            print(f"Failures: {'; '.join(health.notes)}")
            sys.exit(1)

    elif args.discover:
        from pipeline.discovery_runner import DiscoveryRunner
        summary = DiscoveryRunner().run()
        print(f"\nDiscovery complete: {summary['qualified']} qualified, {summary['rejected']} rejected")
        print(f"Tickers: {', '.join(summary['tickers']) or 'none'}")

    elif args.pipeline:
        if args.tickers:
            logger.info("Pipeline: debug mode with %d explicit tickers", len(args.tickers))
        from pipeline.runner import PipelineRunner
        runner = PipelineRunner()
        health = runner.run(tickers=args.tickers or None)
        print(f"\nPipeline: {health.tickers_succeeded}/{health.tickers_attempted} ok "
              f"in {health.duration_seconds:.1f}s")
        if health.tickers_failed:
            print(f"Failures: {'; '.join(health.notes)}")
            sys.exit(1)

    elif args.schedule:
        if args.tickers:
            logger.warning("--tickers is ignored with --schedule")
        from pipeline.scheduler import start_scheduler
        start_scheduler(hour=args.hour, minute=args.minute)


if __name__ == "__main__":
    main()
