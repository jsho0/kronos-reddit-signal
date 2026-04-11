"""
Next-day price backfill script.

For every signal that has price_at_signal but no price_next_day,
fetches the actual closing price for the trading day after signal_date
and writes it back to the DB.

This populates the Accuracy tab in the dashboard.

Run daily (after market close) or on-demand:
    venv\Scripts\python.exe scripts/backfill_next_day_price.py

Dry-run (show what would be updated, no writes):
    venv\Scripts\python.exe scripts/backfill_next_day_price.py --dry-run
"""
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: F401 — loads .env

import pandas as pd
import yfinance as yf
from sqlalchemy import select

from storage.db import get_session, init_db
from storage.models import Signal
from storage.store import SignalStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backfill")


def get_next_trading_day_close(ticker: str, signal_date: str) -> float | None:
    """
    Return the closing price for the first trading day after signal_date.
    Returns None if no data is available (weekend, holiday, too recent).
    """
    start = pd.Timestamp(signal_date) + pd.Timedelta(days=1)
    end = start + pd.Timedelta(days=7)  # look up to a week ahead for holidays

    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            return None
        # First available trading day after signal_date
        return float(df["Close"].iloc[0])
    except Exception as e:
        logger.warning("%s: yfinance error: %s", ticker, e)
        return None


def run(dry_run: bool = False):
    init_db()
    store = SignalStore()

    # Find all signals missing price_next_day
    with get_session() as session:
        rows = session.execute(
            select(Signal).where(
                Signal.price_at_signal.isnot(None),
                Signal.price_next_day.is_(None),
            )
        ).scalars().all()
        pending = [(r.ticker, r.signal_date) for r in rows]

    if not pending:
        logger.info("Nothing to backfill — all signals already have price_next_day")
        return

    logger.info("Found %d signals to backfill", len(pending))

    updated = 0
    skipped = 0

    for ticker, signal_date in pending:
        price = get_next_trading_day_close(ticker, signal_date)
        if price is None:
            logger.debug("%s %s: no next-day price yet (too recent or holiday)", ticker, signal_date)
            skipped += 1
            continue

        if dry_run:
            logger.info("[DRY RUN] %s %s: would set price_next_day=%.2f", ticker, signal_date, price)
        else:
            store.update_next_day_price(ticker, signal_date, price)
            logger.info("%s %s: price_next_day=%.2f", ticker, signal_date, price)
        updated += 1

    action = "Would update" if dry_run else "Updated"
    logger.info("%s %d signals, skipped %d (no data yet)", action, updated, skipped)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill next-day prices for accuracy tracking")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be updated without writing")
    args = parser.parse_args()
    run(dry_run=args.dry_run)
