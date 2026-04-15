"""
Earnings proximity data source.

Adjusts signal confidence based on how close the next earnings date is.
Close to earnings = high uncertainty = pull score toward neutral.

Score:
  1-3 days until earnings  → 0.50 + slight neutralization note
  4-7 days until earnings  → mild 0.48 (small bearish tilt — pre-earnings drift risk)
  Otherwise                → 0.50 (no effect)

This source has a small weight (0.05) so its effect is a gentle dampener,
not a directional signal. Its real job is to flag uncertainty in reasoning.
"""
import logging
from datetime import date

import yfinance as yf

from data_sources import DataSourceResult

logger = logging.getLogger(__name__)

ENABLED = True
WEIGHT = 0.05
NAME = "Earnings Proximity"


def fetch(ticker: str, ohlcv_df=None) -> DataSourceResult:
    try:
        t = yf.Ticker(ticker)
        cal = t.calendar

        if cal is None or cal.empty:
            return DataSourceResult(name=NAME)

        today = date.today()
        next_earnings = None

        if "Earnings Date" in cal.index:
            for val in cal.loc["Earnings Date"].values:
                try:
                    import pandas as pd
                    d = pd.Timestamp(val).date()
                    if d >= today:
                        next_earnings = d
                        break
                except Exception:
                    continue

        if next_earnings is None:
            return DataSourceResult(name=NAME)

        days = (next_earnings - today).days

        if days <= 3:
            return DataSourceResult(
                name=NAME,
                score=0.50,
                reasoning=[f"Earnings in {days}d — signal reliability reduced"],
                raw={"days_until_earnings": days},
            )
        elif days <= 7:
            return DataSourceResult(
                name=NAME,
                score=0.48,
                reasoning=[f"Earnings in {days}d"],
                raw={"days_until_earnings": days},
            )
        else:
            return DataSourceResult(
                name=NAME,
                score=0.50,
                raw={"days_until_earnings": days},
            )

    except Exception as exc:
        logger.debug("earnings: %s failed: %s", ticker, exc)
        return DataSourceResult(name=NAME)
