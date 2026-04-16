"""
Options flow data source.

Uses put/call volume ratio from the nearest-expiry options chain.
High put buying → bearish. High call buying → bullish.

Score mapping:
  P/C > 1.8  → 0.30 (heavy put buying, bearish)
  P/C > 1.3  → 0.40
  P/C 0.7-1.3 → 0.50 (neutral)
  P/C < 0.7  → 0.60
  P/C < 0.4  → 0.70 (heavy call buying, bullish)
"""
import logging

import yfinance as yf

from data_sources import DataSourceResult

logger = logging.getLogger(__name__)

ENABLED = True
WEIGHT = 0.10
NAME = "Options Flow"


def fetch(ticker: str, ohlcv_df=None, as_of_date=None) -> DataSourceResult:
    if as_of_date is not None:
        from datetime import date
        if as_of_date < date.today():
            logger.debug("options_flow: as_of_date ignored (yfinance options are live-only)")
    try:
        t = yf.Ticker(ticker)
        dates = t.options
        if not dates:
            return DataSourceResult(name=NAME)

        chain = t.option_chain(dates[0])
        call_vol = float(chain.calls["volume"].fillna(0).sum())
        put_vol = float(chain.puts["volume"].fillna(0).sum())

        if call_vol + put_vol < 10:
            return DataSourceResult(name=NAME)

        pc = put_vol / max(call_vol, 1.0)

        if pc > 1.8:
            score = 0.30
            reasoning = [f"Heavy put buying (P/C {pc:.2f}) — bearish options flow"]
        elif pc > 1.3:
            score = 0.40
            reasoning = [f"Elevated put/call ratio ({pc:.2f})"]
        elif pc < 0.4:
            score = 0.70
            reasoning = [f"Heavy call buying (P/C {pc:.2f}) — bullish options flow"]
        elif pc < 0.7:
            score = 0.60
            reasoning = [f"Call-heavy options flow (P/C {pc:.2f})"]
        else:
            score = 0.50
            reasoning = []

        return DataSourceResult(
            name=NAME,
            score=score,
            reasoning=reasoning,
            raw={"pc_ratio": round(pc, 3), "call_vol": call_vol, "put_vol": put_vol},
        )

    except Exception as exc:
        logger.debug("options_flow: %s failed: %s", ticker, exc)
        return DataSourceResult(name=NAME)
