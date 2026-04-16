"""
Macro regime data source.

Combines VIX level and SPY 20-day trend into a macro risk score.
High VIX + SPY downtrend = risk-off = bearish context for individual stocks.

VIX thresholds:
  < 15  → low fear, bullish (+0.08)
  15-20 → neutral
  20-25 → elevated fear (-0.06)
  > 25  → risk-off (-0.14)

SPY vs 20MA:
  above → +0.04
  below → -0.04
"""
import logging
from datetime import timedelta

import yfinance as yf

from data_sources import DataSourceResult

logger = logging.getLogger(__name__)

ENABLED = True
WEIGHT = 0.08
NAME = "Macro Regime"

# Cache VIX + SPY within same process to avoid duplicate downloads per ticker
# Keyed by as_of_date string so different dates don't collide
_cache: dict = {}


def _get_macro(as_of_date=None) -> tuple[float, float]:
    """Returns (vix_level, spy_vs_20ma_pct). Cached per process run per date."""
    from datetime import date as date_cls
    effective_date = as_of_date if as_of_date is not None else date_cls.today()
    cache_key = str(effective_date)

    if cache_key in _cache:
        entry = _cache[cache_key]
        return entry["vix"], entry["spy_push"]

    vix_push = 0.0
    spy_push = 0.0
    vix_val = 18.0  # neutral fallback

    if as_of_date is not None:
        vix_start = (as_of_date - timedelta(days=7)).isoformat()
        vix_end = as_of_date.isoformat()
        spy_start = (as_of_date - timedelta(days=90)).isoformat()
        spy_end = as_of_date.isoformat()
        vix_kwargs = {"start": vix_start, "end": vix_end}
        spy_kwargs = {"start": spy_start, "end": spy_end}
    else:
        vix_kwargs = {"period": "5d"}
        spy_kwargs = {"period": "60d"}

    try:
        vix_df = yf.download("^VIX", interval="1d", progress=False, auto_adjust=True, **vix_kwargs)
        if not vix_df.empty:
            vix_val = float(vix_df["Close"].to_numpy()[-1])
    except Exception as exc:
        logger.debug("macro: VIX fetch failed: %s", exc)

    try:
        spy_df = yf.download("SPY", interval="1d", progress=False, auto_adjust=True, **spy_kwargs)
        if not spy_df.empty and len(spy_df) >= 20:
            spy_close = spy_df["Close"]
            ma20 = float(spy_close.rolling(20).mean().to_numpy()[-1])
            last = float(spy_close.to_numpy()[-1])
            spy_push = 0.04 if last > ma20 else -0.04
    except Exception as exc:
        logger.debug("macro: SPY fetch failed: %s", exc)

    if vix_val < 15:
        vix_push = 0.08
    elif vix_val < 20:
        vix_push = 0.0
    elif vix_val < 25:
        vix_push = -0.06
    else:
        vix_push = -0.14

    _cache[cache_key] = {"vix": vix_val, "spy_push": spy_push, "vix_push": vix_push}
    return vix_val, vix_push + spy_push


def fetch(ticker: str, ohlcv_df=None, as_of_date=None) -> DataSourceResult:
    try:
        vix_val, total_push = _get_macro(as_of_date)
        score = max(0.0, min(1.0, 0.5 + total_push))

        reasoning = []
        if vix_val > 25:
            reasoning.append(f"VIX risk-off ({vix_val:.1f}) — macro headwind")
        elif vix_val > 20:
            reasoning.append(f"VIX elevated ({vix_val:.1f})")
        elif vix_val < 15:
            reasoning.append(f"VIX low ({vix_val:.1f}) — low fear environment")

        spy_push = _cache.get("spy_push", 0.0)
        if spy_push < 0:
            reasoning.append("SPY below 20MA (broad market downtrend)")

        return DataSourceResult(
            name=NAME,
            score=score,
            reasoning=reasoning,
            raw={"vix": round(vix_val, 2), "total_push": round(total_push, 3)},
        )

    except Exception as exc:
        logger.debug("macro: fetch failed: %s", exc)
        return DataSourceResult(name=NAME)
