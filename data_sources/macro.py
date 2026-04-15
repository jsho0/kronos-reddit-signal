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

import yfinance as yf

from data_sources import DataSourceResult

logger = logging.getLogger(__name__)

ENABLED = True
WEIGHT = 0.08
NAME = "Macro Regime"

# Cache VIX + SPY within same process to avoid duplicate downloads per ticker
_cache: dict = {}


def _get_macro() -> tuple[float, float]:
    """Returns (vix_level, spy_vs_20ma_pct). Cached per process run."""
    if "vix" in _cache:
        return _cache["vix"], _cache["spy_push"]

    vix_push = 0.0
    spy_push = 0.0
    vix_val = 18.0  # neutral fallback

    try:
        vix_df = yf.download("^VIX", period="5d", interval="1d", progress=False, auto_adjust=True)
        if not vix_df.empty:
            vix_val = float(vix_df["Close"].to_numpy()[-1])
    except Exception as exc:
        logger.debug("macro: VIX fetch failed: %s", exc)

    try:
        spy_df = yf.download("SPY", period="60d", interval="1d", progress=False, auto_adjust=True)
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

    _cache["vix"] = vix_val
    _cache["spy_push"] = spy_push
    _cache["vix_push"] = vix_push
    return vix_val, vix_push + spy_push


def fetch(ticker: str, ohlcv_df=None) -> DataSourceResult:
    try:
        vix_val, total_push = _get_macro()
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
