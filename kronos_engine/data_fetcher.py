import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_ohlcv(ticker: str, lookback_days: int = 200) -> pd.DataFrame:
    """
    Fetch daily OHLCV from yfinance and format for Kronos input.

    Returns DataFrame with columns: [open, high, low, close, volume, amount]
    Also includes amount_proxy_quality column for diagnostic use —
    DROP this column before passing to Kronos (see fetch_ohlcv_for_kronos()).

    NOTE on `amount`:
    Kronos was trained on Chinese A-share data where `amount` = actual transaction
    value in yuan (sum of shares*price per tick). yfinance provides no equivalent.

    We approximate as: amount = ((high + low + close) / 3) * volume
    This is the VWAP numerator — the closest proxy available from daily bars.
    It is more accurate than close*volume on volatile days but remains an
    approximation. Prediction accuracy on US equities will be lower than Kronos
    paper benchmarks (which used true amount on A-share data). This is a known
    limitation tracked via price_at_signal / price_next_day in the DB.

    For tickers with daily H-L spread > 3% of close, the amount proxy is least
    reliable — these bars are flagged as "degraded" in amount_proxy_quality.

    Raises ValueError if fewer than 50 rows returned (insufficient for indicators).
    """
    df = yf.download(ticker, period=f"{lookback_days}d", interval="1d", progress=False)

    if df.empty:
        raise ValueError(f"{ticker}: yfinance returned empty DataFrame")

    # Flatten MultiIndex columns if present (yfinance sometimes returns them)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna()

    if len(df) < 50:
        raise ValueError(
            f"{ticker}: insufficient data ({len(df)} bars, need >= 50)"
        )

    # Typical price proxy — better than close*volume on intraday-volatile tickers
    df["amount"] = ((df["high"] + df["low"] + df["close"]) / 3) * df["volume"]

    # Flag bars where H-L spread > 3% of close — amount proxy is least reliable here
    wide_bars = (df["high"] - df["low"]) / df["close"] > 0.03
    df["amount_proxy_quality"] = "good"
    df.loc[wide_bars, "amount_proxy_quality"] = "degraded"

    degraded_pct = wide_bars.mean()
    if degraded_pct > 0.15:
        logger.warning(
            "%s: %.0f%% of bars have H-L spread > 3%% — amount proxy reliability reduced",
            ticker, degraded_pct * 100
        )

    return df


def fetch_ohlcv_for_kronos(ticker: str, lookback_days: int = 200) -> pd.DataFrame:
    """
    Convenience wrapper: returns OHLCV with amount column, without the
    amount_proxy_quality diagnostic column. Safe to pass directly to Kronos.
    """
    df = fetch_ohlcv(ticker, lookback_days)
    return df[["open", "high", "low", "close", "volume", "amount"]]


def fetch_ohlcv_batch(
    tickers: list[str], lookback_days: int = 200
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV for multiple tickers concurrently using ThreadPoolExecutor.
    Returns dict: {ticker: DataFrame}. Silently skips tickers that fail.
    """
    results: dict[str, pd.DataFrame] = {}

    with ThreadPoolExecutor(max_workers=min(len(tickers), 8)) as executor:
        futures = {
            executor.submit(fetch_ohlcv, ticker, lookback_days): ticker
            for ticker in tickers
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                results[ticker] = future.result()
            except Exception as e:
                logger.warning("%s: OHLCV fetch failed: %s", ticker, e)

    return results
