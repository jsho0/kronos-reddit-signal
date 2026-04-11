import logging
from dataclasses import dataclass

import pandas as pd
import ta

logger = logging.getLogger(__name__)


@dataclass
class TechnicalIndicators:
    ticker: str
    rsi_14: float
    macd_signal: str        # "bullish_cross" | "bearish_cross" | "neutral"
    bb_position: str        # "above_upper" | "below_lower" | "inside"
    atr_14: float
    adx_14: float
    avg_volume_ratio: float
    price_vs_52w_high: float
    price_vs_200ma: float | None  # None if fewer than 200 bars


def compute_technicals(ticker: str, ohlcv_df: pd.DataFrame) -> TechnicalIndicators:
    """
    Compute technical indicators from OHLCV DataFrame.
    Requires at least 50 rows. Handles shorter history gracefully.
    """
    df = ohlcv_df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # RSI 14
    rsi_series = ta.momentum.RSIIndicator(close, window=14).rsi()
    rsi_14 = float(rsi_series.dropna().iloc[-1]) if not rsi_series.dropna().empty else 50.0

    # MACD
    macd_ind = ta.trend.MACD(close)
    macd_line = macd_ind.macd()
    macd_sig = macd_ind.macd_signal()
    if not macd_line.dropna().empty and not macd_sig.dropna().empty:
        curr_diff = float(macd_line.iloc[-1]) - float(macd_sig.iloc[-1])
        prev_diff = float(macd_line.iloc[-2]) - float(macd_sig.iloc[-2])
        if prev_diff < 0 and curr_diff > 0:
            macd_signal = "bullish_cross"
        elif prev_diff > 0 and curr_diff < 0:
            macd_signal = "bearish_cross"
        else:
            macd_signal = "neutral"
    else:
        macd_signal = "neutral"

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close)
    bb_upper = bb.bollinger_hband()
    bb_lower = bb.bollinger_lband()
    last_close = float(close.iloc[-1])
    if not bb_upper.dropna().empty:
        if last_close > float(bb_upper.iloc[-1]):
            bb_position = "above_upper"
        elif last_close < float(bb_lower.iloc[-1]):
            bb_position = "below_lower"
        else:
            bb_position = "inside"
    else:
        bb_position = "inside"

    # ATR 14
    atr_series = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    atr_14 = float(atr_series.dropna().iloc[-1]) if not atr_series.dropna().empty else 0.0

    # ADX 14
    adx_series = ta.trend.ADXIndicator(high, low, close, window=14).adx()
    adx_14 = float(adx_series.dropna().iloc[-1]) if not adx_series.dropna().empty else 0.0

    # Volume ratio: current vs 20-day avg
    vol_ma20 = volume.rolling(20).mean()
    avg_volume_ratio = (
        float(volume.iloc[-1]) / float(vol_ma20.iloc[-1])
        if not vol_ma20.dropna().empty and float(vol_ma20.iloc[-1]) > 0
        else 1.0
    )

    # Price vs 52-week high
    lookback_52w = min(252, len(close))
    high_52w = float(close.iloc[-lookback_52w:].max())
    price_vs_52w_high = (last_close - high_52w) / high_52w if high_52w > 0 else 0.0

    # Price vs 200-day MA (None if not enough data)
    if len(close) >= 200:
        ma200 = float(close.rolling(200).mean().iloc[-1])
        price_vs_200ma = (last_close - ma200) / ma200 if ma200 > 0 else None
    else:
        logger.debug("%s: fewer than 200 bars, skipping 200MA", ticker)
        price_vs_200ma = None

    return TechnicalIndicators(
        ticker=ticker,
        rsi_14=rsi_14,
        macd_signal=macd_signal,
        bb_position=bb_position,
        atr_14=atr_14,
        adx_14=adx_14,
        avg_volume_ratio=avg_volume_ratio,
        price_vs_52w_high=price_vs_52w_high,
        price_vs_200ma=price_vs_200ma,
    )
