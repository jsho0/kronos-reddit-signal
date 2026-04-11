"""
Phase A smoke test: validates data_fetcher + technicals without loading Kronos.
Run this before attempting to load the Kronos model.

Usage:
    venv\Scripts\python.exe smoke_test_phase_a.py
"""
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("smoke_test")


def test_data_fetcher():
    logger.info("=== Testing data_fetcher ===")
    from kronos_engine.data_fetcher import fetch_ohlcv, fetch_ohlcv_for_kronos

    df = fetch_ohlcv("AAPL", lookback_days=250)
    assert len(df) >= 50, f"Expected >= 50 rows, got {len(df)}"
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "amount", "amount_proxy_quality"]

    # Verify amount is ((H+L+C)/3) * V
    expected_amount = ((df["high"] + df["low"] + df["close"]) / 3) * df["volume"]
    assert (abs(df["amount"] - expected_amount) < 1e-6).all(), "Amount proxy formula incorrect"

    # Verify amount_proxy_quality values
    assert set(df["amount_proxy_quality"].unique()).issubset({"good", "degraded"})

    # Kronos-safe version drops quality column
    kronos_df = fetch_ohlcv_for_kronos("AAPL")
    assert "amount_proxy_quality" not in kronos_df.columns
    assert list(kronos_df.columns) == ["open", "high", "low", "close", "volume", "amount"]

    degraded_pct = (df["amount_proxy_quality"] == "degraded").mean()
    logger.info(
        "AAPL: %d bars, %.1f%% degraded amount proxy, last close=%.2f",
        len(df), degraded_pct * 100, df["close"].iloc[-1]
    )
    logger.info("data_fetcher: PASS")


def test_technicals():
    logger.info("=== Testing technicals ===")
    from kronos_engine.data_fetcher import fetch_ohlcv
    from kronos_engine.technicals import compute_technicals

    df = fetch_ohlcv("AAPL", lookback_days=250)
    t = compute_technicals("AAPL", df)

    assert 0 <= t.rsi_14 <= 100, f"RSI out of range: {t.rsi_14}"
    assert t.macd_signal in ("bullish_cross", "bearish_cross", "neutral")
    assert t.bb_position in ("above_upper", "below_lower", "inside")
    assert t.atr_14 >= 0
    assert t.adx_14 >= 0
    assert t.avg_volume_ratio > 0

    logger.info(
        "AAPL technicals: RSI=%.1f MACD=%s BB=%s ADX=%.1f vol_ratio=%.2f 200MA=%.4f",
        t.rsi_14, t.macd_signal, t.bb_position, t.adx_14, t.avg_volume_ratio,
        t.price_vs_200ma if t.price_vs_200ma is not None else float("nan")
    )
    logger.info("technicals: PASS")


def test_batch():
    logger.info("=== Testing batch fetch ===")
    from kronos_engine.data_fetcher import fetch_ohlcv_batch

    results = fetch_ohlcv_batch(["AAPL", "TSLA", "NVDA"])
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    for ticker, df in results.items():
        assert len(df) >= 50, f"{ticker}: too few rows"
    logger.info("Batch fetched %d tickers: PASS", len(results))


if __name__ == "__main__":
    errors = []
    for test_fn in [test_data_fetcher, test_technicals, test_batch]:
        try:
            test_fn()
        except Exception as e:
            logger.error("FAILED: %s — %s", test_fn.__name__, e)
            errors.append((test_fn.__name__, e))

    print()
    if errors:
        print(f"FAILED {len(errors)} test(s):")
        for name, err in errors:
            print(f"  {name}: {err}")
        sys.exit(1)
    else:
        print("All Phase A tests passed. Ready to set up Kronos.")
