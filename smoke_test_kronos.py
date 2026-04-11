"""
Kronos smoke test: validates that Kronos loads, runs inference, and produces
non-trivial output on real AAPL OHLCV data.

This test intentionally uses the smallest model (mini) so it can run on CPU
without a GPU. Expect ~60-120 seconds on first run (HuggingFace download).
Subsequent runs use the local HuggingFace cache and are much faster.

Usage:
    venv\Scripts\python.exe smoke_test_kronos.py
"""
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("smoke_test_kronos")


def test_kronos_import():
    logger.info("=== Testing Kronos import ===")
    # config.py inserts kronos_src/ into sys.path
    import config  # noqa: F401
    from model import Kronos, KronosTokenizer, KronosPredictor  # type: ignore
    logger.info("Kronos import: PASS (Kronos=%s, KronosTokenizer=%s)", Kronos, KronosTokenizer)
    return Kronos, KronosTokenizer, KronosPredictor


def test_kronos_load(Kronos, KronosTokenizer, KronosPredictor):
    logger.info("=== Loading Kronos-mini (CPU) ===")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-2k")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
    predictor = KronosPredictor(model, tokenizer, max_context=2048)
    logger.info("Kronos-mini loaded: PASS")
    return predictor


def test_kronos_predict(predictor):
    logger.info("=== Testing Kronos prediction on AAPL ===")
    import pandas as pd
    from kronos_engine.data_fetcher import fetch_ohlcv_for_kronos

    df = fetch_ohlcv_for_kronos("AAPL", lookback_days=120)
    assert len(df) >= 50, f"Too few bars: {len(df)}"

    context = df.tail(100).copy()
    last_close = float(context["close"].iloc[-1])
    horizon = 5

    # Build timestamps
    if isinstance(context.index, pd.DatetimeIndex):
        x_timestamp = context.index.to_series().reset_index(drop=True)
        last_date = context.index[-1]
    else:
        last_date = pd.Timestamp.today().normalize()
        x_timestamp = pd.Series(pd.bdate_range(end=last_date, periods=len(context)))

    y_timestamp = pd.Series(
        pd.bdate_range(start=last_date + pd.offsets.BDay(1), periods=horizon)
    )

    logger.info("Running predict() with context=%d bars, horizon=%d days...", len(context), horizon)
    pred_df = predictor.predict(
        df=context,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=horizon,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=True,
    )

    assert pred_df is not None, "predict() returned None"
    assert len(pred_df) == horizon, f"Expected {horizon} rows, got {len(pred_df)}"
    assert "close" in pred_df.columns, "Missing 'close' column in prediction"
    assert "volume" in pred_df.columns, "Missing 'volume' column in prediction"

    predicted_close = float(pred_df["close"].iloc[-1])
    # Non-trivial: prediction should be within 50% of last close
    # (a nonsense model would produce 0 or wildly different values)
    assert 0.5 * last_close < predicted_close < 2.0 * last_close, (
        f"Predicted close {predicted_close:.2f} looks wrong vs last close {last_close:.2f}"
    )

    pct_change = (predicted_close - last_close) / last_close * 100
    logger.info(
        "AAPL: last_close=%.2f predicted_close=%.2f (%+.2f%% over %d days)",
        last_close, predicted_close, pct_change, horizon
    )
    logger.info("Kronos predict: PASS")
    return pred_df


def test_predictor_module():
    logger.info("=== Testing kronos_engine.predictor module ===")
    from kronos_engine.predictor import predict
    from kronos_engine.data_fetcher import fetch_ohlcv_for_kronos

    df = fetch_ohlcv_for_kronos("AAPL", lookback_days=120)
    result = predict("AAPL", df, horizon_days=5, model_size="mini", n_mc_samples=3)

    assert result.ticker == "AAPL"
    assert result.direction in ("bullish", "bearish", "neutral")
    assert 0.0 <= result.confidence <= 1.0
    assert result.horizon_days == 5
    assert result.predicted_close > 0

    logger.info(
        "predictor.predict: direction=%s pct_change=%+.2f%% confidence=%.2f predicted_close=%.2f",
        result.direction, result.pct_change * 100, result.confidence, result.predicted_close
    )
    logger.info("predictor module: PASS")


if __name__ == "__main__":
    errors = []

    try:
        Kronos, KronosTokenizer, KronosPredictor = test_kronos_import()
    except Exception as e:
        logger.error("FAILED: test_kronos_import — %s", e)
        errors.append(("test_kronos_import", e))
        print(f"\nFAILED 1 test(s). Aborting — fix import before continuing.")
        sys.exit(1)

    try:
        predictor = test_kronos_load(Kronos, KronosTokenizer, KronosPredictor)
    except Exception as e:
        logger.error("FAILED: test_kronos_load — %s", e)
        errors.append(("test_kronos_load", e))
        print(f"\nFAILED 1 test(s). Aborting — fix model load before continuing.")
        sys.exit(1)

    for test_fn, args in [
        (test_kronos_predict, (predictor,)),
        (test_predictor_module, ()),
    ]:
        try:
            test_fn(*args)
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
        print("All Kronos smoke tests passed. Ready for Phase B.")
