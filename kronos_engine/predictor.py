"""
Kronos predictor module.

Kronos is NOT on PyPI. Before using this module you must install it:

    git submodule add https://github.com/shiyu-coder/Kronos kronos_src

OR clone it manually and add `sys.path.insert(0, "kronos_src")` (config.py does this).

Model zoo (NeoQuasar HuggingFace org):
  mini  — 4.1M params, 2048-token context — uses Kronos-Tokenizer-2k  — RECOMMENDED for dev/CPU
  small — 24.7M params, 512-token context — uses Kronos-Tokenizer-base
  base  —              512-token context — uses Kronos-Tokenizer-base

IMPORTANT: tokenizer pairings are NOT interchangeable.
Wrong tokenizer = silent garbage output, no error raised.
"""
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from kronos_engine.output_schema import KronosPrediction

logger = logging.getLogger(__name__)

# Enforced tokenizer pairing. Wrong pairing = silent bad output.
MODEL_ZOO: dict[str, tuple[str, str, int]] = {
    "mini":  ("NeoQuasar/Kronos-mini",  "NeoQuasar/Kronos-Tokenizer-2k",   2048),
    "small": ("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base", 512),
    "base":  ("NeoQuasar/Kronos-base",  "NeoQuasar/Kronos-Tokenizer-base", 512),
}

# Module-level cache so we don't reload the model on every call
_loaded_model = None
_loaded_tokenizer = None
_loaded_size = None


def load_kronos(model_size: str = "mini"):
    """
    Load Kronos model and tokenizer. Cached at module level after first call.

    Args:
        model_size: "mini" | "small" | "base"
                    Use "mini" for local dev — 4M params, runs on CPU.
                    Use "small" or "base" for production if accuracy matters.

    Returns:
        (predictor, tokenizer, max_context_length)

    Raises:
        KeyError: if model_size is not in MODEL_ZOO
        ImportError: if Kronos is not installed (kronos_src not in sys.path)
    """
    global _loaded_model, _loaded_tokenizer, _loaded_size

    if model_size not in MODEL_ZOO:
        raise KeyError(
            f"Unknown model_size '{model_size}'. Valid options: {list(MODEL_ZOO.keys())}"
        )

    # Return cached model if same size
    if _loaded_size == model_size and _loaded_model is not None:
        return _loaded_model, _loaded_tokenizer, MODEL_ZOO[model_size][2]

    try:
        from model import Kronos, KronosTokenizer, KronosPredictor  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Kronos is not installed. Run:\n"
            "  git submodule add https://github.com/shiyu-coder/Kronos kronos_src\n"
            "See config.py for sys.path setup."
        ) from e

    model_id, tokenizer_id, max_ctx = MODEL_ZOO[model_size]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading Kronos-%s from HuggingFace (device=%s)...", model_size, device)
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_id)
    model = Kronos.from_pretrained(model_id).to(device)
    predictor = KronosPredictor(model, tokenizer, max_context=max_ctx)

    _loaded_model = predictor
    _loaded_tokenizer = tokenizer
    _loaded_size = model_size

    logger.info("Kronos-%s loaded (max_context=%d)", model_size, max_ctx)
    return predictor, tokenizer, max_ctx


def predict(
    ticker: str,
    ohlcv_df: pd.DataFrame,
    horizon_days: int = 5,
    model_size: str = "mini",
    n_mc_samples: int = 10,
) -> KronosPrediction:
    """
    Run Kronos prediction on ohlcv_df.

    Uses n_mc_samples Monte Carlo passes to estimate prediction confidence.
    Confidence = 1 - normalized std dev of predicted_close across MC passes.
    Higher confidence = tighter agreement across MC samples.

    Args:
        ohlcv_df: DataFrame with [open, high, low, close, volume, amount].
                  Do NOT include amount_proxy_quality column.
        horizon_days: number of future days to predict.
        n_mc_samples: number of MC passes for confidence estimation.

    Returns:
        KronosPrediction dataclass.
    """
    predictor, tokenizer, max_ctx = load_kronos(model_size)

    # Truncate context to model's max window
    context = ohlcv_df.tail(max_ctx).copy()
    last_close = float(context["close"].iloc[-1])

    # Build timestamps for context (x) and forecast horizon (y).
    # ohlcv_df index is expected to be DatetimeIndex (set by fetch_ohlcv).
    # If the index is a plain RangeIndex, synthesise daily business-day timestamps.
    if isinstance(context.index, pd.DatetimeIndex):
        x_timestamp = context.index.to_series().reset_index(drop=True)
        last_date = context.index[-1]
    else:
        # Fallback: generate synthetic daily timestamps ending today
        last_date = pd.Timestamp.today().normalize()
        x_timestamp = pd.Series(
            pd.bdate_range(end=last_date, periods=len(context))
        )

    y_timestamp = pd.Series(
        pd.bdate_range(start=last_date + pd.offsets.BDay(1), periods=horizon_days)
    )

    # Kronos sample_count gives multiple stochastic draws in one call.
    # When sample_count > 1, predict() returns a single df that is the mean
    # of all samples — we run n_mc_samples as separate calls for spread estimation.
    mc_closes = []
    raw_dfs = []

    for _ in range(n_mc_samples):
        try:
            result_df = predictor.predict(
                df=context,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=horizon_days,
                T=1.0,
                top_p=0.9,
                sample_count=1,
                verbose=False,
            )
            raw_dfs.append(result_df)
            predicted_close = float(result_df["close"].iloc[-1])
            mc_closes.append(predicted_close)
        except Exception as e:
            logger.warning("%s: MC pass failed: %s", ticker, e)

    if not mc_closes:
        raise RuntimeError(f"{ticker}: all {n_mc_samples} Kronos MC passes failed")

    avg_predicted_close = float(np.mean(mc_closes))
    std_predicted_close = float(np.std(mc_closes))

    # Confidence: low std dev = high confidence
    # Normalize by last_close so it's scale-independent
    normalized_spread = std_predicted_close / max(abs(last_close), 1e-9)
    confidence = float(np.clip(1.0 - normalized_spread * 10, 0.0, 1.0))

    pct_change = (avg_predicted_close - last_close) / max(abs(last_close), 1e-9)

    if pct_change > 0.001:
        direction = "bullish"
    elif pct_change < -0.001:
        direction = "bearish"
    else:
        direction = "neutral"

    # Use the median MC pass result as the representative raw_df
    median_idx = len(raw_dfs) // 2
    best_raw_df = raw_dfs[median_idx] if raw_dfs else pd.DataFrame()

    return KronosPrediction(
        ticker=ticker,
        predicted_close=avg_predicted_close,
        predicted_high=float(best_raw_df["high"].iloc[-1]) if not best_raw_df.empty else avg_predicted_close,
        predicted_low=float(best_raw_df["low"].iloc[-1]) if not best_raw_df.empty else avg_predicted_close,
        predicted_volume=float(best_raw_df["volume"].iloc[-1]) if not best_raw_df.empty else 0.0,
        direction=direction,
        pct_change=pct_change,
        confidence=confidence,
        horizon_days=horizon_days,
        timestamp=datetime.utcnow(),
        raw_df=best_raw_df,
    )


def predict_batch(
    tickers_data: dict[str, pd.DataFrame],
    horizon_days: int = 5,
    model_size: str = "mini",
) -> list[KronosPrediction]:
    """
    Run predictions for multiple tickers. Returns list of KronosPrediction.
    Silently skips tickers that fail — logs the error and continues.
    """
    results = []
    for ticker, ohlcv_df in tickers_data.items():
        try:
            pred = predict(ticker, ohlcv_df, horizon_days, model_size)
            results.append(pred)
            logger.info(
                "%s: %s %+.2f%% (confidence=%.2f)",
                ticker, pred.direction, pred.pct_change * 100, pred.confidence
            )
        except Exception as e:
            logger.error("%s: Kronos prediction failed: %s", ticker, e)
    return results
