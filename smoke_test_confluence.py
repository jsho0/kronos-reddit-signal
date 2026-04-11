"""
Confluence engine smoke test.

Tests all sub-scorers and the full pipeline with known inputs.
No network calls needed — all inputs are constructed in-memory.

Usage:
    venv\Scripts\python.exe smoke_test_confluence.py
"""
import logging
import sys
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("smoke_test_confluence")


def make_kronos(direction="bullish", pct_change=0.05, confidence=0.85, horizon=5):
    from kronos_engine.output_schema import KronosPrediction
    import pandas as pd
    return KronosPrediction(
        ticker="TEST",
        predicted_close=105.0,
        predicted_high=106.0,
        predicted_low=104.0,
        predicted_volume=1_000_000.0,
        direction=direction,
        pct_change=pct_change,
        confidence=confidence,
        horizon_days=horizon,
        timestamp=datetime.utcnow(),
        raw_df=pd.DataFrame(),
    )


def make_sentiment(label="positive", signed_score=0.6, post_count=10):
    from reddit_scraper.sentiment import TickerSentiment
    return TickerSentiment(
        ticker="TEST",
        label=label,
        score=abs(signed_score),
        signed_score=signed_score,
        post_count=post_count,
    )


def make_technicals(rsi=55.0, macd="neutral", bb="inside", adx=20.0, vol_ratio=1.0):
    from kronos_engine.technicals import TechnicalIndicators
    return TechnicalIndicators(
        ticker="TEST",
        rsi_14=rsi,
        macd_signal=macd,
        bb_position=bb,
        atr_14=3.0,
        adx_14=adx,
        avg_volume_ratio=vol_ratio,
        price_vs_52w_high=-0.05,
        price_vs_200ma=0.02,
    )


def test_strong_buy():
    logger.info("=== Testing STRONG_BUY scenario ===")
    from confluence.engine import ConfluenceEngine

    engine = ConfluenceEngine()
    result = engine.score(
        ticker="AAPL",
        kronos=make_kronos(direction="bullish", pct_change=0.08, confidence=0.92),
        sentiment=make_sentiment(label="positive", signed_score=0.75, post_count=20),
        technicals=make_technicals(rsi=28.0, macd="bullish_cross", bb="inside", adx=30.0, vol_ratio=2.0),
    )

    logger.info("  confluence=%.3f label=%s", result.confluence_score, result.label)
    logger.info("  kronos=%.3f reddit=%.3f tech=%.3f", result.kronos_score, result.reddit_score, result.technicals_score)
    logger.info("  reasoning: %s", result.reasoning)

    assert result.label in ("STRONG_BUY", "BUY"), f"Expected bullish label, got {result.label}"
    assert result.confluence_score > 0.6, f"Expected high score, got {result.confluence_score:.3f}"
    assert any("Kronos" in r for r in result.reasoning), "Missing Kronos reasoning"
    assert any("Reddit" in r or "reddit" in r.lower() for r in result.reasoning), "Missing Reddit reasoning"
    assert any("RSI" in r for r in result.reasoning), "Missing RSI reasoning"
    logger.info("strong_buy: PASS")


def test_strong_sell():
    logger.info("=== Testing STRONG_SELL scenario ===")
    from confluence.engine import ConfluenceEngine

    engine = ConfluenceEngine()
    result = engine.score(
        ticker="TSLA",
        kronos=make_kronos(direction="bearish", pct_change=-0.08, confidence=0.90),
        sentiment=make_sentiment(label="negative", signed_score=-0.70, post_count=15),
        technicals=make_technicals(rsi=75.0, macd="bearish_cross", bb="above_upper", adx=28.0),
    )

    logger.info("  confluence=%.3f label=%s", result.confluence_score, result.label)
    assert result.label in ("STRONG_SELL", "SELL"), f"Expected bearish label, got {result.label}"
    assert result.confluence_score < 0.4, f"Expected low score, got {result.confluence_score:.3f}"
    logger.info("strong_sell: PASS")


def test_hold_neutral():
    logger.info("=== Testing HOLD scenario (mixed signals) ===")
    from confluence.engine import ConfluenceEngine

    engine = ConfluenceEngine()
    result = engine.score(
        ticker="MSFT",
        kronos=make_kronos(direction="neutral", pct_change=0.001, confidence=0.5),
        sentiment=make_sentiment(label="neutral", signed_score=0.02, post_count=5),
        technicals=make_technicals(rsi=50.0, macd="neutral", bb="inside", adx=15.0),
    )

    logger.info("  confluence=%.3f label=%s", result.confluence_score, result.label)
    assert result.label == "HOLD", f"Expected HOLD, got {result.label}"
    assert 0.42 <= result.confluence_score <= 0.58, (
        f"Expected near-neutral score, got {result.confluence_score:.3f}"
    )
    logger.info("hold_neutral: PASS")


def test_missing_inputs():
    logger.info("=== Testing graceful degradation (missing inputs) ===")
    from confluence.engine import ConfluenceEngine

    engine = ConfluenceEngine()

    # No Reddit data
    result = engine.score(
        ticker="NVDA",
        kronos=make_kronos(direction="bullish", pct_change=0.05, confidence=0.80),
        sentiment=None,
        technicals=make_technicals(rsi=45.0),
    )
    assert result.confluence_score != 0.0, "Should not be zero without Reddit"
    logger.info("  no reddit: confluence=%.3f label=%s", result.confluence_score, result.label)

    # No Kronos
    result2 = engine.score(
        ticker="AMD",
        kronos=None,
        sentiment=make_sentiment(label="positive", signed_score=0.5),
        technicals=make_technicals(rsi=40.0),
    )
    assert any("unavailable" in r for r in result2.reasoning), "Should note Kronos unavailable"
    logger.info("  no kronos: confluence=%.3f label=%s", result2.confluence_score, result2.label)

    # Nothing at all
    result3 = engine.score(ticker="XX", kronos=None, sentiment=None, technicals=None)
    assert result3.label == "HOLD", f"All-neutral should be HOLD, got {result3.label}"
    assert abs(result3.confluence_score - 0.5) < 0.01
    logger.info("  all missing: confluence=%.3f label=%s", result3.confluence_score, result3.label)

    logger.info("missing_inputs: PASS")


def test_reasoning_dedup():
    logger.info("=== Testing reasoning deduplication ===")
    from confluence.engine import ConfluenceEngine

    engine = ConfluenceEngine()
    result = engine.score(
        ticker="META",
        kronos=make_kronos(direction="bullish", pct_change=0.04, confidence=0.75),
        sentiment=make_sentiment(label="positive", signed_score=0.4, post_count=8),
        technicals=make_technicals(rsi=32.0, macd="bullish_cross"),
    )

    # Reasoning should have no duplicates
    assert len(result.reasoning) == len(set(result.reasoning)), "Duplicate reasoning bullets found"
    logger.info("  reasoning bullets (%d): %s", len(result.reasoning), result.reasoning)
    logger.info("reasoning_dedup: PASS")


def test_score_boundaries():
    logger.info("=== Testing score boundary clamping ===")
    from confluence.engine import ConfluenceEngine

    engine = ConfluenceEngine()

    # Extreme bullish — should not exceed 1.0
    result = engine.score(
        ticker="TEST",
        kronos=make_kronos(pct_change=0.50, confidence=1.0),   # way beyond 10% cap
        sentiment=make_sentiment(signed_score=1.0),
        technicals=make_technicals(rsi=20.0, macd="bullish_cross", bb="below_lower", adx=40.0, vol_ratio=3.0),
    )
    assert 0.0 <= result.confluence_score <= 1.0, f"Score out of range: {result.confluence_score}"
    assert 0.0 <= result.kronos_score <= 1.0
    assert 0.0 <= result.reddit_score <= 1.0
    assert 0.0 <= result.technicals_score <= 1.0

    logger.info("  extreme bullish: confluence=%.3f", result.confluence_score)
    logger.info("score_boundaries: PASS")


if __name__ == "__main__":
    errors = []
    for test_fn in [
        test_strong_buy,
        test_strong_sell,
        test_hold_neutral,
        test_missing_inputs,
        test_reasoning_dedup,
        test_score_boundaries,
    ]:
        try:
            test_fn()
        except Exception as e:
            logger.error("FAILED: %s — %s", test_fn.__name__, e)
            import traceback; traceback.print_exc()
            errors.append((test_fn.__name__, e))

    print()
    if errors:
        print(f"FAILED {len(errors)} test(s):")
        for name, err in errors:
            print(f"  {name}: {err}")
        sys.exit(1)
    else:
        print("All confluence smoke tests passed.")
