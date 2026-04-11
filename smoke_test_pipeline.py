"""
Pipeline smoke test.

Patches out Kronos and Reddit so the full pipeline runs end-to-end
without network calls or GPU. Verifies the signal is written to DB
with the right shape.

Usage:
    venv\Scripts\python.exe smoke_test_pipeline.py
"""
import logging
import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("smoke_test_pipeline")

# Use in-memory DB for all tests
import os
os.environ["DATABASE_URL"] = "sqlite://"


def make_fake_ohlcv(n=120):
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    df = pd.DataFrame({
        "open":   [150.0 + i * 0.05 for i in range(n)],
        "high":   [152.0 + i * 0.05 for i in range(n)],
        "low":    [149.0 + i * 0.05 for i in range(n)],
        "close":  [151.0 + i * 0.05 for i in range(n)],
        "volume": [5_000_000.0] * n,
        "amount": [755_000_000.0] * n,
        "amount_proxy_quality": ["good"] * n,
    }, index=dates)
    return df


def make_fake_kronos_pred(ticker="AAPL"):
    from kronos_engine.output_schema import KronosPrediction
    return KronosPrediction(
        ticker=ticker,
        predicted_close=157.0,
        predicted_high=158.5,
        predicted_low=156.0,
        predicted_volume=5_500_000.0,
        direction="bullish",
        pct_change=0.04,
        confidence=0.82,
        horizon_days=5,
        timestamp=datetime.utcnow(),
        raw_df=pd.DataFrame(),
    )


def test_full_pipeline_run():
    logger.info("=== Testing full pipeline run (mocked Kronos + Reddit) ===")

    fake_ohlcv = make_fake_ohlcv()
    fake_pred = make_fake_kronos_pred("AAPL")

    from reddit_scraper.scraper import ScrapeResult, RedditPost
    fake_post = RedditPost(
        post_id="t3_test001",
        ticker="AAPL",
        title="AAPL looking strong",
        body="Bullish on earnings",
        score=500,
        num_comments=80,
        subreddit="stocks",
        post_created_utc="2026-04-10T10:00:00+00:00",
        url="https://reddit.com/r/stocks/test001",
    )
    fake_scrape = ScrapeResult(ticker="AAPL", posts=[fake_post])

    from reddit_scraper.sentiment import TickerSentiment, SentimentResult
    fake_sentiment = TickerSentiment(
        ticker="AAPL",
        label="positive",
        score=0.65,
        signed_score=0.65,
        post_count=1,
        per_post=[SentimentResult(label="positive", score=0.65, signed_score=0.65)],
    )

    with patch("pipeline.runner.fetch_ohlcv", return_value=fake_ohlcv), \
         patch("pipeline.runner.kronos_predict", return_value=fake_pred), \
         patch.object(__import__("pipeline.runner", fromlist=["RedditScraper"]).RedditScraper, "fetch", return_value=fake_scrape), \
         patch("pipeline.runner.analyze_ticker", return_value=fake_sentiment):

        from pipeline.runner import PipelineRunner
        runner = PipelineRunner(ticker_timeout=30)
        health = runner.run(tickers=["AAPL"])

    assert health.tickers_attempted == 1
    assert health.tickers_succeeded == 1
    assert health.tickers_failed == 0
    assert health.kronos_errors == 0

    logger.info(
        "Run: %d/%d ok in %.1fs",
        health.tickers_succeeded, health.tickers_attempted, health.duration_seconds,
    )
    logger.info("full_pipeline_run: PASS")
    return health


def test_signal_in_db():
    logger.info("=== Verifying signal was written to DB ===")
    from storage.store import SignalStore
    from datetime import date

    store = SignalStore()
    today = date.today().isoformat()
    signal = store.get_signal("AAPL", today)

    assert signal is not None, "Signal not found in DB"
    assert signal.ticker == "AAPL"
    assert signal.classifier_label in ("STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL")
    assert signal.confluence_score is not None
    assert signal.kronos_direction == "bullish"
    assert signal.reddit_sentiment == "positive"
    assert signal.price_at_signal is not None
    assert signal.rsi_14 is not None  # technicals ran against real OHLCV

    logger.info(
        "Signal: %s | confluence=%.3f | label=%s | rsi=%.1f",
        signal.ticker, signal.confluence_score,
        signal.classifier_label, signal.rsi_14,
    )
    logger.info("signal_in_db: PASS")


def test_kronos_failure_is_graceful():
    logger.info("=== Testing graceful Kronos failure ===")

    fake_ohlcv = make_fake_ohlcv()

    from reddit_scraper.scraper import ScrapeResult
    from reddit_scraper.sentiment import TickerSentiment

    with patch("pipeline.runner.fetch_ohlcv", return_value=fake_ohlcv), \
         patch("pipeline.runner.kronos_predict", side_effect=RuntimeError("GPU OOM")), \
         patch.object(__import__("pipeline.runner", fromlist=["RedditScraper"]).RedditScraper, "fetch", return_value=ScrapeResult(ticker="TSLA", posts=[])), \
         patch("pipeline.runner.analyze_ticker", return_value=TickerSentiment("TSLA", "neutral", 0.0, 0.0, 0)):

        from pipeline.runner import PipelineRunner
        runner = PipelineRunner(ticker_timeout=30)
        health = runner.run(tickers=["TSLA"])

    # Should succeed overall — Kronos failure doesn't kill the whole ticker
    assert health.tickers_succeeded == 1, f"Expected success even with Kronos failure"
    assert health.kronos_errors == 1

    logger.info("kronos_failure_graceful: PASS (kronos_errors=%d)", health.kronos_errors)


def test_ticker_timeout():
    logger.info("=== Testing per-ticker timeout ===")
    import time

    fake_ohlcv = make_fake_ohlcv()

    def slow_kronos(*args, **kwargs):
        time.sleep(10)  # will be killed by 2s timeout

    with patch("pipeline.runner.fetch_ohlcv", return_value=fake_ohlcv), \
         patch("pipeline.runner.kronos_predict", side_effect=slow_kronos):

        from pipeline.runner import PipelineRunner
        runner = PipelineRunner(ticker_timeout=2)
        health = runner.run(tickers=["SLOW"])

    assert health.tickers_failed == 1
    assert any("Timed out" in n for n in health.notes), f"Expected timeout note, got: {health.notes}"
    logger.info("ticker_timeout: PASS")


def test_pipeline_run_recorded():
    logger.info("=== Verifying pipeline run was recorded ===")
    from storage.store import SignalStore

    store = SignalStore()
    runs = store.get_pipeline_runs(limit=10)
    assert len(runs) > 0, "No pipeline runs recorded"

    latest = runs[0]
    assert latest.tickers_attempted > 0
    assert latest.duration_seconds > 0

    logger.info(
        "Latest run: %d/%d ok in %.1fs",
        latest.tickers_succeeded, latest.tickers_attempted, latest.duration_seconds,
    )
    logger.info("pipeline_run_recorded: PASS")


if __name__ == "__main__":
    errors = []
    # Order matters: run + db check must come before other tests share the in-memory db
    for test_fn in [
        test_full_pipeline_run,
        test_signal_in_db,
        test_kronos_failure_is_graceful,
        test_ticker_timeout,
        test_pipeline_run_recorded,
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
        print("All pipeline smoke tests passed.")
