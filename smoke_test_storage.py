"""
Storage smoke test: validates models, WAL mode, upsert, cache, pipeline run recording.
Uses an in-memory SQLite database so nothing is written to disk.

Usage:
    venv\Scripts\python.exe smoke_test_storage.py
"""
import logging
import sys
from datetime import date

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("smoke_test_storage")

IN_MEMORY_URL = "sqlite://"   # in-memory SQLite, gone after process exits


def test_init():
    logger.info("=== Testing db init + WAL mode ===")
    import tempfile, os
    from storage.db import init_db
    import sqlalchemy

    # WAL mode requires a real file — in-memory SQLite always uses "memory" journal mode
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        tmp_path = f.name
    try:
        file_url = f"sqlite:///{tmp_path}"
        # Use a fresh engine (not the cached module-level one) for this check
        engine = sqlalchemy.create_engine(file_url, connect_args={"check_same_thread": False})

        @sqlalchemy.event.listens_for(engine, "connect")
        def _set_pragmas(dbapi_conn, _):
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA busy_timeout=10000")
            cur.close()

        from storage.models import Base
        Base.metadata.create_all(engine)

        with engine.connect() as conn:
            mode = conn.execute(sqlalchemy.text("PRAGMA journal_mode")).scalar()
        assert mode == "wal", f"Expected WAL mode, got: {mode}"
        logger.info("WAL mode confirmed: %s", mode)
        engine.dispose()
    finally:
        os.unlink(tmp_path)

    # Verify in-memory init works too (used by all other tests)
    init_db(IN_MEMORY_URL)
    logger.info("db init: PASS")


def test_upsert_signal():
    logger.info("=== Testing upsert_signal ===")
    from storage.db import get_session, init_db
    from storage.models import Signal
    from storage.store import SignalStore
    from sqlalchemy import select

    init_db(IN_MEMORY_URL)
    store = SignalStore()
    today = date.today().isoformat()

    signal_data = {
        "ticker": "AAPL",
        "signal_date": today,
        "kronos_direction": "bullish",
        "kronos_pct_change": 0.021,
        "kronos_confidence": 0.87,
        "kronos_horizon_days": 5,
        "kronos_predicted_close": 265.87,
        "reddit_sentiment": "positive",
        "reddit_score": 0.65,
        "reddit_post_count": 12,
        "reddit_catalyst_status": "SUCCESS",
        "rsi_14": 58.3,
        "macd_signal": "neutral",
        "bb_position": "inside",
        "atr_14": 4.2,
        "adx_14": 22.1,
        "avg_volume_ratio": 1.1,
        "confluence_score": 0.74,
        "classifier_label": "BUY",
        "price_at_signal": 260.48,
    }

    row_id = store.upsert_signal(signal_data)
    assert row_id > 0, "Expected positive row id"

    # Upsert same ticker/date with updated confidence — should update not insert
    updated_data = {**signal_data, "kronos_confidence": 0.95, "classifier_label": "STRONG_BUY"}
    row_id2 = store.upsert_signal(updated_data)

    with get_session(IN_MEMORY_URL) as session:
        rows = session.execute(select(Signal)).scalars().all()
        assert len(rows) == 1, f"Expected 1 row after upsert, got {len(rows)}"
        assert rows[0].kronos_confidence == 0.95, "Upsert should update existing row"
        assert rows[0].classifier_label == "STRONG_BUY"

    logger.info("upsert_signal: PASS (row_id=%d)", row_id)


def test_reddit_posts():
    logger.info("=== Testing Reddit post insert + dedup ===")
    from storage.db import get_session, init_db
    from storage.models import RedditPost
    from storage.store import SignalStore
    from sqlalchemy import select

    init_db(IN_MEMORY_URL)
    store = SignalStore()

    posts = [
        {
            "ticker": "AAPL",
            "post_id": "t3_abc001",
            "title": "AAPL to the moon",
            "body": "Bought calls at open",
            "score": 420,
            "num_comments": 88,
            "subreddit": "wallstreetbets",
            "sentiment_label": "positive",
            "sentiment_score": 0.92,
        },
        {
            "ticker": "AAPL",
            "post_id": "t3_abc002",
            "title": "AAPL earnings miss incoming",
            "body": "Supply chain issues",
            "score": 55,
            "num_comments": 14,
            "subreddit": "stocks",
            "sentiment_label": "negative",
            "sentiment_score": 0.71,
        },
    ]

    inserted = store.insert_reddit_posts(posts)
    assert inserted == 2, f"Expected 2 inserted, got {inserted}"

    # Re-insert same posts — should be skipped
    inserted_again = store.insert_reddit_posts(posts)
    assert inserted_again == 0, f"Expected 0 on re-insert, got {inserted_again}"

    with get_session(IN_MEMORY_URL) as session:
        count = len(session.execute(select(RedditPost)).scalars().all())
    assert count == 2, f"Expected 2 rows, got {count}"

    logger.info("reddit_posts dedup: PASS")


def test_pipeline_run():
    logger.info("=== Testing pipeline run recording ===")
    from storage.db import init_db
    from storage.store import SignalStore

    init_db(IN_MEMORY_URL)
    store = SignalStore()

    run_id = store.record_pipeline_run({
        "tickers_attempted": 6,
        "tickers_succeeded": 5,
        "tickers_failed": 1,
        "kronos_errors": 1,
        "reddit_errors": 0,
        "catalyst_api_dead": 0,
        "catalyst_api_degraded": 0,
        "duration_seconds": 42.3,
        "notes": "TSLA Kronos timeout",
    })
    assert run_id > 0

    runs = store.get_pipeline_runs(limit=5)
    assert len(runs) == 1
    assert runs[0].tickers_attempted == 6
    assert runs[0].duration_seconds == 42.3

    logger.info("pipeline_run: PASS (run_id=%d)", run_id)


def test_ohlcv_cache():
    logger.info("=== Testing OHLCV cache write/read ===")
    from storage.db import init_db
    from storage.store import SignalStore

    init_db(IN_MEMORY_URL)
    store = SignalStore()

    # Build a fake 60-bar OHLCV DataFrame with DatetimeIndex
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=60)
    df = pd.DataFrame({
        "open":  [150.0 + i * 0.1 for i in range(60)],
        "high":  [152.0 + i * 0.1 for i in range(60)],
        "low":   [149.0 + i * 0.1 for i in range(60)],
        "close": [151.0 + i * 0.1 for i in range(60)],
        "volume": [1_000_000.0] * 60,
        "amount": [151_500_000.0] * 60,
        "amount_proxy_quality": ["good"] * 60,
    }, index=dates)

    store.write_ohlcv_cache("FAKE", df)

    # Re-write same data — should not raise or duplicate
    store.write_ohlcv_cache("FAKE", df)

    cached = store.read_ohlcv_cache("FAKE", min_bars=50)
    assert cached is not None, "Expected cached DataFrame"
    assert len(cached) == 60, f"Expected 60 bars, got {len(cached)}"
    assert isinstance(cached.index, pd.DatetimeIndex), "Expected DatetimeIndex"
    assert "close" in cached.columns
    assert "amount_proxy_quality" in cached.columns

    # min_bars check: should return None for a ticker with few bars
    sparse_dates = pd.bdate_range(end=pd.Timestamp.today(), periods=10)
    sparse_df = df.iloc[:10].copy()
    sparse_df.index = sparse_dates
    store.write_ohlcv_cache("SPARSE", sparse_df)
    result = store.read_ohlcv_cache("SPARSE", min_bars=50)
    assert result is None, "Expected None for sparse cache"

    logger.info("ohlcv_cache: PASS (60 bars written and read back)")


def test_accuracy_stats():
    logger.info("=== Testing accuracy stats ===")
    from storage.db import init_db
    from storage.store import SignalStore

    init_db(IN_MEMORY_URL)
    store = SignalStore()

    # Insert 4 signals with known outcomes
    cases = [
        ("AAPL", "2026-04-01", "bullish", 260.0, 265.0),   # correct
        ("TSLA", "2026-04-01", "bearish", 200.0, 195.0),   # correct
        ("NVDA", "2026-04-01", "bullish", 900.0, 880.0),   # wrong
        ("MSFT", "2026-04-01", "bearish", 400.0, 410.0),   # wrong
    ]
    for ticker, sig_date, direction, price_at, price_next in cases:
        store.upsert_signal({
            "ticker": ticker,
            "signal_date": sig_date,
            "kronos_direction": direction,
            "price_at_signal": price_at,
            "price_next_day": price_next,
        })

    stats = store.get_accuracy_stats()
    assert stats["total"] == 4
    assert stats["correct"] == 2
    assert abs(stats["accuracy"] - 0.5) < 1e-9

    logger.info("accuracy_stats: PASS (2/4 = 50%%)")


if __name__ == "__main__":
    errors = []
    for test_fn in [
        test_init,
        test_upsert_signal,
        test_reddit_posts,
        test_pipeline_run,
        test_ohlcv_cache,
        test_accuracy_stats,
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
        print("All storage smoke tests passed.")
