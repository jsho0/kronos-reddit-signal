"""
SQLAlchemy ORM models.

Tables:
  signals        - one row per ticker per pipeline run (the main output)
  reddit_posts   - raw Reddit posts, deduplicated by post_id
  pipeline_runs  - health/metrics for each scheduler run
  ohlcv_cache    - OHLCV bars cached to avoid redundant yfinance fetches

All timestamps stored as UTC ISO strings (TEXT) for portability.
SQLite doesn't have a native datetime type; TEXT is the recommended approach.
"""
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class Base(DeclarativeBase):
    pass


class Signal(Base):
    """
    One row per (ticker, signal_date).
    signal_date is the date the pipeline ran (YYYY-MM-DD string).

    Kronos + Reddit + technicals + classifier all collapsed into one row
    so the dashboard can do a single SELECT to render everything.
    """
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(16), nullable=False)
    signal_date = Column(String(10), nullable=False)  # YYYY-MM-DD

    # Kronos output
    kronos_direction = Column(String(16))          # bullish | bearish | neutral
    kronos_pct_change = Column(Float)              # predicted % change
    kronos_confidence = Column(Float)              # 0.0 - 1.0
    kronos_horizon_days = Column(Integer)
    kronos_predicted_close = Column(Float)

    # Reddit / sentiment
    reddit_sentiment = Column(String(16))          # positive | negative | neutral
    reddit_score = Column(Float)                   # -1.0 to 1.0
    reddit_post_count = Column(Integer)
    reddit_catalyst_status = Column(String(32))    # CatalystExtractionStatus value

    # Technicals (snapshot at signal time)
    rsi_14 = Column(Float)
    macd_signal = Column(String(32))               # bullish_cross | bearish_cross | neutral
    bb_position = Column(String(32))               # above_upper | below_lower | inside
    atr_14 = Column(Float)
    adx_14 = Column(Float)
    avg_volume_ratio = Column(Float)
    price_vs_200ma = Column(Float)                 # None/NULL if < 200 bars
    price_vs_52w_high = Column(Float)

    # Confluence + classifier
    confluence_score = Column(Float)               # 0.0 - 1.0 combined signal strength
    classifier_label = Column(String(32))          # STRONG_BUY | BUY | HOLD | SELL | STRONG_SELL
    classifier_reasoning = Column(Text)            # deduplicated reasoning bullets

    # Accuracy tracking (filled in by next-day price updater)
    price_at_signal = Column(Float)                # close price when signal was generated
    price_next_day = Column(Float)                 # actual next-day close (backfilled)

    created_at = Column(String(32), default=_utcnow)
    updated_at = Column(String(32), default=_utcnow, onupdate=_utcnow)

    __table_args__ = (
        UniqueConstraint("ticker", "signal_date", name="uq_signal_ticker_date"),
        Index("ix_signals_ticker", "ticker"),
        Index("ix_signals_signal_date", "signal_date"),
    )

    def __repr__(self):
        return (
            f"<Signal {self.ticker} {self.signal_date} "
            f"{self.classifier_label} conf={self.kronos_confidence:.2f}>"
        )


class RedditPost(Base):
    """
    Raw Reddit posts. post_id is the Reddit fullname (e.g. 't3_abc123').
    Deduplicated on insert — if we've seen it before, skip.
    Keeps the audit trail for why a sentiment score was assigned.
    """
    __tablename__ = "reddit_posts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(16), nullable=False)
    post_id = Column(String(32), nullable=False, unique=True)  # Reddit fullname
    title = Column(Text)
    body = Column(Text)
    score = Column(Integer)                        # Reddit upvotes
    num_comments = Column(Integer)
    subreddit = Column(String(64))
    post_created_utc = Column(String(32))          # ISO UTC when post was made
    fetched_at = Column(String(32), default=_utcnow)

    # FinBERT output
    sentiment_label = Column(String(16))           # positive | negative | neutral
    sentiment_score = Column(Float)                # raw FinBERT score 0.0 - 1.0

    __table_args__ = (
        Index("ix_reddit_posts_ticker", "ticker"),
        Index("ix_reddit_posts_fetched_at", "fetched_at"),
    )

    def __repr__(self):
        return f"<RedditPost {self.post_id} {self.ticker} {self.sentiment_label}>"


class PipelineRun(Base):
    """
    One row per scheduler run. Records health metrics so we can detect
    degradation over time (e.g. catalyst API going dark, Reddit rate limits).
    """
    __tablename__ = "pipeline_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_at = Column(String(32), nullable=False)    # ISO UTC

    tickers_attempted = Column(Integer, default=0)
    tickers_succeeded = Column(Integer, default=0)
    tickers_failed = Column(Integer, default=0)

    kronos_errors = Column(Integer, default=0)
    reddit_errors = Column(Integer, default=0)
    sentiment_errors = Column(Integer, default=0)

    # True if Claude/catalyst API returned auth error (fail-fast triggered)
    catalyst_api_dead = Column(Integer, default=0)     # 0/1 bool
    # True if >50% of catalyst calls failed (degraded but not dead)
    catalyst_api_degraded = Column(Integer, default=0) # 0/1 bool

    duration_seconds = Column(Float)
    notes = Column(Text)                           # free-text, error summary etc.

    def __repr__(self):
        return (
            f"<PipelineRun {self.run_at} "
            f"{self.tickers_succeeded}/{self.tickers_attempted} ok>"
        )


class OHLCVCache(Base):
    """
    Cached OHLCV bars. One row per (ticker, bar_date).
    Avoids hitting yfinance on every pipeline run for data we already have.

    amount_proxy_quality: "good" | "degraded" (H-L spread > 3%)
    Kronos predictor drops the quality column before inference.
    """
    __tablename__ = "ohlcv_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(16), nullable=False)
    bar_date = Column(String(10), nullable=False)  # YYYY-MM-DD

    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    amount = Column(Float)                         # VWAP numerator proxy
    amount_proxy_quality = Column(String(16))      # good | degraded

    fetched_at = Column(String(32), default=_utcnow)

    __table_args__ = (
        UniqueConstraint("ticker", "bar_date", name="uq_ohlcv_ticker_date"),
        Index("ix_ohlcv_ticker", "ticker"),
    )

    def __repr__(self):
        return f"<OHLCVCache {self.ticker} {self.bar_date} close={self.close}>"
