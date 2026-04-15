"""
Pipeline runner: wires all modules together for one full run.

Per-ticker flow:
  1. Fetch OHLCV (cache-first, falls back to yfinance)
  2. Run Kronos prediction
  3. Fetch Reddit posts
  4. Run FinBERT sentiment
  5. Compute technicals
  6. Run confluence engine
  7. Upsert signal to DB + cache Reddit posts
  8. Record pipeline run metrics

Cross-platform timeout: each ticker runs in a ThreadPoolExecutor future
with a hard timeout. No SIGALRM (Windows-incompatible).

Per-signal commits: each upsert_signal() call is its own transaction,
so a crash on ticker N doesn't roll back tickers 1..N-1.
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

import config
from confluence.engine import ConfluenceEngine, ConfluenceResult
from kronos_engine.data_fetcher import fetch_ohlcv, fetch_ohlcv_for_kronos
from kronos_engine.predictor import predict as kronos_predict
from kronos_engine.technicals import compute_technicals
from reddit_scraper.scraper import RedditScraper
from reddit_scraper.sentiment import analyze_ticker
from storage.db import init_db
from storage.store import SignalStore
from trading.alpaca_trader import AlpacaTrader

logger = logging.getLogger(__name__)

# Hard timeout per ticker (seconds). Kronos on CPU can be slow.
TICKER_TIMEOUT_SECONDS = 120


@dataclass
class PipelineHealth:
    """Metrics collected during a single pipeline run."""
    run_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tickers_attempted: int = 0
    tickers_succeeded: int = 0
    tickers_failed: int = 0
    kronos_errors: int = 0
    reddit_errors: int = 0
    sentiment_errors: int = 0
    catalyst_api_dead: bool = False
    catalyst_api_degraded: bool = False
    duration_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "run_at": self.run_at,
            "tickers_attempted": self.tickers_attempted,
            "tickers_succeeded": self.tickers_succeeded,
            "tickers_failed": self.tickers_failed,
            "kronos_errors": self.kronos_errors,
            "reddit_errors": self.reddit_errors,
            "sentiment_errors": self.sentiment_errors,
            "catalyst_api_dead": int(self.catalyst_api_dead),
            "catalyst_api_degraded": int(self.catalyst_api_degraded),
            "duration_seconds": self.duration_seconds,
            "notes": "; ".join(self.notes) if self.notes else None,
        }


class PipelineRunner:
    """
    Runs the full signal pipeline for a list of tickers.

    Usage:
        runner = PipelineRunner()
        health = runner.run(["AAPL", "TSLA", "NVDA"])
    """

    def __init__(
        self,
        model_size: str = None,
        ticker_timeout: int = TICKER_TIMEOUT_SECONDS,
    ):
        self.model_size = model_size or config.KRONOS_MODEL_SIZE
        self.ticker_timeout = ticker_timeout
        self.store = SignalStore()
        self.scraper = RedditScraper()
        self.confluence = ConfluenceEngine()
        self.trader = AlpacaTrader()
        init_db()

    def run(self, tickers: list[str] = None) -> PipelineHealth:
        """
        Run the full pipeline for all tickers.
        Returns a PipelineHealth with run metrics.
        """
        tickers = tickers or config.WATCHLIST
        health = PipelineHealth()
        t_start = time.monotonic()

        logger.info("Pipeline starting: %d tickers, model=%s", len(tickers), self.model_size)

        for ticker in tickers:
            health.tickers_attempted += 1
            try:
                self._run_ticker(ticker, health)
                health.tickers_succeeded += 1
            except Exception as e:
                health.tickers_failed += 1
                health.notes.append(f"{ticker}: {e}")
                logger.error("%s: pipeline failed: %s", ticker, e)

        health.duration_seconds = time.monotonic() - t_start

        # Record the run
        run_id = self.store.record_pipeline_run(health.to_dict())
        logger.info(
            "Pipeline complete: %d/%d ok in %.1fs (run_id=%d)",
            health.tickers_succeeded, health.tickers_attempted,
            health.duration_seconds, run_id,
        )
        return health

    def _run_ticker(self, ticker: str, health: PipelineHealth):
        """
        Run the full pipeline for a single ticker with a hard timeout.
        Raises on failure so the caller can count it.
        """
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._process_ticker, ticker, health)
            try:
                future.result(timeout=self.ticker_timeout)
            except FutureTimeoutError:
                raise RuntimeError(
                    f"Timed out after {self.ticker_timeout}s"
                )

    def _process_ticker(self, ticker: str, health: PipelineHealth):
        """The actual per-ticker work. Runs inside a thread."""
        logger.info("%s: starting", ticker)
        today = date.today().isoformat()

        # ── 1. OHLCV (cache-first) ──────────────────────────────────────
        ohlcv_df = self.store.read_ohlcv_cache(ticker, min_bars=50)
        if ohlcv_df is None:
            logger.debug("%s: cache miss, fetching from yfinance", ticker)
            ohlcv_df = fetch_ohlcv(ticker, lookback_days=250)
            self.store.write_ohlcv_cache(ticker, ohlcv_df)
        else:
            logger.debug("%s: cache hit (%d bars)", ticker, len(ohlcv_df))

        # Drop quality column for Kronos
        kronos_df = ohlcv_df.drop(columns=["amount_proxy_quality"], errors="ignore")

        # ── 2. Kronos prediction ────────────────────────────────────────
        kronos_pred = None
        try:
            kronos_pred = kronos_predict(
                ticker=ticker,
                ohlcv_df=kronos_df,
                horizon_days=config.PREDICTION_HORIZON_DAYS,
                model_size=self.model_size,
                n_mc_samples=5,
            )
        except Exception as e:
            health.kronos_errors += 1
            logger.warning("%s: Kronos failed: %s", ticker, e)

        # ── 3. Reddit posts ─────────────────────────────────────────────
        reddit_posts = []
        scrape_status = "SKIPPED"
        try:
            result = self.scraper.fetch(
                ticker,
                lookback_hours=config.REDDIT_LOOKBACK_HOURS,
                max_posts=50,
            )
            if result.ok:
                reddit_posts = result.posts
                scrape_status = "SUCCESS"
                # Persist raw posts
                post_dicts = [
                    {
                        "ticker": p.ticker,
                        "post_id": p.post_id,
                        "title": p.title,
                        "body": p.body,
                        "score": p.score,
                        "num_comments": p.num_comments,
                        "subreddit": p.subreddit,
                        "post_created_utc": p.post_created_utc,
                    }
                    for p in reddit_posts
                ]
                self.store.insert_reddit_posts(post_dicts)
            else:
                scrape_status = "API_ERROR"
                health.reddit_errors += 1
                logger.warning("%s: Reddit scrape failed: %s", ticker, result.error)
        except Exception as e:
            scrape_status = "API_ERROR"
            health.reddit_errors += 1
            logger.warning("%s: Reddit exception: %s", ticker, e)

        # ── 4. Sentiment ────────────────────────────────────────────────
        sentiment = None
        try:
            sentiment = analyze_ticker(ticker, reddit_posts)
        except Exception as e:
            health.sentiment_errors += 1
            logger.warning("%s: sentiment failed: %s", ticker, e)

        # ── 5. Technicals ───────────────────────────────────────────────
        technicals = None
        try:
            technicals = compute_technicals(ticker, ohlcv_df)
        except Exception as e:
            logger.warning("%s: technicals failed: %s", ticker, e)

        # ── 6. Confluence ───────────────────────────────────────────────
        conf_result: ConfluenceResult = self.confluence.score(
            ticker=ticker,
            kronos=kronos_pred,
            sentiment=sentiment,
            technicals=technicals,
        )

        # ── 7. Store signal ─────────────────────────────────────────────
        signal_data = _build_signal_dict(
            ticker=ticker,
            signal_date=today,
            conf=conf_result,
            scrape_status=scrape_status,
            price_at_signal=float(ohlcv_df["close"].iloc[-1]),
        )
        signal_id = self.store.upsert_signal(signal_data)

        # ── 8. Paper trade ──────────────────────────────────────────────
        if self.trader.enabled:
            trade_result = self.trader.handle_signal(
                ticker=ticker,
                label=conf_result.label,
                confluence_score=conf_result.confluence_score,
                signal_id=signal_id,
            )
            if trade_result.ok:
                self.store.record_trade({
                    "signal_id": signal_id,
                    "ticker": ticker,
                    "signal_date": today,
                    "signal_label": conf_result.label,
                    "confluence_score": conf_result.confluence_score,
                    "alpaca_order_id": trade_result.order_id,
                    "side": trade_result.action,
                    "qty": trade_result.qty,
                    "position_size_usd": trade_result.position_size_usd,
                    "entry_price": trade_result.price,
                    "status": "open",
                })
                logger.info("%s: paper trade recorded (%s)", ticker, trade_result.action)

        logger.info(
            "%s: %s (confluence=%.3f) reddit_posts=%d",
            ticker, conf_result.label, conf_result.confluence_score, len(reddit_posts),
        )


def _build_signal_dict(
    ticker: str,
    signal_date: str,
    conf: ConfluenceResult,
    scrape_status: str,
    price_at_signal: float,
) -> dict:
    """Flatten ConfluenceResult into a flat dict for SignalStore.upsert_signal."""
    d: dict = {
        "ticker": ticker,
        "signal_date": signal_date,
        "confluence_score": conf.confluence_score,
        "classifier_label": conf.label,
        "classifier_reasoning": "\n".join(conf.reasoning),
        "reddit_catalyst_status": scrape_status,
        "price_at_signal": price_at_signal,
    }

    k = conf.kronos_prediction
    if k is not None:
        d.update({
            "kronos_direction": k.direction,
            "kronos_pct_change": k.pct_change,
            "kronos_confidence": k.confidence,
            "kronos_horizon_days": k.horizon_days,
            "kronos_predicted_close": k.predicted_close,
        })

    s = conf.ticker_sentiment
    if s is not None:
        d.update({
            "reddit_sentiment": s.label,
            "reddit_score": s.signed_score,
            "reddit_post_count": s.post_count,
        })

    t = conf.technicals
    if t is not None:
        d.update({
            "rsi_14": t.rsi_14,
            "macd_signal": t.macd_signal,
            "bb_position": t.bb_position,
            "atr_14": t.atr_14,
            "adx_14": t.adx_14,
            "avg_volume_ratio": t.avg_volume_ratio,
            "price_vs_200ma": t.price_vs_200ma,
            "price_vs_52w_high": t.price_vs_52w_high,
        })

    return d
