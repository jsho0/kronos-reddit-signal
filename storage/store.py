"""
SignalStore: all database read/write operations in one place.

Per-signal commits: each upsert_signal() call opens its own session.
This means a pipeline that processes 50 tickers won't lose all progress
if ticker #37 crashes — the first 36 are already committed.

OHLCV cache: write_ohlcv_cache() uses INSERT OR IGNORE semantics
(via on_conflict_do_nothing) so re-running the pipeline never double-writes bars.
"""
import logging
from datetime import date, datetime, timezone
from typing import Optional

import pandas as pd
from sqlalchemy import select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from storage.db import get_session
from storage.models import Experiment, OHLCVCache, PipelineRun, RedditPost, Signal, Trade

logger = logging.getLogger(__name__)


class SignalStore:
    """
    Thin wrapper around the ORM models.
    All methods open and close their own session (per-signal commit pattern).
    """

    # ------------------------------------------------------------------ #
    #  Signals                                                             #
    # ------------------------------------------------------------------ #

    def upsert_signal(self, data: dict) -> int:
        """
        Insert or update a signal row for (ticker, signal_date).
        Returns the row id.

        data keys mirror Signal columns. Unknown keys are ignored.
        signal_date defaults to today (UTC) if not provided.
        """
        if "signal_date" not in data:
            data["signal_date"] = date.today().isoformat()

        allowed = {c.name for c in Signal.__table__.columns}
        filtered = {k: v for k, v in data.items() if k in allowed}
        filtered["updated_at"] = datetime.now(timezone.utc).isoformat()

        with get_session() as session:
            stmt = (
                sqlite_insert(Signal)
                .values(**filtered)
                .on_conflict_do_update(
                    index_elements=["ticker", "signal_date"],
                    set_=filtered,
                )
            )
            result = session.execute(stmt)
            # Fetch the id after upsert
            row = session.execute(
                select(Signal).where(
                    Signal.ticker == filtered["ticker"],
                    Signal.signal_date == filtered["signal_date"],
                )
            ).scalar_one()
            return row.id

    def get_signal(self, ticker: str, signal_date: str) -> Optional[Signal]:
        """Return a single Signal row or None."""
        with get_session() as session:
            return session.execute(
                select(Signal).where(
                    Signal.ticker == ticker,
                    Signal.signal_date == signal_date,
                )
            ).scalar_one_or_none()

    def get_recent_signals(self, days: int = 30) -> list[Signal]:
        """Return all signals from the last N days, newest first."""
        with get_session() as session:
            cutoff = (
                pd.Timestamp.today() - pd.Timedelta(days=days)
            ).strftime("%Y-%m-%d")
            rows = session.execute(
                select(Signal)
                .where(Signal.signal_date >= cutoff)
                .order_by(Signal.signal_date.desc())
            ).scalars().all()
            # Detach from session so callers can use them after close
            session.expunge_all()
            return rows

    def update_next_day_price(self, ticker: str, signal_date: str, price: float):
        """
        Back-fill price_next_day for accuracy tracking.
        Called by a separate daily job, not the main pipeline.
        """
        with get_session() as session:
            session.execute(
                update(Signal)
                .where(Signal.ticker == ticker, Signal.signal_date == signal_date)
                .values(price_next_day=price)
            )

    # ------------------------------------------------------------------ #
    #  Reddit posts                                                        #
    # ------------------------------------------------------------------ #

    def insert_reddit_posts(self, posts: list[dict]) -> int:
        """
        Bulk-insert Reddit posts, skipping duplicates by post_id.
        Returns the count of rows actually inserted.
        """
        if not posts:
            return 0

        allowed = {c.name for c in RedditPost.__table__.columns}
        inserted = 0

        with get_session() as session:
            for post in posts:
                filtered = {k: v for k, v in post.items() if k in allowed}
                stmt = (
                    sqlite_insert(RedditPost)
                    .values(**filtered)
                    .on_conflict_do_nothing(index_elements=["post_id"])
                )
                result = session.execute(stmt)
                if result.rowcount:
                    inserted += 1

        logger.debug("Inserted %d/%d Reddit posts", inserted, len(posts))
        return inserted

    def get_reddit_posts(self, ticker: str, since_hours: int = 48) -> list[RedditPost]:
        """Return recent Reddit posts for a ticker."""
        with get_session() as session:
            cutoff = (
                pd.Timestamp.utcnow() - pd.Timedelta(hours=since_hours)
            ).isoformat()
            rows = session.execute(
                select(RedditPost)
                .where(
                    RedditPost.ticker == ticker,
                    RedditPost.fetched_at >= cutoff,
                )
                .order_by(RedditPost.fetched_at.desc())
            ).scalars().all()
            session.expunge_all()
            return rows

    # ------------------------------------------------------------------ #
    #  Pipeline runs                                                       #
    # ------------------------------------------------------------------ #

    def record_pipeline_run(self, metrics: dict) -> int:
        """
        Insert a PipelineRun row. Called at the end of each scheduler run.
        metrics keys mirror PipelineRun columns.
        Returns the row id.
        """
        if "run_at" not in metrics:
            metrics["run_at"] = datetime.now(timezone.utc).isoformat()

        allowed = {c.name for c in PipelineRun.__table__.columns}
        filtered = {k: v for k, v in metrics.items() if k in allowed}

        with get_session() as session:
            row = PipelineRun(**filtered)
            session.add(row)
            session.flush()
            run_id = row.id
        return run_id

    def get_pipeline_runs(self, limit: int = 30) -> list[PipelineRun]:
        """Return the most recent pipeline runs."""
        with get_session() as session:
            rows = session.execute(
                select(PipelineRun)
                .order_by(PipelineRun.run_at.desc())
                .limit(limit)
            ).scalars().all()
            session.expunge_all()
            return rows

    # ------------------------------------------------------------------ #
    #  OHLCV cache                                                         #
    # ------------------------------------------------------------------ #

    def write_ohlcv_cache(self, ticker: str, ohlcv_df: pd.DataFrame):
        """
        Write OHLCV bars to cache. Skips bars already stored (INSERT OR IGNORE).

        ohlcv_df must have a DatetimeIndex and columns:
            open, high, low, close, volume, amount, amount_proxy_quality
        """
        if ohlcv_df.empty:
            return

        rows = []
        for bar_date, row in ohlcv_df.iterrows():
            date_str = (
                bar_date.strftime("%Y-%m-%d")
                if hasattr(bar_date, "strftime")
                else str(bar_date)
            )
            rows.append({
                "ticker": ticker,
                "bar_date": date_str,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "amount": float(row["amount"]),
                "amount_proxy_quality": row.get("amount_proxy_quality", "good"),
            })

        with get_session() as session:
            for r in rows:
                stmt = (
                    sqlite_insert(OHLCVCache)
                    .values(**r)
                    .on_conflict_do_nothing(index_elements=["ticker", "bar_date"])
                )
                session.execute(stmt)

        logger.debug("OHLCV cache: wrote %d bars for %s", len(rows), ticker)

    def read_ohlcv_cache(self, ticker: str, min_bars: int = 50) -> Optional[pd.DataFrame]:
        """
        Load cached OHLCV bars for a ticker.
        Returns None if fewer than min_bars are cached (caller should fetch fresh).
        Returns DataFrame with DatetimeIndex and columns matching fetch_ohlcv() output.
        """
        with get_session() as session:
            rows = session.execute(
                select(OHLCVCache)
                .where(OHLCVCache.ticker == ticker)
                .order_by(OHLCVCache.bar_date.asc())
            ).scalars().all()

            if len(rows) < min_bars:
                return None

            # Extract data while still inside session to avoid DetachedInstanceError
            data = [
                {
                    "bar_date": r.bar_date,
                    "open": r.open,
                    "high": r.high,
                    "low": r.low,
                    "close": r.close,
                    "volume": r.volume,
                    "amount": r.amount,
                    "amount_proxy_quality": r.amount_proxy_quality,
                }
                for r in rows
            ]

        df = pd.DataFrame(data)
        df.index = pd.to_datetime(df["bar_date"])
        df.index.name = None
        df = df.drop(columns=["bar_date"])
        return df

    # ------------------------------------------------------------------ #
    #  Accuracy stats (for dashboard)                                      #
    # ------------------------------------------------------------------ #

    def get_accuracy_stats(self) -> dict:
        """
        Return a dict with Kronos directional accuracy stats.
        Only counts signals where price_next_day is available.
        """
        with get_session() as session:
            rows = session.execute(
                select(Signal).where(
                    Signal.price_at_signal.isnot(None),
                    Signal.price_next_day.isnot(None),
                )
            ).scalars().all()

            if not rows:
                return {"total": 0, "correct": 0, "accuracy": None}

            # Extract inside session to avoid DetachedInstanceError
            cases = [
                (r.kronos_direction, r.price_at_signal, r.price_next_day)
                for r in rows
            ]

        correct = 0
        for direction, price_at, price_next in cases:
            actual_dir = (
                "bullish" if price_next > price_at
                else "bearish" if price_next < price_at
                else "neutral"
            )
            if actual_dir == direction:
                correct += 1

        return {
            "total": len(cases),
            "correct": correct,
            "accuracy": correct / len(cases),
        }

    # ------------------------------------------------------------------ #
    #  Trades                                                              #
    # ------------------------------------------------------------------ #

    def record_trade(self, trade_data: dict) -> int:
        """Insert a new trade row. Returns the row id."""
        if "opened_at" not in trade_data:
            trade_data["opened_at"] = datetime.now(timezone.utc).isoformat()

        allowed = {c.name for c in Trade.__table__.columns}
        filtered = {k: v for k, v in trade_data.items() if k in allowed}

        with get_session() as session:
            row = Trade(**filtered)
            session.add(row)
            session.flush()
            trade_id = row.id
        return trade_id

    def close_trade(self, alpaca_order_id: str, exit_price: float, pnl_usd: float, pnl_pct: float):
        """Mark a trade as closed and record exit price + P&L."""
        from sqlalchemy import update
        with get_session() as session:
            session.execute(
                update(Trade)
                .where(Trade.alpaca_order_id == alpaca_order_id)
                .values(
                    status="closed",
                    exit_price=exit_price,
                    pnl_usd=pnl_usd,
                    pnl_pct=pnl_pct,
                    closed_at=datetime.now(timezone.utc).isoformat(),
                )
            )

    def get_trades(self, status: str = None, limit: int = 100) -> list[Trade]:
        """Return trades, optionally filtered by status."""
        with get_session() as session:
            q = select(Trade).order_by(Trade.opened_at.desc()).limit(limit)
            if status:
                q = q.where(Trade.status == status)
            rows = session.execute(q).scalars().all()
            # Extract attributes inside session
            result = []
            for r in rows:
                result.append({
                    "id": r.id,
                    "ticker": r.ticker,
                    "signal_date": r.signal_date,
                    "signal_label": r.signal_label,
                    "confluence_score": r.confluence_score,
                    "side": r.side,
                    "qty": r.qty,
                    "position_size_usd": r.position_size_usd,
                    "entry_price": r.entry_price,
                    "exit_price": r.exit_price,
                    "pnl_usd": r.pnl_usd,
                    "pnl_pct": r.pnl_pct,
                    "status": r.status,
                    "opened_at": r.opened_at,
                    "closed_at": r.closed_at,
                    "alpaca_order_id": r.alpaca_order_id,
                })
            return result

    # ------------------------------------------------------------------ #
    #  Experiments                                                         #
    # ------------------------------------------------------------------ #

    def create_experiment(self, data: dict) -> int:
        """Insert a new experiment row. Returns id."""
        allowed = {c.name for c in Experiment.__table__.columns}
        filtered = {k: v for k, v in data.items() if k in allowed}
        with get_session() as session:
            row = Experiment(**filtered)
            session.add(row)
            session.flush()
            return row.id

    def get_active_experiments(self) -> list[Experiment]:
        """Return all experiments with status='active'."""
        with get_session() as session:
            rows = session.execute(
                select(Experiment).where(Experiment.status == "active")
            ).scalars().all()
            session.expunge_all()
            return rows

    def get_pending_experiments(self) -> list[Experiment]:
        """Return all experiments with status='proposed'."""
        with get_session() as session:
            rows = session.execute(
                select(Experiment).where(Experiment.status == "proposed")
            ).scalars().all()
            session.expunge_all()
            return rows

    def update_experiment(self, experiment_id: int, data: dict):
        """Partial update an experiment row by id."""
        from sqlalchemy import update as sa_update
        allowed = {c.name for c in Experiment.__table__.columns}
        filtered = {k: v for k, v in data.items() if k in allowed}
        with get_session() as session:
            session.execute(
                sa_update(Experiment).where(Experiment.id == experiment_id).values(**filtered)
            )

    def get_all_experiments(self, limit: int = 50) -> list[dict]:
        """Return experiments as dicts for dashboard display."""
        with get_session() as session:
            rows = session.execute(
                select(Experiment).order_by(Experiment.proposed_at.desc()).limit(limit)
            ).scalars().all()
            result = []
            for r in rows:
                result.append({
                    "id": r.id,
                    "source_name": r.source_name,
                    "description": r.description,
                    "status": r.status,
                    "accuracy_before": r.accuracy_before,
                    "accuracy_after": r.accuracy_after,
                    "n_signals_before": r.n_signals_before,
                    "n_signals_after": r.n_signals_after,
                    "lesson": r.lesson,
                    "proposed_at": r.proposed_at,
                    "activated_at": r.activated_at,
                    "completed_at": r.completed_at,
                })
            return result

    def get_trade_stats(self) -> dict:
        """Return aggregate paper trading stats."""
        trades = self.get_trades(limit=10000)
        closed = [t for t in trades if t["status"] == "closed" and t["pnl_usd"] is not None]
        open_trades = [t for t in trades if t["status"] == "open"]

        if not closed:
            return {
                "total_trades": len(trades),
                "open_positions": len(open_trades),
                "closed_trades": 0,
                "total_pnl_usd": 0.0,
                "win_rate": None,
                "avg_pnl_usd": None,
            }

        winners = [t for t in closed if t["pnl_usd"] > 0]
        total_pnl = sum(t["pnl_usd"] for t in closed)

        return {
            "total_trades": len(trades),
            "open_positions": len(open_trades),
            "closed_trades": len(closed),
            "total_pnl_usd": total_pnl,
            "win_rate": len(winners) / len(closed),
            "avg_pnl_usd": total_pnl / len(closed),
        }
