"""
Discovery runner: runs the full Reddit discovery → qualification → watchlist update cycle.

Runs at 6am ET (one hour before the main pipeline).

Flow:
  1. Scrape Reddit hot/top/rising across target subreddits
  2. Extract ticker candidates ranked by buzz score
  3. For each candidate above min_buzz: run qualification gate
     (market validity + options + Claude thesis analysis)
  4. Qualified tickers → upsert into discovered_tickers table
     - New ticker: status=active, priority=NEW, streak=1
     - Existing ticker seen again: increment streak, update priority
  5. Apply decay to tickers NOT seen today
     - Missed 1 day → COOLING
     - Missed 3 days → DROPPED
"""
import logging
from datetime import date, datetime, timezone

import config
from reddit_scraper.discovery import RedditDiscovery
from reddit_scraper.qualifier import qualify
from storage.db import init_db
from storage.store import SignalStore

logger = logging.getLogger(__name__)


def _compute_priority(consecutive_days: int) -> str:
    if consecutive_days >= config.PRIORITY_HIGH_DAYS:
        return "HIGH"
    elif consecutive_days >= config.PRIORITY_MEDIUM_DAYS:
        return "MEDIUM"
    return "NEW"


class DiscoveryRunner:
    def __init__(self):
        self.store = SignalStore()
        self.discovery = RedditDiscovery()
        init_db()

    def run(self) -> dict:
        """
        Execute the full discovery cycle.
        Returns a summary dict with metrics.
        """
        today = date.today().isoformat()
        t0 = datetime.now(timezone.utc)

        logger.info("discovery: starting run for %s", today)

        # Step 1: scrape Reddit
        candidates = self.discovery.run(
            subreddits=config.DISCOVERY_SUBREDDITS,
            post_limit=config.DISCOVERY_POST_LIMIT,
            min_buzz=config.DISCOVERY_MIN_BUZZ,
            lookback_hours=config.DISCOVERY_LOOKBACK_HRS,
        )

        logger.info("discovery: %d candidates above buzz threshold", len(candidates))

        # Step 2: qualify each candidate
        qualified: list[dict] = []
        rejected = 0
        seen_today: set[str] = set()

        for candidate in candidates:
            try:
                result = qualify(candidate, self.discovery)
                if result is None:
                    rejected += 1
                    continue

                ticker = result["ticker"]
                seen_today.add(ticker)

                # Step 3: upsert into discovered_tickers
                existing = self.store.get_discovered_ticker(ticker)

                if existing is None or existing.status == "dropped":
                    # New ticker or returning after drop
                    row_data = {
                        "ticker": ticker,
                        "first_seen": today,
                        "last_seen": today,
                        "consecutive_days": 1,
                        "peak_streak": 1,
                        "total_days_seen": 1,
                        "priority": "NEW",
                        "status": "active",
                        "last_buzz_score": result["buzz_score"],
                        "avg_buzz_score": result["buzz_score"],
                        "mention_count": result["mention_count"],
                        **{k: result[k] for k in (
                            "company_name", "sector", "industry", "market_cap",
                            "description", "website", "thesis_quality", "layman_summary",
                            "bull_case", "bear_case", "key_catalyst", "analysis_confidence",
                            "stocktwits_count", "short_ratio", "short_float",
                            "post_summaries", "triggering_post_url",
                        ) if k in result},
                    }
                else:
                    # Existing ticker seen again — only increment streak/total if it's a new day
                    is_new_day = existing.last_seen != today
                    new_consecutive = existing.consecutive_days + (1 if is_new_day else 0)
                    new_peak = max(existing.peak_streak, new_consecutive)
                    new_total = existing.total_days_seen + (1 if is_new_day else 0)

                    # Running average buzz (only update if new day, else just refresh latest)
                    prev_avg = existing.avg_buzz_score or result["buzz_score"]
                    if is_new_day and new_total > 0:
                        new_avg = (prev_avg * existing.total_days_seen + result["buzz_score"]) / new_total
                    else:
                        new_avg = prev_avg

                    row_data = {
                        "ticker": ticker,
                        "last_seen": today,
                        "consecutive_days": new_consecutive,
                        "peak_streak": new_peak,
                        "total_days_seen": new_total,
                        "priority": _compute_priority(new_consecutive),
                        "status": "active",
                        "last_buzz_score": result["buzz_score"],
                        "avg_buzz_score": round(new_avg, 3),
                        "mention_count": result["mention_count"],
                        # Refresh analysis
                        "thesis_quality": result["thesis_quality"],
                        "layman_summary": result["layman_summary"],
                        "bull_case": result["bull_case"],
                        "bear_case": result["bear_case"],
                        "key_catalyst": result["key_catalyst"],
                        "analysis_confidence": result["analysis_confidence"],
                        "stocktwits_count": result["stocktwits_count"],
                        "post_summaries": result["post_summaries"],
                        "triggering_post_url": result["triggering_post_url"],
                    }

                self.store.upsert_discovered_ticker(row_data)
                qualified.append(result)
                logger.info(
                    "discovery: qualified %s (%s, buzz=%.1f, quality=%d)",
                    ticker,
                    row_data["priority"],
                    result["buzz_score"],
                    result["thesis_quality"],
                )

            except Exception as exc:
                logger.error("discovery: error qualifying %s: %s", candidate.ticker, exc)
                rejected += 1

        # Step 4: apply decay to tickers not seen today
        self.store.apply_discovery_decay(seen_today, today)

        duration = (datetime.now(timezone.utc) - t0).total_seconds()
        summary = {
            "date": today,
            "candidates_found": len(candidates),
            "qualified": len(qualified),
            "rejected": rejected,
            "tickers": [q["ticker"] for q in qualified],
            "duration_seconds": round(duration, 1),
        }

        logger.info(
            "discovery: complete in %.1fs — %d qualified, %d rejected. Tickers: %s",
            duration, len(qualified), rejected, ", ".join(summary["tickers"]),
        )
        return summary
