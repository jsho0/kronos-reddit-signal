"""
Reddit scraper using the public JSON API (no credentials required).

Reddit serves public subreddit data as JSON to any client with a descriptive
User-Agent. This is how the Reddit mobile app works. No API key needed.

Endpoint pattern:
    https://www.reddit.com/r/{subreddit}/search.json
    ?q={ticker} OR ${ticker}&sort=new&t=week&limit=100

Rate limits: ~60 req/min unauthenticated. We fetch one ticker at a time
(one request per subreddit batch) so this is not a concern in practice.

TODO: Replace with PRAW for higher rate limits and more reliable access.
      See TODOS.md for steps.
"""
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)

REDDIT_BASE = "https://www.reddit.com"
DEFAULT_USER_AGENT = "kronos-signal-bot/0.1 (personal research tool)"

DEFAULT_SUBREDDITS = [
    "wallstreetbets",
    "stocks",
    "investing",
    "SecurityAnalysis",
    "options",
]

# Polite delay between requests to avoid triggering rate limits
_REQUEST_DELAY_SECONDS = 1.0


@dataclass
class RedditPost:
    post_id: str           # Reddit fullname, e.g. "t3_abc123"
    ticker: str
    title: str
    body: str
    score: int             # Reddit upvote score
    num_comments: int
    subreddit: str
    post_created_utc: str  # ISO UTC string
    url: str


@dataclass
class ScrapeResult:
    ticker: str
    posts: list[RedditPost] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class RedditScraper:
    """
    Fetches ticker-related Reddit posts via the public JSON API.

    Usage:
        scraper = RedditScraper()
        result = scraper.fetch("AAPL", lookback_hours=48, max_posts=50)
        for post in result.posts:
            print(post.title, post.score)
    """

    def __init__(
        self,
        user_agent: str = None,
        subreddits: list[str] = None,
        request_delay: float = _REQUEST_DELAY_SECONDS,
    ):
        self.user_agent = user_agent or os.getenv(
            "REDDIT_USER_AGENT", DEFAULT_USER_AGENT
        )
        self.subreddits = subreddits or _parse_subreddits()
        self.request_delay = request_delay
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": self.user_agent})

    def fetch(
        self,
        ticker: str,
        lookback_hours: int = 48,
        max_posts: int = 100,
    ) -> ScrapeResult:
        """
        Fetch recent Reddit posts mentioning ticker.

        Searches all configured subreddits as a combined multireddit.
        Uses Reddit's search endpoint with time_filter=week, then filters
        to lookback_hours locally for consistency.

        Args:
            ticker: stock ticker, e.g. "AAPL"
            lookback_hours: only return posts newer than this
            max_posts: cap total posts returned

        Returns:
            ScrapeResult with deduplicated posts sorted newest-first.
        """
        cutoff_ts = (
            datetime.now(timezone.utc).timestamp() - lookback_hours * 3600
        )

        subreddit_str = "+".join(self.subreddits)
        query = f"{ticker} OR ${ticker}"
        url = f"{REDDIT_BASE}/r/{subreddit_str}/search.json"
        params = {
            "q": query,
            "sort": "new",
            "t": "week",
            "limit": 100,
            "restrict_sr": "true",
        }

        seen_ids: set[str] = set()
        posts: list[RedditPost] = []
        after: str | None = None  # pagination cursor

        try:
            while len(posts) < max_posts:
                if after:
                    params["after"] = after

                time.sleep(self.request_delay)
                resp = self._session.get(url, params=params, timeout=15)

                if resp.status_code == 429:
                    logger.warning("Reddit rate limit hit for %s, backing off 30s", ticker)
                    time.sleep(30)
                    continue

                resp.raise_for_status()
                data = resp.json()

                children = data.get("data", {}).get("children", [])
                if not children:
                    break

                for child in children:
                    p = child.get("data", {})
                    created_utc = float(p.get("created_utc", 0))
                    if created_utc < cutoff_ts:
                        continue

                    post_id = f"t3_{p.get('id', '')}"
                    if post_id in seen_ids:
                        continue
                    seen_ids.add(post_id)

                    posts.append(RedditPost(
                        post_id=post_id,
                        ticker=ticker,
                        title=p.get("title", ""),
                        body=p.get("selftext", ""),
                        score=int(p.get("score", 0)),
                        num_comments=int(p.get("num_comments", 0)),
                        subreddit=p.get("subreddit", ""),
                        post_created_utc=datetime.fromtimestamp(
                            created_utc, tz=timezone.utc
                        ).isoformat(),
                        url=p.get("url", ""),
                    ))

                    if len(posts) >= max_posts:
                        break

                after = data.get("data", {}).get("after")
                if not after:
                    break

        except requests.RequestException as e:
            logger.error("HTTP error fetching %s: %s", ticker, e)
            return ScrapeResult(ticker=ticker, error=str(e))
        except Exception as e:
            logger.error("Unexpected error fetching %s: %s", ticker, e)
            return ScrapeResult(ticker=ticker, error=str(e))

        logger.info(
            "%s: fetched %d posts from r/%s (lookback=%dh)",
            ticker, len(posts), subreddit_str, lookback_hours,
        )
        return ScrapeResult(ticker=ticker, posts=posts)

    def fetch_batch(
        self,
        tickers: list[str],
        lookback_hours: int = 48,
        max_posts_per_ticker: int = 100,
    ) -> dict[str, ScrapeResult]:
        """
        Fetch posts for multiple tickers sequentially.
        Returns dict of ticker -> ScrapeResult.
        """
        results = {}
        for ticker in tickers:
            results[ticker] = self.fetch(ticker, lookback_hours, max_posts_per_ticker)
        return results


def _parse_subreddits() -> list[str]:
    raw = os.getenv("REDDIT_SUBREDDITS", "")
    if raw:
        return [s.strip() for s in raw.split(",") if s.strip()]
    return DEFAULT_SUBREDDITS
