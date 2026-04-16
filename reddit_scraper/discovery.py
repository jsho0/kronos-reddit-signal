"""
Reddit discovery: scrapes hot/top/rising posts from target subreddits,
extracts ticker mentions, and ranks them by buzz score.

This replaces the old ticker-by-ticker search pattern.
Instead of N requests for N tickers, we make a small number of
subreddit-level pulls and extract all candidates from those results.

Buzz score formula (per ticker):
  For each post mentioning the ticker:
    upvote_weight  = log(1 + max(0, post_score))
    comment_weight = log(1 + num_comments) * 0.5
    recency_weight = 1.0 if < 6h ago else 0.8 if < 12h else 0.6
    mention_weight = log(1 + mention_count)
    buzz += (upvote_weight + comment_weight) * recency_weight * mention_weight

Higher buzz = more community attention weighted by engagement and recency.
"""
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import requests

from reddit_scraper.ticker_extractor import count_mentions, extract_tickers

logger = logging.getLogger(__name__)

REDDIT_BASE = "https://www.reddit.com"
DEFAULT_USER_AGENT = "kronos-signal-bot/0.1 (discovery mode)"
_REQUEST_DELAY = 1.2   # seconds between requests


@dataclass
class DiscoveredPost:
    post_id: str
    subreddit: str
    title: str
    body: str
    score: int
    num_comments: int
    url: str
    created_utc: float
    permalink: str


@dataclass
class BuzzCandidate:
    ticker: str
    buzz_score: float
    mention_count: int           # total mentions across all posts
    post_count: int              # distinct posts mentioning this ticker
    top_posts: list[DiscoveredPost] = field(default_factory=list)


class RedditDiscovery:
    """
    Scrapes Reddit feeds to discover trending tickers.

    Usage:
        discovery = RedditDiscovery()
        candidates = discovery.run(subreddits=[...], min_buzz=5.0)
        for c in candidates:
            print(c.ticker, c.buzz_score)
    """

    def __init__(self, user_agent: str = DEFAULT_USER_AGENT):
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": user_agent})

    def run(
        self,
        subreddits: list[str],
        post_limit: int = 100,
        min_buzz: float = 5.0,
        lookback_hours: int = 24,
    ) -> list[BuzzCandidate]:
        """
        Main entry point. Scrapes all feeds, extracts tickers, ranks by buzz.

        Returns BuzzCandidates sorted by buzz_score descending, filtered to
        those above min_buzz.
        """
        cutoff = datetime.now(timezone.utc).timestamp() - lookback_hours * 3600
        all_posts: list[DiscoveredPost] = []

        for sub in subreddits:
            for feed in ("hot", "top", "rising"):
                posts = self._fetch_feed(sub, feed, post_limit, cutoff)
                all_posts.extend(posts)
                logger.debug("discovery: r/%s/%s → %d posts", sub, feed, len(posts))

        logger.info("discovery: %d total posts scraped across %d subreddits", len(all_posts), len(subreddits))

        candidates = self._rank_tickers(all_posts)
        filtered = [c for c in candidates if c.buzz_score >= min_buzz]

        logger.info(
            "discovery: %d unique tickers found, %d above buzz threshold %.1f",
            len(candidates), len(filtered), min_buzz,
        )
        return filtered

    def _fetch_feed(
        self,
        subreddit: str,
        feed: str,
        limit: int,
        cutoff_ts: float,
    ) -> list[DiscoveredPost]:
        """Fetch one feed (hot/top/rising) from one subreddit."""
        url = f"{REDDIT_BASE}/r/{subreddit}/{feed}.json"
        params = {"limit": min(limit, 100), "raw_json": 1}
        if feed == "top":
            params["t"] = "day"

        try:
            time.sleep(_REQUEST_DELAY)
            resp = self._session.get(url, params=params, timeout=15)
            if resp.status_code == 429:
                logger.warning("discovery: rate limited on r/%s/%s, waiting 30s", subreddit, feed)
                time.sleep(30)
                resp = self._session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("discovery: failed r/%s/%s: %s", subreddit, feed, exc)
            return []

        posts = []
        for child in data.get("data", {}).get("children", []):
            p = child.get("data", {})
            created = float(p.get("created_utc", 0))
            if created < cutoff_ts:
                continue
            posts.append(DiscoveredPost(
                post_id=f"t3_{p.get('id', '')}",
                subreddit=p.get("subreddit", subreddit),
                title=p.get("title", ""),
                body=p.get("selftext", "") or "",
                score=int(p.get("score", 0)),
                num_comments=int(p.get("num_comments", 0)),
                url=p.get("url", ""),
                created_utc=created,
                permalink=f"https://www.reddit.com{p.get('permalink', '')}",
            ))
        return posts

    def _rank_tickers(self, posts: list[DiscoveredPost]) -> list[BuzzCandidate]:
        """Extract tickers from all posts and compute buzz scores."""
        now = datetime.now(timezone.utc).timestamp()

        # ticker → {posts: list, total_mentions: int, buzz: float}
        ticker_data: dict[str, dict] = {}

        for post in posts:
            full_text = post.title + " " + post.body
            tickers = extract_tickers(full_text)
            if not tickers:
                continue

            hours_ago = (now - post.created_utc) / 3600
            recency = 1.0 if hours_ago < 6 else (0.8 if hours_ago < 12 else 0.6)
            upvote_w = math.log1p(max(0, post.score))
            comment_w = math.log1p(post.num_comments) * 0.5

            for ticker in tickers:
                mentions = count_mentions(full_text, ticker)
                mention_w = math.log1p(mentions)
                buzz_contrib = (upvote_w + comment_w) * recency * mention_w

                if ticker not in ticker_data:
                    ticker_data[ticker] = {"posts": [], "total_mentions": 0, "buzz": 0.0}

                ticker_data[ticker]["buzz"] += buzz_contrib
                ticker_data[ticker]["total_mentions"] += mentions
                if post not in ticker_data[ticker]["posts"]:
                    ticker_data[ticker]["posts"].append(post)

        candidates = []
        for ticker, data in ticker_data.items():
            # Sort posts by score descending, keep top 5
            top_posts = sorted(data["posts"], key=lambda p: p.score, reverse=True)[:5]
            candidates.append(BuzzCandidate(
                ticker=ticker,
                buzz_score=round(data["buzz"], 3),
                mention_count=data["total_mentions"],
                post_count=len(data["posts"]),
                top_posts=top_posts,
            ))

        return sorted(candidates, key=lambda c: c.buzz_score, reverse=True)

    def fetch_post_comments(self, subreddit: str, post_id: str, limit: int = 20) -> list[str]:
        """
        Fetch top comments for a post. Used by the qualifier for context.
        post_id should be the bare ID (without t3_ prefix).
        """
        bare_id = post_id.replace("t3_", "")
        url = f"{REDDIT_BASE}/r/{subreddit}/comments/{bare_id}.json"
        params = {"limit": limit, "sort": "top", "raw_json": 1}

        try:
            time.sleep(_REQUEST_DELAY)
            resp = self._session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            if not isinstance(data, list) or len(data) < 2:
                return []

            comments = []
            for child in data[1].get("data", {}).get("children", []):
                body = child.get("data", {}).get("body", "")
                if body and body != "[deleted]" and body != "[removed]":
                    comments.append(body.strip())
                if len(comments) >= limit:
                    break
            return comments

        except Exception as exc:
            logger.debug("discovery: comments fetch failed for %s: %s", post_id, exc)
            return []
