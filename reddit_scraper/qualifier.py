"""
Ticker qualification gate.

For each buzz candidate, runs a multi-stage filter before it enters the pipeline:

Stage 1 — Market validity (yfinance)
  - Ticker resolves to a real stock
  - Market cap >= MIN_MARKET_CAP ($100M)
  - Average daily volume >= MIN_AVG_VOLUME
  - Has an options chain (institutional interest / active market)

Stage 2 — Cross-reference (StockTwits + short interest)
  - StockTwits message volume (public API, no auth)
  - Short interest ratio from yfinance info

Stage 3 — Claude thesis quality
  - Reads the top Reddit post + comments
  - Returns quality_score (0-10), bull/bear case, layman summary
  - Filters pump attempts and low-quality posts

Minimum to pass all stages:
  market_cap >= 100M, has_options, quality_score >= 5, not is_pump

Cache: company info is cached in yfinance and refreshed here per run.
"""
import json
import logging
import os
import time

import yfinance as yf
import anthropic

from reddit_scraper.discovery import BuzzCandidate, RedditDiscovery

logger = logging.getLogger(__name__)

MIN_MARKET_CAP = 100_000_000    # $100M
MIN_AVG_VOLUME = 500_000        # avg daily shares traded
MIN_QUALITY_SCORE = 5           # Claude 0-10 quality threshold


QUALIFIER_PROMPT = """You are a financial analyst evaluating a Reddit post to determine if it represents a genuine stock opportunity.

## Stock: {ticker} ({company_name})
Sector: {sector}
Market Cap: ${market_cap_str}
Current Price: ${current_price}
52-week range: ${low_52w} - ${high_52w}
Short Interest: {short_ratio} days to cover
Has Options: {has_options}
StockTwits messages (24h): {stocktwits_count}

## Reddit Post
Subreddit: r/{subreddit}
Title: {title}
Upvotes: {upvotes} | Comments: {num_comments}

{body}

## Top Comments
{comments}

## Your Task
Evaluate whether this represents a genuine investment opportunity worth deeper analysis.

Respond with ONLY a valid JSON object, no other text:
{{
  "quality_score": <integer 0-10>,
  "is_pump": <true or false>,
  "bull_case": "<1-2 sentence bull case in plain English>",
  "bear_case": "<1-2 sentence bear case in plain English>",
  "layman_summary": "<2-3 sentence plain English explanation of why this stock is being discussed right now>",
  "key_catalyst": "<the main reason for current attention: earnings/news/technical/squeeze/etc>",
  "confidence": "<low|medium|high>"
}}

Quality score guide:
0-3: No thesis — memes, single mentions, hype without substance
4-5: Thin thesis — some reasoning but speculative or vague
6-7: Solid thesis — clear reasoning, identifiable catalyst
8-10: Strong thesis — specific data, risk/reward analysis, well-reasoned

Mark is_pump=true if: coordinated call-to-action, suspicious uniformity of comments, unrealistic price targets, known pump patterns."""


def _get_stocktwits_count(ticker: str) -> int:
    """Return 24h message count from StockTwits public API. Returns 0 on failure."""
    try:
        import requests
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
        resp = requests.get(url, timeout=8, headers={"User-Agent": "kronos-signal-bot/0.1"})
        if resp.status_code == 200:
            data = resp.json()
            messages = data.get("messages", [])
            return len(messages)
    except Exception:
        pass
    return 0


def _validate_market(ticker: str) -> dict | None:
    """
    Validate ticker via yfinance. Returns info dict or None if invalid.
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info

        # Must have price data
        price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
        if not price or price <= 0:
            return None

        market_cap = info.get("marketCap") or 0
        if market_cap < MIN_MARKET_CAP:
            logger.debug("qualifier: %s rejected — market cap $%dM", ticker, market_cap // 1_000_000)
            return None

        avg_volume = info.get("averageVolume") or info.get("averageDailyVolume10Day") or 0
        if avg_volume < MIN_AVG_VOLUME:
            logger.debug("qualifier: %s rejected — avg volume %d", ticker, avg_volume)
            return None

        # Check for options chain
        has_options = bool(t.options)

        return {
            "ticker": ticker,
            "company_name": info.get("longName") or info.get("shortName") or ticker,
            "sector": info.get("sector") or "Unknown",
            "industry": info.get("industry") or "Unknown",
            "market_cap": market_cap,
            "current_price": price,
            "low_52w": info.get("fiftyTwoWeekLow") or price,
            "high_52w": info.get("fiftyTwoWeekHigh") or price,
            "short_ratio": info.get("shortRatio") or 0,
            "short_float": info.get("shortPercentOfFloat") or 0,
            "description": info.get("longBusinessSummary") or "",
            "website": info.get("website") or "",
            "has_options": has_options,
            "avg_volume": avg_volume,
        }
    except Exception as exc:
        logger.debug("qualifier: yfinance failed for %s: %s", ticker, exc)
        return None


def _call_claude(prompt: str) -> dict | None:
    """Call Claude and parse JSON response."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("qualifier: ANTHROPIC_API_KEY not set, skipping Claude qualification")
        return None

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("qualifier: Claude returned invalid JSON: %s", exc)
        return None
    except Exception as exc:
        logger.warning("qualifier: Claude call failed: %s", exc)
        return None


def qualify(
    candidate: BuzzCandidate,
    discovery: RedditDiscovery,
) -> dict | None:
    """
    Run the full qualification pipeline for one buzz candidate.

    Returns a qualification dict if the ticker passes all stages, None if rejected.

    The returned dict contains everything needed to populate DiscoveredTicker:
      market info, Claude analysis, buzz score, top post summaries.
    """
    ticker = candidate.ticker
    logger.info("qualifier: evaluating %s (buzz=%.1f, %d posts)", ticker, candidate.buzz_score, candidate.post_count)

    # Stage 1: market validity
    market = _validate_market(ticker)
    if market is None:
        logger.debug("qualifier: %s failed market validation", ticker)
        return None

    if not market["has_options"]:
        logger.debug("qualifier: %s has no options chain, skipping", ticker)
        return None

    # Stage 2: cross-reference
    stocktwits_count = _get_stocktwits_count(ticker)

    # Stage 3: Claude thesis quality
    # Use the highest-score post for qualification
    top_post = candidate.top_posts[0] if candidate.top_posts else None
    if top_post is None:
        return None

    comments = discovery.fetch_post_comments(
        subreddit=top_post.subreddit,
        post_id=top_post.post_id,
        limit=15,
    )
    comments_text = "\n\n".join(f"• {c[:300]}" for c in comments[:10]) if comments else "No comments available."

    body_preview = (top_post.body[:1500] + "...") if len(top_post.body) > 1500 else top_post.body
    if not body_preview.strip():
        body_preview = "(No body — title post)"

    prompt = QUALIFIER_PROMPT.format(
        ticker=ticker,
        company_name=market["company_name"],
        sector=market["sector"],
        market_cap_str=f"{market['market_cap'] / 1_000_000:.0f}M",
        current_price=f"{market['current_price']:.2f}",
        low_52w=f"{market['low_52w']:.2f}",
        high_52w=f"{market['high_52w']:.2f}",
        short_ratio=f"{market['short_ratio']:.1f}" if market["short_ratio"] else "N/A",
        has_options="Yes" if market["has_options"] else "No",
        stocktwits_count=stocktwits_count,
        subreddit=top_post.subreddit,
        title=top_post.title,
        upvotes=top_post.score,
        num_comments=top_post.num_comments,
        body=body_preview,
        comments=comments_text,
    )

    analysis = _call_claude(prompt)

    if analysis is None:
        # Claude unavailable — apply a lenient fallback based on buzz alone
        quality_score = min(10, int(candidate.buzz_score / 3))
        analysis = {
            "quality_score": quality_score,
            "is_pump": False,
            "bull_case": "Unable to analyze — Claude API unavailable.",
            "bear_case": "Unable to analyze — Claude API unavailable.",
            "layman_summary": f"{ticker} is getting Reddit attention with a buzz score of {candidate.buzz_score:.1f}.",
            "key_catalyst": "Unknown",
            "confidence": "low",
        }

    quality_score = int(analysis.get("quality_score", 0))
    is_pump = bool(analysis.get("is_pump", False))

    if is_pump:
        logger.info("qualifier: %s flagged as pump attempt, rejected", ticker)
        return None

    if quality_score < MIN_QUALITY_SCORE:
        logger.info("qualifier: %s quality score %d < %d, rejected", ticker, quality_score, MIN_QUALITY_SCORE)
        return None

    logger.info("qualifier: %s PASSED (quality=%d, buzz=%.1f)", ticker, quality_score, candidate.buzz_score)

    # Build post summaries for dashboard
    post_summaries = []
    for post in candidate.top_posts[:3]:
        post_summaries.append({
            "post_id": post.post_id,
            "subreddit": post.subreddit,
            "title": post.title,
            "score": post.score,
            "num_comments": post.num_comments,
            "permalink": post.permalink,
            "created_utc": post.created_utc,
        })

    return {
        "ticker": ticker,
        # Market info
        "company_name": market["company_name"],
        "sector": market["sector"],
        "industry": market["industry"],
        "market_cap": market["market_cap"],
        "description": market["description"],
        "website": market["website"],
        # Buzz
        "buzz_score": candidate.buzz_score,
        "mention_count": candidate.mention_count,
        "post_count": candidate.post_count,
        # Claude analysis
        "thesis_quality": quality_score,
        "bull_case": analysis.get("bull_case", ""),
        "bear_case": analysis.get("bear_case", ""),
        "layman_summary": analysis.get("layman_summary", ""),
        "key_catalyst": analysis.get("key_catalyst", ""),
        "analysis_confidence": analysis.get("confidence", "low"),
        # Cross-reference signals
        "stocktwits_count": stocktwits_count,
        "short_ratio": market["short_ratio"],
        "short_float": market["short_float"],
        # Top posts for dashboard
        "post_summaries": json.dumps(post_summaries),
        "triggering_post_url": top_post.permalink,
    }
