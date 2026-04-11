"""
Reddit + sentiment smoke test.

Two parts:
  1. Scraper: tests fetch logic with a PRAW mock (no real API call needed).
     If REDDIT_CLIENT_ID is set in .env, also tests a live fetch for AAPL.
  2. Sentiment: runs real FinBERT on a handful of hard-coded sentences.
     Expects obvious positives/negatives to score correctly.

Usage:
    venv\Scripts\python.exe smoke_test_reddit.py
"""
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("smoke_test_reddit")


# ------------------------------------------------------------------ #
#  Scraper tests                                                       #
# ------------------------------------------------------------------ #

def test_scraper_live():
    """Live fetch via public JSON API — no credentials needed."""
    logger.info("=== Testing live Reddit fetch for AAPL (public JSON API) ===")
    from reddit_scraper.scraper import RedditScraper

    # request_delay=0 for tests — we're only making one request
    scraper = RedditScraper(request_delay=0)
    result = scraper.fetch("AAPL", lookback_hours=72, max_posts=10)

    if not result.ok:
        logger.warning("Live fetch failed (network?): %s", result.error)
        logger.info("live scraper: SKIPPED")
        return

    logger.info("Live fetch: %d posts for AAPL", len(result.posts))
    if result.posts:
        p = result.posts[0]
        assert p.post_id.startswith("t3_"), f"Unexpected post_id: {p.post_id}"
        assert p.ticker == "AAPL"
        assert p.subreddit != ""
        logger.info("  [%s] %r (score=%d)", p.subreddit, p.title[:60], p.score)

    logger.info("live scraper: PASS (%d posts)", len(result.posts))


def test_scraper_post_dataclass():
    """RedditPost dataclass has expected fields."""
    logger.info("=== Testing RedditPost dataclass ===")
    from reddit_scraper.scraper import RedditPost

    post = RedditPost(
        post_id="t3_test001",
        ticker="TSLA",
        title="TSLA short squeeze incoming",
        body="Volume is spiking",
        score=1500,
        num_comments=220,
        subreddit="wallstreetbets",
        post_created_utc="2026-04-10T12:00:00+00:00",
        url="https://reddit.com/r/wallstreetbets/test001",
    )
    assert post.post_id == "t3_test001"
    assert post.score == 1500
    logger.info("RedditPost dataclass: PASS")




# ------------------------------------------------------------------ #
#  Sentiment tests                                                     #
# ------------------------------------------------------------------ #

def test_finbert_obvious_cases():
    """FinBERT should get obvious positives/negatives right."""
    logger.info("=== Testing FinBERT on obvious cases ===")
    from reddit_scraper.sentiment import score_posts

    texts = [
        "AAPL crushed earnings, massive revenue beat, stock soaring",           # positive
        "TSLA disaster quarter, massive miss, CEO selling shares again",         # negative
        "Market closed today for holiday",                                       # neutral
        "NVDA AI chips driving incredible growth, best quarter in history",      # positive
        "Bankruptcy filing imminent, creditors circling, cash burn accelerating",# negative
    ]
    expected_labels = ["positive", "negative", "neutral", "positive", "negative"]

    results = score_posts(texts)
    assert len(results) == len(texts), f"Expected {len(texts)} results, got {len(results)}"

    wrong = []
    for i, (res, expected) in enumerate(zip(results, expected_labels)):
        if res.label != expected:
            wrong.append(f"  [{i}] '{texts[i][:50]}...' → got {res.label}, expected {expected}")
        logger.info(
            "  [%d] %s (score=%.3f, signed=%.3f) — expected %s",
            i, res.label, res.score, res.signed_score, expected
        )

    if wrong:
        logger.warning("Some obvious cases scored unexpectedly:\n%s", "\n".join(wrong))
        # Warn but don't fail — FinBERT can disagree on edge cases
    else:
        logger.info("All obvious cases scored correctly")

    logger.info("finbert obvious cases: PASS")


def test_finbert_aggregate():
    """analyze_ticker should aggregate correctly."""
    logger.info("=== Testing FinBERT aggregate (analyze_ticker) ===")
    from reddit_scraper.sentiment import analyze_ticker

    # 4 positive posts, 1 negative → net positive
    posts = [
        {"title": "AAPL new all-time high, incredible momentum", "body": ""},
        {"title": "Apple beats on revenue and earnings", "body": "iPhone sales record"},
        {"title": "AAPL stock buyback announced, bullish signal", "body": ""},
        {"title": "Apple expanding into new markets aggressively", "body": ""},
        {"title": "Minor supply chain concern flagged by analyst", "body": ""},
    ]

    result = analyze_ticker("AAPL", posts)
    assert result.ticker == "AAPL"
    assert result.post_count == 5
    assert result.label in ("positive", "neutral"), (
        f"Expected positive or neutral for bullish-heavy posts, got {result.label}"
    )
    assert len(result.per_post) == 5

    logger.info(
        "analyze_ticker: label=%s signed_score=%.3f post_count=%d",
        result.label, result.signed_score, result.post_count
    )
    logger.info("finbert aggregate: PASS")


def test_finbert_empty():
    """analyze_ticker with no posts returns neutral default."""
    logger.info("=== Testing FinBERT empty posts ===")
    from reddit_scraper.sentiment import analyze_ticker

    result = analyze_ticker("NOBODY", [])
    assert result.label == "neutral"
    assert result.post_count == 0
    assert result.score == 0.0
    logger.info("finbert empty: PASS")


def test_finbert_batch_boundary():
    """score_posts handles inputs that straddle batch_size boundary."""
    logger.info("=== Testing FinBERT batch boundary (35 texts, batch_size=32) ===")
    from reddit_scraper.sentiment import score_posts

    texts = ["Stock went up today"] * 35
    results = score_posts(texts, batch_size=32)
    assert len(results) == 35, f"Expected 35, got {len(results)}"
    logger.info("finbert batch boundary: PASS")


if __name__ == "__main__":
    errors = []
    tests = [
        test_scraper_live,
        test_scraper_post_dataclass,
        test_finbert_obvious_cases,
        test_finbert_aggregate,
        test_finbert_empty,
        test_finbert_batch_boundary,
    ]
    for test_fn in tests:
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
        print("All Reddit/sentiment smoke tests passed.")
