from reddit_scraper.scraper import RedditScraper, RedditPost, ScrapeResult
from reddit_scraper.sentiment import TickerSentiment, SentimentResult, analyze_ticker, score_posts

__all__ = [
    "RedditScraper", "RedditPost", "ScrapeResult",
    "TickerSentiment", "SentimentResult", "analyze_ticker", "score_posts",
]

try:
    from reddit_scraper.discovery import discover, DiscoveredTicker, DiscoveryResult
    __all__ += ["discover", "DiscoveredTicker", "DiscoveryResult"]
except ImportError:
    pass
