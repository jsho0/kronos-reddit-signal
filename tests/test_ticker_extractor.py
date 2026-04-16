"""
Regression tests for reddit_scraper.ticker_extractor.

Found by /qa on 2026-04-15
Report: .gstack/qa-reports/qa-report-kronos-signal-2026-04-15.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from reddit_scraper.ticker_extractor import extract_tickers, count_mentions


class TestExtractTickers:
    def test_basic_extraction(self):
        tickers = extract_tickers("I love AAPL and TSLA is going to the moon")
        assert "AAPL" in tickers
        assert "TSLA" in tickers

    def test_blocklist_filters_common_words(self):
        tickers = extract_tickers("THE ETF AND OR FOR CEO CFO IPO ALL")
        assert tickers == []

    def test_dollar_prefix_bypasses_blocklist(self):
        # Dollar-prefixed tickers should always be included
        tickers = extract_tickers("$THE is a legit ticker when dollar-prefixed")
        assert "THE" in tickers

    def test_empty_text(self):
        assert extract_tickers("") == []

    def test_deduplication(self):
        tickers = extract_tickers("AAPL AAPL AAPL TSLA")
        assert tickers.count("AAPL") == 1

    def test_no_six_plus_char_tickers(self):
        # Only 1-5 char uppercase sequences should match
        tickers = extract_tickers("TOOLONG is not a ticker")
        assert "TOOLONG" not in tickers


class TestCountMentions:
    def test_basic_count(self):
        assert count_mentions("AAPL AAPL TSLA", "AAPL") == 2

    def test_dollar_prefix_counted(self):
        # $AAPL counts as 1 dollar mention + 1 bare mention (regex sees AAPL inside $AAPL)
        # plus the standalone AAPL = 3 total
        assert count_mentions("$AAPL went up, AAPL is great", "AAPL") == 3
        # Dollar-only (no bare): 1 bare (inside $) + 1 dollar = 2
        assert count_mentions("$AAPL is interesting", "AAPL") == 2

    def test_zero_mentions(self):
        assert count_mentions("TSLA is great", "AAPL") == 0

    def test_empty_text(self):
        assert count_mentions("", "AAPL") == 0
