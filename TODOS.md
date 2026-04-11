# TODOS

## Deferred / Needs Implementation

### Reddit API (PRAW)
The scraper currently uses Reddit's public JSON endpoints (no auth required).
PRAW should be implemented for higher rate limits and more reliable access.

Steps when ready:
1. Create a Reddit app at reddit.com/prefs/apps (accept the Responsible Builder Policy first)
2. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env
3. Rewrite `reddit_scraper/scraper.py` to use PRAW in read-only mode
4. The rest of the pipeline (sentiment, storage, confluence) is unaffected

### Next-day price backfill
`storage/store.py` has `update_next_day_price()` ready.
Need a daily job that fetches yesterday's close and backfills `price_next_day`
for signals from the prior trading day. This enables Kronos accuracy tracking.

### Ticker list update script
`data/us_tickers.txt` (S&P 500 + Russell 1000) is static.
Need a quarterly script (`scripts/update_tickers.py`) to refresh it.
