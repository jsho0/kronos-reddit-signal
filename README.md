# Kronos Reddit Signal

A self-improving stock signal system. Discovers tickers from Reddit buzz, qualifies them with Claude, runs a time-series foundation model + FinBERT sentiment + technical indicators, and evolves its own data sources weekly.

Live dashboard: [http://83.136.219.215:8501](http://83.136.219.215:8501)

---

## What It Does

Every weekday morning, two jobs run back-to-back:

**6am ET — Discovery** (builds the dynamic watchlist):
1. Scrapes Reddit (`wallstreetbets`, `stocks`, `investing`, `options`, and more) for recent posts
2. Extracts ticker symbols, scores buzz per ticker (mentions, velocity, engagement)
3. Fetches basic fundamentals (market cap, sector) via yfinance
4. Sends each candidate to **Claude** for a qualify/reject decision with a bull/bear thesis
5. Saves qualified tickers to SQLite with priority tier (NEW / MEDIUM / HIGH / COOLING)

**7am ET — Pipeline** (runs signals on whatever discovery qualified):
1. Downloads OHLCV price history (yfinance, cached in SQLite)
2. Runs **Kronos** (a time-series foundation model) to predict direction and magnitude over the next 5 days
3. Scrapes Reddit for posts mentioning the ticker
4. Runs **FinBERT** on those posts to produce a signed sentiment score
5. Computes **technical indicators** (RSI, MACD, Bollinger Bands, ADX, volume ratio)
6. Queries **data source plugins** (options flow, macro regime, earnings proximity, and any auto-generated sources)
7. Combines everything into a single **confluence score** (0.0–1.0) and a label: `STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL`
8. Stores the signal to SQLite and optionally executes a **paper trade** via Alpaca

Every Sunday at 11pm ET, the **meta runner** evolves the system:
- Evaluates recent experiments for accuracy improvement
- Distills lessons into a cognition store
- Asks Claude to propose and write a brand new data source module
- Activates it on the next pipeline run

---

## Architecture

```
reddit_scraper/
  ticker_extractor.py  Regex-based ticker extraction from post text
  discovery.py         Reddit scraper for discovery (multi-sub, hot+new+top feeds)
  qualifier.py         Claude qualification gate — bull/bear thesis, priority scoring
  scraper.py           Per-ticker Reddit fetch for pipeline sentiment
  sentiment.py         FinBERT sentiment scorer
pipeline/
  discovery_runner.py  Orchestrates the full discovery flow
  runner.py            Runs Kronos + signals on discovered tickers
  scheduler.py         APScheduler — discovery 6am, pipeline 7am ET weekdays
kronos_engine/         Kronos model wrapper, OHLCV fetcher, technicals
confluence/            Scoring engine — combines all signals, plugin-aware
data_sources/          Plugin directory — each .py is a data source
  options_flow.py      Put/call ratio from yfinance options chain
  macro.py             VIX level + SPY 20-day trend
  earnings.py          Proximity to next earnings date
  <generated>.py       Auto-written by the Researcher agent
meta/                  ASI-Evolve layer
  researcher.py        Claude agent — proposes new data source modules
  analyzer.py          Evaluates experiment outcomes, distills lessons
  cognition.py         Persistent JSON lesson store
  runner.py            Orchestrates the weekly evolution cycle
storage/               SQLAlchemy models + SignalStore (SQLite, WAL mode)
trading/               Alpaca paper trading integration
dashboard/             Streamlit dashboard (5 tabs)
tests/                 Regression tests (pytest)
```

### Confluence Score

Weights auto-adjust based on active plugins. With the three default plugins active:

| Source | Base Weight | Effective Weight |
|---|---|---|
| Kronos | 0.50 | 0.385 |
| Reddit sentiment | 0.25 | 0.193 |
| Technicals | 0.25 | 0.193 |
| Options Flow | — | 0.077 |
| Macro Regime | — | 0.062 |
| Earnings Proximity | — | 0.038 |

Plugins cap at 30% total so core signals always dominate.

### Labels

| Label | Confluence Score |
|---|---|
| STRONG_BUY | >= 0.68 |
| BUY | >= 0.54 |
| HOLD | >= 0.46 |
| SELL | >= 0.32 |
| STRONG_SELL | < 0.32 |

---

## Self-Evolving Data Sources

Every Sunday night:

1. The **Analyzer** checks if any active experiment has been running 7+ days. It compares directional accuracy before and after activation, asks Claude to write a one-sentence lesson, and saves it to `data/cognition.json`. If accuracy dropped, the source is automatically disabled.

2. The **Researcher** reads all lessons + current accuracy + existing sources, then asks Claude to write a brand new `data_sources/*.py` module. The code is validated (AST parse, required exports check) before being written to disk.

3. On the next pipeline run, the new module is live.

To trigger the cycle manually:

```bash
python -c "from meta.runner import MetaRunner; MetaRunner().run()"
```

---

## Setup

### Requirements

- Python 3.11+
- 4GB+ RAM (Kronos-mini runs on CPU)
- Anthropic API key (required for discovery qualification and meta runner)
- Reddit account (optional, improves rate limits)
- Alpaca account (optional, for paper trading)

### Install

```bash
git clone https://github.com/jsho0/kronos-reddit-signal
cd kronos-reddit-signal
python -m venv venv
venv/Scripts/pip install -r requirements.txt   # Windows
# venv/bin/pip install -r requirements.txt      # Linux/Mac
```

### Environment

Copy `.env.example` to `.env` and fill in:

```env
ANTHROPIC_API_KEY=sk-ant-...        # required — discovery qualification + meta runner
REDDIT_CLIENT_ID=                   # optional — higher rate limits
REDDIT_CLIENT_SECRET=               # optional

# Discovery tuning (all optional, these are the defaults)
DISCOVERY_SUBREDDITS=wallstreetbets,stocks,investing,options,SecurityAnalysis,StockMarket,pennystocks,ValueInvesting
DISCOVERY_POST_LIMIT=100            # posts per subreddit per feed
DISCOVERY_LOOKBACK_HRS=24
DISCOVERY_MIN_BUZZ=5.0              # min buzz score to qualify
DISCOVERY_MIN_MARKET_CAP=100000000  # $100M minimum

PREDICTION_HORIZON_DAYS=5
KRONOS_MODEL_SIZE=mini              # mini (fast, CPU) | small | base

ALPACA_API_KEY=                     # optional, paper trading
ALPACA_SECRET_KEY=                  # optional
PAPER_TRADING_ENABLED=false
POSITION_SIZE_USD=1000

DATABASE_URL=sqlite:///./signal_store.db
```

### Kronos Model

Kronos is not on PyPI. Clone it as a submodule:

```bash
git submodule add https://github.com/shiyu-coder/Kronos kronos_src
```

Or clone manually and copy the `model/` folder to `kronos_src/`.

### Run

```bash
# Run discovery + pipeline once and exit
python main.py --once

# Run discovery only (build/update the dynamic watchlist)
python main.py --discover

# Run pipeline only on already-discovered tickers
python main.py --pipeline

# Debug: run pipeline on specific tickers, skip discovery
python main.py --pipeline --tickers AAPL TSLA NVDA

# Start daily scheduler (blocks — discovery 6am + pipeline 7am ET weekdays)
python main.py --schedule

# Launch dashboard
streamlit run dashboard/app.py
```

### Tests

```bash
venv/bin/python -m pytest tests/ -v   # Linux/Mac
venv\Scripts\python.exe -m pytest tests/ -v   # Windows
```

---

## Dashboard

Five tabs:

- **Discovery** — tickers qualified today, grouped by priority (HIGH / BUILDING / COOLING), with Claude's bull/bear thesis and buzz scores for each
- **Ticker Detail** — deep dive: company snapshot, Claude's analysis, Reddit post summaries, Kronos model inputs, confluence reasoning, confluence history chart
- **Pipeline Health** — run history, success rates, duration trends
- **Accuracy** — Kronos directional accuracy over time as next-day prices fill in
- **Paper Trading** — open positions, closed trades, P&L (requires Alpaca keys)

---

## VPS Deployment (Hostinger / systemd)

The live instance runs on a Hostinger Debian VPS as a systemd service.

`/etc/systemd/system/kronos-dashboard.service`:

```ini
[Unit]
Description=Kronos Reddit Signal Dashboard
After=network.target

[Service]
User=josh
WorkingDirectory=/home/kronos-reddit-signal
ExecStart=/home/kronos-reddit-signal/venv/bin/streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

Deploy updates:

```bash
git pull
sudo systemctl restart kronos-dashboard
```

To run the pipeline alongside the dashboard, use `python main.py --schedule` in a separate tmux session or second systemd service.

---

## Data Sources Plugin API

Drop a file in `data_sources/` with these exports:

```python
from data_sources import DataSourceResult

ENABLED = True
WEIGHT = 0.10          # 0.05–0.20, contributes this fraction to confluence score
NAME = "My Source"

def fetch(ticker: str, ohlcv_df=None) -> DataSourceResult:
    try:
        # your logic here
        return DataSourceResult(name=NAME, score=0.60, reasoning=["reason"], raw={})
    except Exception as exc:
        return DataSourceResult(name=NAME)   # neutral fallback, never raise
```

Picked up automatically on the next pipeline run. No registration needed.

---

## Tech Stack

| Component | Library |
|---|---|
| Time-series prediction | [Kronos](https://github.com/shiyu-coder/Kronos) (NeoQuasar/Kronos-mini) |
| Sentiment analysis | [FinBERT](https://huggingface.co/ProsusAI/finbert) via HuggingFace |
| Market data | yfinance |
| Technical indicators | ta |
| Reddit scraping | Public JSON API (no auth) / PRAW optional |
| Ticker qualification | Anthropic Claude (Sonnet) |
| Database | SQLite (WAL mode) via SQLAlchemy |
| Dashboard | Streamlit + Plotly |
| Paper trading | alpaca-py |
| Meta runner / LLM | Anthropic Claude (Sonnet for proposals, Haiku for analysis) |
| Scheduler | APScheduler |

---

## Roadmap

- [ ] PRAW migration (higher Reddit rate limits)
- [ ] Next-day price backfill cron job (populates Accuracy tab)
- [ ] Dashboard tab for experiments (show proposed/active/completed data sources)
- [ ] Tier 1: weight optimizer (nudge Kronos/Reddit/Technicals weights based on accuracy)
