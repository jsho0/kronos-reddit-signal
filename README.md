# Kronos Reddit Signal

A self-improving stock signal system. Combines a time-series foundation model, Reddit sentiment, and technical indicators into a directional confluence score — and evolves its own data sources weekly using Claude.

Live dashboard: [http://83.136.219.215:8501](http://83.136.219.215:8501)

---

## What It Does

Every weekday at 8am ET, the pipeline runs across a 30-ticker watchlist:

1. Downloads OHLCV price history (yfinance, cached in SQLite)
2. Runs **Kronos** (a time-series foundation model) to predict direction and magnitude over the next 5 days
3. Scrapes Reddit (`wallstreetbets`, `stocks`, `investing`, `options`) for posts mentioning the ticker
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
kronos_engine/       Kronos model wrapper, OHLCV fetcher, technicals
reddit_scraper/      Reddit public JSON scraper + FinBERT sentiment
confluence/          Scoring engine — combines all signals, plugin-aware
data_sources/        Plugin directory — each .py is a data source
  options_flow.py    Put/call ratio from yfinance options chain
  macro.py           VIX level + SPY 20-day trend
  earnings.py        Proximity to next earnings date
  <generated>.py     Auto-written by the Researcher agent
meta/                ASI-Evolve layer
  researcher.py      Claude agent — proposes new data source modules
  analyzer.py        Evaluates experiment outcomes, distills lessons
  cognition.py       Persistent JSON lesson store
  runner.py          Orchestrates the weekly evolution cycle
pipeline/            Runner + APScheduler (daily + weekly jobs)
storage/             SQLAlchemy models + SignalStore (SQLite, WAL mode)
trading/             Alpaca paper trading integration
dashboard/           Streamlit dashboard
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

The most interesting part of this project. Every Sunday night:

1. The **Analyzer** checks if any active experiment has been running 7+ days. It compares directional accuracy before and after activation, asks Claude-Haiku to write a one-sentence lesson, and saves it to `data/cognition.json`. If accuracy dropped, the source is automatically disabled.

2. The **Researcher** reads all lessons + current accuracy + existing sources, then asks Claude-Sonnet to write a brand new `data_sources/*.py` module. The code is validated (AST parse, required exports check) before being written to disk.

3. On the next pipeline run, the new module is live.

Over time the system builds up a library of signals it proposed, tested, and learned from — without you writing any code.

To trigger the cycle manually:

```bash
python -c "from meta.runner import MetaRunner; MetaRunner().run()"
```

---

## Setup

### Requirements

- Python 3.11+
- 4GB+ RAM (Kronos-mini runs on CPU)
- Reddit account (optional, improves rate limits)
- Alpaca account (optional, for paper trading)
- Anthropic API key (required for meta runner)

### Install

```bash
git clone https://github.com/jsho0/kronos-reddit-signal
cd kronos-reddit-signal
python -m venv venv
venv/Scripts/pip install -r requirements.txt   # Windows
# source venv/bin/activate && pip install -r requirements.txt  # Linux/Mac
```

### Environment

Copy `.env.example` to `.env` and fill in:

```env
ANTHROPIC_API_KEY=sk-ant-...        # required for meta runner
REDDIT_CLIENT_ID=                   # optional
REDDIT_CLIENT_SECRET=               # optional

WATCHLIST=AAPL,TSLA,NVDA,MSFT,AMD,META,GOOG,...
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
# Single run (all tickers)
python main.py --once

# Single run (specific tickers)
python main.py --once --tickers AAPL TSLA NVDA

# Start daily scheduler (blocks, runs 8am ET weekdays + meta runner Sundays)
python main.py --schedule

# Launch dashboard
streamlit run dashboard/app.py
```

---

## Dashboard

Five tabs:

- **Signals** — today's full watchlist with labels, confluence scores, and sub-scores
- **Ticker Detail** — deep dive on any ticker: signal banner, reasoning bullets, confluence history, technicals
- **Accuracy** — Kronos directional accuracy over time as `price_next_day` fills in
- **Paper Trading** — open positions, closed trades, P&L (requires Alpaca keys)
- **Pipeline Health** — run history, error rates, duration trends

---

## VPS Deployment (Hostinger / systemd)

The live instance runs on a Hostinger Debian VPS as a systemd service.

`/etc/systemd/system/kronos-dashboard.service`:

```ini
[Unit]
Description=Kronos Reddit Signal Dashboard
After=network.target

[Service]
User=root
WorkingDirectory=/root/kronos-reddit-signal
ExecStart=/root/kronos-reddit-signal/venv/bin/streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

Deploy updates:

```bash
git pull
sudo systemctl restart kronos-dashboard
```

To run the pipeline as a separate process alongside the dashboard, use `python main.py --schedule` in a second service or tmux session.

---

## Data Sources Plugin API

Adding a new data source manually is one file. Drop it in `data_sources/` with these exports:

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

It will be picked up automatically on the next pipeline run. No registration needed.

---

## Tech Stack

| Component | Library |
|---|---|
| Time-series prediction | [Kronos](https://github.com/shiyu-coder/Kronos) (NeoQuasar/Kronos-mini) |
| Sentiment analysis | [FinBERT](https://huggingface.co/ProsusAI/finbert) via HuggingFace |
| Market data | yfinance |
| Technical indicators | ta |
| Reddit scraping | Public JSON API (no auth) / PRAW optional |
| Database | SQLite (WAL mode) via SQLAlchemy |
| Dashboard | Streamlit + Plotly |
| Paper trading | alpaca-py |
| Meta runner / LLM | Anthropic Claude (Sonnet for proposals, Haiku for analysis) |
| Scheduler | APScheduler |

---

## Roadmap

- [ ] PRAW migration (higher Reddit rate limits)
- [ ] Next-day price backfill cron job (populates Accuracy tab)
- [ ] Quarterly ticker list refresh script
- [ ] Dashboard tab for experiments (show proposed/active/completed data sources)
- [ ] Tier 1: weight optimizer (nudge Kronos/Reddit/Technicals weights based on accuracy)
