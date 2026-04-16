# Kronos Reddit Signal — Implementation Plan

This document is a feed-to-Claude-Code implementation guide. Execute tasks in order.
Each task is self-contained. Read the referenced files before making changes.

---

## Context

Kronos Reddit Signal is a self-improving stock signal system. It scrapes Reddit for
ticker buzz, qualifies candidates with Claude, runs Kronos (time-series model) +
Qwen sentiment + technicals, combines them into a confluence score, and paper trades
via Alpaca. A weekly meta runner proposes and activates new data source plugins.

Key directories:
- `reddit_scraper/` — scraping, sentiment, discovery, qualifier
- `pipeline/` — runner.py orchestrates per-ticker flow
- `confluence/` — scoring engine
- `data_sources/` — plugin directory (options_flow, macro, earnings + auto-generated)
- `storage/` — SQLAlchemy models (models.py) + SignalStore (store.py)
- `meta/` — researcher.py writes plugins, analyzer.py evaluates them
- `trading/` — Alpaca paper trading
- `dashboard/` — Streamlit app (5 tabs)

---

## Already Completed — Do Not Redo

- **Sentiment swap**: `reddit_scraper/sentiment.py` already uses Qwen via OpenRouter
  (not FinBERT). Public interface (`score_posts`, `analyze_ticker`, `SentimentResult`,
  `TickerSentiment`) is unchanged. Smoke tests updated.

---

## Task 1: Lookahead Audit + `as_of_date` Parameter

**Goal:** Every data fetch must be reproducible at a past date. When `as_of_date` is
supplied, no source may return data with a timestamp after that date. This is a
precondition for the forward-test harness and any future backtesting.

### 1a. Update the plugin API signature

**File:** `data_sources/__init__.py`

Read this file first. Find the `fetch` function type hint or docstring in the plugin
API. Add `as_of_date` to the documented signature:

```python
def fetch(ticker: str, ohlcv_df=None, as_of_date=None) -> DataSourceResult:
```

Document that `as_of_date` is a `datetime.date | None`. When `None`, treat as today.

### 1b. Update each built-in data source

**Files:** `data_sources/options_flow.py`, `data_sources/macro.py`,
`data_sources/earnings.py`

For each file, change the `fetch` signature to:
```python
def fetch(ticker: str, ohlcv_df=None, as_of_date=None) -> DataSourceResult:
```

**options_flow.py:** yfinance options chains are always current-day only — no
historical access. Accept `as_of_date` but add a log warning if it is set to a past
date: `logger.debug("options_flow: as_of_date ignored (yfinance options are live-only)")`.
No other changes needed.

**macro.py:** The `_get_macro()` helper uses `yf.download` with `period="5d"`. When
`as_of_date` is provided, replace `period="5d"` with `end=as_of_date.isoformat()` and
`start=(as_of_date - timedelta(days=7)).isoformat()` so VIX/SPY data is point-in-time.
Do the same for the 60-day SPY download. The `_cache` dict is keyed by process run —
change the cache key to include `as_of_date` so different dates don't collide:
`cache_key = str(as_of_date)`. Pass `as_of_date` through from `fetch` to `_get_macro`.

**earnings.py:** The source uses `date.today()` to compute days until earnings. Replace
`today = date.today()` with `today = as_of_date if as_of_date is not None else date.today()`.

### 1c. Update the confluence engine to pass `as_of_date`

**File:** `confluence/engine.py`

Read this file. Find where plugins are called (likely a loop over loaded modules calling
`module.fetch(ticker, ohlcv_df)`). Change to:
```python
module.fetch(ticker, ohlcv_df=ohlcv_df, as_of_date=as_of_date)
```

Add `as_of_date=None` parameter to the `score(...)` method signature and thread it
through.

### 1d. Trim OHLCV to `as_of_date` in the pipeline

**File:** `pipeline/runner.py`

In `_process_ticker`, after OHLCV is loaded (step 1), add:
```python
if as_of_date is not None:
    ohlcv_df = ohlcv_df[ohlcv_df.index <= pd.Timestamp(as_of_date)]
```

Add `as_of_date=None` to `_process_ticker` and `_run_ticker` signatures. Pass it
through to `self.confluence.score(...)`.

When called live (normal pipeline run), pass `as_of_date=date.today()` from
`PipelineRunner.run()`. This has no behavioral effect today but establishes the
discipline.

### 1e. Add a lookahead-bias test

**File:** `tests/test_lookahead.py` (create new file)

Write a pytest test that:
1. Picks `as_of_date = date(2024, 1, 15)` (a past date)
2. Calls `macro.fetch("SPY", as_of_date=as_of_date)` and asserts the raw VIX/SPY
   data in the result was fetched for that date range (check `result.raw` keys)
3. Calls `earnings.fetch("AAPL", as_of_date=as_of_date)` and asserts no exception
4. Documents that options_flow is exempt (live-only)

---

## Task 2: Signal Versioning + Forward-Test Harness

**Goal:** Every signal stored in the DB gets a version hash encoding the methodology
active at that moment. This lets us compare before/after for any change (sentiment
swap, new plugin, weight change) and start a clean forward-test dataset today.

### 2a. Add `signal_version` column to Signal model

**File:** `storage/models.py`

Add to the `Signal` class after `created_at`:
```python
signal_version = Column(String(64))  # hash of active methodology
```

Also add a `plugin_scores_json` column (used by Task 3 and Task 6):
```python
plugin_scores_json = Column(Text)  # JSON dict: {plugin_name: score}
```

**File:** `storage/db.py` (or wherever `init_db` lives — read it first)

SQLite supports `ALTER TABLE` for adding columns. After `Base.metadata.create_all(engine)`,
add migration logic that runs on startup:

```python
from sqlalchemy import text, inspect
inspector = inspect(engine)
existing = [c["name"] for c in inspector.get_columns("signals")]
with engine.connect() as conn:
    if "signal_version" not in existing:
        conn.execute(text("ALTER TABLE signals ADD COLUMN signal_version TEXT"))
    if "plugin_scores_json" not in existing:
        conn.execute(text("ALTER TABLE signals ADD COLUMN plugin_scores_json TEXT"))
    conn.commit()
```

### 2b. Compute and store signal version in the pipeline

**File:** `pipeline/runner.py`

Add a helper function:

```python
def _compute_signal_version(confluence_engine, model_size: str) -> str:
    import hashlib, json
    from data_sources import load_plugins
    plugins = [(p.NAME, p.WEIGHT, getattr(p, "SHADOW_MODE", False))
               for p in confluence_engine.plugins]
    payload = json.dumps({
        "model": model_size,
        "plugins": sorted(plugins),
    }, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]
```

Call this once per `PipelineRunner.run()` after `self.confluence = ConfluenceEngine()`
and store as `self.signal_version`. Pass it into `_build_signal_dict`.

In `_build_signal_dict`, add `"signal_version": signal_version` to the dict.

Also capture plugin scores from `ConfluenceResult` (see Task 3 for how they get there)
and store as `"plugin_scores_json": json.dumps(conf.plugin_scores)` if available.

### 2c. Create the forward-test script

**File:** `scripts/forward_test.py` (create new file)

```python
"""
Forward-test evaluator.

Queries signals from the DB, groups by signal_version, and reports:
- Hit rate per label tier (did STRONG_BUY actually go up next day?)
- Mean signed return per tier
- Sample sizes

Usage:
    python scripts/forward_test.py
    python scripts/forward_test.py --version abc123def456
    python scripts/forward_test.py --days 30
"""
```

The script should:
1. Connect to SQLite via SQLAlchemy (read `storage/db.py` for the engine pattern)
2. Query signals where `price_next_day IS NOT NULL` (these have outcomes)
3. Group by `signal_version` and `classifier_label`
4. For each group compute:
   - `hit_rate`: fraction where `price_next_day > price_at_signal` for BUY signals,
     or `price_next_day < price_at_signal` for SELL signals
   - `mean_return_pct`: mean of `(price_next_day - price_at_signal) / price_at_signal`
   - `n`: sample count
5. Print a table. Warn if n < 30 for any group (not statistically significant yet).

CLI args: `--version` (filter to a specific hash), `--days N` (last N days only).

---

## Task 3: Shadow Mode for New Data Sources

**Goal:** Claude-generated data sources run for 30+ signals at zero weight before being
promoted to full weight. At promotion time, a correlation check ensures the new source
isn't redundant with existing ones.

### 3a. Add `SHADOW_MODE` to the plugin API

**File:** `data_sources/__init__.py`

Document in the plugin API that modules may export `SHADOW_MODE = True`. When True,
the confluence engine runs the plugin (to accumulate scores) but contributes zero
weight to the final confluence score. Default is False (not present = active).

### 3b. Update the confluence engine to respect `SHADOW_MODE`

**File:** `confluence/engine.py`

When iterating over plugins, check `getattr(module, "SHADOW_MODE", False)`.
If True: call `fetch(...)`, store the score in a separate `shadow_scores` dict, but
do NOT include it in the weighted sum.

Attach both active plugin scores and shadow scores to `ConfluenceResult`:
```python
@dataclass
class ConfluenceResult:
    ...
    plugin_scores: dict  # {plugin_name: score} for ALL plugins (active + shadow)
```

This populates `plugin_scores_json` in the DB (Task 2) and enables the feature
importance tab (Task 6) and correlation check (Task 3d).

### 3c. Update the Researcher to generate shadow-mode plugins

**File:** `meta/researcher.py`

Find where the generated plugin template is constructed. Add `SHADOW_MODE = True` as
a required export in the generated code template, placed below `ENABLED = True`.

Also add `SHADOW_SIGNAL_COUNT = 0` to the template — this gets updated by the analyzer.

### 3d. Update the Analyzer to handle shadow promotion

**File:** `meta/analyzer.py`

Add a new function `_try_promote_shadow(experiment)`:

1. Query `plugin_scores_json` from the last 30+ signals where the plugin was active
   (use `activated_at` from the Experiment row to filter by date)
2. If fewer than 30 signals exist, return — not ready
3. Extract the shadow plugin's scores as a list. Extract scores for each existing
   active plugin as a list (same signal rows, same order)
4. Compute Pearson correlation between the shadow plugin's scores and each active
   plugin's scores
5. If any correlation > 0.85: archive the experiment with lesson
   "Redundant with {existing_plugin_name} (r={corr:.2f})" and set `ENABLED = False`
   in the module file
6. Otherwise: set `SHADOW_MODE = False` in the module file (regex replace, same
   pattern as `_disable_source`) and update experiment status to "active"

Add `SHADOW_MODE` to the `Experiment` model:

**File:** `storage/models.py`

Add to `Experiment`:
```python
shadow_signal_count = Column(Integer, default=0)
promoted_at = Column(String(32))
```

Run the same ALTER TABLE migration in `db.py` for these columns.

### 3e. Update the meta runner to call shadow promotion check

**File:** `meta/runner.py`

Read this file. After the Analyzer runs, add a step that checks all experiments in
shadow status and calls `_try_promote_shadow` for each one.

---

## Task 4: Tier-Based Position Sizing

**Goal:** Replace flat $1000 position sizing with label-based tiers.
STRONG_BUY = 1x base, BUY = 0.5x base, HOLD/SELL/STRONG_SELL = no entry.

**File:** `trading/alpaca_trader.py`

In `_open_position`, the current code uses `notional=self.position_size_usd` flat.
Change to:

```python
TIER_MULTIPLIERS = {
    "STRONG_BUY": 1.0,
    "BUY": 0.5,
}

# in _open_position:
multiplier = TIER_MULTIPLIERS.get(label, 0.0)
if multiplier == 0.0:
    return TradeResult(ticker=ticker, action="skipped", ...)
notional = self.position_size_usd * multiplier
```

Define `TIER_MULTIPLIERS` at module level. Update the docstring to document the tiers.
No other changes needed — `label` is already passed to `_open_position`.

---

## Task 5: Karma/Engagement Weighting for Reddit Posts

**Goal:** Downweight low-karma posts (likely bots or low-signal noise) and upweight
high-engagement posts in the sentiment aggregate.

**File:** `reddit_scraper/sentiment.py`

In `analyze_ticker`, the current aggregate is a simple mean of signed scores:
```python
mean_signed = sum(r.signed_score for r in per_post) / len(per_post)
```

Change to a weighted mean using post karma. The `posts` items have `.score`
(Reddit upvotes) and `.num_comments` or `.get("score")` for dict posts.

```python
import math

weights = []
for p in posts:
    karma = p.score if hasattr(p, "score") else p.get("score", 1)
    karma = max(karma, 1)  # floor at 1 to avoid zero weights
    weights.append(math.log1p(karma))

total_weight = sum(weights)
mean_signed = sum(
    r.signed_score * w for r, w in zip(per_post, weights)
) / total_weight
```

This uses `log1p(karma)` so a post with 1000 upvotes gets ~7x the weight of a post
with 1 upvote, not 1000x (logarithmic dampening). The rest of `analyze_ticker` is
unchanged.

---

## Task 6: Feature Importance Tab in Dashboard

**Goal:** Add a "Signal Breakdown" tab showing which plugins are actually contributing
to confluence scores across recent signals.

**File:** `dashboard/app.py`

Read this file first to understand the tab structure. Add a new tab "Signal Breakdown"
(or append to the existing tab list).

The tab should:

1. Query the last N signals (default 60, user-selectable via slider) where
   `plugin_scores_json IS NOT NULL`
2. Parse `plugin_scores_json` for each signal into a dict
3. Build a DataFrame: rows = signals, columns = plugin names, values = scores
4. Display:
   - **Mean score by plugin** — horizontal bar chart (Plotly). Color bars green if
     mean > 0.5 (bullish lean) red if < 0.5 (bearish lean), gray if ~neutral.
   - **Score distribution** — box plot per plugin showing spread
   - **Plugin activation table** — which plugins are currently ENABLED and which are
     in SHADOW_MODE (read from `data_sources/` directory dynamically)

For reading plugin metadata, scan `data_sources/*.py` and import each to read
`ENABLED`, `SHADOW_MODE`, `WEIGHT`, `NAME`. Do this with `importlib` the same way
`confluence/engine.py` loads plugins.

If no signals have `plugin_scores_json` yet (before Task 3 runs), show a placeholder
message: "No plugin score data yet — run the pipeline after enabling shadow mode tracking."

---

## Implementation Order

Execute tasks strictly in this order. Each one builds on the previous.

1. Task 1 (lookahead audit) — establishes data integrity for everything after
2. Task 2 (signal versioning + forward-test) — starts the evaluation clock
3. Task 3 (shadow mode) — needs `plugin_scores_json` from Task 2
4. Task 4 (position sizing) — independent, do anytime after Task 1
5. Task 5 (karma weighting) — independent, do anytime after Task 1
6. Task 6 (feature importance tab) — needs `plugin_scores_json` populated from Task 3

After completing all tasks, run:
```bash
python -m pytest tests/ -v
python smoke_test_pipeline.py
python smoke_test_reddit.py
```

---

## Constraints and Notes

- **Do not change** the public interfaces of `ConfluenceResult`, `TickerSentiment`,
  `SentimentResult`, or `SignalStore.upsert_signal` in breaking ways. Add fields,
  don't remove them.
- **SQLite migrations** must be additive (ALTER TABLE ADD COLUMN only). Never DROP
  or rename columns — the existing data is valuable.
- **All data source plugins** must never raise exceptions — always return
  `DataSourceResult(name=NAME)` as the neutral fallback.
- **The `as_of_date` param** is optional everywhere. When None, behavior is identical
  to current (uses today). Do not break existing callers.
- **ConfluenceEngine** is instantiated fresh each pipeline run so newly written
  plugins are picked up automatically. Don't cache it across runs.
- The project runs on a Hostinger Debian VPS. Use Unix paths. The venv is at
  `venv/` relative to project root.
