"""
Researcher agent: uses Claude to propose new data source modules.

Given current accuracy stats + existing sources + cognition store lessons,
asks Claude to write a complete Python module that implements the DataSourceResult
interface and adds a new signal to the confluence engine.

The generated code is:
  1. Validated via ast.parse() before writing to disk
  2. Written to data_sources/{source_name}.py
  3. Logged as a new Experiment record with status="proposed"

Activation: the meta runner promotes proposed → active on the next weekly cycle.
"""
import ast
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import anthropic

from meta.cognition import format_for_prompt

logger = logging.getLogger(__name__)

DATA_SOURCES_DIR = Path(__file__).parent.parent / "data_sources"

# One existing source to show Claude the expected pattern
PATTERN_EXAMPLE = """
EXAMPLE — options_flow.py (follow this pattern exactly):

```python
import logging
import yfinance as yf
from data_sources import DataSourceResult

logger = logging.getLogger(__name__)

ENABLED = True
WEIGHT = 0.10
NAME = "Options Flow"

def fetch(ticker: str, ohlcv_df=None) -> DataSourceResult:
    try:
        # ... implementation ...
        return DataSourceResult(name=NAME, score=0.55, reasoning=["..."], raw={})
    except Exception as exc:
        logger.debug("options_flow: %s failed: %s", ticker, exc)
        return DataSourceResult(name=NAME)
```
"""

RESEARCHER_PROMPT = """You are a quantitative trading researcher building signal data sources for a stock prediction system.

## System Overview

The system generates BUY/SELL signals by combining:
- Kronos time-series model (50% weight)
- Reddit Qwen sentiment (25% weight)
- Technical indicators — RSI, MACD, Bollinger Bands, ADX, volume (25% weight)

These weights are then adjusted to accommodate your new data source.

## Current Accuracy

{accuracy_stats}

## Already Implemented Data Sources

{existing_sources}

## Lessons from Previous Experiments

{lessons}

## Your Task

Propose ONE new data source. Write it as a complete Python module following the pattern below.

{pattern}

Rules:
1. Import ONLY from: yfinance, pandas, numpy, ta, requests, and Python standard library
2. No API keys required — use only free, publicly accessible data
3. WEIGHT must be between 0.05 and 0.15
4. The fetch() function must NEVER raise an exception — wrap everything in try/except
5. On any failure, return DataSourceResult(name=NAME) (neutral score 0.5)
6. score must be in [0.0, 1.0] where 0.5 = neutral, >0.5 = bullish, <0.5 = bearish
7. Keep it focused: one clear signal, not a kitchen sink

Good signal ideas (pick the most promising one that isn't already implemented):
- Short interest ratio (yfinance: ticker.info["shortRatio"] or "shortPercentOfFloat")
- Analyst price target vs current price (yfinance: ticker.info["targetMeanPrice"])
- Institutional ownership change (yfinance: ticker.institutional_holders)
- 52-week high/low position (already have price_vs_52w_high in technicals — try something else)
- Sector momentum relative to the individual stock
- Historical earnings beat/miss pattern

Respond with ONLY the Python code. No explanation, no markdown fences, no preamble. Just the raw Python.
"""


def _get_existing_sources() -> str:
    sources = []
    for path in DATA_SOURCES_DIR.glob("*.py"):
        if path.name == "__init__.py":
            continue
        sources.append(path.stem)
    if not sources:
        return "None yet."
    return ", ".join(sorted(sources))


def _get_accuracy_summary(store) -> str:
    try:
        stats = store.get_accuracy_stats()
        if stats["total"] == 0:
            return "No accuracy data yet (system is new)."
        return (
            f"Directional accuracy: {stats['accuracy']:.1%} "
            f"({stats['correct']}/{stats['total']} signals resolved)"
        )
    except Exception:
        return "Accuracy stats unavailable."


def _extract_source_name(code: str) -> str:
    """Pull the module name from the NAME = '...' line."""
    match = re.search(r'^NAME\s*=\s*["\'](.+?)["\']', code, re.MULTILINE)
    if match:
        raw = match.group(1).lower().replace(" ", "_").replace("/", "_")
        raw = re.sub(r"[^a-z0-9_]", "", raw)
        return raw[:40]
    return "researcher_source"


def propose(store) -> dict | None:
    """
    Ask Claude to propose a new data source module.
    Returns a dict with keys: source_name, description, module_path, code
    or None if proposal failed.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("researcher: ANTHROPIC_API_KEY not set")
        return None

    accuracy = _get_accuracy_summary(store)
    existing = _get_existing_sources()
    lessons = format_for_prompt(limit=6)

    prompt = RESEARCHER_PROMPT.format(
        accuracy_stats=accuracy,
        existing_sources=existing,
        lessons=lessons,
        pattern=PATTERN_EXAMPLE,
    )

    logger.info("researcher: calling Claude to propose new data source...")

    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        code = response.content[0].text.strip()
    except Exception as exc:
        logger.error("researcher: Claude API call failed: %s", exc)
        return None

    # Strip markdown fences if Claude added them anyway
    if code.startswith("```"):
        lines = code.split("\n")
        code = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    # Validate: must be parseable Python
    try:
        ast.parse(code)
    except SyntaxError as exc:
        logger.error("researcher: generated code has syntax error: %s", exc)
        return None

    # Must contain required exports
    if "ENABLED" not in code or "WEIGHT" not in code or "def fetch(" not in code:
        logger.error("researcher: generated code missing required exports")
        return None

    source_name = _extract_source_name(code)

    # Avoid overwriting existing sources
    target_path = DATA_SOURCES_DIR / f"{source_name}.py"
    if target_path.exists():
        source_name = f"{source_name}_v2"
        target_path = DATA_SOURCES_DIR / f"{source_name}.py"

    # Write to disk
    target_path.write_text(code, encoding="utf-8")
    logger.info("researcher: wrote new data source → %s", target_path)

    # Extract NAME constant for description
    name_match = re.search(r'^NAME\s*=\s*["\'](.+?)["\']', code, re.MULTILINE)
    display_name = name_match.group(1) if name_match else source_name

    return {
        "source_name": source_name,
        "description": f"Auto-proposed by Researcher agent: {display_name}",
        "module_path": f"data_sources/{source_name}.py",
        "code": code,
        "status": "proposed",
        "proposed_at": datetime.now(timezone.utc).isoformat(),
    }
