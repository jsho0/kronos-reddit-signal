"""
Analyzer agent: evaluates completed experiments and distills lessons.

After a data source has been active for MIN_ACTIVE_DAYS, the Analyzer:
  1. Compares accuracy before vs after activation
  2. Calls Claude to write a 1-2 sentence lesson
  3. Stores the lesson in the cognition store
  4. Marks the experiment as "completed"

If accuracy improved → keep the module (ENABLED stays True)
If accuracy dropped → set ENABLED = False in the module, archive experiment
"""
import logging
import os
import re
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

import anthropic

from meta.cognition import add_lesson

logger = logging.getLogger(__name__)

MIN_ACTIVE_DAYS = 7   # how long a source must run before evaluation
DATA_SOURCES_DIR = Path(__file__).parent.parent / "data_sources"

ANALYZER_PROMPT = """You are analyzing the performance of a new data source added to a stock signal system.

## Data Source: {source_name}
Description: {description}

## Accuracy Before Activation
{n_before} signals evaluated. Directional accuracy: {acc_before}

## Accuracy After Activation
{n_after} signals evaluated. Directional accuracy: {acc_after}
Accuracy delta: {delta:+.1%}

## Task
Write ONE concise sentence (max 25 words) that captures what this experiment taught us.
Focus on: did it help, hurt, or make no difference, and why that might be.

Respond with ONLY the lesson sentence. No preamble, no label, just the sentence.
"""


def _get_accuracy_for_period(store, since_date: str) -> tuple[float | None, int]:
    """Return (accuracy, n_signals) for signals since since_date with known outcomes."""
    try:
        from sqlalchemy import select
        from storage.db import get_session
        from storage.models import Signal

        with get_session() as session:
            rows = session.execute(
                select(Signal).where(
                    Signal.signal_date >= since_date,
                    Signal.price_at_signal.isnot(None),
                    Signal.price_next_day.isnot(None),
                )
            ).scalars().all()

            if not rows:
                return None, 0

            cases = [(r.kronos_direction, r.price_at_signal, r.price_next_day) for r in rows]

        correct = sum(
            1 for direction, p_at, p_next in cases
            if (p_next > p_at and direction == "bullish")
            or (p_next < p_at and direction == "bearish")
        )
        return correct / len(cases), len(cases)
    except Exception as exc:
        logger.warning("analyzer: accuracy query failed: %s", exc)
        return None, 0


def _call_claude(prompt: str) -> str | None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as exc:
        logger.warning("analyzer: Claude call failed: %s", exc)
        return None


def _disable_source(source_name: str):
    """Set ENABLED = False in the module file."""
    path = DATA_SOURCES_DIR / f"{source_name}.py"
    if not path.exists():
        return
    code = path.read_text(encoding="utf-8")
    updated = re.sub(r"^ENABLED\s*=\s*True", "ENABLED = False", code, flags=re.MULTILINE)
    path.write_text(updated, encoding="utf-8")
    logger.info("analyzer: disabled %s (accuracy dropped)", source_name)


def analyze_experiment(experiment, store) -> bool:
    """
    Evaluate a single active experiment.
    Returns True if the experiment was completed, False if not ready yet.
    """
    if not experiment.activated_at:
        return False

    activated = datetime.fromisoformat(experiment.activated_at.replace("Z", "+00:00"))
    days_active = (datetime.now(timezone.utc) - activated).days

    if days_active < MIN_ACTIVE_DAYS:
        logger.debug(
            "analyzer: %s only active %d/%d days, skipping",
            experiment.source_name, days_active, MIN_ACTIVE_DAYS,
        )
        return False

    # Accuracy before: 30 days before activation
    before_date = (activated - timedelta(days=30)).strftime("%Y-%m-%d")
    activation_date = activated.strftime("%Y-%m-%d")
    acc_before, n_before = _get_accuracy_for_period(store, before_date)

    # Accuracy after: since activation
    acc_after, n_after = _get_accuracy_for_period(store, activation_date)

    if n_after < 5:
        logger.info("analyzer: %s has only %d post-activation signals, waiting", experiment.source_name, n_after)
        return False

    delta = (acc_after or 0.0) - (acc_before or 0.0)

    # Ask Claude for a lesson
    prompt = ANALYZER_PROMPT.format(
        source_name=experiment.source_name,
        description=experiment.description or "No description",
        n_before=n_before,
        acc_before=f"{acc_before:.1%}" if acc_before is not None else "unknown",
        n_after=n_after,
        acc_after=f"{acc_after:.1%}" if acc_after is not None else "unknown",
        delta=delta,
    )
    lesson_text = _call_claude(prompt) or (
        f"Added {experiment.source_name}: accuracy changed by {delta:+.1%} over {n_after} signals."
    )

    # Store lesson
    add_lesson(
        source_name=experiment.source_name,
        title=f"{experiment.source_name} experiment ({days_active}d)",
        lesson=lesson_text,
        accuracy_delta=delta,
    )

    # If accuracy dropped meaningfully, disable the source
    keep = acc_before is None or delta >= -0.02
    if not keep:
        _disable_source(experiment.source_name)

    # Mark experiment complete
    store.update_experiment(experiment.id, {
        "status": "completed" if keep else "archived",
        "accuracy_before": acc_before,
        "accuracy_after": acc_after,
        "n_signals_before": n_before,
        "n_signals_after": n_after,
        "lesson": lesson_text,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })

    logger.info(
        "analyzer: %s → %s (delta %+.1%%, lesson: %s)",
        experiment.source_name,
        "kept" if keep else "disabled",
        delta * 100,
        lesson_text[:60],
    )
    return True
