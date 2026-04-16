"""
Analyzer agent: evaluates experiments and distills lessons.

Decision logic based on signal count:

  n_after < MIN_CONFIDENT_SIGNALS (30):
    - Accuracy dropped → ROTATE: disable temporarily, preserve data, bench it
    - Accuracy flat/up  → keep running, check again next week

  n_after >= MIN_CONFIDENT_SIGNALS:
    - Accuracy dropped → ARCHIVE: permanently disable, lesson written
    - Accuracy flat/up  → COMPLETE: keep enabled, lesson written

"Rotating" means: benched while we try something new, but not dead.
The runner re-evaluates rotating experiments once their signal count
crosses MIN_CONFIDENT_SIGNALS.

Status lifecycle:
  proposed → active → rotating → active (re-evaluated) → completed | archived
                               → completed | archived   (if confident)
"""
import logging
import os
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

from openai import OpenAI

from meta.cognition import add_lesson

logger = logging.getLogger(__name__)

MIN_ACTIVE_DAYS = 7          # days before first evaluation
MIN_CONFIDENT_SIGNALS = 30   # signals needed before permanent keep/kill decision

DATA_SOURCES_DIR = Path(__file__).parent.parent / "data_sources"

ANALYZER_PROMPT = """You are analyzing the performance of a new data source added to a stock signal system.

## Data Source: {source_name}
Description: {description}

## Accuracy Before Activation
{n_before} signals evaluated. Directional accuracy: {acc_before}

## Accuracy After Activation
{n_after} signals evaluated. Directional accuracy: {acc_after}
Accuracy delta: {delta:+.1%}
Confidence: {confidence}

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
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        response = client.chat.completions.create(
            model="qwen/qwen-2.5-7b-instruct",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("analyzer: Qwen call failed: %s", exc)
        return None


def _disable_source(source_name: str):
    """Set ENABLED = False in the module file (reversible)."""
    path = DATA_SOURCES_DIR / f"{source_name}.py"
    if not path.exists():
        return
    code = path.read_text(encoding="utf-8")
    updated = re.sub(r"^ENABLED\s*=\s*True", "ENABLED = False", code, flags=re.MULTILINE)
    path.write_text(updated, encoding="utf-8")
    logger.info("analyzer: disabled %s", source_name)


def _enable_source(source_name: str):
    """Set ENABLED = True in the module file."""
    path = DATA_SOURCES_DIR / f"{source_name}.py"
    if not path.exists():
        return
    code = path.read_text(encoding="utf-8")
    updated = re.sub(r"^ENABLED\s*=\s*False", "ENABLED = True", code, flags=re.MULTILINE)
    path.write_text(updated, encoding="utf-8")
    logger.info("analyzer: re-enabled %s", source_name)


def _promote_from_shadow(source_name: str):
    """Set SHADOW_MODE = False in the module file."""
    path = DATA_SOURCES_DIR / f"{source_name}.py"
    if not path.exists():
        return
    code = path.read_text(encoding="utf-8")
    updated = re.sub(r"^SHADOW_MODE\s*=\s*True", "SHADOW_MODE = False", code, flags=re.MULTILINE)
    path.write_text(updated, encoding="utf-8")
    logger.info("analyzer: promoted %s from shadow to active", source_name)


def _try_promote_shadow(experiment, store) -> bool:
    """
    Check if a shadow-mode plugin has enough signals to evaluate for promotion.

    Steps:
    1. Require 30+ signals since activation
    2. Compute Pearson correlation with every active plugin's scores
    3. If any correlation > 0.85 → archive as redundant
    4. Otherwise → promote (set SHADOW_MODE = False)

    Returns True if a decision was made (promoted or archived).
    """
    if not experiment.activated_at:
        return False

    activated = datetime.fromisoformat(experiment.activated_at.replace("Z", "+00:00"))
    activation_date = activated.strftime("%Y-%m-%d")

    try:
        import json as _json
        from sqlalchemy import select
        from storage.db import get_session
        from storage.models import Signal

        with get_session() as session:
            rows = session.execute(
                select(Signal).where(
                    Signal.signal_date >= activation_date,
                    Signal.plugin_scores_json.isnot(None),
                )
            ).scalars().all()
    except Exception as exc:
        logger.warning("analyzer: shadow check DB query failed for %s: %s", experiment.source_name, exc)
        return False

    if len(rows) < MIN_CONFIDENT_SIGNALS:
        # Update shadow_signal_count for dashboard visibility
        store.update_experiment(experiment.id, {"shadow_signal_count": len(rows)})
        logger.debug(
            "analyzer: shadow %s has %d/%d signals, waiting",
            experiment.source_name, len(rows), MIN_CONFIDENT_SIGNALS,
        )
        return False

    # Extract score vectors from plugin_scores_json
    shadow_scores = []
    active_scores: dict[str, list] = {}

    for row in rows:
        try:
            scores = _json.loads(row.plugin_scores_json)
        except Exception:
            continue
        if experiment.source_name not in scores:
            continue
        shadow_scores.append(scores[experiment.source_name])
        for plugin_name, score in scores.items():
            if plugin_name != experiment.source_name:
                active_scores.setdefault(plugin_name, []).append(score)

    if len(shadow_scores) < MIN_CONFIDENT_SIGNALS:
        logger.debug("analyzer: shadow %s: not enough score data yet (%d)", experiment.source_name, len(shadow_scores))
        return False

    # Pearson correlation check
    def _pearson(xs, ys):
        n = min(len(xs), len(ys))
        if n < 2:
            return 0.0
        xs, ys = xs[:n], ys[:n]
        mx = sum(xs) / n
        my = sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        dx = sum((x - mx) ** 2 for x in xs) ** 0.5
        dy = sum((y - my) ** 2 for y in ys) ** 0.5
        if dx == 0 or dy == 0:
            return 0.0
        return num / (dx * dy)

    redundant_with = None
    max_corr = 0.0
    for plugin_name, scores_list in active_scores.items():
        if len(scores_list) < MIN_CONFIDENT_SIGNALS:
            continue
        corr = abs(_pearson(shadow_scores, scores_list))
        if corr > max_corr:
            max_corr = corr
            if corr > 0.85:
                redundant_with = plugin_name

    now = datetime.now(timezone.utc).isoformat()

    if redundant_with:
        lesson = f"Redundant with {redundant_with} (r={max_corr:.2f}) — archived without promotion."
        _disable_source(experiment.source_name)
        store.update_experiment(experiment.id, {
            "status": "archived",
            "shadow_signal_count": len(shadow_scores),
            "lesson": lesson,
            "completed_at": now,
        })
        add_lesson(
            source_name=experiment.source_name,
            title=f"{experiment.source_name} (shadow, redundant)",
            lesson=lesson,
            accuracy_delta=0.0,
        )
        logger.info("analyzer: shadow %s archived — %s", experiment.source_name, lesson)
        return True

    # Passed correlation check — promote
    _promote_from_shadow(experiment.source_name)
    store.update_experiment(experiment.id, {
        "status": "active",
        "shadow_signal_count": len(shadow_scores),
        "promoted_at": now,
        "activated_at": experiment.activated_at,  # preserve original activation date
    })
    logger.info(
        "analyzer: shadow %s promoted to active (%d signals, max_corr=%.2f)",
        experiment.source_name, len(shadow_scores), max_corr,
    )
    return True


def _write_lesson(experiment, acc_before, n_before, acc_after, n_after, delta, confident, days_active):
    confidence_str = f"confident ({n_after} signals)" if confident else f"preliminary ({n_after} signals, need {MIN_CONFIDENT_SIGNALS})"
    prompt = ANALYZER_PROMPT.format(
        source_name=experiment.source_name,
        description=experiment.description or "No description",
        n_before=n_before,
        acc_before=f"{acc_before:.1%}" if acc_before is not None else "unknown",
        n_after=n_after,
        acc_after=f"{acc_after:.1%}" if acc_after is not None else "unknown",
        delta=delta,
        confidence=confidence_str,
    )
    lesson_text = _call_claude(prompt) or (
        f"{experiment.source_name}: accuracy {delta:+.1%} over {n_after} signals ({confidence_str})."
    )
    add_lesson(
        source_name=experiment.source_name,
        title=f"{experiment.source_name} ({days_active}d, {'confident' if confident else 'preliminary'})",
        lesson=lesson_text,
        accuracy_delta=delta,
    )
    return lesson_text


def analyze_experiment(experiment, store) -> bool:
    """
    Evaluate a single active experiment.
    Returns True if action was taken (rotated, completed, or archived).
    Returns False if not ready yet or no action needed.
    """
    if not experiment.activated_at:
        return False

    activated = datetime.fromisoformat(experiment.activated_at.replace("Z", "+00:00"))
    days_active = (datetime.now(timezone.utc) - activated).days

    if days_active < MIN_ACTIVE_DAYS:
        logger.debug(
            "analyzer: %s only %d/%d days active, skipping",
            experiment.source_name, days_active, MIN_ACTIVE_DAYS,
        )
        return False

    before_date = (activated - timedelta(days=30)).strftime("%Y-%m-%d")
    activation_date = activated.strftime("%Y-%m-%d")
    acc_before, n_before = _get_accuracy_for_period(store, before_date)
    acc_after, n_after = _get_accuracy_for_period(store, activation_date)

    if n_after < 5:
        logger.info("analyzer: %s only %d post-activation signals, waiting", experiment.source_name, n_after)
        return False

    delta = (acc_after or 0.0) - (acc_before or 0.0)
    confident = n_after >= MIN_CONFIDENT_SIGNALS
    negative = delta < -0.02

    if not confident and not negative:
        # Looking good so far, not enough data — let it keep running
        logger.debug("analyzer: %s positive/neutral so far (%d signals), continuing", experiment.source_name, n_after)
        return False

    # Write a lesson regardless of outcome
    lesson_text = _write_lesson(experiment, acc_before, n_before, acc_after, n_after, delta, confident, days_active)

    if confident:
        # Enough data — make permanent decision
        if negative:
            _disable_source(experiment.source_name)
            final_status = "archived"
            outcome = "archived (confident, accuracy dropped)"
        else:
            final_status = "completed"
            outcome = "completed (confident, accuracy held)"

        store.update_experiment(experiment.id, {
            "status": final_status,
            "accuracy_before": acc_before,
            "accuracy_after": acc_after,
            "n_signals_before": n_before,
            "n_signals_after": n_after,
            "lesson": lesson_text,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
    else:
        # Not enough data but early signs are negative — rotate out, try something new
        _disable_source(experiment.source_name)
        store.update_experiment(experiment.id, {
            "status": "rotating",
            "accuracy_before": acc_before,
            "accuracy_after": acc_after,
            "n_signals_before": n_before,
            "n_signals_after": n_after,
            "lesson": lesson_text,
        })
        outcome = f"rotating (preliminary, {n_after}/{MIN_CONFIDENT_SIGNALS} signals, delta {delta:+.1%})"

    logger.info("analyzer: %s → %s", experiment.source_name, outcome)
    return True


def recheck_rotating(experiment, store) -> bool:
    """
    Re-evaluate a rotating experiment once it has accumulated enough signals.
    If confident now, make permanent decision. Otherwise leave it rotating.
    Returns True if a permanent decision was made.
    """
    if not experiment.activated_at:
        return False

    activated = datetime.fromisoformat(experiment.activated_at.replace("Z", "+00:00"))
    activation_date = activated.strftime("%Y-%m-%d")
    acc_after, n_after = _get_accuracy_for_period(store, activation_date)

    if n_after < MIN_CONFIDENT_SIGNALS:
        logger.debug(
            "analyzer: rotating %s still only %d/%d signals, waiting",
            experiment.source_name, n_after, MIN_CONFIDENT_SIGNALS,
        )
        return False

    # We now have enough data — make the permanent call
    acc_before = experiment.accuracy_before or 0.0
    delta = (acc_after or 0.0) - acc_before
    days_active = (datetime.now(timezone.utc) - activated).days

    lesson_text = _write_lesson(experiment, acc_before, experiment.n_signals_before or 0,
                                acc_after, n_after, delta, confident=True, days_active=days_active)

    if delta < -0.02:
        # Confirmed bad — leave disabled, archive
        final_status = "archived"
        outcome = f"archived (confirmed after {n_after} signals)"
    else:
        # Actually fine — re-enable and complete
        _enable_source(experiment.source_name)
        final_status = "completed"
        outcome = f"completed (recovered after {n_after} signals, delta {delta:+.1%})"

    store.update_experiment(experiment.id, {
        "status": final_status,
        "accuracy_after": acc_after,
        "n_signals_after": n_after,
        "lesson": lesson_text,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })

    logger.info("analyzer: rotating %s → %s", experiment.source_name, outcome)
    return True
