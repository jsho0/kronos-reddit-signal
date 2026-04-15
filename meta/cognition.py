"""
Cognition store: persistent lessons learned from experiments.

Simple JSON file at data/cognition.json.
No FAISS needed at this scale — lessons are short text, we just pass them all to the prompt.

Each lesson:
  source_name: str       — which data source the lesson is about
  title: str
  lesson: str            — what was learned (1-3 sentences)
  accuracy_delta: float  — accuracy change (positive = improvement)
  recorded_at: str       — ISO UTC
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

COGNITION_PATH = Path(__file__).parent.parent / "data" / "cognition.json"


def load_lessons() -> list[dict]:
    if not COGNITION_PATH.exists():
        return []
    try:
        with open(COGNITION_PATH) as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("cognition: failed to load: %s", exc)
        return []


def add_lesson(source_name: str, title: str, lesson: str, accuracy_delta: float = 0.0):
    lessons = load_lessons()
    entry = {
        "source_name": source_name,
        "title": title,
        "lesson": lesson,
        "accuracy_delta": accuracy_delta,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    lessons.append(entry)
    COGNITION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(COGNITION_PATH, "w") as f:
        json.dump(lessons, f, indent=2)
    logger.info("cognition: added lesson '%s'", title)


def format_for_prompt(limit: int = 8) -> str:
    lessons = load_lessons()
    if not lessons:
        return "No lessons recorded yet."
    recent = lessons[-limit:]
    lines = []
    for l in recent:
        delta_str = f" (acc delta: {l['accuracy_delta']:+.1%})" if l.get("accuracy_delta") else ""
        lines.append(f"- [{l.get('source_name', '?')}] {l.get('lesson', '')}{delta_str}")
    return "\n".join(lines)
