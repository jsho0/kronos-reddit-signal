"""
Qwen sentiment analyzer (via OpenRouter).

Replaces FinBERT with qwen/qwen3-8b via OpenRouter for WSB-aware sentiment.
Falls back to neutral on any API error so the pipeline never hard-fails.

Key details:
- Posts are scored in batches of up to BATCH_SIZE using a single prompt per batch.
- temperature=0 and a pinned model version for deterministic, debuggable results.
- Same public interface as the old FinBERT module: score_posts / analyze_ticker.
"""
import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

QWEN_MODEL = "qwen/qwen3-8b"
BATCH_SIZE = 20  # posts per API call

_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    return _client


_SYSTEM_PROMPT = (
    "You are a financial sentiment classifier. "
    "You understand Reddit slang, WSB memes, and sarcasm. "
    "For each post, output exactly one word: positive, negative, or neutral. "
    "No explanations."
)

_USER_TEMPLATE = (
    "Classify the sentiment of each post toward the stock being discussed. "
    "Reply with one word per line in the same order: positive, negative, or neutral.\n\n"
    "{posts}"
)


@dataclass
class SentimentResult:
    label: str     # "positive" | "negative" | "neutral"
    score: float   # confidence proxy: 1.0 for positive/negative, 0.0 for neutral
    signed_score: float = 0.0  # +score, -score, or 0


@dataclass
class TickerSentiment:
    ticker: str
    label: str
    score: float
    signed_score: float
    post_count: int
    per_post: list[SentimentResult] = field(default_factory=list)


def _parse_labels(raw: str, expected: int) -> list[str]:
    """Extract one label per line from model output, pad/trim to expected count."""
    valid = {"positive", "negative", "neutral"}
    lines = [l.strip().lower() for l in raw.strip().splitlines() if l.strip()]
    labels = [l if l in valid else "neutral" for l in lines]
    # pad or trim to match expected count
    while len(labels) < expected:
        labels.append("neutral")
    return labels[:expected]


def score_posts(texts: list[str], batch_size: int = BATCH_SIZE) -> list[SentimentResult]:
    """
    Score a list of text strings via Qwen on OpenRouter.
    Returns one SentimentResult per input, in the same order.
    Falls back to neutral on API failure.
    """
    if not texts:
        return []

    results: list[SentimentResult] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        labels = _score_batch(batch)
        for label in labels:
            if label == "positive":
                results.append(SentimentResult(label="positive", score=1.0, signed_score=1.0))
            elif label == "negative":
                results.append(SentimentResult(label="negative", score=1.0, signed_score=-1.0))
            else:
                results.append(SentimentResult(label="neutral", score=0.0, signed_score=0.0))

    return results


def _score_batch(texts: list[str]) -> list[str]:
    """Call Qwen for a single batch. Returns a list of label strings."""
    numbered = "\n".join(f"{i+1}. {t[:400]}" for i, t in enumerate(texts))
    prompt = _USER_TEMPLATE.format(posts=numbered)
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=QWEN_MODEL,
            temperature=0,
            max_tokens=len(texts) * 4,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        raw = response.choices[0].message.content or ""
        return _parse_labels(raw, len(texts))
    except Exception as exc:
        logger.warning("sentiment: Qwen batch failed, defaulting to neutral: %s", exc)
        return ["neutral"] * len(texts)


def analyze_ticker(
    ticker: str,
    posts,  # list[RedditPost] or list[dict] with title/body keys
    batch_size: int = BATCH_SIZE,
) -> TickerSentiment:
    """
    Score all posts for a ticker and return an aggregate TickerSentiment.

    Aggregate logic:
    - signed_score = mean of per-post signed scores
    - label = "positive" if mean > 0.05, "negative" if mean < -0.05, else "neutral"
    - score = abs(signed_score)
    """
    if not posts:
        return TickerSentiment(
            ticker=ticker, label="neutral", score=0.0, signed_score=0.0, post_count=0
        )

    texts = []
    for p in posts:
        if hasattr(p, "title"):
            title = p.title or ""
            body = p.body or ""
        else:
            title = p.get("title", "")
            body = p.get("body", "")
        text = (title + " " + body).strip()
        texts.append(text if text else title)

    per_post = score_posts(texts, batch_size=batch_size)

    mean_signed = sum(r.signed_score for r in per_post) / len(per_post)

    if mean_signed > 0.05:
        agg_label = "positive"
    elif mean_signed < -0.05:
        agg_label = "negative"
    else:
        agg_label = "neutral"

    return TickerSentiment(
        ticker=ticker,
        label=agg_label,
        score=abs(mean_signed),
        signed_score=mean_signed,
        post_count=len(posts),
        per_post=per_post,
    )
