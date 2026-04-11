"""
FinBERT sentiment analyzer.

Model: ProsusAI/finbert (financial domain BERT, 3-class: positive/negative/neutral)
Loaded lazily on first call and cached at module level.

Key details:
- Truncate to 512 TOKENS (not 512 chars). The tokenizer handles this.
- Batch size 32: good balance of throughput vs memory on CPU.
- Input = title + " " + body. Title carries most of the signal.
- Returns per-post label + score, plus an aggregate for the ticker.
"""
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

FINBERT_MODEL = "ProsusAI/finbert"

_pipeline = None  # cached transformers pipeline


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline as hf_pipeline
        logger.info("Loading FinBERT (%s)...", FINBERT_MODEL)
        _pipeline = hf_pipeline(
            "text-classification",
            model=FINBERT_MODEL,
            tokenizer=FINBERT_MODEL,
            truncation=True,
            max_length=512,       # 512 tokens, not chars
            padding=True,
            top_k=None,           # return all 3 class scores
        )
        logger.info("FinBERT loaded")
    return _pipeline


@dataclass
class SentimentResult:
    label: str     # "positive" | "negative" | "neutral"
    score: float   # magnitude of the winning label, 0.0-1.0
    # Signed composite: positive=+score, negative=-score, neutral=0
    # Useful for averaging across posts.
    signed_score: float = 0.0


@dataclass
class TickerSentiment:
    ticker: str
    label: str           # majority label across all posts
    score: float         # mean |signed_score| of all posts
    signed_score: float  # mean signed_score (negative = net bearish)
    post_count: int
    per_post: list[SentimentResult] = field(default_factory=list)


def score_posts(texts: list[str], batch_size: int = 32) -> list[SentimentResult]:
    """
    Run FinBERT on a list of text strings.
    Returns one SentimentResult per input, in the same order.

    texts: list of strings (title + body concatenated by caller)
    """
    if not texts:
        return []

    pipe = _get_pipeline()
    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # pipeline returns list of list of dicts when top_k=None
        batch_output = pipe(batch)
        for post_scores in batch_output:
            # post_scores: [{"label": "positive", "score": 0.9}, ...]
            best = max(post_scores, key=lambda x: x["score"])
            label = best["label"].lower()
            raw_score = float(best["score"])
            signed = raw_score if label == "positive" else (-raw_score if label == "negative" else 0.0)
            results.append(SentimentResult(
                label=label,
                score=raw_score,
                signed_score=signed,
            ))

    return results


def analyze_ticker(
    ticker: str,
    posts,  # list[RedditPost] or list[dict] with title/body keys
    batch_size: int = 32,
) -> TickerSentiment:
    """
    Score all posts for a ticker and return an aggregate TickerSentiment.

    Aggregate logic:
    - signed_score = mean of per-post signed scores
    - label = "positive" if mean > 0.05, "negative" if mean < -0.05, else "neutral"
    - score = abs(signed_score)  — strength of the aggregate signal

    The 0.05 threshold prevents a single loud post from flipping a neutral corpus.
    """
    if not posts:
        return TickerSentiment(
            ticker=ticker,
            label="neutral",
            score=0.0,
            signed_score=0.0,
            post_count=0,
        )

    # Build text inputs: title + body. FinBERT truncates to 512 tokens.
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
