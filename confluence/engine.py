"""
Confluence engine: combines Kronos, Reddit sentiment, and technicals
into a single directional score and classifier label.

Score breakdown (all sub-scores normalized 0.0-1.0):
  kronos_score     weight=0.50  — predicted direction + confidence + magnitude
  reddit_score     weight=0.25  — FinBERT signed sentiment
  technicals_score weight=0.25  — RSI, MACD, BB, ADX, volume

Final confluence_score = weighted sum, range 0.0-1.0.
0.5 = neutral. >0.5 = net bullish. <0.5 = net bearish.

Label thresholds (applied after determining net direction):
  STRONG_BUY  : confluence >= 0.72
  BUY         : confluence >= 0.58
  HOLD        : confluence >= 0.42
  SELL        : confluence >= 0.28
  STRONG_SELL : confluence <  0.28

Reasoning bullets are collected as a set() and deduplicated before output.
"""
import logging
from dataclasses import dataclass, field

from kronos_engine.output_schema import KronosPrediction
from kronos_engine.technicals import TechnicalIndicators
from reddit_scraper.sentiment import TickerSentiment

logger = logging.getLogger(__name__)

# Confluence weights — must sum to 1.0
WEIGHT_KRONOS = 0.50
WEIGHT_REDDIT = 0.25
WEIGHT_TECHNICALS = 0.25

# Label thresholds
LABEL_THRESHOLDS = [
    (0.68, "STRONG_BUY"),
    (0.54, "BUY"),
    (0.46, "HOLD"),
    (0.32, "SELL"),
    (0.00, "STRONG_SELL"),
]


@dataclass
class ConfluenceResult:
    ticker: str
    confluence_score: float        # 0.0-1.0, 0.5=neutral
    label: str                     # STRONG_BUY | BUY | HOLD | SELL | STRONG_SELL
    reasoning: list[str]           # deduplicated bullets explaining the label

    # Sub-scores for dashboard transparency
    kronos_score: float
    reddit_score: float
    technicals_score: float

    # Pass-through for storage
    kronos_prediction: KronosPrediction | None = None
    ticker_sentiment: TickerSentiment | None = None
    technicals: TechnicalIndicators | None = None


class ConfluenceEngine:
    """
    Combines Kronos, Reddit, and technicals into a single signal.

    All three inputs are optional — the engine degrades gracefully if one
    is unavailable (e.g. Kronos timeout, no Reddit posts).
    Missing inputs default to neutral (0.5 sub-score, no reasoning bullets).
    """

    def score(
        self,
        ticker: str,
        kronos: KronosPrediction | None,
        sentiment: TickerSentiment | None,
        technicals: TechnicalIndicators | None,
    ) -> ConfluenceResult:
        reasoning: set[str] = set()

        kronos_score = _score_kronos(kronos, reasoning)
        reddit_score = _score_reddit(sentiment, reasoning)
        tech_score = _score_technicals(technicals, reasoning)

        confluence = (
            WEIGHT_KRONOS * kronos_score
            + WEIGHT_REDDIT * reddit_score
            + WEIGHT_TECHNICALS * tech_score
        )
        confluence = float(max(0.0, min(1.0, confluence)))

        label = _label(confluence)
        sorted_reasoning = sorted(reasoning)  # deterministic order

        logger.info(
            "%s: confluence=%.3f (%s) | kronos=%.3f reddit=%.3f tech=%.3f",
            ticker, confluence, label, kronos_score, reddit_score, tech_score,
        )

        return ConfluenceResult(
            ticker=ticker,
            confluence_score=confluence,
            label=label,
            reasoning=sorted_reasoning,
            kronos_score=kronos_score,
            reddit_score=reddit_score,
            technicals_score=tech_score,
            kronos_prediction=kronos,
            ticker_sentiment=sentiment,
            technicals=technicals,
        )


# ------------------------------------------------------------------ #
#  Sub-scorers                                                         #
# ------------------------------------------------------------------ #

def _score_kronos(k: KronosPrediction | None, reasoning: set) -> float:
    """
    Map Kronos prediction to 0.0-1.0.

    Formula:
      base = 0.5 (neutral center)
      direction_push = pct_change clamped to ±10% → mapped to ±0.35
      confidence_scale = k.confidence (already 0-1)
      score = 0.5 + direction_push * confidence_scale

    A bullish 5% prediction with 0.9 confidence → ~0.66
    A bearish 3% prediction with 0.5 confidence → ~0.42
    """
    if k is None:
        reasoning.add("Kronos unavailable — using neutral prior")
        return 0.5

    # Clamp pct_change to ±10% range, map to ±0.35 push
    clamped = max(-0.10, min(0.10, k.pct_change))
    direction_push = (clamped / 0.10) * 0.35
    score = 0.5 + direction_push * k.confidence
    score = float(max(0.0, min(1.0, score)))

    if k.direction == "bullish":
        reasoning.add(
            f"Kronos predicts +{k.pct_change*100:.1f}% over {k.horizon_days}d "
            f"(confidence {k.confidence:.0%})"
        )
    elif k.direction == "bearish":
        reasoning.add(
            f"Kronos predicts {k.pct_change*100:.1f}% over {k.horizon_days}d "
            f"(confidence {k.confidence:.0%})"
        )
    else:
        reasoning.add(f"Kronos sees neutral price action (confidence {k.confidence:.0%})")

    return score


def _score_reddit(s: TickerSentiment | None, reasoning: set) -> float:
    """
    Map TickerSentiment to 0.0-1.0.

    signed_score is already in [-1, 1].
    Map: 0.5 + signed_score * 0.4
    So max bullish (+1.0) → 0.9, max bearish (-1.0) → 0.1.

    No posts → neutral 0.5 with no reasoning bullet.
    """
    if s is None or s.post_count == 0:
        return 0.5

    score = 0.5 + s.signed_score * 0.4
    score = float(max(0.0, min(1.0, score)))

    label_str = s.label
    if s.post_count == 1:
        post_str = "1 post"
    else:
        post_str = f"{s.post_count} posts"

    if s.label == "positive":
        reasoning.add(f"Reddit sentiment positive across {post_str} (score {s.signed_score:+.2f})")
    elif s.label == "negative":
        reasoning.add(f"Reddit sentiment negative across {post_str} (score {s.signed_score:+.2f})")
    # neutral: no bullet — not worth mentioning

    return score


def _score_technicals(t: TechnicalIndicators | None, reasoning: set) -> float:
    """
    Score technicals using RSI, MACD, BB position, ADX, and volume.

    Each indicator votes +1 (bullish), -1 (bearish), or 0 (neutral).
    Final score = 0.5 + (sum of votes / max_votes) * 0.35

    Votes and thresholds:
      RSI < 30           → +1 (oversold, mean reversion)
      RSI > 70           → -1 (overbought)
      30 <= RSI <= 70    → 0
      MACD bullish_cross → +1
      MACD bearish_cross → -1
      BB below_lower     → +1 (oversold)
      BB above_upper     → -1 (overbought)
      ADX > 25           → magnitude vote: amplifies direction by 0.5
      Volume ratio > 1.5 → magnitude vote: +0.5 (unusual activity)
    """
    if t is None:
        reasoning.add("Technicals unavailable — using neutral prior")
        return 0.5

    votes = 0.0
    max_votes = 3.0  # RSI + MACD + BB

    # RSI
    if t.rsi_14 < 30:
        votes += 1
        reasoning.add(f"RSI oversold ({t.rsi_14:.1f})")
    elif t.rsi_14 > 70:
        votes -= 1
        reasoning.add(f"RSI overbought ({t.rsi_14:.1f})")

    # MACD
    if t.macd_signal == "bullish_cross":
        votes += 1
        reasoning.add("MACD bullish crossover")
    elif t.macd_signal == "bearish_cross":
        votes -= 1
        reasoning.add("MACD bearish crossover")

    # Bollinger Bands
    if t.bb_position == "below_lower":
        votes += 1
        reasoning.add("Price below lower Bollinger Band (oversold)")
    elif t.bb_position == "above_upper":
        votes -= 1
        reasoning.add("Price above upper Bollinger Band (overbought)")

    # ADX amplifier: strong trend → weight votes more
    if t.adx_14 > 25:
        votes *= 1.3
        max_votes *= 1.3

    # Volume amplifier: unusual volume → slightly more conviction
    if t.avg_volume_ratio > 1.5:
        reasoning.add(f"Volume spike ({t.avg_volume_ratio:.1f}x average)")
        # nudge in direction of existing votes
        if votes > 0:
            votes += 0.3
        elif votes < 0:
            votes -= 0.3

    score = 0.5 + (votes / max_votes) * 0.35
    return float(max(0.0, min(1.0, score)))


def _label(score: float) -> str:
    for threshold, label in LABEL_THRESHOLDS:
        if score >= threshold:
            return label
    return "STRONG_SELL"
