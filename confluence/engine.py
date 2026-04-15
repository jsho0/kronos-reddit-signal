"""
Confluence engine: combines Kronos, Reddit, technicals, and data source plugins
into a single directional score and classifier label.

Core weights (before plugin adjustment):
  kronos_score     0.50
  reddit_score     0.25
  technicals_score 0.25

Active data source plugins each declare their own WEIGHT. The core weights are
scaled proportionally so all weights always sum to 1.0.

Example with options_flow (0.10) + macro (0.08) + earnings (0.05) active:
  plugin total = 0.23 → core scaled to 0.77
  kronos:     0.50 × 0.77 = 0.385
  reddit:     0.25 × 0.77 = 0.193
  technicals: 0.25 × 0.77 = 0.193

Final confluence_score = weighted sum, range 0.0–1.0.
0.5 = neutral. >0.5 = net bullish. <0.5 = net bearish.

Label thresholds:
  STRONG_BUY  : confluence >= 0.68
  BUY         : confluence >= 0.54
  HOLD        : confluence >= 0.46
  SELL        : confluence >= 0.32
  STRONG_SELL : confluence <  0.32

Plugins cap at MAX_PLUGIN_WEIGHT_TOTAL so core signals never get diluted below 70%.
"""
import importlib
import logging
import pkgutil
from dataclasses import dataclass, field
from pathlib import Path

from kronos_engine.output_schema import KronosPrediction
from kronos_engine.technicals import TechnicalIndicators
from reddit_scraper.sentiment import TickerSentiment

logger = logging.getLogger(__name__)

# Core weights — must sum to 1.0
WEIGHT_KRONOS = 0.50
WEIGHT_REDDIT = 0.25
WEIGHT_TECHNICALS = 0.25

# Plugins cannot claim more than this share of the total weight
MAX_PLUGIN_WEIGHT_TOTAL = 0.30

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
    confluence_score: float        # 0.0–1.0, 0.5=neutral
    label: str                     # STRONG_BUY | BUY | HOLD | SELL | STRONG_SELL
    reasoning: list[str]           # deduplicated bullets

    # Sub-scores for transparency
    kronos_score: float
    reddit_score: float
    technicals_score: float
    extra_scores: dict = field(default_factory=dict)   # {source_name: score}

    # Pass-through for storage
    kronos_prediction: KronosPrediction | None = None
    ticker_sentiment: TickerSentiment | None = None
    technicals: TechnicalIndicators | None = None


def _load_data_sources() -> list:
    """
    Scan the data_sources package and return all enabled plugin modules.
    Returns empty list if the package doesn't exist or no modules are enabled.
    """
    sources = []
    try:
        import data_sources as ds_pkg
        for finder, name, ispkg in pkgutil.iter_modules(ds_pkg.__path__):
            try:
                mod = importlib.import_module(f"data_sources.{name}")
                if getattr(mod, "ENABLED", False) and hasattr(mod, "fetch") and hasattr(mod, "WEIGHT"):
                    sources.append(mod)
                    logger.debug("confluence: loaded plugin '%s' (weight=%.2f)", name, mod.WEIGHT)
            except Exception as exc:
                logger.warning("confluence: failed to load plugin '%s': %s", name, exc)
    except ImportError:
        pass
    return sources


class ConfluenceEngine:
    """
    Combines Kronos, Reddit, technicals, and optional data source plugins
    into a single signal.

    All inputs are optional — missing inputs default to neutral (0.5).
    Plugins are scanned from data_sources/ at construction time.
    """

    def __init__(self):
        self._plugins = _load_data_sources()
        if self._plugins:
            names = [getattr(p, "NAME", p.__name__) for p in self._plugins]
            logger.info("confluence: %d plugin(s) active: %s", len(self._plugins), ", ".join(names))

    def score(
        self,
        ticker: str,
        kronos: KronosPrediction | None,
        sentiment: TickerSentiment | None,
        technicals: TechnicalIndicators | None,
        ohlcv_df=None,
    ) -> ConfluenceResult:
        reasoning: set[str] = set()

        kronos_score = _score_kronos(kronos, reasoning)
        reddit_score = _score_reddit(sentiment, reasoning)
        tech_score = _score_technicals(technicals, reasoning)

        # Run plugins
        extra_scores: dict[str, float] = {}
        plugin_weight_total = 0.0
        plugin_contributions: list[tuple[float, float]] = []  # (score, weight)

        for plugin in self._plugins:
            weight = float(getattr(plugin, "WEIGHT", 0.0))
            name = getattr(plugin, "NAME", plugin.__name__)
            try:
                result = plugin.fetch(ticker=ticker, ohlcv_df=ohlcv_df)
                for bullet in result.reasoning:
                    reasoning.add(bullet)
                extra_scores[name] = result.score
                plugin_contributions.append((result.score, weight))
                plugin_weight_total += weight
            except Exception as exc:
                logger.warning("confluence: plugin '%s' crashed on %s: %s", name, ticker, exc)
                extra_scores[name] = 0.5
                plugin_contributions.append((0.5, weight))
                plugin_weight_total += weight

        # Cap plugin weight so core stays >= 70%
        plugin_weight_total = min(plugin_weight_total, MAX_PLUGIN_WEIGHT_TOTAL)
        core_scale = 1.0 - plugin_weight_total

        # Recompute per-plugin weights proportionally if they exceed cap
        raw_total = sum(w for _, w in plugin_contributions)
        if raw_total > 0:
            plugin_contributions = [
                (s, w / raw_total * plugin_weight_total) for s, w in plugin_contributions
            ]

        # Weighted sum
        confluence = (
            WEIGHT_KRONOS * core_scale * kronos_score
            + WEIGHT_REDDIT * core_scale * reddit_score
            + WEIGHT_TECHNICALS * core_scale * tech_score
            + sum(s * w for s, w in plugin_contributions)
        )
        confluence = float(max(0.0, min(1.0, confluence)))

        label = _label(confluence)
        sorted_reasoning = sorted(reasoning)

        logger.info(
            "%s: confluence=%.3f (%s) | kronos=%.3f reddit=%.3f tech=%.3f plugins=%s",
            ticker, confluence, label, kronos_score, reddit_score, tech_score,
            {k: f"{v:.3f}" for k, v in extra_scores},
        )

        return ConfluenceResult(
            ticker=ticker,
            confluence_score=confluence,
            label=label,
            reasoning=sorted_reasoning,
            kronos_score=kronos_score,
            reddit_score=reddit_score,
            technicals_score=tech_score,
            extra_scores=extra_scores,
            kronos_prediction=kronos,
            ticker_sentiment=sentiment,
            technicals=technicals,
        )


# ------------------------------------------------------------------ #
#  Sub-scorers                                                         #
# ------------------------------------------------------------------ #

def _score_kronos(k: KronosPrediction | None, reasoning: set) -> float:
    if k is None:
        reasoning.add("Kronos unavailable — using neutral prior")
        return 0.5

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
    if s is None or s.post_count == 0:
        return 0.5

    score = 0.5 + s.signed_score * 0.4
    score = float(max(0.0, min(1.0, score)))

    post_str = "1 post" if s.post_count == 1 else f"{s.post_count} posts"

    if s.label == "positive":
        reasoning.add(f"Reddit sentiment positive across {post_str} (score {s.signed_score:+.2f})")
    elif s.label == "negative":
        reasoning.add(f"Reddit sentiment negative across {post_str} (score {s.signed_score:+.2f})")

    return score


def _score_technicals(t: TechnicalIndicators | None, reasoning: set) -> float:
    if t is None:
        reasoning.add("Technicals unavailable — using neutral prior")
        return 0.5

    votes = 0.0
    max_votes = 3.0

    if t.rsi_14 < 30:
        votes += 1
        reasoning.add(f"RSI oversold ({t.rsi_14:.1f})")
    elif t.rsi_14 > 70:
        votes -= 1
        reasoning.add(f"RSI overbought ({t.rsi_14:.1f})")

    if t.macd_signal == "bullish_cross":
        votes += 1
        reasoning.add("MACD bullish crossover")
    elif t.macd_signal == "bearish_cross":
        votes -= 1
        reasoning.add("MACD bearish crossover")

    if t.bb_position == "below_lower":
        votes += 1
        reasoning.add("Price below lower Bollinger Band (oversold)")
    elif t.bb_position == "above_upper":
        votes -= 1
        reasoning.add("Price above upper Bollinger Band (overbought)")

    if t.adx_14 > 25:
        votes *= 1.3
        max_votes *= 1.3

    if t.avg_volume_ratio > 1.5:
        reasoning.add(f"Volume spike ({t.avg_volume_ratio:.1f}x average)")
        if votes > 0:
            votes += 0.3
        elif votes < 0:
            votes -= 0.3

    return float(max(0.0, min(1.0, 0.5 + (votes / max_votes) * 0.35)))


def _label(score: float) -> str:
    for threshold, label in LABEL_THRESHOLDS:
        if score >= threshold:
            return label
    return "STRONG_SELL"
