from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd


@dataclass
class KronosPrediction:
    ticker: str
    predicted_close: float
    predicted_high: float
    predicted_low: float
    predicted_volume: float
    direction: str           # "bullish" | "bearish" | "neutral"
    pct_change: float        # predicted % change from last close
    confidence: float        # 0.0–1.0, derived from MC prediction std dev
    horizon_days: int
    timestamp: datetime
    raw_df: pd.DataFrame = field(default_factory=pd.DataFrame)
