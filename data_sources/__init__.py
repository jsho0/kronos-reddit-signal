"""
Data source plugin system.

Each module in this directory (besides __init__.py) is a plugin.
Required exports per module:
  ENABLED: bool        — set False to disable without deleting
  SHADOW_MODE: bool    — optional; when True, scores are recorded but weight=0 (default False)
  WEIGHT: float        — contribution to confluence score (0.05–0.20)
  NAME: str            — display name for dashboard/reasoning
  fetch(ticker: str, ohlcv_df: pd.DataFrame | None, as_of_date: date | None) -> DataSourceResult

fetch() signature:
  def fetch(ticker: str, ohlcv_df=None, as_of_date=None) -> DataSourceResult

  as_of_date: datetime.date | None
    When provided, the plugin must not return data with a timestamp after this date.
    When None, treat as today (live mode). This enables point-in-time reproducibility.

DataSourceResult:
  name:      str
  score:     float  # 0.0–1.0, 0.5 = neutral
  reasoning: list[str]
  raw:       dict   # arbitrary raw values for storage/debugging
"""
from dataclasses import dataclass, field


@dataclass
class DataSourceResult:
    name: str
    score: float = 0.5
    reasoning: list = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    def __post_init__(self):
        self.score = float(max(0.0, min(1.0, self.score)))
