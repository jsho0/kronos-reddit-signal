import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Kronos model path injection
KRONOS_SRC = Path(__file__).parent / "kronos_src"
if KRONOS_SRC.exists():
    sys.path.insert(0, str(KRONOS_SRC))

# ------------------------------------------------------------------ #
#  Kronos model                                                        #
# ------------------------------------------------------------------ #
KRONOS_MODEL_SIZE = os.getenv("KRONOS_MODEL_SIZE", "mini")
PREDICTION_HORIZON_DAYS = int(os.getenv("PREDICTION_HORIZON_DAYS", "5"))

# MC samples by priority tier (used in discovery mode)
MC_SAMPLES_BY_PRIORITY = {
    "NEW":     5,
    "MEDIUM":  10,
    "HIGH":    20,
    "COOLING": 5,
}

# ------------------------------------------------------------------ #
#  Reddit discovery                                                    #
# ------------------------------------------------------------------ #
DISCOVERY_SUBREDDITS = os.getenv(
    "DISCOVERY_SUBREDDITS",
    "wallstreetbets,stocks,investing,options,SecurityAnalysis,StockMarket,pennystocks,ValueInvesting"
).split(",")

DISCOVERY_POST_LIMIT   = int(os.getenv("DISCOVERY_POST_LIMIT", "100"))   # posts per subreddit per feed
DISCOVERY_LOOKBACK_HRS = int(os.getenv("DISCOVERY_LOOKBACK_HRS", "24"))  # hours to look back
DISCOVERY_MIN_BUZZ     = float(os.getenv("DISCOVERY_MIN_BUZZ", "5.0"))   # min buzz score to qualify
DISCOVERY_MIN_MARKET_CAP = int(os.getenv("DISCOVERY_MIN_MARKET_CAP", "100000000"))  # $100M

# Reddit sentiment lookback for pipeline (per-ticker sentiment fetch)
REDDIT_LOOKBACK_HOURS = int(os.getenv("REDDIT_LOOKBACK_HOURS", "48"))
REDDIT_SUBREDDITS = os.getenv(
    "REDDIT_SUBREDDITS",
    "wallstreetbets,stocks,investing,SecurityAnalysis,options"
).split(",")

# ------------------------------------------------------------------ #
#  Priority / decay thresholds                                        #
# ------------------------------------------------------------------ #
PRIORITY_HIGH_DAYS    = int(os.getenv("PRIORITY_HIGH_DAYS", "4"))    # streak to reach HIGH
PRIORITY_MEDIUM_DAYS  = int(os.getenv("PRIORITY_MEDIUM_DAYS", "2"))  # streak to reach MEDIUM
PRIORITY_COOLING_DAYS = int(os.getenv("PRIORITY_COOLING_DAYS", "1")) # days missed → COOLING
PRIORITY_DROP_DAYS    = int(os.getenv("PRIORITY_DROP_DAYS", "3"))    # days missed → DROPPED

# ------------------------------------------------------------------ #
#  Storage                                                             #
# ------------------------------------------------------------------ #
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./signal_store.db")

# ------------------------------------------------------------------ #
#  Alpaca paper trading                                                #
# ------------------------------------------------------------------ #
ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
PAPER_TRADING_ENABLED = os.getenv("PAPER_TRADING_ENABLED", "false").lower() == "true"
POSITION_SIZE_USD = float(os.getenv("POSITION_SIZE_USD", "1000"))
