import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Kronos is not on PyPI. It must be cloned from GitHub and added to sys.path.
# Option A (submodule): git submodule add https://github.com/shiyu-coder/Kronos kronos_src
# Option B (manual): clone the repo and copy the `model/` folder here as `kronos_src/`
# Then this path injection makes `from kronos_src.model import ...` work.
KRONOS_SRC = Path(__file__).parent / "kronos_src"
if KRONOS_SRC.exists():
    sys.path.insert(0, str(KRONOS_SRC))

WATCHLIST = os.getenv("WATCHLIST", "AAPL,TSLA,NVDA,MSFT,AMD,META").split(",")
PREDICTION_HORIZON_DAYS = int(os.getenv("PREDICTION_HORIZON_DAYS", "5"))
REDDIT_LOOKBACK_HOURS = int(os.getenv("REDDIT_LOOKBACK_HOURS", "48"))
REDDIT_SUBREDDITS = os.getenv(
    "REDDIT_SUBREDDITS",
    "wallstreetbets,stocks,investing,SecurityAnalysis,options"
).split(",")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./signal_store.db")
KRONOS_MODEL_SIZE = os.getenv("KRONOS_MODEL_SIZE", "mini")
