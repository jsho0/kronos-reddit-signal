"""
Ticker extraction and false-positive filtering.

Finds potential stock tickers in text using regex, then filters against
a comprehensive blocklist of common words, financial jargon, and acronyms
that are not tradable equity symbols.

The blocklist is intentionally conservative — valid tickers that look like
real words (COST, LOVE, REAL) can slip through. The qualifier stage does
the final validation via yfinance.

Add new false positives to BLOCKLIST. It's just a set — easy to extend.
"""
import re
from typing import Generator

# Regex: 1-5 uppercase letters, optionally preceded by $ (e.g. $AAPL)
_TICKER_RE = re.compile(r'\$([A-Z]{1,5})\b|\b([A-Z]{2,5})\b')

BLOCKLIST: set[str] = {
    # English words commonly written in caps on Reddit
    "A", "I", "AM", "AN", "AS", "AT", "BE", "BY", "DO", "GO",
    "HE", "IF", "IN", "IS", "IT", "ME", "MY", "NO", "OF", "OK",
    "ON", "OR", "SO", "TO", "UP", "US", "WE",
    "ALL", "AND", "ARE", "BUT", "CAN", "DID", "FOR", "GET", "GOT",
    "HAD", "HAS", "HIM", "HIS", "HOW", "ITS", "LET", "LOL", "MAY",
    "NEW", "NOT", "NOW", "ONE", "OUR", "OUT", "OWN", "SAY", "SEE",
    "SET", "SHE", "THE", "TOO", "TWO", "USE", "WAS", "WAY", "WHO",
    "WHY", "YEP", "YES", "YET", "YOU",
    "ALSO", "BACK", "BEEN", "BOTH", "CALL", "CAME", "COME", "DOES",
    "DONE", "DOWN", "EACH", "EVEN", "EVER", "FEEL", "FROM", "GIVE",
    "GOES", "GOOD", "HAVE", "HERE", "HIGH", "HOLD", "JUST", "KEEP",
    "KNOW", "LAST", "LIKE", "LONG", "LOOK", "MADE", "MAKE", "MANY",
    "MORE", "MOST", "MOVE", "MUCH", "MUST", "NAME", "NEED", "NEXT",
    "ONLY", "OPEN", "OVER", "PAST", "PLAY", "REAL", "SAME", "SAID",
    "SELL", "SHOW", "SIDE", "SOME", "STAY", "SUCH", "SURE", "TAKE",
    "THAN", "THAT", "THEM", "THEN", "THEY", "THIS", "THUS", "TIME",
    "TYPE", "VERY", "VIEW", "WANT", "WELL", "WENT", "WERE", "WHAT",
    "WHEN", "WILL", "WITH", "WORK", "YOUR",
    "ABOUT", "AFTER", "AGAIN", "AHEAD", "AMONG", "ASKED", "BEING",
    "BELOW", "COULD", "ENDED", "EVERY", "FIRST", "GIVEN", "GOING",
    "GREAT", "LARGE", "LATER", "LEVEL", "LOWER", "MIGHT", "MONEY",
    "NEVER", "OFTEN", "OTHER", "QUITE", "RALLY", "RAISE", "RIGHT",
    "RISEN", "ROUND", "SINCE", "STILL", "STOCK", "THEIR", "THESE",
    "THING", "THINK", "THOSE", "THREE", "TODAY", "UNDER", "UNTIL",
    "USING", "WATCH", "WEEKS", "WHICH", "WHILE", "WOULD",

    # Trading / finance jargon
    "ATH", "ATL", "ATM", "OTM", "ITM", "IV", "DD", "DTE", "EPS",
    "PE", "PB", "ROE", "FCF", "DCF", "YOY", "QOQ", "TTM", "NTA",
    "NAV", "AUM", "ETF", "REIT", "SPAC", "IPO", "SPO", "FPO",
    "BULL", "BEAR", "MOON", "PUMP", "DUMP", "BAGS", "DIPS", "FOMO",
    "YOLO", "HODL", "FUD", "SHILL", "APES", "APES", "MEME", "GANG",
    "PUTS", "CALL", "CALLS", "THETA", "DELTA", "GAMMA", "VEGA",
    "HEDGE", "SWAP", "REPO", "BOND", "DEBT", "LOAN", "CASH",
    "LOSS", "GAIN", "RISK", "SAFE", "LONG", "SHORT", "COVER",
    "FLOAT", "SHARES", "PRICE", "VALUE", "RATIO", "YIELD",

    # Exchanges and regulators
    "NYSE", "AMEX", "CBOE", "CFTC", "FINRA", "FDIC", "SIPC",
    "SEC", "IMF", "BIS", "ECB", "BOJ", "BOE", "RBI",

    # Macro / economic terms
    "GDP", "CPI", "PPI", "PCE", "PMI", "ISM", "ADP", "NFP",
    "VIX", "SPX", "NDX", "RUT", "DJI", "DJIA", "JOLTS",
    "FOMC", "FED", "REPO", "TARP", "QE", "QT", "ZIRP", "NIRP",

    # Country / geography
    "USA", "EUR", "USD", "GBP", "JPY", "CNY", "CHF", "CAD",
    "AUD", "NZD", "HKD", "NYC", "DC", "EU", "UK", "US",

    # Roles / titles
    "CEO", "CFO", "CTO", "COO", "CMO", "CLO", "CRO", "EVP", "SVP",
    "VP", "MD", "GM", "HR", "PR", "IR", "RD",

    # Common abbreviations
    "TBH", "IMO", "IMHO", "TIL", "ELI", "TLDR", "FAQ", "AMA",
    "OP", "OC", "OG", "PM", "DM", "EOD", "EOW", "EOY",
    "AI", "ML", "IT", "API", "SDK", "SaaS", "IaaS", "PaaS",
    "EV", "AR", "VR", "NFT", "DAO", "DeFi", "DYOR",

    # Explicit false positives from the instruction doc
    "WHY", "AND", "PDT", "IEA",

    # Other known noise
    "ALL", "ANY", "ASK", "BIG", "BUY", "DAY", "DIP", "DIV",
    "EOD", "EST", "ETA", "EV", "FYI", "GAP", "GUY", "HIT",
    "HOT", "LOT", "LOW", "MAX", "MID", "MOM", "NET", "OTC",
    "POP", "POS", "POV", "PUT", "QED", "RAW", "RED", "REG",
    "REV", "ROI", "RUN", "RUT", "TAX", "TGT", "TOP", "TOS",
    "TOS", "TSA", "USD", "WAR", "WIN", "WTF", "YOY", "YTD",
}

# Symbols that look like words but ARE valid tradable tickers — never block these
WHITELIST: set[str] = {
    "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "TLT", "HYG",
    "XLF", "XLK", "XLE", "XLV", "XLU", "XLI", "XLB", "XLP",
    "VIX",  # tradable via derivatives
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOG", "META",
}


def extract_tickers(text: str) -> list[str]:
    """
    Extract potential ticker symbols from text.
    Returns deduplicated list, preserving first-appearance order.
    Dollar-prefixed mentions ($AAPL) are always included regardless of blocklist.
    """
    seen: dict[str, bool] = {}  # ticker -> dollar_prefixed
    for match in _TICKER_RE.finditer(text):
        dollar_group = match.group(1)  # $TICKER match
        plain_group = match.group(2)   # plain TICKER match
        if dollar_group:
            # Dollar-prefixed: always include, skip blocklist
            t = dollar_group.upper()
            seen[t] = True
        elif plain_group:
            t = plain_group.upper()
            if t not in seen:
                seen[t] = False

    results = []
    for ticker, dollar_prefixed in seen.items():
        if dollar_prefixed or ticker in WHITELIST:
            results.append(ticker)
        elif ticker not in BLOCKLIST and len(ticker) >= 2:
            results.append(ticker)
    return results


def count_mentions(text: str, ticker: str) -> int:
    """Count how many times a ticker is mentioned in text (case-insensitive on $)."""
    plain = len(re.findall(rf'\b{re.escape(ticker)}\b', text))
    dollar = len(re.findall(rf'\${re.escape(ticker)}\b', text))
    return plain + dollar
