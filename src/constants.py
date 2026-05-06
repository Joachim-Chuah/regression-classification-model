from datetime import date

RANDOM_STATE = 42

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",  # mega-cap tech
    "JPM", "GS",                                # financials
    "JNJ", "UNH",                               # healthcare
    "XOM", "CVX",                               # energy
    "SPY", "QQQ",                               # broad market ETFs
]

TICKER_SECTOR_ETF = {
    "AAPL": "XLK", "MSFT": "XLK", "GOOGL": "XLK", "AMZN": "XLK", "META": "XLK",
    "JPM": "XLF", "GS": "XLF",
    "JNJ": "XLV", "UNH": "XLV",
    "XOM": "XLE", "CVX": "XLE",
    "SPY": "SPY",  "QQQ": "QQQ",
}

DEFAULT_HORIZON = 20        # trading days forward
NEUTRAL_THRESHOLD = 0.02    # ±2% — returns inside this band are labeled "neutral"

SWING_HORIZON = 5           # trading days forward for swing model
SWING_NEUTRAL_THRESHOLD = 0.01  # ±1% neutral band for 5-day returns

DATA_START = "2015-01-01"
DATA_END   = date.today().isoformat()   # always pull through today

TRAIN_START = "2015-01-01"
TRAIN_END   = "2024-12-31"   # 10 full years: 2015-2024 incl. 2022 bear + 2023 recovery
VAL_START   = "2025-01-01"
VAL_END     = "2025-06-30"   # H1 2025 — isotonic calibration set
TEST_START  = "2025-07-01"   # H2 2025 onward — held-out test set
