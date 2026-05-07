from datetime import date

RANDOM_STATE = 42

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",  # Mag7 (NVDA/TSLA excluded — see below)
    "JPM", "GS", "MA",                          # financials
    "JNJ", "UNH",                               # healthcare
    "XOM", "CVX",                               # energy
    "DIS", "NFLX",                              # communication services
    "HD", "NKE",                                # consumer discretionary
    "SPY", "QQQ",                               # broad market ETFs
    # NVDA excluded: AI-boom regime creates train/test distribution shift
    # TSLA excluded: news-driven; structured features have near-zero predictive power
]

TICKER_SECTOR_ETF = {
    "AAPL": "XLK", "MSFT": "XLK", "GOOGL": "XLK", "AMZN": "XLK", "META": "XLK",
    "JPM": "XLF", "GS": "XLF", "MA": "XLF",
    "JNJ": "XLV", "UNH": "XLV",
    "XOM": "XLE", "CVX": "XLE",
    "SPY": "SPY",  "QQQ": "QQQ",
}

DEFAULT_HORIZON = 20        # trading days forward
NEUTRAL_THRESHOLD = 0.02    # ±2% — returns inside this band are labeled "neutral"

SWING_HORIZON = 5           # trading days forward for swing model
SWING_NEUTRAL_THRESHOLD = 0.01  # ±1% neutral band for 5-day returns

DATA_START = "2000-01-01"
DATA_END   = date.today().isoformat()   # always pull through today

TRAIN_START = "2000-01-01"
TRAIN_END   = "2024-12-31"   # 25 full years: includes 2004-06 rate cycle + 2008 crisis
VAL_START   = "2025-01-01"
VAL_END     = "2025-06-30"   # H1 2025 — isotonic calibration set
TEST_START  = "2025-07-01"   # H2 2025 onward — held-out test set
