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

TRAIN_START = "2015-01-01"
TRAIN_END = "2023-06-30"   # includes 2022 crash + 2023 recovery w/ inverted curve
VAL_START = "2023-07-01"
VAL_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2024-12-31"
