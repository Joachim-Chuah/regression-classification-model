from datetime import date

RANDOM_STATE = 42

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",  # Mag7 (NVDA/TSLA excluded — see below)
    "JPM", "GS", "MA", "V",                    # financials
    "JNJ", "UNH", "MRK", "ABBV",              # healthcare / pharma
    "XOM", "CVX",                               # energy
    "AVGO", "PANW",                             # semis/infra + cybersecurity
    "DIS", "NFLX",                              # communication services
    "HD", "NKE",                                # consumer discretionary
    "SPY", "QQQ",                               # broad market ETFs
    # NVDA excluded: AI-boom regime creates train/test distribution shift
    # TSLA excluded: news-driven; structured features have near-zero predictive power
    # LLY excluded: GLP-1 regime shift 2022-23 causes same distribution problem as NVDA
    # CRWD excluded: IPO 2019, insufficient history pre-bull-market
]

TICKER_SECTOR_ETF = {
    # ── existing training tickers ───────────────────────────────────────────
    "AAPL": "XLK", "MSFT": "XLK", "GOOGL": "XLK", "AMZN": "XLK", "META": "XLK",
    "JPM": "XLF", "GS": "XLF", "MA": "XLF",
    "JNJ": "XLV", "UNH": "XLV",
    "XOM": "XLE", "CVX": "XLE",
    "SPY": "SPY",  "QQQ": "QQQ",

    # ── Semiconductors / AI hardware (SOXX basket → XLK proxy) ─────────────
    "NVDA": "XLK", "AMD": "XLK", "INTC": "XLK", "MU": "XLK", "ARM": "XLK",
    "MRVL": "XLK", "AVGO": "XLK", "QCOM": "XLK", "TXN": "XLK",
    "AMAT": "XLK", "LRCX": "XLK", "KLAC": "XLK", "ADI": "XLK",
    "ASML": "XLK", "TSM": "XLK",

    # ── Big Tech / AI infra / SaaS (XLK basket) ────────────────────────────
    "ORCL": "XLK", "CRM": "XLK", "ADBE": "XLK", "CSCO": "XLK",
    "DELL": "XLK", "HPE": "XLK",

    # ── Cybersecurity (HACK basket → XLK proxy) ────────────────────────────
    "CRWD": "XLK", "PANW": "XLK", "ZS": "XLK", "FTNT": "XLK",
    "OKTA": "XLK", "S": "XLK", "CYBR": "XLK", "TENB": "XLK",
    "NET": "XLK", "CHKP": "XLK",

    # ── AI software / data (predict watchlist) ──────────────────────────────
    "PLTR": "XLK", "NOW": "XLK", "SNOW": "XLK",

    # ── Energy (XLE basket) ─────────────────────────────────────────────────
    "COP": "XLE", "EOG": "XLE", "SLB": "XLE", "MPC": "XLE",
    "PSX": "XLE", "VLO": "XLE", "HAL": "XLE", "OXY": "XLE",

    # ── Healthcare (XLV basket) ─────────────────────────────────────────────
    "LLY": "XLV", "ABBV": "XLV", "MRK": "XLV", "PFE": "XLV",
    "BMY": "XLV", "TMO": "XLV", "DHR": "XLV", "CVS": "XLV",

    # ── Biotech (XBI basket → XLV proxy, closest available) ────────────────
    "MRNA": "XLV", "BIIB": "XLV", "REGN": "XLV", "VRTX": "XLV",
    "GILD": "XLV", "ALNY": "XLV", "BMRN": "XLV",

    # ── Financials (XLF basket) ─────────────────────────────────────────────
    "WFC": "XLF", "BLK": "XLF", "SCHW": "XLF", "C": "XLF",
    "AXP": "XLF", "V": "XLF", "BAC": "XLF", "MS": "XLF",
    # Industrials/Defense/Clean Energy/EV/Consumer/Real Estate/Materials
    # baskets intentionally omitted — their sector ETFs (XLI, ITA, ICLN,
    # DRIV, XLY, XLRE, XLB) are not in SECTOR_ETFS so rel_strength_vs_sector
    # would be None anyway. Add those ETFs to data.SECTOR_ETFS to unlock.
}

DEFAULT_HORIZON = 20        # trading days forward
NEUTRAL_THRESHOLD = 0.02    # ±2% — returns inside this band are labeled "neutral"

SWING_HORIZON = 5           # trading days forward for swing model
SWING_NEUTRAL_THRESHOLD = 0.01  # ±1% neutral band for 5-day returns

DAILY_HORIZON = 3           # trading days forward for daily model
DAILY_NEUTRAL_THRESHOLD = 0.005  # ±0.5% neutral band for 3-day returns

DATA_START = "2000-01-01"
DATA_END   = date.today().isoformat()   # always pull through today

TRAIN_START = "2000-01-01"
TRAIN_END   = "2024-12-31"   # 25 full years: includes 2004-06 rate cycle + 2008 crisis
VAL_START   = "2025-01-01"
VAL_END     = "2025-06-30"   # H1 2025 — isotonic calibration set
TEST_START  = "2025-07-01"   # H2 2025 onward — held-out test set
