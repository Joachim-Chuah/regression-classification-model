import pandas as pd
import yfinance as yf
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


def pull(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Pull adjusted OHLCV from yfinance, cache to data/raw/ as parquet."""
    cache_path = RAW_DIR / f"{ticker}_{start}_{end}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    # yfinance returns MultiIndex columns when downloading a single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = df.columns.str.lower()
    df.index.name = "date"
    df = df[["open", "high", "low", "close", "volume"]]

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    return df
