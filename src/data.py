import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLE"]


def pull(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Pull adjusted OHLCV from yfinance, cache to data/raw/ as parquet.

    Caching policy: historical data (end < today) is cached permanently since
    it never changes. Requests with end == today are never cached — the market
    may still be open and the day's close isn't final until after 4pm ET.
    """
    today = date.today().isoformat()
    safe = ticker.replace("^", "").replace("=", "").replace("-", "_")
    cache_path = RAW_DIR / f"{safe}_{start}_{end}.parquet"

    use_cache = end < today   # never cache today's data
    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    # yfinance returns MultiIndex columns when downloading a single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = df.columns.str.lower()
    df.index.name = "date"
    df = df[["open", "high", "low", "close", "volume"]]

    if use_cache:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
    return df


def _zscore(series: pd.Series, window: int = 252) -> pd.Series:
    """Trailing z-score: (value - rolling_mean) / rolling_std.

    Converts absolute levels (VIX=30, yield=4.5%) into regime-agnostic
    deviations from recent history. A reading of +2 means 2 std above the
    trailing year — far more informative to a tree model than the raw level.
    """
    mean = series.rolling(window).mean()
    std = series.rolling(window).std().replace(0, np.nan)
    return (series - mean) / std


def pull_macro(start: str, end: str) -> pd.DataFrame:
    """Pull market-wide macro series and sector ETF returns.

    Returns a DataFrame indexed by date with columns:
        vix_zscore_252d, vix_change_5d,
        spy_return_20d, spy_vs_200ma,
        yield_10y_zscore_252d, yield_change_20d, yield_curve_zscore_252d,
        XLK_return_20d, XLF_return_20d, XLV_return_20d, XLE_return_20d

    VIX and 10Y yield are expressed as trailing 252-day z-scores so the
    model sees relative stress (±σ from recent history) rather than absolute
    levels that shift with the macro regime.

    All values at time t use only data available at or before t.
    Forward-filled so it aligns cleanly with any equity ticker.
    """
    vix = pull("^VIX", start, end)["close"].rename("vix")
    tnx = pull("^TNX", start, end)["close"].rename("yield_10y")   # 10-year yield
    irx = pull("^IRX", start, end)["close"].rename("yield_3m")    # 13-week T-bill
    spy = pull("SPY",  start, end)["close"].rename("spy_close")

    macro = pd.concat([vix, tnx, irx, spy], axis=1).ffill()

    macro["vix_zscore_252d"]        = _zscore(macro["vix"])
    macro["vix_change_5d"]          = macro["vix"].pct_change(5)
    macro["spy_return_20d"]         = macro["spy_close"].pct_change(20)
    macro["spy_vs_200ma"]           = macro["spy_close"] / macro["spy_close"].rolling(200).mean() - 1
    macro["yield_10y_zscore_252d"]  = _zscore(macro["yield_10y"])
    macro["yield_change_20d"]       = macro["yield_10y"].diff(20)
    raw_curve                        = macro["yield_10y"] - macro["yield_3m"]
    macro["yield_curve_zscore_252d"] = _zscore(raw_curve)   # inverted but for how long?

    for etf in SECTOR_ETFS:
        etf_close = pull(etf, start, end)["close"]
        macro[f"{etf}_return_20d"] = etf_close.pct_change(20).reindex(macro.index).ffill()

    keep = [
        "vix_zscore_252d", "vix_change_5d",
        "spy_return_20d", "spy_vs_200ma",
        "yield_10y_zscore_252d", "yield_change_20d", "yield_curve_zscore_252d",
        "XLK_return_20d", "XLF_return_20d", "XLV_return_20d", "XLE_return_20d",
    ]
    return macro[keep]
