import os

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date
from fredapi import Fred
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


def pull_fred(series_id: str, start: str, end: str) -> pd.Series:
    """Pull a FRED series with parquet caching (same policy as pull()).

    Requires FRED_API_KEY env var. Free key at:
    https://fred.stlouisfed.org/docs/api/api_key.html
    then: export FRED_API_KEY=your_key_here
    """
    cache_path = RAW_DIR / f"fred_{series_id}_{start}_{end}.parquet"

    if cache_path.exists():
        return pd.read_parquet(cache_path)["value"]

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY not set. Get a free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html "
            "then: export FRED_API_KEY=your_key_here"
        )

    fred = Fred(api_key=api_key)
    series = fred.get_series(series_id, observation_start=start, observation_end=end)
    series.name = series_id
    series.index.name = "date"
    if series.index.tz is not None:
        series.index = series.index.tz_localize(None)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    series.to_frame("value").to_parquet(cache_path)

    return series


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

    # Market breadth proxies (yfinance, no API key required)
    iwm_ret = pull("IWM", start, end)["close"].pct_change(20).reindex(macro.index).ffill()
    xlp_ret = pull("XLP", start, end)["close"].pct_change(20).reindex(macro.index).ffill()
    macro["iwm_vs_spy_20d"] = iwm_ret - macro["spy_return_20d"]   # small cap breadth
    macro["xlp_vs_spy_20d"] = xlp_ret - macro["spy_return_20d"]   # defensive rotation

    keep = [
        "vix_zscore_252d", "vix_change_5d",
        "spy_return_20d", "spy_vs_200ma",
        "yield_10y_zscore_252d", "yield_change_20d", "yield_curve_zscore_252d",
        "XLK_return_20d", "XLF_return_20d", "XLV_return_20d", "XLE_return_20d",
        "iwm_vs_spy_20d", "xlp_vs_spy_20d",
    ]

    fred_key = os.environ.get("FRED_API_KEY")
    if fred_key:
        # Fed funds effective rate (daily). FRED publishes with ~1-day lag → shift(1).
        dff = pull_fred("DFF", start, end).shift(1)
        dff_d = dff.reindex(macro.index, method="ffill")
        macro["fedfunds"]           = dff_d
        macro["fedfunds_change_1y"] = dff_d.diff(252)

        # CPI YoY (monthly). FRED timestamps to the reference month start, but the
        # release arrives ~2-3 weeks later → shift 1 month before forward-filling.
        cpi = pull_fred("CPIAUCSL", start, end)
        cpi_m   = cpi.resample("MS").last()
        cpi_yoy = cpi_m.pct_change(12)
        macro["cpi_yoy"]      = cpi_yoy.shift(1).reindex(macro.index, method="ffill")
        macro["cpi_momentum"] = cpi_yoy.diff(3).shift(1).reindex(macro.index, method="ffill")

        # Unemployment rate (monthly, same release-lag logic as CPI).
        unrate = pull_fred("UNRATE", start, end)
        unrate_m = unrate.resample("MS").last()
        macro["unemployment"]           = unrate_m.shift(1).reindex(macro.index, method="ffill")
        macro["unemployment_change_1y"] = unrate_m.diff(12).shift(1).reindex(macro.index, method="ffill")

        # HY credit spread (daily). Widening = risk-off / financial stress.
        hy = pull_fred("BAMLH0A0HYM2", start, end).shift(1)
        hy_d = hy.reindex(macro.index, method="ffill")
        macro["hy_spread"]           = hy_d
        macro["hy_spread_change_20d"] = hy_d.diff(20)

        # Chicago Fed National Financial Conditions Index (weekly).
        # Positive = tighter-than-average conditions, negative = looser.
        nfci = pull_fred("NFCI", start, end)
        macro["nfci"] = nfci.shift(1).reindex(macro.index, method="ffill")

        keep += [
            "fedfunds", "fedfunds_change_1y",
            "cpi_yoy", "cpi_momentum",
            "unemployment", "unemployment_change_1y",
            "hy_spread", "hy_spread_change_20d",
            "nfci",
        ]
    else:
        print(
            "  [data] FRED_API_KEY not set — skipping fed funds / CPI / unemployment / credit features.\n"
            "         Get a free key: https://fred.stlouisfed.org/docs/api/api_key.html\n"
            "         Then: export FRED_API_KEY=your_key_here"
        )

    return macro[keep]


def pull_earnings_dates(ticker: str) -> pd.DatetimeIndex:
    """Return historical + upcoming earnings dates for a ticker.

    Cached to data/raw/earnings_{ticker}.parquet. Re-fetches when the cache
    has no future dates remaining (i.e. the last known earnings has passed).
    Returns an empty DatetimeIndex if yfinance has no data for the ticker.

    No leakage risk: earnings dates are announced publicly 2-4 weeks in
    advance, so knowing the next earnings date at time t is realistic.
    """
    safe = ticker.replace("^", "").replace("=", "").replace("-", "_")
    cache_path = RAW_DIR / f"earnings_{safe}.parquet"
    today = pd.Timestamp.today().normalize()

    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        dates = pd.DatetimeIndex(cached["date"])
        if len(dates) > 0 and dates.max() >= today:
            return dates.sort_values()

    try:
        t = yf.Ticker(ticker)
        ed = t.earnings_dates
        if ed is None or ed.empty:
            return pd.DatetimeIndex([])
        dates = pd.DatetimeIndex(ed.index)
        if dates.tz is not None:
            dates = dates.tz_localize(None)
        dates = dates.normalize().sort_values()
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"date": dates}).to_parquet(cache_path, index=False)
        return dates
    except Exception:
        return pd.DatetimeIndex([])
