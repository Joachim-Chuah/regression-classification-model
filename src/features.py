import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # EWM with com=window-1 matches the Wilder smoothing convention
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series]:
    fast_ema = close.ewm(span=fast, adjust=False).mean()
    slow_ema = close.ewm(span=slow, adjust=False).mean()
    macd_line = (fast_ema - slow_ema) / close  # normalised by price
    macd_hist = macd_line - macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, macd_hist


def _bollinger_position(close: pd.Series, window: int = 20) -> pd.Series:
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    band_width = (2 * std * 2).replace(0, np.nan)  # upper - lower = 4 * std
    return (close - (mid - 2 * std)) / band_width  # 0 = at lower band, 1 = at upper


def _atr_pct(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(com=window - 1, min_periods=window).mean()
    return atr / close  # express as fraction of price so it's scale-independent


def compute_features(
    df: pd.DataFrame,
    macro: pd.DataFrame | None = None,
    sector_etf: str | None = None,
    earnings_dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """
    Compute features from OHLCV DataFrame, optionally joined with macro series.

    All features at time t use only data available at or before t.
    No in-place mutation of the input.

    Parameters
    ----------
    df         : DataFrame with [open, high, low, close, volume] columns, date index.
    macro      : Optional DataFrame from data.pull_macro() with market-wide features.
    sector_etf : Optional sector ETF ticker (e.g. "XLK") for sector-relative strength.

    Returns
    -------
    DataFrame of features with the same index as df.
    """
    close = df["close"]
    volume = df["volume"]
    daily_returns = close.pct_change()

    features = pd.DataFrame(index=df.index)

    # Momentum: N-day price return ending at t
    features["momentum_5d"]  = close.pct_change(5)
    features["momentum_10d"] = close.pct_change(10)
    features["momentum_21d"] = close.pct_change(21)

    # RSI
    features["rsi_14"] = _rsi(close, window=14)

    # Rolling realised volatility (annualised)
    features["rolling_vol_21d"] = daily_returns.rolling(21).std() * (252**0.5)

    # Volume z-score relative to trailing 21-day window
    vol_mean = volume.rolling(21).mean()
    vol_std  = volume.rolling(21).std().replace(0, np.nan)
    features["volume_zscore_21d"] = (volume - vol_mean) / vol_std

    # MACD line and histogram (both normalised by close price)
    features["macd_line"], features["macd_hist"] = _macd(close)

    # Bollinger Band position: 0 = at lower band, 1 = at upper band
    features["bb_position"] = _bollinger_position(close)

    # ATR as a fraction of close price — measures current volatility regime
    features["atr_pct"] = _atr_pct(df)

    # 200-day MA signal — is this stock above its long-term trend?
    features["vs_200ma"] = close / close.rolling(200).mean() - 1

    if macro is not None:
        m = macro.reindex(df.index).ffill()

        features["vix_zscore_252d"]       = m["vix_zscore_252d"]
        features["vix_change_5d"]         = m["vix_change_5d"]
        features["spy_return_20d"]        = m["spy_return_20d"]
        features["spy_vs_200ma"]          = m["spy_vs_200ma"]
        features["yield_10y_zscore_252d"] = m["yield_10y_zscore_252d"]
        features["yield_change_20d"]          = m["yield_change_20d"]
        features["yield_curve_zscore_252d"]   = m["yield_curve_zscore_252d"]

        # Broad market relative strength
        features["rel_strength_20d"] = close.pct_change(20) - m["spy_return_20d"]

        # Sector-relative strength (more specific than vs SPY)
        if sector_etf and f"{sector_etf}_return_20d" in m.columns:
            features["rel_strength_vs_sector"] = (
                close.pct_change(20) - m[f"{sector_etf}_return_20d"]
            )

        # Pass-through macro columns (breadth + FRED, present when available)
        for col in [
            "iwm_vs_spy_20d", "xlp_vs_spy_20d",
            "fedfunds", "fedfunds_change_1y",
            "cpi_yoy", "cpi_momentum",
            "unemployment", "unemployment_change_1y",
            "hy_spread", "hy_spread_change_20d",
            "nfci",
        ]:
            if col in m.columns:
                features[col] = m[col]

    if earnings_dates is not None and len(earnings_dates) > 0:
        sorted_ed = np.sort(np.asarray(earnings_dates, dtype="datetime64[D]"))
        idx_days  = df.index.values.astype("datetime64[D]")
        # searchsorted side="right" means: first earnings date strictly after idx_day
        pos = np.searchsorted(sorted_ed, idx_days, side="right")
        raw_days = np.full(len(idx_days), np.nan)
        valid = pos < len(sorted_ed)
        raw_days[valid] = (sorted_ed[pos[valid]] - idx_days[valid]).astype(float)
        features["days_to_earnings"] = pd.Series(
            np.clip(raw_days, 0, 90), index=df.index
        )

    return features
