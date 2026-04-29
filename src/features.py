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


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features from OHLCV DataFrame.

    All features at time t use only data available at or before t.
    No in-place mutation of the input.

    Parameters
    ----------
    df : DataFrame with [open, high, low, close, volume] columns, date index.

    Returns
    -------
    DataFrame of features with the same index.
    """
    close = df["close"]
    volume = df["volume"]
    daily_returns = close.pct_change()

    features = pd.DataFrame(index=df.index)

    # Momentum: N-day price return ending at t
    features["momentum_5d"] = close.pct_change(5)
    features["momentum_10d"] = close.pct_change(10)
    features["momentum_21d"] = close.pct_change(21)

    # RSI
    features["rsi_14"] = _rsi(close, window=14)

    # Rolling realised volatility (annualised)
    features["rolling_vol_21d"] = daily_returns.rolling(21).std() * (252**0.5)

    # Volume z-score relative to trailing 21-day window
    vol_mean = volume.rolling(21).mean()
    vol_std = volume.rolling(21).std().replace(0, np.nan)
    features["volume_zscore_21d"] = (volume - vol_mean) / vol_std

    return features
