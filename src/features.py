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


def _shift_bday(series: pd.Series, n: int) -> pd.Series:
    """Return a copy of series with its index shifted forward by n business days."""
    return pd.Series(series.values, index=series.index + pd.tseries.offsets.BDay(n))


def compute_features(
    df: pd.DataFrame,
    macro: pd.DataFrame | None = None,
    sector_etf: str | None = None,
    earnings_dates: pd.DatetimeIndex | None = None,
    external: dict | None = None,
) -> pd.DataFrame:
    """
    Compute features from OHLCV DataFrame, optionally joined with macro series
    and external signals (short interest, news sentiment, analyst grades, etc.).

    All features at time t use only data available at or before t.
    No in-place mutation of the input.

    Parameters
    ----------
    df       : DataFrame with [open, high, low, close, volume] columns, date index.
    macro    : Optional DataFrame from data.pull_macro() with market-wide features.
    sector_etf : Optional sector ETF ticker (e.g. "XLK") for sector-relative strength.
    external : Optional dict of external signals:
               {
                 "short_volume"        : pd.DataFrame  # col: short_volume_ratio
                 "short_interest"      : pd.DataFrame  # cols: short_interest, days_to_cover
                 "news_sentiment"      : pd.DataFrame  # col: news_sentiment
                 "analyst_grades"      : pd.DataFrame  # col: grade_score
                 "insider_trades"      : pd.DataFrame  # col: insider_score
                 "option_snapshot"     : dict           # put_call_oi_ratio, iv_atm, iv_skew
                 "price_target_upside" : float          # (consensus_pt - price) / price
               }

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
        features["vix_level"]             = m["vix_level"]
        features["spy_return_20d"]        = m["spy_return_20d"]
        features["spy_vs_200ma"]          = m["spy_vs_200ma"]
        features["spy_return_52w"]        = m["spy_return_52w"]
        features["spy_drawdown_52w"]      = m["spy_drawdown_52w"]
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

        # Pass-through macro columns (breadth + credit + FRED, present when available)
        for col in [
            "iwm_vs_spy_20d", "xlp_vs_spy_20d",
            "hyg_lqd_ratio", "hyg_lqd_change_20d",
            "fedfunds", "fedfunds_change_1y",
            "cpi_yoy", "cpi_momentum",
            "unemployment", "unemployment_change_1y",
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

    # -----------------------------------------------------------------------
    # External signals (short interest, news sentiment, analyst grades, etc.)
    # NaN when data unavailable — XGBoost handles missing natively.
    # -----------------------------------------------------------------------
    ext = external or {}

    def _dedup(s: pd.Series) -> pd.Series:
        """Drop duplicate index entries, keeping the last value per date."""
        return s[~s.index.duplicated(keep="last")]

    # Short sale volume ratio (5-day rolling avg, 1-day publication lag)
    sv_df = ext.get("short_volume")
    if sv_df is not None and not sv_df.empty and "short_volume_ratio" in sv_df.columns:
        sv = _shift_bday(_dedup(sv_df["short_volume_ratio"]), 1)
        sv_daily = sv.reindex(df.index).ffill()
        features["short_volume_ratio_5d"] = sv_daily.rolling(5, min_periods=1).mean()
    else:
        features["short_volume_ratio_5d"] = np.nan

    # Days-to-cover from biweekly short interest (forward-filled)
    si_df = ext.get("short_interest")
    if si_df is not None and not si_df.empty and "days_to_cover" in si_df.columns:
        features["days_to_cover"] = _dedup(si_df["days_to_cover"]).reindex(df.index).ffill()
    else:
        features["days_to_cover"] = np.nan

    # News sentiment — rolling 3-day and 20-day averages
    ns_df = ext.get("news_sentiment")
    if ns_df is not None and not ns_df.empty and "news_sentiment" in ns_df.columns:
        sent = _dedup(ns_df["news_sentiment"]).reindex(df.index)
        features["news_sentiment_3d"]  = sent.rolling(3,  min_periods=1).mean()
        features["news_sentiment_20d"] = sent.rolling(20, min_periods=1).mean()
    else:
        features["news_sentiment_3d"]  = np.nan
        features["news_sentiment_20d"] = np.nan

    # Analyst grade momentum — net score (upgrades minus downgrades) over trailing 30 days
    # Aggregate multiple same-day events first, then shift by 1 business day.
    ag_df = ext.get("analyst_grades")
    if ag_df is not None and not ag_df.empty and "grade_score" in ag_df.columns:
        grades_agg = ag_df["grade_score"].groupby(level=0).sum()
        grades = _shift_bday(grades_agg, 1).groupby(level=0).sum()  # re-agg after shift collision
        grades_daily = grades.reindex(df.index).fillna(0.0)
        features["analyst_upgrade_net_30d"] = grades_daily.rolling(30, min_periods=1).sum()
    else:
        features["analyst_upgrade_net_30d"] = np.nan

    # Insider transaction net signal — rolling 30-day sum
    # Aggregate same-day transactions, shift by 2 business days (Form 4 filing deadline).
    it_df = ext.get("insider_trades")
    if it_df is not None and not it_df.empty and "insider_score" in it_df.columns:
        insider_agg = it_df["insider_score"].groupby(level=0).sum()
        insider = _shift_bday(insider_agg, 2).groupby(level=0).sum()  # re-agg after shift collision
        insider_daily = insider.reindex(df.index).fillna(0.0)
        features["insider_net_buy_30d"] = insider_daily.rolling(30, min_periods=1).sum()
    else:
        features["insider_net_buy_30d"] = np.nan

    # Options snapshot (inference-time only — NaN during training)
    opt = ext.get("option_snapshot") or {}
    features["put_call_oi_ratio"] = opt.get("put_call_oi_ratio")   # None → NaN
    features["iv_atm"]            = opt.get("iv_atm")
    features["iv_skew"]           = opt.get("iv_skew")

    # Price-target upside (inference-time only — NaN during training)
    features["price_target_upside"] = ext.get("price_target_upside")  # None → NaN

    return features
