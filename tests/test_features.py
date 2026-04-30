import numpy as np
import pandas as pd
import pytest

from src.features import compute_features


def _make_ohlcv(n: int, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.date_range("2019-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n))
    return pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.random.randint(1_000_000, 10_000_000, size=n).astype(float),
        },
        index=dates,
    )


@pytest.fixture
def sample_ohlcv():
    return _make_ohlcv(100)


@pytest.fixture
def long_ohlcv():
    """300-day OHLCV — enough for the 252-day rolling window features."""
    return _make_ohlcv(300)


def test_no_lookahead_bias(sample_ohlcv):
    """Corrupting future rows must not change features computed at earlier rows."""
    features_original = compute_features(sample_ohlcv)

    corrupted = sample_ohlcv.copy()
    corrupted.iloc[50:] *= 999
    features_corrupted = compute_features(corrupted)

    pd.testing.assert_frame_equal(
        features_original.iloc[:50],
        features_corrupted.iloc[:50],
    )


def test_index_matches_input(sample_ohlcv):
    assert compute_features(sample_ohlcv).index.equals(sample_ohlcv.index)


def test_no_inplace_mutation(sample_ohlcv):
    original = sample_ohlcv["close"].copy()
    compute_features(sample_ohlcv)
    pd.testing.assert_series_equal(sample_ohlcv["close"], original)


def test_expected_columns_present(sample_ohlcv):
    features = compute_features(sample_ohlcv)
    expected = [
        "momentum_5d",
        "momentum_10d",
        "momentum_21d",
        "rsi_14",
        "rolling_vol_21d",
        "volume_zscore_21d",
        "macd_line",
        "macd_hist",
        "bb_position",
        "atr_pct",
        "vs_200ma",
        "pct_from_52w_high",
        "pct_from_52w_low",
    ]
    for col in expected:
        assert col in features.columns, f"Missing feature: {col}"


def test_rsi_bounds(sample_ohlcv):
    rsi = compute_features(sample_ohlcv)["rsi_14"].dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all()


def test_pct_from_52w_high_nonpositive(long_ohlcv):
    valid = compute_features(long_ohlcv)["pct_from_52w_high"].dropna()
    assert len(valid) > 0
    assert (valid <= 1e-10).all()  # allow for floating point at exact high


def test_pct_from_52w_low_nonnegative(long_ohlcv):
    valid = compute_features(long_ohlcv)["pct_from_52w_low"].dropna()
    assert len(valid) > 0
    assert (valid >= -1e-10).all()  # allow for floating point at exact low
