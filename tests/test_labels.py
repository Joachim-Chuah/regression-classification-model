import numpy as np
import pandas as pd
import pytest

from src.labels import (
    compute_sample_weights,
    make_3class_labels,
    make_3class_labels_vol_scaled,
    make_labels,
    make_returns,
)


@pytest.fixture
def close():
    dates = pd.date_range("2020-01-01", periods=50, freq="B")
    return pd.Series(np.arange(1.0, 51.0), index=dates)


# --- make_labels (binary) ---

def test_last_n_rows_are_nan(close):
    horizon = 5
    assert make_labels(close, horizon=horizon).iloc[-horizon:].isna().all()


def test_labels_are_binary(close):
    labels = make_labels(close, horizon=5).dropna()
    assert set(labels.unique()).issubset({0.0, 1.0})


def test_strictly_increasing_series_labels_all_up(close):
    labels = make_labels(close, horizon=5).dropna()
    assert (labels == 1.0).all()


def test_index_preserved(close):
    assert make_labels(close, horizon=5).index.equals(close.index)


def test_valid_label_count(close):
    horizon = 5
    labels = make_labels(close, horizon=horizon)
    assert labels.notna().sum() == len(close) - horizon


# --- make_returns ---

def test_returns_last_n_rows_are_nan(close):
    assert make_returns(close, horizon=5).iloc[-5:].isna().all()


def test_returns_are_continuous(close):
    returns = make_returns(close, horizon=5).dropna()
    assert returns.dtype == float
    assert not set(returns.unique()).issubset({0.0, 1.0})


def test_returns_positive_for_increasing_series(close):
    returns = make_returns(close, horizon=5).dropna()
    assert (returns > 0).all()


def test_returns_index_preserved(close):
    assert make_returns(close, horizon=5).index.equals(close.index)


# --- make_3class_labels ---

def test_3class_last_n_rows_are_nan(close):
    assert make_3class_labels(close, horizon=5).iloc[-5:].isna().all()


def test_3class_only_valid_values(close):
    labels = make_3class_labels(close, horizon=5, threshold=0.02).dropna()
    assert set(labels.unique()).issubset({0.0, 1.0, 2.0})


def test_3class_increasing_series_labels_up(close):
    # close is 1, 2, 3, ... — 5-day forward returns are large and positive
    labels = make_3class_labels(close, horizon=5, threshold=0.02).dropna()
    assert (labels == 2.0).all()


def test_3class_flat_series_labels_neutral():
    dates = pd.date_range("2020-01-01", periods=20, freq="B")
    flat = pd.Series(np.ones(20) * 100.0, index=dates)
    labels = make_3class_labels(flat, horizon=5, threshold=0.02).dropna()
    assert (labels == 1.0).all()


def test_3class_index_preserved(close):
    assert make_3class_labels(close, horizon=5).index.equals(close.index)


def test_3class_valid_count(close):
    horizon = 5
    labels = make_3class_labels(close, horizon=horizon)
    assert labels.notna().sum() == len(close) - horizon


def test_3class_vol_scaled_only_valid_values(close):
    atr = pd.Series(np.full(len(close), 0.02), index=close.index)
    labels = make_3class_labels_vol_scaled(close, atr_pct=atr, horizon=5).dropna()
    assert set(labels.unique()).issubset({0.0, 1.0, 2.0})


def test_3class_vol_scaled_increasing_series_labels_up(close):
    atr = pd.Series(np.full(len(close), 0.02), index=close.index)
    labels = make_3class_labels_vol_scaled(close, atr_pct=atr, horizon=5).dropna()
    assert (labels == 2.0).all()


def test_3class_vol_scaled_uses_min_threshold_on_tiny_atr():
    dates = pd.date_range("2020-01-01", periods=20, freq="B")
    close_small_move = pd.Series(np.linspace(100.0, 100.4, 20), index=dates)
    atr_tiny = pd.Series(np.full(20, 0.0001), index=dates)
    labels = make_3class_labels_vol_scaled(
        close_small_move,
        atr_pct=atr_tiny,
        horizon=5,
        k=1.0,
        min_threshold=0.01,  # 1% neutral floor
    ).dropna()
    assert (labels == 1.0).all()


def test_compute_sample_weights_mean_one():
    returns = np.array([-0.05, 0.01, 0.08, -0.03, 0.15])
    w = compute_sample_weights(returns, clip_low=0.01, clip_high=0.10)
    assert abs(w.mean() - 1.0) < 1e-10


def test_compute_sample_weights_clip_respected():
    returns = np.array([0.0001, 0.50])  # below clip_low and above clip_high
    w = compute_sample_weights(returns, clip_low=0.01, clip_high=0.10)
    raw = np.clip(np.abs(returns), 0.01, 0.10)
    np.testing.assert_allclose(w, raw / raw.mean())


def test_compute_sample_weights_accepts_series():
    s = pd.Series([-0.03, 0.05, -0.08, 0.02])
    w = compute_sample_weights(s, clip_low=0.01, clip_high=0.10)
    assert len(w) == len(s)
    assert abs(w.mean() - 1.0) < 1e-10
