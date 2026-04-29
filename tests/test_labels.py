import numpy as np
import pandas as pd
import pytest

from src.labels import make_labels


@pytest.fixture
def close():
    dates = pd.date_range("2020-01-01", periods=50, freq="B")
    return pd.Series(np.arange(1.0, 51.0), index=dates)


def test_last_n_rows_are_nan(close):
    horizon = 5
    assert make_labels(close, horizon=horizon).iloc[-horizon:].isna().all()


def test_labels_are_binary(close):
    labels = make_labels(close, horizon=5).dropna()
    assert set(labels.unique()).issubset({0.0, 1.0})


def test_strictly_increasing_series_labels_all_up(close):
    # close is 1, 2, 3, ... so every forward return is positive
    labels = make_labels(close, horizon=5).dropna()
    assert (labels == 1.0).all()


def test_index_preserved(close):
    assert make_labels(close, horizon=5).index.equals(close.index)


def test_valid_label_count(close):
    horizon = 5
    labels = make_labels(close, horizon=horizon)
    assert labels.notna().sum() == len(close) - horizon
