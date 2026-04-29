import numpy as np
import pandas as pd
import pytest

from src.splits import time_split, walk_forward_splits


@pytest.fixture
def sample_df():
    dates = pd.date_range("2015-01-01", "2024-12-31", freq="B")
    return pd.DataFrame({"value": np.random.randn(len(dates))}, index=dates)


def test_time_split_no_overlap(sample_df):
    train, val, test = time_split(sample_df, "2022-12-31", "2023-12-31")
    assert train.index.max() <= pd.Timestamp("2022-12-31")
    assert val.index.min() > pd.Timestamp("2022-12-31")
    assert val.index.max() <= pd.Timestamp("2023-12-31")
    assert test.index.min() > pd.Timestamp("2023-12-31")


def test_time_split_exhaustive(sample_df):
    train, val, test = time_split(sample_df, "2022-12-31", "2023-12-31")
    assert len(train) + len(val) + len(test) == len(sample_df)


def test_walk_forward_no_future_leakage(sample_df):
    for train, test in walk_forward_splits(sample_df, min_train_size=500, test_size=63, step=63):
        assert train.index.max() < test.index.min()


def test_walk_forward_expanding_window(sample_df):
    sizes = [
        len(train)
        for train, _ in walk_forward_splits(sample_df, min_train_size=500, test_size=63, step=63)
    ]
    assert all(sizes[i] > sizes[i - 1] for i in range(1, len(sizes)))


def test_time_split_works_on_series(sample_df):
    s = sample_df["value"]
    train, val, test = time_split(s, "2022-12-31", "2023-12-31")
    assert isinstance(train, pd.Series)
    assert len(train) + len(val) + len(test) == len(s)
