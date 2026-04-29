from typing import Iterator, Tuple, Union

import pandas as pd


_Indexable = Union[pd.DataFrame, pd.Series]


def time_split(
    data: _Indexable,
    train_end: str,
    val_end: str,
) -> Tuple[_Indexable, _Indexable, _Indexable]:
    """Hard cutoff split into train / val / test. Works on DataFrame or Series."""
    train = data[data.index <= train_end]
    val = data[(data.index > train_end) & (data.index <= val_end)]
    test = data[data.index > val_end]
    return train, val, test


def walk_forward_splits(
    data: _Indexable,
    min_train_size: int,
    test_size: int,
    step: int,
) -> Iterator[Tuple[_Indexable, _Indexable]]:
    """
    Expanding-window walk-forward splits.

    Yields (train, test) pairs. The train window grows by `step` each iteration.
    No row ever appears in a test fold before it has appeared in a train fold.
    """
    n = len(data)
    start = min_train_size
    while start + test_size <= n:
        yield data.iloc[:start], data.iloc[start : start + test_size]
        start += step
