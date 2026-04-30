import pandas as pd

from src.constants import NEUTRAL_THRESHOLD


def make_returns(close: pd.Series, horizon: int = 5) -> pd.Series:
    """
    N-day forward return for regression targets.

    Returns the raw signed float return: close[t + horizon] / close[t] - 1.
    The last `horizon` rows will be NaN. Always dropna() before training.
    """
    return close.shift(-horizon) / close - 1


def make_labels(close: pd.Series, horizon: int = 5) -> pd.Series:
    """
    Binary label: 1 if close[t + horizon] > close[t], else 0.

    The last `horizon` rows will be NaN because the forward price is unknown.
    Always dropna() before training.
    """
    forward_return = close.shift(-horizon) / close - 1
    labels = (forward_return > 0).astype(float)
    # Restore NaN for rows where the forward price doesn't exist
    labels = labels.where(forward_return.notna())
    return labels


def make_3class_labels(
    close: pd.Series,
    horizon: int = 5,
    threshold: float = NEUTRAL_THRESHOLD,
) -> pd.Series:
    """
    3-class label based on forward return magnitude:
        0 = down   (return < -threshold)
        1 = neutral (return within ±threshold)
        2 = up     (return > +threshold)

    Removing the noise of small moves forces the model to identify
    high-conviction setups rather than guessing direction on flat days.
    The last `horizon` rows will be NaN. Always dropna() before training.
    """
    forward_return = close.shift(-horizon) / close - 1
    labels = pd.Series(1.0, index=close.index)   # default: neutral
    labels[forward_return >  threshold] = 2.0    # up
    labels[forward_return < -threshold] = 0.0    # down
    labels = labels.where(forward_return.notna())
    return labels
