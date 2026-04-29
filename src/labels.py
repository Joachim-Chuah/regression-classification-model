import pandas as pd


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
