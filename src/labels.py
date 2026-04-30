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


def make_3class_labels_vol_scaled(
    close: pd.Series,
    atr_pct: pd.Series,
    horizon: int = 5,
    k: float = 1.0,
    min_threshold: float = 0.005,
) -> pd.Series:
    """
    3-class label with a volatility-scaled neutral zone.

    Neutral threshold at time t is:
        threshold_t = max(min_threshold, k * atr_pct_t)

    This keeps the neutral band tighter in low-vol regimes and wider in
    high-vol regimes, so the model is not forced to classify tiny noisy moves
    as directional signals.
    """
    forward_return = close.shift(-horizon) / close - 1
    dynamic_threshold = (k * atr_pct).clip(lower=min_threshold)

    labels = pd.Series(1.0, index=close.index)  # default: neutral
    labels[forward_return > dynamic_threshold] = 2.0
    labels[forward_return < -dynamic_threshold] = 0.0
    labels = labels.where(forward_return.notna())
    return labels
