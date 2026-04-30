import numpy as np
import pandas as pd

from src.evaluate import signal_quality_table, topk_precision_table


def test_signal_quality_table_threshold_metrics():
    y = pd.Series([2, 2, 1, 0, 2, 0], dtype=float)
    p_up = np.array([0.90, 0.70, 0.60, 0.30, 0.55, 0.20])
    forward_return = pd.Series([0.05, 0.03, 0.01, -0.02, 0.02, -0.01], dtype=float)

    table = signal_quality_table(
        y=y,
        p_up=p_up,
        forward_return=forward_return,
        thresholds=[0.60],
    )

    row = table.iloc[0]
    assert row["n_calls"] == 3
    assert np.isclose(row["coverage"], 3 / 6)
    assert np.isclose(row["precision_up"], 2 / 3)
    assert np.isclose(row["mean_forward_return"], (0.05 + 0.03 + 0.01) / 3)


def test_topk_precision_table_computes_top_decile_slice():
    y = pd.Series([2, 0, 2, 1, 0, 2, 1, 0, 2, 0], dtype=float)
    p_up = np.array([0.95, 0.10, 0.85, 0.40, 0.20, 0.80, 0.35, 0.15, 0.75, 0.05])
    forward_return = pd.Series([0.04, -0.01, 0.03, 0.00, -0.02, 0.02, -0.01, -0.01, 0.01, -0.03], dtype=float)

    table = topk_precision_table(
        y=y,
        p_up=p_up,
        forward_return=forward_return,
        topk_fracs=[0.20],
    )

    row = table.iloc[0]
    assert row["n_calls"] == 2
    assert np.isclose(row["precision_up"], 1.0)
    assert np.isclose(row["min_p_up"], 0.85)
    assert np.isclose(row["mean_forward_return"], (0.04 + 0.03) / 2)
