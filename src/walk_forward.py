"""
Walk-forward cross-validation for the 3-class classifier.

Expanding-window: trains on all data up to (test_year - 1), evaluates on
test_year. Each fold is a full calendar year so the market regime is clearly
labelled. Five folds cover COVID crash (2020), low-rate bull (2021), bear /
rate-shock (2022), recovery (2023), and bull (2024).

Consistent performance across all five means the model has real signal.
Strong performance in only one or two means it learned a regime-specific
pattern that will fail in production.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, precision_score, recall_score
from xgboost import XGBClassifier

from src.constants import (
    DEFAULT_HORIZON, DEFAULT_TICKERS, NEUTRAL_THRESHOLD,
    RANDOM_STATE, TICKER_SECTOR_ETF,
)
from src.data import pull, pull_macro
from src.features import compute_features
from src.labels import make_3class_labels

_REGIME_LABELS = {
    2020: "COVID crash",
    2021: "low-rate bull",
    2022: "bear/rate shock",
    2023: "recovery",
    2024: "bull",
}

_XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    objective="multi:softprob",
    num_class=3,
    random_state=RANDOM_STATE,
    eval_metric="mlogloss",
    verbosity=0,
)


def walk_forward_cv(
    tickers: list[str] = DEFAULT_TICKERS,
    horizon: int = DEFAULT_HORIZON,
    test_years: list[int] | None = None,
) -> pd.DataFrame:
    """
    Expanding-window walk-forward evaluation of the 3-class classifier.

    Builds the full dataset once (parquet cache makes this fast after the
    first run), then for each test year trains on all prior data and
    evaluates on that year alone. Returns a summary DataFrame.
    """
    if test_years is None:
        test_years = [2020, 2021, 2022, 2023, 2024]

    print("=== Walk-Forward Cross-Validation ===")
    print(f"  Tickers: {len(tickers)}  |  Horizon: {horizon}d  |  Years: {test_years}\n")

    # Build the full 2015-2024 dataset once
    print("  Building full dataset (using parquet cache where available)...")
    macro = pull_macro("2015-01-01", "2024-12-31")
    frames = []
    for ticker in tickers:
        df = pull(ticker, start="2015-01-01", end="2024-12-31")
        sector_etf = TICKER_SECTOR_ETF.get(ticker)
        X_ticker = compute_features(df, macro=macro, sector_etf=sector_etf)
        y_ticker = make_3class_labels(df["close"], horizon, NEUTRAL_THRESHOLD)
        combined = X_ticker.join(y_ticker.rename("label")).dropna()
        frames.append(combined)
    all_data = pd.concat(frames).sort_index()
    X_all = all_data.drop(columns=["label"])
    y_all = all_data["label"]
    print(f"  Total rows available: {len(X_all):,}\n")

    results = []
    for test_year in test_years:
        train_end  = f"{test_year - 1}-12-31"
        test_start = f"{test_year}-01-01"
        test_end   = f"{test_year}-12-31"

        X_train = X_all[X_all.index <= train_end]
        y_train = y_all[y_all.index <= train_end]
        X_test  = X_all[(X_all.index >= test_start) & (X_all.index <= test_end)]
        y_test  = y_all[(y_all.index >= test_start) & (y_all.index <= test_end)]

        if len(X_train) < 2000 or len(X_test) < 200:
            print(f"  {test_year}: insufficient data, skipping")
            continue

        model = XGBClassifier(**_XGB_PARAMS)
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        y_up_bin       = (y_test == 2).astype(int)
        p_up           = y_proba[:, 2]
        baseline_brier = brier_score_loss(y_up_bin, np.full(len(y_up_bin), y_up_bin.mean()))
        model_brier    = brier_score_loss(y_up_bin, p_up)
        brier_skill    = 1 - model_brier / baseline_brier if baseline_brier > 0 else float("nan")

        up_prec = precision_score(y_test == 2, y_pred == 2, zero_division=0)
        up_rec  = recall_score(y_test == 2, y_pred == 2, zero_division=0)
        acc     = accuracy_score(y_test, y_pred)

        regime = _REGIME_LABELS.get(test_year, "")
        print(
            f"  {test_year} ({regime:15s}):  "
            f"acc={acc:.1%}  up_prec={up_prec:.1%}  up_rec={up_rec:.1%}  "
            f"brier={brier_skill:+.4f}  "
            f"actual_up={y_up_bin.mean():.1%}  pred_up={(y_pred == 2).mean():.1%}"
        )

        results.append({
            "year":        test_year,
            "regime":      regime,
            "train_rows":  len(X_train),
            "test_rows":   len(X_test),
            "accuracy":    acc,
            "up_precision": up_prec,
            "up_recall":   up_rec,
            "brier_skill": brier_skill,
            "actual_up":   float(y_up_bin.mean()),
            "pred_up":     float((y_pred == 2).mean()),
        })

    summary = pd.DataFrame(results)

    print(f"\n  {'Metric':<20} {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    print("  " + "-" * 58)
    for col in ["accuracy", "up_precision", "up_recall", "brier_skill"]:
        print(
            f"  {col:<20} {summary[col].mean():>8.3f}  {summary[col].std():>8.3f}"
            f"  {summary[col].min():>8.3f}  {summary[col].max():>8.3f}"
        )

    return summary


if __name__ == "__main__":
    results = walk_forward_cv()
    print("\nFull results:")
    print(results.to_string(index=False))
