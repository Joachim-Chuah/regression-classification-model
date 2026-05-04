import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression as SigmoidCalibrator
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier, XGBRegressor

from src.constants import (
    DEFAULT_HORIZON, DEFAULT_TICKERS, NEUTRAL_THRESHOLD,
    RANDOM_STATE, TICKER_SECTOR_ETF, TRAIN_END, VAL_END,
)
from src.data import pull, pull_earnings_dates, pull_macro
from src.evaluate import evaluate, evaluate_3class, evaluate_regressor, threshold_analysis
from src.features import compute_features
from src.labels import (
    compute_sample_weights,
    make_3class_labels,
    make_3class_labels_vol_scaled,
    make_labels,
    make_returns,
)
from src.serialize import save_artifact
from src.splits import time_split


# ---------------------------------------------------------------------------
# 3-class calibration wrapper
# Re-exported here so existing pickled artifacts (which reference
# src.train.CalibratedXGB3Class) still deserialise without error.
# ---------------------------------------------------------------------------

from src.models import CalibratedXGB3Class  # noqa: F401, E402


# ---------------------------------------------------------------------------
# Shared dataset builder
# ---------------------------------------------------------------------------

def _build(
    tickers: list[str],
    start: str,
    end: str,
    target: str,   # "label" | "3class" | "return"
    horizon: int = DEFAULT_HORIZON,
    label_mode: str = "fixed",  # "fixed" | "vol_scaled"
    vol_k: float = 1.0,
    vol_min_threshold: float = 0.005,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Returns (X, y, forward_return). forward_return is always the raw
    signed N-day return, regardless of target — used for sample weighting."""
    macro = pull_macro(start, end)
    frames = []
    for ticker in tickers:
        df = pull(ticker, start=start, end=end)
        sector_etf = TICKER_SECTOR_ETF.get(ticker)
        ed = pull_earnings_dates(ticker)
        X = compute_features(df, macro=macro, sector_etf=sector_etf, earnings_dates=ed)
        r = make_returns(df["close"], horizon)
        if target == "3class":
            if label_mode == "vol_scaled":
                y = make_3class_labels_vol_scaled(
                    df["close"],
                    atr_pct=X["atr_pct"],
                    horizon=horizon,
                    k=vol_k,
                    min_threshold=vol_min_threshold,
                )
            else:
                y = make_3class_labels(df["close"], horizon, NEUTRAL_THRESHOLD)
        elif target == "label":
            y = make_labels(df["close"], horizon)
        else:
            y = r
        combined = X.join(y.rename(target)).join(r.rename("__fwd_ret")).dropna()
        frames.append(combined)
        print(f"  {ticker}: {len(combined)} rows")
    all_data = pd.concat(frames).sort_index()
    return (
        all_data.drop(columns=[target, "__fwd_ret"]),
        all_data[target],
        all_data["__fwd_ret"],
    )


# ---------------------------------------------------------------------------
# 3-class classifier (preferred)
# ---------------------------------------------------------------------------

def train_classifier_3class(
    tickers: list[str] = DEFAULT_TICKERS,
    version: str = "1.2.0",
    horizon: int = DEFAULT_HORIZON,
    label_mode: str = "fixed",  # "fixed" | "vol_scaled"
    vol_k: float = 1.0,
    vol_min_threshold: float = 0.005,
    weight_clip: tuple[float, float] | None = None,
) -> CalibratedXGB3Class:
    """
    Train a 3-class XGBoost classifier with isotonic probability calibration.

    Classes: 0=down, 1=neutral, 2=up  (neutral = return within ±NEUTRAL_THRESHOLD)
    predict_proba(X) returns shape (N, 3): [P(down), P(neutral), P(up)]
    For Rylo: use proba[:,2] as P(up) and proba[:,0] as P(down).

    Calibration is fit on the held-out val set so it is never exposed to
    training data. This makes the probability magnitudes meaningful.
    """
    print("=== 3-Class Classifier — Loading data ===")
    X, y, fwd_ret = _build(
        tickers,
        "2015-01-01",
        "2024-12-31",
        target="3class",
        horizon=horizon,
        label_mode=label_mode,
        vol_k=vol_k,
        vol_min_threshold=vol_min_threshold,
    )
    print(f"  Total: {len(X)} rows across {len(tickers)} tickers")
    if label_mode == "vol_scaled":
        print(
            f"  Label mode: vol_scaled (k={vol_k:.2f}, min={vol_min_threshold:.2%})  |  "
            f"down:{(y==0).mean():.1%}  neutral:{(y==1).mean():.1%}  up:{(y==2).mean():.1%}\n"
        )
    else:
        print(
            f"  Label mode: fixed (±{NEUTRAL_THRESHOLD:.0%})  |  "
            f"down:{(y==0).mean():.1%}  neutral:{(y==1).mean():.1%}  up:{(y==2).mean():.1%}\n"
        )

    X_train, X_val, X_test = time_split(X, TRAIN_END, VAL_END)
    y_train, y_val, y_test = time_split(y, TRAIN_END, VAL_END)
    fwd_train, _, _ = time_split(fwd_ret, TRAIN_END, VAL_END)
    print(f"  Train: {len(X_train)} rows  |  Val: {len(X_val)} rows  |  Test: {len(X_test)} rows")
    if weight_clip is not None:
        print(f"  Sample weights: clip abs(return) to [{weight_clip[0]:.2%}, {weight_clip[1]:.2%}], normalised to mean=1\n")
    else:
        print()

    model = XGBClassifier(
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
    w_train = compute_sample_weights(fwd_train, weight_clip[0], weight_clip[1]) if weight_clip else None
    model.fit(X_train, y_train, sample_weight=w_train)

    print("=== 3-Class Classifier — Feature Importance ===")
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    print(importances.sort_values(ascending=False).to_string())
    print()

    # --- Isotonic calibration on val set ---
    print("=== 3-Class Classifier — Calibrating on val set ===")
    raw_val_proba = model.predict_proba(X_val)
    calibrators = []
    for k in range(3):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_val_proba[:, k], (y_val == k).astype(float))
        calibrators.append(iso)
    calibrated = CalibratedXGB3Class(model, calibrators)
    print("  Done.\n")

    print("=== 3-Class Classifier — Test ===")
    evaluate_3class(calibrated, X_test, y_test)
    threshold_analysis(calibrated, X_test, y_test)

    save_artifact(
        calibrated, list(X_train.columns),
        version=version, ticker="multi", model_name="xgb_clf3",
    )
    return calibrated


# ---------------------------------------------------------------------------
# Regressor
# ---------------------------------------------------------------------------

def train_regressor(
    tickers: list[str] = DEFAULT_TICKERS,
    version: str = "0.3.0",
    horizon: int = DEFAULT_HORIZON,
) -> XGBRegressor:
    print("\n=== Regressor — Loading data ===")
    X, y, _ = _build(tickers, "2015-01-01", "2024-12-31", target="return", horizon=horizon)
    print(f"  Total: {len(X)} rows across {len(tickers)} tickers\n")

    X_train, _, X_test = time_split(X, TRAIN_END, VAL_END)
    y_train, _, y_test = time_split(y, TRAIN_END, VAL_END)

    model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    print("=== Regressor — Feature Importance ===")
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    print(importances.sort_values(ascending=False).to_string())
    print()

    print("=== Regressor — Test ===")
    evaluate_regressor(model, X_test, y_test)

    save_artifact(
        model, list(X_train.columns),
        version=version, ticker="multi", model_name="xgb_reg",
    )
    return model


# ---------------------------------------------------------------------------
# Train both
# ---------------------------------------------------------------------------

def train_both(
    tickers: list[str] = DEFAULT_TICKERS,
    horizon: int = DEFAULT_HORIZON,
) -> tuple[XGBClassifier, XGBRegressor]:
    clf = train_classifier_3class(tickers=tickers, horizon=horizon)
    reg = train_regressor(tickers=tickers, horizon=horizon)
    return clf, reg


if __name__ == "__main__":
    train_both()
