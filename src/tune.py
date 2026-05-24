"""
Optuna hyperparameter search for the 3-class XGBoost classifier.

Objective: minimise val-set log-loss (raw XGBoost, no calibration).
Log-loss is a proper scoring rule — it cannot be gamed by collapsing
predictions to low-coverage high-precision subsets the way precision alone can.

After the search the best params are used to retrain the full model
(with temperature scaling calibration on val) and save a new versioned artifact.

Usage
-----
    # Tune daily (20-day) model — 50 trials
    python -m src.tune

    # Tune swing (5-day) model
    python -m src.tune --model swing

    # More trials
    python -m src.tune --trials 100
"""

import argparse

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

from src.constants import (
    DATA_END, DATA_START,
    DAILY_HORIZON, DAILY_NEUTRAL_THRESHOLD,
    DEFAULT_HORIZON, DEFAULT_TICKERS, NEUTRAL_THRESHOLD,
    RANDOM_STATE, SWING_HORIZON, SWING_NEUTRAL_THRESHOLD,
    TRAIN_END, VAL_END,
)
from src.evaluate import evaluate_3class, threshold_analysis
from src.models import CalibratedXGB3Class, fit_temperature
from src.serialize import save_artifact
from src.splits import time_split
from src.train import _build

optuna.logging.set_verbosity(optuna.logging.WARNING)

_SEARCH_SPACE = {
    "n_estimators":      (100, 600),
    "max_depth":         (3, 7),
    "learning_rate":     (0.01, 0.20),   # log scale
    "subsample":         (0.6, 1.0),
    "colsample_bytree":  (0.5, 1.0),
    "min_child_weight":  (3, 20),
    "reg_alpha":         (0.0, 2.0),
    "reg_lambda":        (0.5, 5.0),
}

_FIXED_PARAMS = dict(
    objective="multi:softprob",
    num_class=3,
    random_state=RANDOM_STATE,
    eval_metric="mlogloss",
    verbosity=0,
)


def _make_objective(X_train, y_train, X_val, y_val):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", *_SEARCH_SPACE["n_estimators"]),
            "max_depth":        trial.suggest_int("max_depth", *_SEARCH_SPACE["max_depth"]),
            "learning_rate":    trial.suggest_float("learning_rate", *_SEARCH_SPACE["learning_rate"], log=True),
            "subsample":        trial.suggest_float("subsample", *_SEARCH_SPACE["subsample"]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *_SEARCH_SPACE["colsample_bytree"]),
            "min_child_weight": trial.suggest_int("min_child_weight", *_SEARCH_SPACE["min_child_weight"]),
            "reg_alpha":        trial.suggest_float("reg_alpha", *_SEARCH_SPACE["reg_alpha"]),
            "reg_lambda":       trial.suggest_float("reg_lambda", *_SEARCH_SPACE["reg_lambda"]),
            **_FIXED_PARAMS,
        }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)
        return log_loss(y_val, proba)

    return objective


def tune(
    model: str = "daily",
    n_trials: int = 50,
) -> dict:
    """
    Run Optuna search and retrain with best params.

    Parameters
    ----------
    model    : "daily" (20d) or "swing" (5d)
    n_trials : number of Optuna trials

    Returns the best hyperparameter dict.
    """
    if model == "swing":
        horizon           = SWING_HORIZON
        neutral_threshold = SWING_NEUTRAL_THRESHOLD
        model_name        = "xgb_clf3_5d"
        clf_version       = "1.2.0"
    elif model == "daily":
        horizon           = DAILY_HORIZON
        neutral_threshold = DAILY_NEUTRAL_THRESHOLD
        model_name        = "xgb_clf3_3d"
        clf_version       = "1.0.0"
    else:  # leaps
        horizon           = DEFAULT_HORIZON
        neutral_threshold = NEUTRAL_THRESHOLD
        model_name        = "xgb_clf3"
        clf_version       = "1.4.0"

    print(f"=== Optuna Tuning — {model} model (horizon={horizon}d, ±{neutral_threshold:.0%} neutral) ===")
    print(f"  Trials: {n_trials}\n")

    print("  Building dataset...")
    X, y, _ = _build(
        DEFAULT_TICKERS,
        DATA_START,
        DATA_END,
        target="3class",
        horizon=horizon,
        neutral_threshold=neutral_threshold,
    )
    X_train, X_val, X_test = time_split(X, TRAIN_END, VAL_END)
    y_train, y_val, y_test = time_split(y, TRAIN_END, VAL_END)
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}\n")

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(_make_objective(X_train, y_train, X_val, y_val), n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"\n  Best val log-loss: {study.best_value:.6f}")
    print("  Best params:")
    for k, v in best.items():
        print(f"    {k:<22} {v}")

    # Baseline: default params log-loss
    default_model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, **_FIXED_PARAMS,
    )
    default_model.fit(X_train, y_train)
    default_ll = log_loss(y_val, default_model.predict_proba(X_val))
    improvement = (default_ll - study.best_value) / default_ll * 100
    print(f"\n  Default log-loss: {default_ll:.6f}  →  improvement: {improvement:+.2f}%\n")

    # Retrain with best params + temperature scaling calibration on val
    print(f"  Retraining with best params + temperature scaling...")
    best_model = XGBClassifier(**{**best, **_FIXED_PARAMS})
    best_model.fit(X_train, y_train)
    raw_val = best_model.predict_proba(X_val)
    T = fit_temperature(raw_val, y_val.values)
    print(f"  Temperature: {T:.4f}  ({'softening' if T > 1 else 'sharpening'} raw probabilities)")
    calibrated = CalibratedXGB3Class(best_model, T)

    print(f"\n=== Test set evaluation ({model_name} v{clf_version}) ===")
    evaluate_3class(calibrated, X_test, y_test)
    threshold_analysis(calibrated, X_test, y_test)

    save_artifact(
        calibrated, list(X_train.columns),
        version=clf_version, ticker="multi", model_name=model_name,
    )
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for XGBoost classifier.")
    parser.add_argument("--model", choices=["leaps", "swing", "daily"], default="leaps",
                        help="leaps=20d (default) | swing=5d | daily=3d")
    parser.add_argument("--trials", type=int, default=50,
                        help="Number of Optuna trials (default: 50).")
    args = parser.parse_args()
    tune(model=args.model, n_trials=args.trials)
