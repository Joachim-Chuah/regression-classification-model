import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    brier_score_loss,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def signal_quality_table(
    y: pd.Series,
    p_up: np.ndarray,
    forward_return: pd.Series | np.ndarray,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """
    Compute threshold-based precision/coverage/return metrics for "up" calls.

    Parameters
    ----------
    y              : 3-class labels (0=down, 1=neutral, 2=up)
    p_up           : predicted P(up)
    forward_return : realized forward return aligned to y/p_up
    thresholds     : confidence cutoffs to evaluate
    """
    if thresholds is None:
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]

    y_arr = np.asarray(y)
    p_up_arr = np.asarray(p_up)
    ret_arr = np.asarray(forward_return)
    actual_up = y_arr == 2
    base_rate = float(actual_up.mean())

    rows = []
    for t in thresholds:
        mask = p_up_arr >= t
        n_calls = int(mask.sum())
        precision = float(actual_up[mask].mean()) if n_calls > 0 else float("nan")
        coverage = float(mask.mean())
        mean_return = float(ret_arr[mask].mean()) if n_calls > 0 else float("nan")
        rows.append(
            {
                "threshold": float(t),
                "n_calls": n_calls,
                "coverage": coverage,
                "precision_up": precision,
                "mean_forward_return": mean_return,
                "lift_vs_base": (precision / base_rate - 1) if n_calls > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def topk_precision_table(
    y: pd.Series,
    p_up: np.ndarray,
    forward_return: pd.Series | np.ndarray,
    topk_fracs: list[float] | None = None,
) -> pd.DataFrame:
    """Evaluate precision/return on the highest-confidence P(up) slice."""
    if topk_fracs is None:
        topk_fracs = [0.10, 0.20]

    y_arr = np.asarray(y)
    p_up_arr = np.asarray(p_up)
    ret_arr = np.asarray(forward_return)
    actual_up = y_arr == 2
    n = len(p_up_arr)
    order = np.argsort(-p_up_arr)

    rows = []
    for frac in topk_fracs:
        k = max(1, int(np.floor(n * frac)))
        idx = order[:k]
        rows.append(
            {
                "topk_frac": float(frac),
                "n_calls": int(k),
                "precision_up": float(actual_up[idx].mean()),
                "mean_forward_return": float(ret_arr[idx].mean()),
                "min_p_up": float(p_up_arr[idx].min()),
            }
        )
    return pd.DataFrame(rows)


def evaluate(model, X: pd.DataFrame, y: pd.Series, plot: bool = False) -> dict:
    """
    Print accuracy, precision, recall, Brier score, and Brier skill score.
    Brier skill score > 0 means the model beats a naive base-rate prediction.
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    base_rate = float(y.mean())
    baseline_brier = brier_score_loss(y, np.full(len(y), base_rate))
    model_brier = brier_score_loss(y, y_proba)

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "brier_score": model_brier,
        "brier_baseline": baseline_brier,
        "brier_skill": 1 - model_brier / baseline_brier,
    }

    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        prob_true, prob_pred = calibration_curve(y, y_proba, n_bins=10)
        axes[0].plot(prob_pred, prob_true, marker="o", label="model")
        axes[0].plot([0, 1], [0, 1], linestyle="--", label="perfect")
        axes[0].set_title("Calibration Curve")
        axes[0].set_xlabel("Mean predicted probability")
        axes[0].set_ylabel("Fraction of positives")
        axes[0].legend()

        ConfusionMatrixDisplay.from_predictions(y, y_pred, ax=axes[1])
        axes[1].set_title("Confusion Matrix")

        plt.tight_layout()
        plt.show()

    return metrics


def evaluate_3class(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Evaluate a 3-class model (0=down, 1=neutral, 2=up).

    Prints per-class precision/recall/F1 and Brier skill for the
    "up" class — the signal that matters most for Rylo.
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)  # shape (N, 3)

    # Class distribution
    for cls, name in [(0, "down"), (1, "neutral"), (2, "up")]:
        actual_pct = (y == cls).mean()
        pred_pct = (y_pred == cls).mean()
        print(f"  {name:7s}: actual {actual_pct:.1%}  predicted {pred_pct:.1%}")
    print()

    print(classification_report(
        y, y_pred,
        target_names=["down", "neutral", "up"],
        zero_division=0,
    ))

    # Brier skill for the "up" class — the key metric for Rylo
    y_up = (y == 2).astype(int)
    p_up = y_proba[:, 2]
    baseline_brier = brier_score_loss(y_up, np.full(len(y_up), y_up.mean()))
    model_brier = brier_score_loss(y_up, p_up)
    brier_skill_up = 1 - model_brier / baseline_brier

    print(f"  Brier skill (up class): {brier_skill_up:.4f}")

    return {
        "accuracy": accuracy_score(y, y_pred),
        "brier_skill_up": brier_skill_up,
    }


def threshold_analysis(model, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Precision and coverage for the 'up' call at varying P(up) thresholds.

    Answers the key Rylo question: at what confidence cutoff does the signal
    become precise enough to act on, and how many signals survive that cut?
    'Lift' = precision / base_rate - 1 (how much better than guessing).
    """
    p_up = model.predict_proba(X)[:, 2]
    actual_up = (y == 2).values
    base_rate = float(actual_up.mean())
    df = signal_quality_table(
        y=y,
        p_up=p_up,
        # For this generic helper, use binary reward proxy (1 if up else 0).
        # Walk-forward reports realized returns via forward-return labels.
        forward_return=actual_up.astype(float),
        thresholds=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
    )

    print(f"\n=== Confidence Threshold Analysis (base rate: {base_rate:.1%}) ===")
    print(f"  {'Threshold':>9}  {'Precision':>9}  {'Coverage':>9}  {'N calls':>8}  {'Lift':>7}")
    print("  " + "-" * 53)
    for _, row in df.iterrows():
        prec_str = f"{row['precision_up']:.1%}" if not np.isnan(row["precision_up"]) else "  n/a"
        lift_str = f"{row['lift_vs_base']:+.1%}" if not np.isnan(row["lift_vs_base"]) else "  n/a"
        print(f"    {row['threshold']:.2f}       {prec_str}      {row['coverage']:.1%}"
              f"     {row['n_calls']:>6,}    {lift_str}")

    return df


def evaluate_regressor(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Print MAE, RMSE, R², and directional accuracy for a regression model.
    Directional accuracy: fraction of predictions where sign(pred) == sign(actual).
    A naive baseline of predicting 0 every time gives directional_accuracy = 0.5.
    """
    y_pred = model.predict(X)

    metrics = {
        "mae": mean_absolute_error(y, y_pred),
        "rmse": mean_squared_error(y, y_pred) ** 0.5,
        "r2": r2_score(y, y_pred),
        "directional_accuracy": float((np.sign(y_pred) == np.sign(y)).mean()),
    }

    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return metrics
