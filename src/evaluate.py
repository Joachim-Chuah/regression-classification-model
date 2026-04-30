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

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    rows = []
    for t in thresholds:
        mask = p_up >= t
        n = int(mask.sum())
        precision = float(actual_up[mask].mean()) if n > 0 else float("nan")
        coverage = float(mask.mean())
        lift = precision / base_rate - 1 if n > 0 else float("nan")
        rows.append({"threshold": t, "precision": precision, "coverage": coverage,
                     "n_calls": n, "lift": lift})

    df = pd.DataFrame(rows)

    print(f"\n=== Confidence Threshold Analysis (base rate: {base_rate:.1%}) ===")
    print(f"  {'Threshold':>9}  {'Precision':>9}  {'Coverage':>9}  {'N calls':>8}  {'Lift':>7}")
    print("  " + "-" * 53)
    for _, row in df.iterrows():
        prec_str = f"{row['precision']:.1%}" if not np.isnan(row["precision"]) else "  n/a"
        lift_str = f"{row['lift']:+.1%}" if not np.isnan(row["lift"]) else "  n/a"
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
