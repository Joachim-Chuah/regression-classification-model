import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    brier_score_loss,
    precision_score,
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
