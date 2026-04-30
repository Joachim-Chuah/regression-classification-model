import numpy as np
import pandas as pd
from xgboost import XGBClassifier


class CalibratedXGB3Class:
    """XGBoost 3-class classifier with per-class isotonic calibration.

    Isotonic regression is fit on the held-out val set so it never sees
    training data — fitting it on training data would overfit and produce
    overconfident probabilities. After calibration, probabilities are
    renormalised to sum to 1.
    """

    def __init__(self, base: XGBClassifier, calibrators: list):
        self.base = base
        self.calibrators = calibrators   # list[IsotonicRegression], one per class

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.base.predict_proba(X)   # (N, 3)
        cal = np.column_stack([
            self.calibrators[k].predict(raw[:, k]) for k in range(3)
        ])
        cal = np.clip(cal, 0.0, 1.0)
        row_sums = cal.sum(axis=1, keepdims=True)
        return cal / np.where(row_sums == 0, 1.0, row_sums)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    @property
    def feature_importances_(self):
        return self.base.feature_importances_
