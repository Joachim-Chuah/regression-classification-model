import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from xgboost import XGBClassifier


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(raw_proba: np.ndarray, y: np.ndarray) -> float:
    """
    Find temperature T that minimises NLL of temperature-scaled probabilities on (raw_proba, y).

    T > 1 softens an overconfident model toward uniform.
    T < 1 sharpens an underconfident model.
    T = 1 leaves raw probabilities unchanged.

    Unlike per-class isotonic regression, temperature scaling operates on all
    classes simultaneously via softmax, so there is no renormalisation artifact
    and no risk of one class dominating after independent calibration.
    """
    log_p = np.log(np.clip(raw_proba, 1e-10, 1.0))
    y_int = y.astype(int)
    n = len(y_int)

    def nll(T: float) -> float:
        scaled = _softmax(log_p / T)
        return -np.log(scaled[np.arange(n), y_int] + 1e-10).mean()

    res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
    return float(res.x)


class CalibratedXGB3Class:
    """XGBoost 3-class classifier with temperature scaling calibration.

    Temperature T is fit on the held-out val set so it never sees training data.
    Calibration is applied via softmax over temperature-scaled log-probabilities,
    ensuring probabilities always sum to 1 without a separate renormalisation step.
    """

    def __init__(self, base: XGBClassifier, temperature: float):
        self.base = base
        self.temperature = temperature

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.base.predict_proba(X)          # (N, 3)
        log_p = np.log(np.clip(raw, 1e-10, 1.0)) / self.temperature
        return _softmax(log_p)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    @property
    def feature_importances_(self):
        return self.base.feature_importances_
