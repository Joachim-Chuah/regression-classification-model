import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.constants import DEFAULT_HORIZON, RANDOM_STATE, TRAIN_END, VAL_END
from src.data import pull
from src.evaluate import evaluate
from src.features import compute_features
from src.labels import make_labels
from src.serialize import save_artifact
from src.splits import time_split


def build_dataset(
    ticker: str, start: str, end: str, horizon: int = DEFAULT_HORIZON
) -> tuple[pd.DataFrame, pd.Series]:
    df = pull(ticker, start=start, end=end)
    X = compute_features(df)
    y = make_labels(df["close"], horizon=horizon)
    combined = X.join(y.rename("label")).dropna()
    return combined.drop(columns=["label"]), combined["label"]


def train(ticker: str = "AAPL", version: str = "0.1.0") -> object:
    X, y = build_dataset(ticker, start="2015-01-01", end="2024-12-31")

    X_train, X_val, X_test = time_split(X, TRAIN_END, VAL_END)
    y_train, y_val, y_test = time_split(y, TRAIN_END, VAL_END)

    # Fit the base pipeline on train
    base = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
    base.fit(X_train, y_train)

    print("=== Validation (uncalibrated) ===")
    evaluate(base, X_val, y_val)

    # Calibrate on val using prefit — keeps train/val/test boundaries clean
    calibrated = CalibratedClassifierCV(base, cv="prefit", method="isotonic")
    calibrated.fit(X_val, y_val)

    print("=== Test (calibrated) ===")
    evaluate(calibrated, X_test, y_test)

    save_artifact(calibrated, list(X_train.columns), version=version, ticker=ticker)
    return calibrated


if __name__ == "__main__":
    train()
