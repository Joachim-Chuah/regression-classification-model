from datetime import datetime, timezone
from pathlib import Path

import joblib

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"


def save_artifact(
    model, feature_names: list, version: str, ticker: str, model_name: str = "model"
) -> Path:
    """Save model + metadata as a single joblib artifact."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "feature_names": feature_names,
        "version": version,
        "ticker": ticker,
        "model_name": model_name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    path = ARTIFACTS_DIR / f"v{version}_{ticker}_{model_name}.pkl"
    joblib.dump(artifact, path)
    print(f"Saved: {path}")
    return path


def load_artifact(path: str) -> dict:
    return joblib.load(path)
