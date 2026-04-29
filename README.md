# r-c-model

A standalone ML project for predicting short-term stock direction (up/down over N trading days). Produces a serialized `.pkl` model artifact designed to slot into a separate inference service.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run the pipeline

```bash
# Pull data, train, evaluate, and save artifact to artifacts/
python -m src.train
```

By default this trains on `AAPL` using 2015–2022 as train, 2023 as val, and 2024 as test.

## Run tests

```bash
python -m pytest tests/ -v
```

## Regenerate data

`data/` is gitignored. Raw OHLCV is pulled from yfinance and cached as parquet on first run. To re-pull, delete the relevant file in `data/raw/` and rerun the pipeline.

## Artifact format

Saved to `artifacts/v{version}_{ticker}_logreg.pkl` via joblib:

```python
{
    "model": calibrated_pipeline,   # exposes predict_proba(X)
    "feature_names": [...],
    "version": "0.1.0",
    "ticker": "AAPL",
    "trained_at": "2024-...",
}
```
