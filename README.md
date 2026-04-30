# r-c-model

A standalone ML project for predicting short-term stock direction and return magnitude over N trading days. Produces serialized `.pkl` artifacts designed to slot into a separate inference service (Rylo).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# macOS only — required for XGBoost
brew install libomp
```

## How to Run (Current Workflow)
Run after 4:30pm ET on a trading day to get signals based on the latest settled close.

Run these from the `regression-classification-model` directory.

### 1) Train both models and save artifacts

```bash
python -m src.train
```

This trains:
- 3-class classifier (`xgb_clf3`, isotonic-calibrated)
- regressor (`xgb_reg`)

Artifacts are saved under `artifacts/`.

### 2) Run walk-forward evaluation and export metrics CSV

```bash
python -m src.walk_forward
```

This prints per-year fold metrics and writes:
- `artifacts/metrics/walk_forward_summary.csv`

### 3) Run the daily scanner

```bash
# default watchlist, LEAPS mode
python predict.py

# swing mode
python predict.py --mode swing

# custom tickers
python predict.py AAPL NVDA MSFT
```

### 4) Run tests

```bash
# full suite (includes integration tests that may hit live network data)
pytest tests/ -v

# stable local quick check (no integration tests)
pytest tests/ -m "not integration" -v
```

---

**Threshold guide** (backtested precision on 2024 test set):

| Signal | P(up) | Historical precision |
|---|---|---|
| ★★ STRONG UP | ≥ 0.70 | 83% |
| ★ HIGH UP | ≥ 0.60 | 76% — recommended entry bar |
| up | ≥ 0.50 | 57% — directional lean only |

Always check IV rank on your broker before entering options. The model does not price volatility.

## Train the models

```bash
# Train 3-class classifier + regressor on 13 tickers, save artifacts to artifacts/
python -m src.train
```

Training covers 2015–2023 (train + val) and evaluates on 2024 (test). You do not need to retrain before running `predict.py` — the saved artifacts are loaded directly.

**Retrain when:**
- You want to include a new year of market data (quarterly refresh)
- You change `DEFAULT_HORIZON` in `src/constants.py` (e.g. to 5 days for swing trading)
- You add new features to `src/features.py`

## Walk-forward cross-validation

Validates the model across 5 distinct market regimes (2020–2024) rather than a single test year.

```bash
python -m src.walk_forward
```

## Tests

```bash
# Fast unit tests — no network required (~1 second)
python -m pytest tests/ -m "not integration" -q

# Full suite including network/data freshness checks
# Run after 4:30pm ET to verify today's EOD data is live
python -m pytest tests/test_data.py -m integration -v

# All tests
python -m pytest tests/ -v
```

## Data

`data/` is gitignored. Raw OHLCV is pulled from yfinance and cached as parquet on first run. Historical data (any date before today) is cached permanently. Today's data is never cached — it is always re-fetched so you always get the latest close.

To re-pull historical data for a ticker, delete the relevant file in `data/raw/` and rerun.

## Artifact format

Saved to `artifacts/v{version}_{ticker}_{model_name}.pkl` via joblib:

```python
{
    "model":         calibrated_model,   # exposes predict_proba(X) and predict(X)
    "feature_names": [...],              # must match compute_features() output
    "version":       "1.1.0",
    "ticker":        "multi",
    "model_name":    "xgb_clf3",
    "trained_at":    "2025-...",
}
```

The classifier (`xgb_clf3`) returns `predict_proba(X)` with shape `(N, 3)`:
- `[:, 0]` → P(down) — 20-day return < −2%
- `[:, 1]` → P(neutral) — 20-day return within ±2%
- `[:, 2]` → P(up) — 20-day return > +2%

The regressor (`xgb_reg`) returns the expected 20-day forward return as a signed float.
