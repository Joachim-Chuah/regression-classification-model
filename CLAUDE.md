# CLAUDE.md

Guidance for Claude (and other AI assistants) working in this repo.

## Project: r-c-model

A standalone ML project for learning how to build a regression/classification model that predicts short-term stock direction (up/down over N days). The goal is both educational and practical: produce a serialized model artifact that can later be dropped into a separate production service.

**This repo is independent.** It does not import from, depend on, or modify any other project. The only "contract" with the outside world is the shape of the final model artifact (see [Integration Target](#integration-target)).

## Integration Target

The trained model will eventually be loaded by `confidence_service.py` in a separate FastAPI app (Rylo). That service already has `model_mode` and `model_version` fields plumbed through its schema, so it can swap between rule-based scoring and a serialized model.

What this means for code in *this* repo:

- The final artifact must be a single `.pkl` file (joblib preferred over raw pickle for sklearn objects).
- It must expose `predict_proba(X)` returning calibrated probabilities.
- Feature names and order must be saved alongside the model, e.g.:
  ```python
  {"model": clf, "feature_names": [...], "version": "0.1.0", "trained_at": "..."}
  ```
- Feature engineering must live in an importable pure function so the production service can call the **exact same code** at inference time. Drift between training-time and inference-time feature computation is the single biggest failure mode for this kind of system.

## Non-Negotiable ML Constraints

These are the rules that separate a real model from a fantasy. Any code suggestion that violates one of these is wrong, regardless of how clean it looks.

1. **No lookahead bias.** Features for time `t` may only use information available *strictly before* `t`. This includes rolling windows, normalization stats, and any kind of target encoding. When in doubt, compute features per-row using only past data.
2. **Train/inference parity.** The function that computes features during training must be the same function used at inference. No "I'll just reimplement this in pandas later." Feature engineering lives in `src/features.py` and is imported everywhere.
3. **Time-based splits only.** Never use `train_test_split` with shuffling on this data. Use `TimeSeriesSplit`, an expanding window, or a hard cutoff date. Random splits leak future information into training.
4. **Labels must respect the horizon.** If predicting 5-day-forward returns, the last 5 rows of any series cannot have valid labels — drop them, don't fill them.
5. **Fit transformers on train only.** Scalers, imputers, encoders are fit on train and *applied* to test. Always use `Pipeline` to enforce this.

## Project Structure

```
r-c-model/
├── CLAUDE.md
├── README.md
├── pyproject.toml          # or requirements.txt
├── data/
│   ├── raw/                # raw yfinance pulls — gitignored
│   └── processed/          # feature matrices — gitignored
├── notebooks/              # exploration only, not production
├── src/
│   ├── __init__.py
│   ├── constants.py        # RANDOM_STATE, default tickers, horizons
│   ├── data.py             # yfinance pulls, caching
│   ├── features.py         # ALL feature engineering — single source of truth
│   ├── labels.py           # forward-return labeling
│   ├── splits.py           # time-based CV utilities
│   ├── train.py            # training entrypoint
│   ├── evaluate.py         # metrics + diagnostic plots
│   └── serialize.py        # save/load model artifacts
├── tests/
│   ├── test_features.py    # critical: catch lookahead bugs
│   ├── test_labels.py
│   └── test_splits.py
└── artifacts/              # serialized .pkl files, versioned
```

## Tech Stack

- Python 3.11+
- `yfinance` — historical price/volume data
- `pandas`, `numpy` — data manipulation
- `scikit-learn` — logistic regression, random forest, gradient boosting, calibration, time-series splits
- `joblib` — model serialization
- `matplotlib` / `seaborn` — diagnostics
- `pytest` — testing

Add libraries only when justified. No deep-learning frameworks for now — the project is about understanding fundamentals, and tabular tree models will likely outperform on this kind of data anyway.

## Workflow

1. **Pull data** (`src/data.py`) — yfinance, cache to `data/raw/`. Document tickers and date ranges used.
2. **Engineer features** (`src/features.py`) — momentum, RSI, rolling volatility, volume z-score, simple sentiment proxies. Every feature is a function of strictly past data.
3. **Generate labels** (`src/labels.py`) — binary up/down at horizon `N` (start with N=5).
4. **Split** (`src/splits.py`) — time-based, e.g. train on 2015–2022, validate on 2023, test on 2024.
5. **Train** (`src/train.py`) — start with logistic regression as an interpretable baseline, then random forest, then gradient boosting. Always wrap in `CalibratedClassifierCV` (or use `method="isotonic"` post-hoc) — Brier score depends on calibrated probabilities.
6. **Evaluate** (`src/evaluate.py`) — accuracy, precision, recall, Brier score, calibration curve, confusion matrix. Always compare against a "predict majority class" and "predict base rate" baseline.
7. **Serialize** (`src/serialize.py`) — save model + feature names + version into a single dict, write to `artifacts/`.

## Evaluation Metrics

- **Accuracy** — sanity check only; misleading on imbalanced data.
- **Precision / Recall** — per class, especially for the "up" class if that's the trade signal.
- **Brier score** — the metric that actually matters for a confidence-style model. Lower is better. Always compare against a baseline of predicting the base rate.
- **Calibration curve** — visual check that predicted 70% really means 70% empirically.

A model that looks accurate but is poorly calibrated is worse than useless for downstream confidence scoring.

## Testing

Every non-trivial function in `src/` must have a corresponding test. Tests live in `tests/` and mirror the module they cover (`src/features.py` → `tests/test_features.py`).

**What to test:**
- `test_features.py` — lookahead bias (corrupt future rows, verify past features unchanged), index preservation, no in-place mutation, expected columns present, RSI bounds.
- `test_labels.py` — last N rows are NaN, labels are binary, label direction is correct for known inputs, index preserved, valid label count.
- `test_splits.py` — no overlap between train/val/test, exhaustive row coverage, walk-forward has no future leakage, walk-forward window expands, works on both DataFrame and Series.

**Rules:**
- Use `pytest` fixtures for reusable sample data. Seed with `np.random.seed(42)`.
- Never use real yfinance calls in tests — construct synthetic DataFrames directly.
- A test that passes but doesn't actually catch bugs is worse than no test. The lookahead bias test in `test_features.py` is the most important test in this repo — keep it sharp.
- Run tests before every commit: `.venv/bin/python -m pytest tests/ -v`

## Conventions

- All feature functions take a DataFrame indexed by date and return a Series or DataFrame with the same index. No in-place mutation.
- Random seeds are fixed (`RANDOM_STATE = 42` in `src/constants.py`) so runs are reproducible.
- Artifacts are versioned: `artifacts/v0.1.0_logreg.pkl`, never `model.pkl`.
- Notebooks are for exploration. Anything that matters gets ported into `src/` and tested.
- Committing `data/` is forbidden — it's gitignored. Document how to regenerate it in `README.md`.

## Self-Check Before Suggesting Code

When in doubt, walk these three questions:

- Does this use any data from after the prediction time? *(lookahead check)*
- Will this exact code run at inference time too? *(parity check)*
- Is the train/test time boundary respected? *(leakage check)*

If any answer is "no" or "I'm not sure," stop and flag it before writing the code.
