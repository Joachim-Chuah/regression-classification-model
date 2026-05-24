# r-c-model

ML signal scanner for short-term stock direction. Produces calibrated probabilities (P(up), P(neutral), P(down)) across three horizons using XGBoost trained on price, macro, and alternative data.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
brew install libomp   # macOS only — required for XGBoost
```

### API keys

Create a `.env` file in the project root:

```
MASSIVE_API_KEY=your_key   # short volume, short interest, news sentiment, options
FMP_API_KEY=your_key       # analyst grades, insider trades, price targets
FRED_API_KEY=your_key      # macro: fed funds, CPI, unemployment, NFCI
```

All three are optional — missing keys mean those features are NaN (XGBoost handles this natively).

## Running

```bash
# LEAPS mode (20-day horizon) — default
python predict.py

# Swing mode (5-day)
python predict.py --mode swing

# Daily mode (3-day)
python predict.py --mode daily

# Write markdown report to artifacts/reports/
python predict.py --report

# Custom tickers
python predict.py AAPL NVDA MSFT
```

Run after 4:30pm ET on a trading day to get signals based on the day's settled close.

## Retraining

```bash
# Retune LEAPS model (50 Optuna trials)
python -m src.tune

# Retune swing or daily
python -m src.tune --model swing
python -m src.tune --model daily
```

Retraining pulls fresh data, re-searches hyperparameters, and saves a new versioned `.pkl` to `artifacts/`.

## Automation (GitHub Actions)

Two workflows run on schedule:

| Workflow | Schedule | What it does |
|---|---|---|
| `predict.yml` | 8am + 8pm EDT daily | Runs all three modes, commits reports |
| `retrain.yml` | 8pm EDT Sunday | Re-tunes all three models, commits new `.pkl` files |

Both can also be triggered manually from the Actions tab. Requires `MASSIVE_API_KEY`, `FMP_API_KEY`, and `FRED_API_KEY` set as GitHub repository secrets.

## Features

| Group | Features |
|---|---|
| Momentum | 5d, 10d, 21d price return; vs 200-day MA |
| Technical | RSI-14, MACD line + histogram, Bollinger position, ATR% |
| Volatility | 21-day realised vol, 14-day ATR |
| Volume | 21-day volume z-score |
| Macro | VIX level + z-score + 5d change; SPY 20d/52w return + 52w drawdown; 10Y yield z-score + 20d change; yield curve z-score; IWM vs SPY; HYG/LQD ratio; fed funds, CPI, unemployment, NFCI |
| Relative strength | Stock vs SPY (20d); stock vs sector ETF (20d) |
| Short data | Short volume ratio (5d avg); days-to-cover |
| Sentiment | News sentiment (3d + 20d avg); analyst upgrade net score (30d); insider net buy score (30d) |
| Options | Put/call OI ratio; ATM IV; IV skew (inference-time only) |
| Analyst | Price target upside (inference-time only) |
| Event | Days to next earnings (capped at 90) |

## Tests

```bash
pytest tests/ -m "not integration" -v   # fast, no network
pytest tests/ -v                         # full suite
```

## Data

`data/` is gitignored. OHLCV is pulled from yfinance and cached as daily parquet files. Alternative data (short volume, sentiment, grades, insider trades) is cached once per day per ticker.

To re-pull: delete the relevant file in `data/raw/` and rerun.

## Artifact format

Saved to `artifacts/v{version}_multi_{model_name}.pkl` via joblib:

```python
{
    "model":          calibrated_model,  # predict_proba(X) → (N, 3)
    "feature_names":  [...],             # must match compute_features() output
    "version":        "1.4.0",
    "trained_at":     "2025-...",
}
```

`predict_proba` columns: `[P(down), P(neutral), P(up)]`

Signal threshold: **P(up) ≥ 0.65** → ~70% historical precision (2024 backtest). Always check IV rank before entering options — the model does not price volatility.
