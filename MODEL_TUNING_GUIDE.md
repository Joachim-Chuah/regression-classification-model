# Model Tuning Guide — r-c-model

A plain-English walkthrough of every decision made while building this model:
what we tried, why, in what order, and what each step taught us.

---

## The Big Picture First

The goal is to predict whether a stock will be up or down over the next 20
trading days (classification), and by how much (regression). The output feeds
into Rylo's confidence scorer.

Two things make this hard:

1. **Stock returns are mostly noise.** The signal-to-noise ratio is extremely
   low. You will never get 80% accuracy. A model that is right 53% of the time
   on direction, consistently, is genuinely useful.

2. **The past does not cleanly repeat.** A pattern that worked in 2019 (low
   rates, low vol) may completely break in 2022 (rate shock, bear market) or
   2024 (high rates, bull market). This is called *regime shift* and it was the
   central problem we had to solve.

Every tuning decision below was made in response to one of these two
constraints.

---

## Step 1 — Logistic Regression Baseline

**What we did:** Trained a simple logistic regression on momentum and RSI
features for a single ticker.

**Why this first:** Logistic regression is the "hello world" of classification
models. It is fast, interpretable, and gives you a floor. If XGBoost later
can't beat logistic regression, something is wrong with the data or the
features — not the model choice.

**Result:** Brier skill ≈ −0.023 (slightly worse than just predicting the
base rate every time).

**What this told us:** Momentum and RSI alone, on a single ticker, carry almost
no predictive signal for 20-day forward returns. The features were too weak,
not the model.

---

## Step 2 — Richer Technical Features

**What we added:**
- `momentum_5d`, `momentum_10d`, `momentum_21d` — price return over multiple
  lookback windows. Multiple windows capture both short-term mean-reversion
  and medium-term trend.
- `rolling_vol_21d` — realised volatility. High-vol regimes behave differently
  from low-vol regimes.
- `volume_zscore_21d` — volume relative to its trailing average. A big price
  move on high volume is more meaningful than the same move on thin volume.
- `rsi_14` — a classic momentum oscillator bounded 0–100. Gives the model a
  sense of whether a stock is "overbought" or "oversold."
- `macd_line`, `macd_hist` — measures the relationship between two exponential
  moving averages. Captures momentum acceleration/deceleration.
- `bb_position` — where the price sits inside its Bollinger Band. 0 = at the
  lower band, 1 = at the upper band. Gives the model a normalised measure of
  recent price extremity.
- `atr_pct` — Average True Range as a fraction of price. Tells the model how
  volatile this specific stock currently is, independent of its price level.
- `vs_200ma` — stock price relative to its 200-day moving average. The single
  most robust regime signal for individual stocks: above = long-term uptrend,
  below = long-term downtrend.

**Why these and not others:** These are the most widely studied, backtested
technical indicators. They are also *scale-independent* by design — RSI is
always 0–100, volume z-score is always in standard deviations, ATR is always
a fraction. A tree model does not care about scale, but scale-independent
features make the model more robust across tickers with very different price
levels (a $3 stock vs a $400 stock).

**The non-lookahead rule:** Every single feature uses only data from time t
and earlier. Rolling windows, exponential moving averages, RSI — all computed
purely from past prices. This is enforced by the test in `test_features.py`
that corrupts future rows and checks that past features are unchanged.

---

## Step 3 — Switch to XGBoost

**What we did:** Replaced logistic regression with an XGBoost gradient boosting
classifier.

**Why XGBoost:** Logistic regression assumes a linear relationship between
features and the log-odds of the outcome. Stock return prediction is non-linear
— RSI matters differently when volume is also high, momentum behaves differently
when volatility is low. XGBoost builds an ensemble of decision trees that can
capture these *interaction effects* automatically without us having to manually
engineer them.

**Why not a neural network:** Neural networks excel at unstructured data
(images, text). For tabular data with tens of features and tens of thousands of
rows, tree ensembles like XGBoost routinely outperform neural networks and are
far more interpretable through feature importance.

**Result:** Brier skill improved from −0.023 to approximately −0.007 on a
single ticker.

---

## Step 4 — Multi-Ticker Training

**What we did:** Trained on 13 tickers simultaneously (mega-cap tech,
financials, healthcare, energy, SPY, QQQ) instead of a single stock.

**Why:** A model trained on one ticker sees ~2,500 rows of daily data over 10
years. That is not a lot. By stacking all tickers (sorted by date, so the
time-series structure is preserved), we get ~29,000 rows. More data → better
generalisation. Critically, the model also learns *cross-sectional* patterns:
what makes a tech stock go up during an interest rate shock is different from
what makes an energy stock go up, and the model learns both.

**The discipline required:** Every row is still labelled correctly for its own
ticker. A row for JPM in March 2020 gets a label based on JPM's own 20-day
forward return, not SPY's. There is no cross-contamination.

**Result:** Brier skill moved from −0.007 to +0.004 (first time we beat the
baseline).

---

## Step 5 — Macro Features

**What we added:**
- `vix` — the CBOE Volatility Index. High VIX = market fear = stocks tend to
  be selling off.
- `vix_change_5d` — the *rate of change* of VIX. Fear accelerating is more
  meaningful than fear being elevated.
- `spy_return_20d` — what the broad market has done recently. Stock returns
  are highly correlated with the market; this lets the model separate alpha
  from beta.
- `spy_vs_200ma` — is the overall market in a bull or bear regime?
- `yield_10y` — the 10-year Treasury yield. Rising rates hurt growth stocks
  but may help financials.
- `yield_change_20d` — the direction and speed of yield moves.
- `yield_curve` — the 10Y minus 3M Treasury spread. An inverted yield curve
  (negative value) historically precedes recessions.
- Sector ETF 20-day returns (`XLK`, `XLF`, `XLV`, `XLE`) — is this stock's
  sector in a relative tailwind or headwind?

**Why macro matters:** A stock does not move in isolation. A great earnings
report during a VIX spike (e.g. Lehman weekend) will still likely go down. The
model needs market context to distinguish company-specific signal from
macro-driven noise.

---

## Step 6 — The Recall Problem (Binary Classifier Collapse)

**What happened:** The binary classifier (predict up/down) achieved recall of
1.0 — it predicted "up" for every single row. This is not a good model; it is
a broken one.

**Why this happens:** When a dataset has more "up" rows than "down" rows (which
is true for long-only stock universes over long time periods), a gradient
boosting model can maximise training accuracy by just predicting "up" always.
Even with class_weight="balanced," the probabilities were all compressed into a
tiny band (0.612 to 0.631) — the model was not discriminating at all.

**Root cause:** Binary up/down labels include a huge number of "noise" rows —
days where the forward return was +0.3% or −0.1%. The model literally cannot
learn to distinguish these because the signal is below the noise floor. Asking
the model to predict whether a stock will be up by 0.1% or down by 0.1% in 20
days is asking it to predict coin flips.

---

## Step 7 — 3-Class Labels with a Neutral Zone

**What we did:** Replaced binary labels with 3-class labels:
- **0 = down** (20-day forward return < −2%)
- **1 = neutral** (return within ±2%)
- **2 = up** (return > +2%)

**Why this fixes the problem:** By explicitly flagging the near-zero-return
rows as "neutral" rather than forcing them to be up or down, we removed the
noise from the training signal. The model now only needs to distinguish
*meaningful* up moves from *meaningful* down moves. The neutral class acts as
a trash bin for the unlearnable rows.

**Why ±2% specifically:** The 20-day forward return distribution has most of
its mass within ±5%. A 2% threshold keeps roughly 25% of rows in each
directional class and 25% as neutral, which gives the model enough examples
of each class to learn from.

**For Rylo:** The 3-class model's output is a probability vector
`[P(down), P(neutral), P(up)]`. For a confidence score, use `P(up)` directly
or combine: `confidence = P(up) × |expected_return|` using the regressor.

---

## Step 8 — The Regime Shift Problem (92% Down Predictions)

**What happened:** After switching to 3-class labels and retraining, the model
predicted "down" for 92% of the 2024 test rows. But 2024 was a bull market
(S&P 500 up ~23%). The model was catastrophically wrong in the opposite
direction from before.

**Root cause diagnosis:** The top features by importance were `vix` (raw level)
and `yield_10y` (raw level). These are *absolute* numbers, not relative ones:

- In 2022 (training data): VIX ≈ 30, yield ≈ 4.5% → market crashed → model
  learned: high VIX + high yields = down.
- In 2024 (test data): VIX ≈ 14, yield ≈ 4.3% → market rallied 23%.

VIX was actually LOW in 2024. But the yield was still high in absolute terms.
The model saw "yield near 4.5%" and fired the "2022 crash" pattern, even though
the context was completely different. This is called *regime-dependent feature
leakage*.

**The fix:** Express macro levels as *deviations from recent history* using a
trailing 252-day z-score:

```
vix_zscore = (vix_today - mean(vix over last 252 days)) / std(vix over last 252 days)
```

This transforms "VIX = 30" into "VIX is 2.1 standard deviations above its
recent average." That is regime-agnostic information. Whether yields are at
1% or 5%, a z-score of +2 means the same thing: rates are elevated relative
to the recent past.

---

## Step 9 — Yield Curve Z-Score + Extended Training

**What remained broken after Step 8:** Even after z-scoring VIX and yield
levels, the raw `yield_curve` (10Y minus 3M spread) was still an absolute
level. During 2022, the curve first *inverted* (went negative) while the
market crashed. During 2024, the curve was *still inverted* but the market
rallied. The model had learned: negative yield curve = down. This was still
firing.

**Fix 1 — Z-score the yield curve too:**
```
yield_curve_zscore = (curve_today - mean(curve over last 252 days)) / std(curve)
```
In late 2022 (first inversion): z-score = −3 (extreme).
In 2024 (stable inversion): z-score ≈ 0 (normal for recent history).
The model now distinguishes "newly inverted" from "long-running inversion."

**Fix 2 — Extend training through mid-2023:**
The original `TRAIN_END = "2022-12-31"` meant the model *never saw* an inverted
yield curve in a bullish market during training. 2023 had exactly that: inverted
curve + market recovery. By moving `TRAIN_END = "2023-06-30"`, the model
learns that inversion alone does not guarantee a crash — the rate of change and
broader context matter more.

**Result after both fixes:**

| | Before (broken) | After (fixed) |
|---|---|---|
| Predicted down | 92.1% | 25.4% |
| Predicted up | 0.5% | 70.9% |
| Accuracy | 22% | 43% |
| Brier skill (up) | −0.49 | −0.07 |
| Regressor MAE | 0.130 | 0.060 |
| Directional accuracy | 0.35 | 0.53 |

---

## Why the Brier Skill Is Still Slightly Negative

A Brier skill of −0.07 means the model's *probability estimates* are slightly
worse than just predicting the base rate (e.g. "I always say P(up) = 51.6%").

This does **not** mean the model is useless. It means the probabilities are
miscalibrated by a small amount. There are two reasons:

1. **Train/test distribution shift:** The model was trained when "up" occurred
   49% of the time. In 2024, "up" occurred 51.6% of the time. The model's base
   rate is slightly off.

2. **The Brier skill for the 3-class model includes neutral rows.** When the
   actual outcome is "neutral," the model is being penalised for emitting any
   P(up) > 0. This is a harder problem than the binary classifier faced, where
   neutral rows were simply dropped.

The fix is post-hoc probability calibration (isotonic regression or Platt
scaling applied after training). This is the next logical step.

---

## The Order in Summary

```
Logistic baseline        → establishes the floor, proves features matter more than model
↓
Rich technical features  → gives the model something to learn from
↓
XGBoost                  → captures non-linear feature interactions
↓
Multi-ticker             → 12× more data, cross-sectional generalisation
↓
Macro features           → market context (VIX, yields, SPY regime)
↓
Recall=1.0 diagnosis     → model is predicting noise; need cleaner labels
↓
3-class labels           → removes near-zero-return noise from training signal
↓
92% down diagnosis       → absolute macro levels are regime-dependent
↓
Z-score normalization    → regime-agnostic features
↓
Yield curve z-score      → same fix applied to the remaining absolute level
↓
Extended training        → model must see inverted curve + bull market to learn it
```

Each step was a response to a specific failure mode, not a random guess.

---

---

# The Brain of the Project: `src/features.py`

This is the most important file in the repository. Here is why and how it works.

## Why This File Is the Brain

Every other file serves one purpose:
- `data.py` — fetches raw prices
- `labels.py` — generates training targets
- `train.py` — calls the model
- `evaluate.py` — measures the output
- `serialize.py` — saves the artifact

`features.py` is the only file that runs at **both** training time and
inference time. The golden rule in the CLAUDE.md is: **train/inference parity**.
Whatever transformation happens to a row of data during training must happen
identically when Rylo calls the model in production. If `features.py` drifts
between training and inference, the model sees completely different numbers
than it was trained on — and it will produce garbage predictions silently,
with no error message.

This is the single biggest failure mode for ML systems in production.

## How It Is Structured

```
src/features.py
│
├── _rsi()               — private helper: Wilder RSI
├── _macd()              — private helper: MACD line + histogram
├── _bollinger_position() — private helper: where price sits in its band
├── _atr_pct()           — private helper: Average True Range / price
│
└── compute_features()   — THE public function
        ├── stock-level features (momentum, RSI, vol, MACD, BB, ATR, 200MA)
        └── macro features (VIX z-score, yield z-score, SPY, sector ETF)
```

The private helpers with a leading underscore (`_`) are not meant to be called
directly — they are implementation details. Only `compute_features()` is the
external contract.

## The Function Signature

```python
def compute_features(
    df: pd.DataFrame,
    macro: pd.DataFrame | None = None,
    sector_etf: str | None = None,
) -> pd.DataFrame:
```

- `df` — an OHLCV DataFrame for one ticker, indexed by date
- `macro` — optional market-wide context from `pull_macro()`. When training,
  this is always provided. At inference, Rylo must call `pull_macro()` for the
  same date range and pass it in.
- `sector_etf` — optional ticker string (e.g. "XLK"). If provided, a
  sector-relative strength feature is added for that ticker.

**Returns:** A DataFrame with the same index as `df` and one column per
feature. Every row at time t uses only data from t and earlier.

## The No-Mutation Contract

The function never modifies its input. It creates a fresh DataFrame at the
start and writes new columns into that. The test `test_no_inplace_mutation`
enforces this. If `compute_features()` mutated its input, calling it twice
on the same data could produce different results — a subtle bug that is very
hard to debug.

## Feature-by-Feature Explanation

### `momentum_5d`, `momentum_10d`, `momentum_21d`
```python
features["momentum_5d"] = close.pct_change(5)
```
The percentage price change over 5, 10, and 21 trading days ending at t.
Three windows because different patterns operate at different speeds:
short-term mean-reversion (5d), medium-term momentum (10d), and monthly
trend (21d ≈ 1 calendar month).

### `rsi_14`
Relative Strength Index over 14 periods. Computed using Wilder's exponential
smoothing (EWM with com=13, matching the original formula). Bounded 0–100.
The RSI test `test_rsi_bounds` checks this invariant explicitly.

RSI above 70 = overbought (recent gains have been fast, may mean-revert).
RSI below 30 = oversold (recent losses have been fast, may bounce).

### `rolling_vol_21d`
```python
daily_returns.rolling(21).std() * (252**0.5)
```
Annualised realised volatility over the past 21 days. Multiplied by √252 to
express as an annual number. High volatility regimes behave differently from
low volatility regimes — the model needs to know which regime it's in.

### `volume_zscore_21d`
```python
(volume - volume.rolling(21).mean()) / volume.rolling(21).std()
```
Volume normalised to its own trailing 21-day mean and standard deviation.
A z-score of +2 means today's volume is 2 standard deviations above the
recent average — a meaningful signal that institution-sized participants are
active.

### `macd_line`, `macd_hist`
The MACD line is the difference between the 12-day and 26-day exponential
moving averages, divided by price to make it scale-independent. The
histogram is the MACD line minus its own 9-day EMA — it measures whether
momentum is accelerating or decelerating.

### `bb_position`
```python
(close - lower_band) / (upper_band - lower_band)
```
Where the price sits inside its 20-day Bollinger Band: 0 = at the lower
band, 1 = at the upper band. Values outside 0–1 mean the price has broken
out of its recent range. Bounded-style features are easier for tree models
to split on than unbounded ones.

### `atr_pct`
Average True Range (the largest of: high−low, |high−prev_close|,
|low−prev_close|) averaged over 14 days, then divided by close price. This
gives a volatility estimate for *this specific stock* right now, independent
of price level. A $3 stock with ATR 10% behaves very differently from a
$400 stock with ATR 0.5%.

### `vs_200ma`
```python
close / close.rolling(200).mean() - 1
```
How far the stock is above or below its 200-day moving average. This is the
most durable regime signal known in technical analysis. Stocks above their
200MA tend to have different return distributions than stocks below it. This
requires 200 days of history before producing a non-NaN value — rows before
that are automatically excluded by `dropna()` during training.

### `vix_zscore_252d`
```python
(vix - vix.rolling(252).mean()) / vix.rolling(252).std()
```
The VIX level expressed as standard deviations above/below its own trailing
year. This replaces the raw VIX level, which is regime-dependent (VIX=30
meant "panic" in 2017 but "normal elevated" in 2022). The z-score is always
in comparable units regardless of the macro regime.

### `yield_10y_zscore_252d`
Same logic as VIX. The raw 10Y yield (4.5% in 2022 = crash; 4.3% in 2024 =
bull market) was causing the model to fire the "2022 crash" pattern during
2024. The z-score says "rates are X sigma above their recent average" rather
than "rates are at 4.5%."

### `yield_change_20d`
The absolute change in 10Y yield over the past 20 days (not z-scored). This
is already relative — it measures the *speed* of rate movement, not the
level. Fast-rising rates are generically bad for equities regardless of
starting level.

### `yield_curve_zscore_252d`
The 10Y minus 3M Treasury spread, z-scored over a 252-day trailing window.
An inversion is not inherently bearish — it depends on *how unusual* the
inversion is relative to recent history. In late 2022 (first inversion):
z-score ≈ −3 (extreme signal). In 2024 (stable inversion): z-score ≈ 0
(the inversion is now the baseline). Without z-scoring, the model saw
"inverted" and always predicted crash.

### `spy_return_20d`, `spy_vs_200ma`
Broad market momentum and regime signal. These give the model context about
whether the market tide is coming in or going out, so it can separate a
stock's alpha from the market's beta.

### `rel_strength_20d`, `rel_strength_vs_sector`
```python
rel_strength_20d = stock_return_20d - spy_return_20d
rel_strength_vs_sector = stock_return_20d - sector_etf_return_20d
```
How much better or worse this stock did versus the market (or its sector)
over the past 20 days. A stock that is up 5% when the market is down 5% is
showing genuine relative strength. A stock that is up 5% when the market
is up 8% is actually underperforming. These normalise for market and sector
moves.

---

## The One Rule That Cannot Break

If you ever change `compute_features()`, you must also retrain the model.
The serialized artifact in `artifacts/` stores the feature names list
alongside the model weights:

```python
{"model": clf, "feature_names": [...], "version": "1.0.0", "trained_at": "..."}
```

Rylo's inference service validates that the features it computes match this
list before calling `model.predict_proba()`. A mismatch throws an error.
This is intentional — silent feature drift would be far worse than a loud error.

---

*Document generated: 2026-04-29*
