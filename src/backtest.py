"""
Walk-forward backtest for the 3-class classifier.

Strategy
--------
  Long  signal: P(up)   >= threshold  → buy at close, exit after `horizon` days
  Short signal: P(down) >= threshold  → short at close, exit after `horizon` days

Each signal is treated as an independent $1 trade. P&L is reported as the
arithmetic sum of returns (not compounded) — honest for equal-weight-per-signal
analysis with no capital constraint.

Fold-level isotonic calibration (same as production) is applied so that
P=0.65 actually means roughly 65% empirical frequency.

Usage
-----
    python -m src.backtest                         # daily model, threshold=0.65
    python -m src.backtest --threshold 0.60        # lower bar, more signals
    python -m src.backtest --model swing           # 5-day model
    python -m src.backtest --no-shorts             # long-only
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier

from src.constants import (
    DATA_END, DATA_START,
    DEFAULT_HORIZON, DEFAULT_TICKERS, NEUTRAL_THRESHOLD,
    RANDOM_STATE, SWING_HORIZON, SWING_NEUTRAL_THRESHOLD,
    TICKER_SECTOR_ETF,
)
from src.data import pull, pull_earnings_dates, pull_macro
from src.features import compute_features
from src.labels import make_3class_labels, make_returns

OUTPUT_DIR = Path(__file__).parent.parent / "artifacts" / "metrics" / "backtest"

_XGB_PARAMS = dict(
    n_estimators=125,
    max_depth=3,
    learning_rate=0.020339596830796842,
    subsample=0.6347264216551218,
    colsample_bytree=0.5424230643104554,
    min_child_weight=12,
    reg_alpha=0.0884116695730035,
    reg_lambda=4.906319185497624,
    objective="multi:softprob",
    num_class=3,
    random_state=RANDOM_STATE,
    eval_metric="mlogloss",
    verbosity=0,
)

_XGB_PARAMS_5D = dict(
    n_estimators=402,
    max_depth=3,
    learning_rate=0.037565091381403744,
    subsample=0.662636073998579,
    colsample_bytree=0.5554426822384888,
    min_child_weight=18,
    reg_alpha=1.3138805023122861,
    reg_lambda=3.5594825218564665,
    objective="multi:softprob",
    num_class=3,
    random_state=RANDOM_STATE,
    eval_metric="mlogloss",
    verbosity=0,
)


# ---------------------------------------------------------------------------
# Collect walk-forward predictions (with per-fold isotonic calibration)
# ---------------------------------------------------------------------------

def _collect_predictions(
    horizon: int,
    neutral_threshold: float,
    test_years: list[int],
    xgb_params: dict,
    cal_window_years: int = 1,
) -> pd.DataFrame:
    """
    Run the walk-forward loop with per-fold isotonic calibration and return
    a DataFrame of every test-set prediction.

    Columns: date, ticker, year, p_up, p_down, p_neutral, forward_return
    """
    print("  Building full dataset (parquet cache)...")
    macro = pull_macro(DATA_START, DATA_END)
    frames = []
    for ticker in DEFAULT_TICKERS:
        df = pull(ticker, start=DATA_START, end=DATA_END)
        sector_etf = TICKER_SECTOR_ETF.get(ticker)
        ed = pull_earnings_dates(ticker)
        X_t = compute_features(df, macro=macro, sector_etf=sector_etf, earnings_dates=ed)
        y_t = make_3class_labels(df["close"], horizon, neutral_threshold)
        r_t = make_returns(df["close"], horizon).rename("forward_return")
        combined = X_t.join(y_t.rename("label")).join(r_t).dropna()
        combined["ticker"] = ticker
        frames.append(combined)

    all_data = pd.concat(frames).sort_index()
    X_all    = all_data.drop(columns=["label", "forward_return", "ticker"])
    y_all    = all_data["label"]
    ret_all  = all_data["forward_return"]
    tick_all = all_data["ticker"]
    print(f"  Total rows: {len(X_all):,}\n")

    records = []
    for test_year in test_years:
        xgb_train_end = f"{test_year - 1 - cal_window_years}-12-31"
        cal_start     = f"{test_year - cal_window_years}-01-01"
        cal_end       = f"{test_year - 1}-12-31"
        test_start    = f"{test_year}-01-01"
        test_end      = f"{test_year}-12-31"

        X_train_xgb = X_all[X_all.index <= xgb_train_end]
        y_train_xgb = y_all[y_all.index <= xgb_train_end]
        X_cal       = X_all[(X_all.index >= cal_start) & (X_all.index <= cal_end)]
        y_cal       = y_all[(y_all.index >= cal_start) & (y_all.index <= cal_end)]
        X_test      = X_all[(X_all.index >= test_start) & (X_all.index <= test_end)]
        r_test      = ret_all[(ret_all.index >= test_start) & (ret_all.index <= test_end)]
        t_test      = tick_all[(tick_all.index >= test_start) & (tick_all.index <= test_end)]

        if len(X_train_xgb) < 1000 or len(X_cal) < 100 or len(X_test) < 100:
            print(f"  {test_year}: insufficient data, skipping")
            continue

        xgb = XGBClassifier(**xgb_params)
        xgb.fit(X_train_xgb, y_train_xgb)

        raw_cal = xgb.predict_proba(X_cal)
        calibrators = []
        for k in range(3):
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(raw_cal[:, k], (y_cal == k).astype(float))
            calibrators.append(iso)

        raw_test = xgb.predict_proba(X_test)
        proba = np.column_stack([
            calibrators[k].predict(raw_test[:, k]) for k in range(3)
        ])

        for i in range(len(X_test)):
            records.append({
                "date":           X_test.index[i],
                "ticker":         t_test.iloc[i],
                "year":           test_year,
                "p_down":         float(proba[i, 0]),
                "p_neutral":      float(proba[i, 1]),
                "p_up":           float(proba[i, 2]),
                "forward_return": float(r_test.iloc[i]),
            })

        print(f"  {test_year}: {len(X_test):,} rows  |  "
              f"P≥0.60 coverage: {(proba[:,2] >= 0.60).mean():.1%}  "
              f"P≥0.65 coverage: {(proba[:,2] >= 0.65).mean():.1%}")

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Strategy simulation
# ---------------------------------------------------------------------------

def _simulate(
    preds: pd.DataFrame,
    threshold: float,
    long_only: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    longs  = preds[preds["p_up"]   >= threshold].copy()
    shorts = preds[preds["p_down"] >= threshold].copy() if not long_only else pd.DataFrame()
    longs["trade_return"]  =  longs["forward_return"]
    if not shorts.empty:
        shorts["trade_return"] = -shorts["forward_return"]
    return longs, shorts


def _sharpe(returns: pd.Series, periods_per_year: float = 252) -> float:
    if len(returns) < 2 or returns.std() == 0:
        return float("nan")
    return (returns.mean() / returns.std()) * np.sqrt(periods_per_year)


def _max_drawdown(cum_pnl: pd.Series) -> float:
    """Max drawdown on arithmetic cumulative P&L series."""
    peak = cum_pnl.cummax()
    return float((cum_pnl - peak).min())


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _print_stats(label: str, trades: pd.DataFrame, horizon: int, base_rate: float) -> None:
    if trades.empty:
        print(f"  {label}: no trades at this threshold")
        return
    r   = trades["trade_return"]
    hit = (r > 0).mean()
    n   = len(trades)
    total_pnl  = r.sum()           # arithmetic: $total_pnl on $1×n invested
    mean_ret   = r.mean()
    median_ret = r.median()
    sharpe     = _sharpe(r, periods_per_year=252 / horizon)
    cum_pnl    = r.cumsum()
    mdd        = _max_drawdown(cum_pnl)

    print(f"  {label} ({n} trades):")
    print(f"    Hit rate          {hit:>8.1%}  (base rate: {base_rate:.1%}  lift: {hit-base_rate:+.1%})")
    print(f"    Mean per trade    {mean_ret:>+8.2%}")
    print(f"    Median per trade  {median_ret:>+8.2%}")
    print(f"    Total P&L         {total_pnl:>+8.1%}  ($total on $1-per-signal)")
    print(f"    Sharpe (per trade){sharpe:>+8.2f}")
    print(f"    Max drawdown      {mdd:>+8.2%}  (arithmetic)")
    print()


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot(
    longs: pd.DataFrame,
    shorts: pd.DataFrame,
    preds: pd.DataFrame,
    threshold: float,
    horizon: int,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Walk-Forward Backtest  |  P≥{threshold:.0%} threshold  |  horizon={horizon}d\n"
        f"Calibrated XGBoost  ·  equal $1 per signal  ·  no costs",
        fontsize=12,
    )

    all_trades = pd.concat([t for t in [longs, shorts] if not t.empty]).sort_values("date")

    # Top left: arithmetic cumulative P&L
    ax = axes[0, 0]
    if not all_trades.empty:
        cum = all_trades["trade_return"].cumsum()
        ax.plot(range(len(cum)), cum.values, color="steelblue", linewidth=1.5)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.fill_between(range(len(cum)), cum.values, 0,
                        where=cum.values >= 0, alpha=0.15, color="green")
        ax.fill_between(range(len(cum)), cum.values, 0,
                        where=cum.values < 0, alpha=0.15, color="red")
        ax.set_title("Cumulative P&L (all signals, $1 per trade)")
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Cumulative return ($)")

    # Top right: cumulative P&L by year
    ax = axes[0, 1]
    colors = {2020: "tab:blue", 2021: "tab:orange", 2022: "tab:red",
              2023: "tab:green", 2024: "tab:purple"}
    for year, grp in all_trades.groupby("year"):
        cum = grp["trade_return"].cumsum()
        ax.plot(range(len(cum)), cum.values, label=str(year), color=colors.get(year))
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Cumulative P&L by Year")
    ax.set_xlabel("Trade #")
    ax.legend(fontsize=8)

    # Bottom left: return distribution
    ax = axes[1, 0]
    if not all_trades.empty:
        ax.hist(all_trades["trade_return"], bins=50,
                color="steelblue", edgecolor="white", linewidth=0.4)
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.axvline(all_trades["trade_return"].mean(), color="orange", linestyle="--",
                   linewidth=1.2, label=f"mean={all_trades['trade_return'].mean():+.2%}")
        ax.set_title("Trade Return Distribution")
        ax.set_xlabel("Return")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    # Bottom right: hit rate by P(up) bucket
    ax = axes[1, 1]
    if not longs.empty or not preds.empty:
        src = longs if not longs.empty else preds
        bins   = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 1.01]
        labels = ["0.50–55", "0.55–60", "0.60–65", "0.65–70", "0.70–75", "0.75+"]
        # Use all predictions to build the bucket chart (not just threshold-filtered)
        all_longs = preds[preds["p_up"] >= 0.50].copy()
        all_longs["bucket"] = pd.cut(all_longs["p_up"], bins=bins, labels=labels, right=False)
        grp = all_longs.groupby("bucket", observed=True)["forward_return"].agg(
            hit_rate=lambda x: (x > 0).mean(),
            count="count",
        ).reset_index()
        base = (preds["forward_return"] > 0).mean()
        bar_colors = ["green" if h > base else "salmon" for h in grp["hit_rate"]]
        bars = ax.bar(grp["bucket"].astype(str), grp["hit_rate"],
                      color=bar_colors, edgecolor="white")
        ax.axhline(base, color="red", linestyle="--", linewidth=1,
                   label=f"base rate {base:.1%}")
        for bar, (_, row) in zip(bars, grp.iterrows()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"n={int(row['count'])}", ha="center", va="bottom", fontsize=7)
        ax.set_title("Long Hit Rate by P(up) Bucket")
        ax.set_xlabel("P(up) range")
        ax.set_ylabel("Hit rate")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", labelsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Chart saved: {output_path.resolve()}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_backtest(
    model: str = "daily",
    threshold: float = 0.65,
    long_only: bool = False,
    test_years: list[int] | None = None,
) -> pd.DataFrame:
    if test_years is None:
        test_years = [2020, 2021, 2022, 2023, 2024]

    if model == "swing":
        horizon           = SWING_HORIZON
        neutral_threshold = SWING_NEUTRAL_THRESHOLD
        xgb_params        = _XGB_PARAMS_5D
    else:
        horizon           = DEFAULT_HORIZON
        neutral_threshold = NEUTRAL_THRESHOLD
        xgb_params        = _XGB_PARAMS

    print(f"\n=== Walk-Forward Backtest ===")
    print(f"  Model      {model}  (horizon={horizon}d, ±{neutral_threshold:.0%} neutral)")
    print(f"  Threshold  P≥{threshold:.0%}")
    print(f"  Sides      {'long only' if long_only else 'long + short'}")
    print(f"  Years      {test_years}\n")

    preds = _collect_predictions(horizon, neutral_threshold, test_years, xgb_params)
    if preds.empty:
        print("No predictions collected — check data availability.")
        return preds

    base_rate_up = (preds["forward_return"] > 0).mean()
    longs, shorts = _simulate(preds, threshold, long_only)

    print(f"\n  Total predictions: {len(preds):,}  |  base rate (up): {base_rate_up:.1%}")
    print(f"  Long signals (P≥{threshold:.0%}): {len(longs):,}  ({len(longs)/len(preds):.1%} of all rows)")
    if not long_only:
        print(f"  Short signals (P≥{threshold:.0%}): {len(shorts):,}  ({len(shorts)/len(preds):.1%} of all rows)")

    print(f"\n=== Results ===\n")
    _print_stats("LONG signals", longs, horizon, base_rate_up)
    if not long_only and not shorts.empty:
        base_rate_dn = (preds["forward_return"] < 0).mean()
        _print_stats("SHORT signals", shorts, horizon, base_rate_dn)

    all_trades = pd.concat([t for t in [longs, shorts] if not t.empty]).sort_values("date")
    if len(all_trades) > len(longs):
        _print_stats("COMBINED", all_trades, horizon, base_rate_up)

    # Per-year breakdown
    if not longs.empty:
        print("  Year-by-year (longs):")
        print(f"  {'Year':<6}  {'Trades':>6}  {'Hit rate':>9}  {'Mean ret':>9}  {'Total P&L':>10}  Regime")
        print("  " + "-" * 60)
        regimes = {2020: "COVID crash", 2021: "bull", 2022: "bear/rates",
                   2023: "recovery", 2024: "bull"}
        for year in test_years:
            yr = longs[longs["year"] == year]
            if yr.empty:
                print(f"  {year:<6}  {'0':>6}  {'—':>9}  {'—':>9}  {'—':>10}  {regimes.get(year,'')}")
                continue
            total_pnl = yr["trade_return"].sum()
            print(
                f"  {year:<6}  {len(yr):>6}  {(yr['trade_return']>0).mean():>9.1%}"
                f"  {yr['trade_return'].mean():>+9.2%}  {total_pnl:>+10.2%}"
                f"  {regimes.get(year,'')}"
            )

    # Save chart
    tag = f"{model}_t{int(threshold*100)}_{'lo' if long_only else 'ls'}"
    _plot(longs, shorts, preds, threshold, horizon, OUTPUT_DIR / f"backtest_{tag}.png")

    return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walk-forward backtest.")
    parser.add_argument("--model", choices=["daily", "swing"], default="daily")
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--no-shorts", action="store_true")
    args = parser.parse_args()
    run_backtest(model=args.model, threshold=args.threshold, long_only=args.no_shorts)
