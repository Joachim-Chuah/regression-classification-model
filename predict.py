#!/usr/bin/env python3
"""
Daily signal scanner — LEAPS and swing trade modes.

Loads the latest trained classifier + regressor, fetches fresh end-of-day
data from yfinance, and prints a ranked signal table + actionable calls.

Usage
-----
    # LEAPS mode (default) — 2-6 month options
    python predict.py
    python predict.py AAPL NVDA TSLA MSFT

    # Swing mode — 1-2 week stock holds
    python predict.py --mode swing
    python predict.py --mode swing AAPL NVDA MSFT

Threshold guide (v1.4.0 Optuna-tuned on 19 tickers, backtested on 2024 test set):
    P(up) ≥ 0.65  →  69.6% precision  92 calls/yr  ← recommended entry bar
    P(up) ≥ 0.60  →  58.3% precision  254 calls/yr  (higher volume, lower bar)
    P(up) ≥ 0.50  →  55.8% precision  (directional lean only)

Always check IV rank before entering options. Model does not price volatility.
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Required so joblib can deserialise CalibratedXGB3Class stored in the artifact
from src.models import CalibratedXGB3Class  # noqa: F401
from src.constants import DEFAULT_TICKERS, TICKER_SECTOR_ETF
from src.data import pull, pull_earnings_dates, pull_macro
from src.features import compute_features

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CALL_THRESHOLD = 0.65   # P≥ this → actionable BUY / SELL call

DEFAULT_WATCHLIST = [
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
    # Semiconductors
    "AMD", "AVGO", "QCOM", "MU", "INTC",
    # Consumer discretionary
    "TSLA", "NKE", "MCD", "SBUX", "HD",
    # Consumer staples
    "WMT", "PG", "KO", "PEP",
    # Financials
    "JPM", "GS", "BAC", "V", "MA",
    # Healthcare
    "JNJ", "UNH", "LLY", "ABBV", "MRK",
    # Energy
    "XOM", "CVX", "SLB",
    # Communication / media
    "NFLX", "DIS",
    # Industrials
    "CAT", "HON",
    # Broad market
    "SPY", "QQQ",
]

# Only XLK / XLF / XLV / XLE are pulled by pull_macro — anything else
# will not get a sector-relative-strength feature (XGBoost handles NaN natively).
SECTOR_MAP = {
    **TICKER_SECTOR_ETF,
    # Tech / semis
    "NVDA": "XLK", "AMD": "XLK", "AVGO": "XLK", "QCOM": "XLK",
    "MU":   "XLK", "INTC": "XLK", "TSLA": "XLK",
    "CRM":  "XLK", "NFLX": "XLK",
    # Financials
    "BAC":  "XLF", "WFC":  "XLF", "MS":   "XLF",
    "V":    "XLF", "MA":   "XLF",
    # Healthcare
    "LLY":  "XLV", "ABBV": "XLV", "MRK":  "XLV",
    "PFE":  "XLV", "AMGN": "XLV",
    # Energy
    "SLB":  "XLE", "EOG":  "XLE",
    # Consumer discretionary (no dedicated sector ETF pulled — NaN OK)
    # Consumer staples (same)
}

TRAINING_TICKERS = set(DEFAULT_TICKERS)
ARTIFACTS_DIR    = Path(__file__).parent / "artifacts"
DATA_START       = "2015-01-01"

# Mode-specific display config
MODE_CONFIG = {
    "leaps": {
        "title":        "LEAPS Signal Scanner",
        "horizon_note": "20-day model  |  target: 2-6 month options",
        "up_action":    "LEAPS call",
        "dn_action":    "LEAPS put",
        "ret_header":   "Exp 20d",
        "ret_scale":    1.0,
        "footer":       "Check IV rank before entering. Model does not price volatility.",
        "precision_note": "P≥0.65 → ~70% precision (2024 backtest)",
    },
    "swing": {
        "title":        "Swing Trade Scanner",
        "horizon_note": "5-day model  |  target: 1-2 week stock holds",
        "up_action":    "buy stock / short-dated call",
        "dn_action":    "short / short-dated put",
        "ret_header":   "Exp 5d",
        "ret_scale":    1.0,
        "footer":       "5-day forward return estimate. Check liquidity before entering.",
        "precision_note": "P≥0.65 → ~72% precision (2024 backtest)",
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_latest(pattern: str) -> tuple[dict, str]:
    candidates = sorted(ARTIFACTS_DIR.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No artifact matching '{pattern}' found in {ARTIFACTS_DIR}.\n"
            "Run `python -m src.train` first to train and save the models."
        )
    path = candidates[-1]
    return joblib.load(path), path.name


def _signal_label(p_up: float, p_down: float) -> str:
    if p_up >= 0.70:   return "★★ STRONG UP  "
    if p_up >= 0.65:   return "★  UP         "
    if p_up >= 0.50:   return "   up         "
    if p_down >= 0.70: return "▼▼ STRONG DOWN"
    if p_down >= 0.65: return "▼  DOWN       "
    if p_down >= 0.50: return "   down       "
    return               "   neutral    "


# ---------------------------------------------------------------------------
# Main scan
# ---------------------------------------------------------------------------

def run(tickers: list[str], mode: str = "leaps") -> pd.DataFrame:
    cfg   = MODE_CONFIG[mode]
    today = date.today().isoformat()

    print(f"\n{'=' * 68}")
    print(f"  {cfg['title']}  —  {today}")
    print(f"{'=' * 68}\n")

    clf_pattern = "*_xgb_clf3_5d.pkl" if mode == "swing" else "*_xgb_clf3.pkl"
    reg_pattern = "*_xgb_reg_5d.pkl"  if mode == "swing" else "*_xgb_reg.pkl"
    try:
        clf_artifact, clf_name = _load_latest(clf_pattern)
        reg_artifact, reg_name = _load_latest(reg_pattern)
    except FileNotFoundError as e:
        sys.exit(str(e))

    clf          = clf_artifact["model"]
    reg          = reg_artifact["model"]
    clf_features = clf_artifact["feature_names"]
    reg_features = reg_artifact["feature_names"]
    clf_version  = clf_artifact.get("version", "?")
    reg_version  = reg_artifact.get("version", "?")

    print(f"  Classifier  v{clf_version}  ({clf_name})")
    print(f"  Regressor   v{reg_version}  ({reg_name})")
    print(f"  Mode        {cfg['horizon_note']}")
    print(f"  Tickers     {len(tickers)} stocks\n")

    print("  Pulling macro data (VIX, yields, SPY, sector ETFs)...")
    try:
        macro = pull_macro(DATA_START, today)
    except Exception as e:
        sys.exit(f"Failed to pull macro data: {e}")

    rows, errors = [], []

    for ticker in tickers:
        try:
            df = pull(ticker, start=DATA_START, end=today)

            if len(df) < 220:
                errors.append(f"  {ticker}: only {len(df)} rows — need ≥220 for all features")
                continue

            sector_etf = SECTOR_MAP.get(ticker)
            ed         = pull_earnings_dates(ticker)
            features   = compute_features(df, macro=macro, sector_etf=sector_etf, earnings_dates=ed)
            latest     = features.iloc[[-1]].copy()

            for col in clf_features:
                if col not in latest.columns:
                    latest[col] = np.nan

            X_clf = latest[clf_features]
            X_reg = latest[reg_features]

            proba   = clf.predict_proba(X_clf)[0]   # [P(down), P(neutral), P(up)]
            exp_ret = float(reg.predict(X_reg)[0]) * cfg["ret_scale"]

            rows.append({
                "ticker":    ticker,
                "p_up":      round(proba[2], 4),
                "p_neutral": round(proba[1], 4),
                "p_down":    round(proba[0], 4),
                "exp_ret":   round(exp_ret, 4),
                "signal":    _signal_label(proba[2], proba[0]),
                "trained":   ticker in TRAINING_TICKERS,
                "as_of":     str(features.index[-1].date()),
            })

        except Exception as e:
            errors.append(f"  {ticker}: {e}")

    if errors:
        print("\n  Warnings:")
        for msg in errors:
            print(msg)

    if not rows:
        print("\n  No results to display.")
        return pd.DataFrame()

    results = pd.DataFrame(rows).sort_values("p_up", ascending=False).reset_index(drop=True)

    # -----------------------------------------------------------------------
    # Full signal table
    # -----------------------------------------------------------------------
    ret_h = cfg["ret_header"]
    print(f"\n  {'#':<3}  {'Ticker':<7}  {'P(up)':>6}  {'P(dn)':>6}  {'Signal':<16}  {ret_h:>8}  {'As of'}")
    print("  " + "-" * 70)

    for i, row in results.iterrows():
        note = "  ⚠" if not row["trained"] else ""
        print(
            f"  {i+1:<3}  {row['ticker']:<7}  {row['p_up']:>5.1%}  {row['p_down']:>6.1%}"
            f"  {row['signal']:<16}  {row['exp_ret']:>+7.1%}  {row['as_of']}{note}"
        )

    # -----------------------------------------------------------------------
    # TODAY'S CALLS — actionable signals at CALL_THRESHOLD
    # -----------------------------------------------------------------------
    buys  = results[results["p_up"]   >= CALL_THRESHOLD].copy()
    sells = results[results["p_down"] >= CALL_THRESHOLD].copy()

    div = "─" * 52
    print(f"\n  {div}")
    print(f"  TODAY'S CALLS  (P≥{CALL_THRESHOLD:.0%}  ·  {cfg['precision_note']})")
    print(f"  {div}")

    if buys.empty and sells.empty:
        print(f"  No calls today — no ticker reached P≥{CALL_THRESHOLD:.0%}.")
        print(f"  Strongest up:  {results.iloc[0]['ticker']}  P(up)={results.iloc[0]['p_up']:.1%}")
        if len(results) > 0:
            top_dn = results.sort_values("p_down", ascending=False).iloc[0]
            print(f"  Strongest dn:  {top_dn['ticker']}  P(dn)={top_dn['p_down']:.1%}")
    else:
        for _, row in buys.iterrows():
            flag = "  ⚠ not in training set" if not row["trained"] else ""
            print(f"  BUY   {row['ticker']:<6}  P(up)={row['p_up']:.0%}  {ret_h}={row['exp_ret']:+.1%}  →  {cfg['up_action']}{flag}")
        for _, row in sells.iterrows():
            flag = "  ⚠ not in training set" if not row["trained"] else ""
            print(f"  SELL  {row['ticker']:<6}  P(dn)={row['p_down']:.0%}  {ret_h}={row['exp_ret']:+.1%}  →  {cfg['dn_action']}{flag}")
        n = len(buys) + len(sells)
        print(f"  {div}")
        print(f"  {n} call{'s' if n != 1 else ''} today")

    print(f"\n  {cfg['footer']}\n")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Signal scanner — LEAPS (2-6mo options) or swing (1-2wk stocks).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "tickers",
        nargs="*",
        help="Tickers to scan (default: built-in watchlist of 40 stocks)",
    )
    parser.add_argument(
        "--mode",
        choices=["leaps", "swing"],
        default="leaps",
        help="leaps = 2-6 month options (default) | swing = 1-2 week stock holds",
    )
    args = parser.parse_args()

    tickers = (
        [t.upper().strip(".,; ") for t in args.tickers if t.strip(".,; ")]
        if args.tickers else DEFAULT_WATCHLIST
    )
    run(tickers, mode=args.mode)
