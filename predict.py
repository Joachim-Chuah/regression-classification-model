#!/usr/bin/env python3
"""
Daily signal scanner — LEAPS and swing trade modes.

Loads the latest trained classifier + regressor, fetches fresh end-of-day
data from yfinance, and prints a ranked signal table for your watchlist.

Usage
-----
    # LEAPS mode (default) — 2-6 month options
    python predict.py
    python predict.py AAPL NVDA TSLA MSFT

    # Swing mode — 1-2 week stock holds
    python predict.py --mode swing
    python predict.py --mode swing AAPL NVDA MSFT

Threshold guide (backtested precision on 2024 test set):
    ★★  P(up) ≥ 0.70  →  83% historical precision  (rare, high conviction)
    ★   P(up) ≥ 0.60  →  76% historical precision  (recommended entry bar)
        P(up) ≥ 0.50  →  57% historical precision  (directional lean only)

Model was trained on 20-day forward returns. In swing mode the same signal
is used — direction is valid, but the magnitude estimate is for 20 days.
For a 1-week hold you will typically capture 30-50% of the 20d estimate.

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
from src.data import pull, pull_macro
from src.features import compute_features

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "AMD",
    "JPM", "GS", "BAC",
    "JNJ", "UNH", "LLY",
    "XOM", "CVX",
    "SPY", "QQQ",
]

# Only XLK / XLF / XLV / XLE are pulled by pull_macro — anything else
# will not get a sector-relative-strength feature (XGBoost handles NaN natively).
SECTOR_MAP = {
    **TICKER_SECTOR_ETF,
    "NVDA": "XLK", "AMD":  "XLK", "TSLA": "XLK",
    "CRM":  "XLK", "NFLX": "XLK", "INTC": "XLK",
    "BAC":  "XLF", "WFC":  "XLF", "MS":   "XLF",
    "V":    "XLF", "MA":   "XLF",
    "PFE":  "XLV", "ABBV": "XLV", "MRK":  "XLV",
    "BMY":  "XLV", "LLY":  "XLV",
    "SLB":  "XLE", "EOG":  "XLE",
}

TRAINING_TICKERS = set(DEFAULT_TICKERS)
ARTIFACTS_DIR    = Path(__file__).parent / "artifacts"
DATA_START       = "2015-01-01"

# Mode-specific display config
MODE_CONFIG = {
    "leaps": {
        "title":        "LEAPS Signal Scanner",
        "horizon_note": "20-day model  |  target: 2-6 month options",
        "up_action":    "consider LEAPS calls",
        "dn_action":    "consider LEAPS puts",
        "ret_header":   "Exp 20d",
        "ret_scale":    1.0,
        "footer":       "Check IV rank before entering. Model does not price volatility.",
    },
    "swing": {
        "title":        "Swing Trade Scanner",
        "horizon_note": "20-day model used as proxy  |  target: 1-2 week stock holds",
        "up_action":    "consider buying stock / short-dated call",
        "dn_action":    "consider shorting / short-dated put",
        "ret_header":   "~1wk est",
        "ret_scale":    0.40,   # rough: 1-week ≈ 40% of a 20-day return
        "footer": (
            "Est 1-wk return = 40% of the 20-day model estimate (rough approximation).\n"
            "  For best swing results retrain with DEFAULT_HORIZON=5 in src/constants.py."
        ),
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
    if p_up >= 0.60:   return "★  HIGH UP    "
    if p_up >= 0.50:   return "   up         "
    if p_down >= 0.70: return "▼▼ STRONG DOWN"
    if p_down >= 0.60: return "▼  HIGH DOWN  "
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

    try:
        clf_artifact, clf_name = _load_latest("*_xgb_clf3.pkl")
        reg_artifact, reg_name = _load_latest("*_xgb_reg.pkl")
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
    print(f"  Tickers     {', '.join(tickers)}\n")

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
            features   = compute_features(df, macro=macro, sector_etf=sector_etf)
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
    # Signal table
    # -----------------------------------------------------------------------
    ret_h = cfg["ret_header"]
    print(f"\n  {'#':<3}  {'Ticker':<7}  {'P(up)':>6}  {'P(dn)':>6}  {'Signal':<16}  {ret_h:>9}  {'As of':<12}  Note")
    print("  " + "-" * 84)

    for i, row in results.iterrows():
        note = "" if row["trained"] else "⚠ not in training set"
        print(
            f"  {i+1:<3}  {row['ticker']:<7}  {row['p_up']:>5.1%}  {row['p_down']:>6.1%}"
            f"  {row['signal']:<16}  {row['exp_ret']:>+8.1%}  {row['as_of']:<12}  {note}"
        )

    # -----------------------------------------------------------------------
    # High-conviction summary
    # -----------------------------------------------------------------------
    high       = results[results["p_up"]   >= 0.60]
    strong_dn  = results[results["p_down"] >= 0.60]

    print(f"\n  {'─' * 68}")

    if not high.empty:
        print(f"\n  HIGH-CONVICTION UP (P≥0.60) — {cfg['up_action']}:")
        for _, row in high.iterrows():
            flag = "" if row["trained"] else "  ⚠ extrapolating"
            print(f"    {row['ticker']}:  P(up)={row['p_up']:.1%}  {ret_h}={row['exp_ret']:+.1%}{flag}")
    else:
        print("\n  No high-conviction UP calls today (P(up) < 0.60 for all tickers).")

    if not strong_dn.empty:
        print(f"\n  HIGH-CONVICTION DOWN (P≥0.60) — {cfg['dn_action']}:")
        for _, row in strong_dn.iterrows():
            flag = "" if row["trained"] else "  ⚠ extrapolating"
            print(f"    {row['ticker']}:  P(dn)={row['p_down']:.1%}  {ret_h}={row['exp_ret']:+.1%}{flag}")

    print(f"\n  ★★ P≥0.70 · ★ P≥0.60 · plain = weaker edge · ▼ = down signal")
    print(f"  {cfg['footer']}\n")

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
        help="Tickers to scan (default: built-in watchlist of 18 stocks)",
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
