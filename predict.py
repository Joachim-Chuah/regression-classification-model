#!/usr/bin/env python3
"""
Daily signal scanner — LEAPS, swing, and daily (3-day) modes.

Loads the latest trained classifier + regressor, fetches fresh end-of-day
data from yfinance, and prints a ranked signal table + actionable calls.
AI & semiconductor tickers are always featured at the top of the output.

Usage
-----
    python predict.py                        # LEAPS mode (20-day)
    python predict.py --mode swing           # swing mode (5-day)
    python predict.py --mode daily           # daily mode (3-day)
    python predict.py --report               # also write markdown report
    python predict.py AAPL NVDA MSFT         # custom tickers, LEAPS mode

Threshold guide (v1.4.0, backtested on 2024 test set):
    P(up) ≥ 0.65  →  ~70% precision  ← recommended entry bar
    P(up) ≥ 0.70  →  ~76% precision  (strong conviction only)

Always check IV rank before entering options. Model does not price volatility.
"""

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models import CalibratedXGB3Class  # noqa: F401
from src.constants import DEFAULT_TICKERS, TICKER_SECTOR_ETF
from src.data import pull, pull_earnings_dates, pull_macro
from src.features import compute_features

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CALL_THRESHOLD = 0.65

# Featured prominently at top of every report regardless of ranking
AI_SEMI_TICKERS = {
    # Semiconductors
    "NVDA", "AMD", "AVGO", "QCOM", "MU", "INTC", "TSM", "AMAT", "MRVL", "ARM",
    # AI software
    "PLTR", "NOW", "CRWD", "SNOW",
    # Mega-cap AI
    "MSFT", "GOOGL", "META", "AMZN",
}

DEFAULT_WATCHLIST = [
    # Mag7 / mega-cap AI
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
    # Semiconductors
    "AMD", "AVGO", "QCOM", "MU", "INTC", "TSM", "AMAT", "MRVL", "ARM",
    # AI Software
    "PLTR", "NOW", "CRWD", "SNOW",
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

SECTOR_MAP = {
    **TICKER_SECTOR_ETF,
    # Semis / tech
    "NVDA": "XLK", "AMD": "XLK", "AVGO": "XLK", "QCOM": "XLK",
    "MU":   "XLK", "INTC": "XLK", "TSM":  "XLK", "AMAT": "XLK",
    "MRVL": "XLK", "ARM":  "XLK",
    # AI software
    "PLTR": "XLK", "NOW":  "XLK", "CRWD": "XLK", "SNOW": "XLK",
    # Other tech
    "TSLA": "XLK", "CRM":  "XLK", "NFLX": "XLK",
    # Financials
    "BAC":  "XLF", "WFC":  "XLF", "MS":   "XLF", "V": "XLF",
    # Healthcare
    "LLY":  "XLV", "ABBV": "XLV", "MRK":  "XLV",
    "PFE":  "XLV", "AMGN": "XLV",
    # Energy
    "SLB":  "XLE", "EOG":  "XLE",
}

TRAINING_TICKERS = set(DEFAULT_TICKERS)
ARTIFACTS_DIR    = Path(__file__).parent / "artifacts"
REPORTS_DIR      = Path(__file__).parent / "artifacts" / "reports"
DATA_START       = "2015-01-01"

MODE_CONFIG = {
    "leaps": {
        "title":          "LEAPS Signal Scanner",
        "horizon_note":   "20-day model  |  target: 2-6 month options",
        "clf_pattern":    "*_xgb_clf3.pkl",
        "reg_pattern":    "*_xgb_reg.pkl",
        "up_action":      "LEAPS call",
        "dn_action":      "LEAPS put",
        "ret_header":     "Exp 20d",
        "footer":         "Check IV rank before entering. Model does not price volatility.",
        "precision_note": "P≥0.65 → ~70% precision (2024 backtest)",
    },
    "swing": {
        "title":          "Swing Trade Scanner",
        "horizon_note":   "5-day model  |  target: 1-2 week stock holds",
        "clf_pattern":    "*_xgb_clf3_5d.pkl",
        "reg_pattern":    "*_xgb_reg_5d.pkl",
        "up_action":      "buy stock / short-dated call",
        "dn_action":      "short / short-dated put",
        "ret_header":     "Exp 5d",
        "footer":         "5-day forward return estimate. Check liquidity before entering.",
        "precision_note": "P≥0.65 → ~72% precision (2024 backtest)",
    },
    "daily": {
        "title":          "Daily Signal Scanner",
        "horizon_note":   "3-day model  |  target: 1-3 day moves",
        "clf_pattern":    "*_xgb_clf3_3d.pkl",
        "reg_pattern":    "*_xgb_reg_5d.pkl",   # 5d regressor as proxy
        "up_action":      "buy stock / 0-1wk call",
        "dn_action":      "short / 0-1wk put",
        "ret_header":     "Exp 5d",
        "footer":         "3-day horizon. Very short — use small size.",
        "precision_note": "P≥0.65 → precision on 3d model",
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
            "Run `python -m src.tune` first to train and save the models."
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


def _write_report(
    results: pd.DataFrame,
    mode: str,
    cfg: dict,
    clf_name: str,
    reg_name: str,
    clf_version: str,
    reg_version: str,
    calls_up: pd.DataFrame,
    calls_dn: pd.DataFrame,
) -> Path:
    now      = datetime.now()
    filename = f"{now.strftime('%Y-%m-%d_%H%M')}_{mode}.md"
    path     = REPORTS_DIR / filename
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    ret_h = cfg["ret_header"]
    lines = [
        f"# Signal Report — {now.strftime('%Y-%m-%d %H:%M')}",
        "",
        f"**Mode**: {cfg['horizon_note']}  ",
        f"**Models**: clf v{clf_version} `{clf_name}` | reg v{reg_version} `{reg_name}`  ",
        f"**Universe**: {len(results)} stocks",
        "",
        "---",
        "",
    ]

    # AI & Semiconductors section
    ai_rows = results[results["ticker"].isin(AI_SEMI_TICKERS)].copy()
    if not ai_rows.empty:
        lines += [
            "## AI & Semiconductors Watch",
            "",
            f"| Ticker | P(up) | P(dn) | Signal | {ret_h} | As of |",
            "|--------|-------|-------|--------|---------|-------|",
        ]
        for _, row in ai_rows.iterrows():
            flag = " ⚠" if not row["trained"] else ""
            lines.append(
                f"| {row['ticker']}{flag} | {row['p_up']:.1%} | {row['p_down']:.1%} "
                f"| {row['signal'].strip()} | {row['exp_ret']:+.1%} | {row['as_of']} |"
            )
        lines.append("")

    # Calls section
    lines += [f"## Calls (P≥{CALL_THRESHOLD:.0%})", ""]
    if calls_up.empty and calls_dn.empty:
        top    = results.iloc[0]
        top_dn = results.sort_values("p_down", ascending=False).iloc[0]
        lines += [
            f"No calls today — no ticker reached P≥{CALL_THRESHOLD:.0%}.",
            "",
            f"- Strongest up: **{top['ticker']}** P(up)={top['p_up']:.1%}",
            f"- Strongest dn: **{top_dn['ticker']}** P(dn)={top_dn['p_down']:.1%}",
        ]
    else:
        for _, row in calls_up.iterrows():
            note = " ⚠ not in training" if not row["trained"] else ""
            lines.append(
                f"- **BUY {row['ticker']}**  P(up)={row['p_up']:.0%}  "
                f"{ret_h}={row['exp_ret']:+.1%}  → {cfg['up_action']}{note}"
            )
        for _, row in calls_dn.iterrows():
            note = " ⚠ not in training" if not row["trained"] else ""
            lines.append(
                f"- **SELL {row['ticker']}**  P(dn)={row['p_down']:.0%}  "
                f"{ret_h}={row['exp_ret']:+.1%}  → {cfg['dn_action']}{note}"
            )
    lines.append("")

    # Full rankings table
    lines += [
        "## Full Rankings",
        "",
        f"| # | Ticker | P(up) | P(dn) | Signal | {ret_h} | As of |",
        "|---|--------|-------|-------|--------|---------|-------|",
    ]
    for i, row in results.iterrows():
        flag = " ⚠" if not row["trained"] else ""
        lines.append(
            f"| {i+1} | {row['ticker']}{flag} | {row['p_up']:.1%} | {row['p_down']:.1%} "
            f"| {row['signal'].strip()} | {row['exp_ret']:+.1%} | {row['as_of']} |"
        )

    lines += [
        "",
        "---",
        f"*{cfg['footer']}*  ",
        f"*Generated: {now.strftime('%Y-%m-%d %H:%M:%S')}*",
    ]

    path.write_text("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# Main scan
# ---------------------------------------------------------------------------

def run(tickers: list[str], mode: str = "leaps", report: bool = False) -> pd.DataFrame:
    cfg   = MODE_CONFIG[mode]
    today = date.today().isoformat()

    print(f"\n{'=' * 68}")
    print(f"  {cfg['title']}  —  {today}")
    print(f"{'=' * 68}\n")

    try:
        clf_artifact, clf_name = _load_latest(cfg["clf_pattern"])
        reg_artifact, reg_name = _load_latest(cfg["reg_pattern"])
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
            X_clf   = latest[clf_features]
            X_reg   = latest[reg_features]
            proba   = clf.predict_proba(X_clf)[0]
            exp_ret = float(reg.predict(X_reg)[0])
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
    ret_h   = cfg["ret_header"]

    # -----------------------------------------------------------------------
    # AI & SEMICONDUCTORS — featured at top regardless of ranking
    # -----------------------------------------------------------------------
    ai_rows = results[results["ticker"].isin(AI_SEMI_TICKERS)].copy()
    if not ai_rows.empty:
        print(f"\n  {'─' * 56}")
        print(f"  AI & SEMICONDUCTORS  ({len(ai_rows)} tickers)")
        print(f"  {'─' * 56}")
        for _, row in ai_rows.iterrows():
            note = "  ⚠" if not row["trained"] else ""
            print(
                f"  {row['ticker']:<6}  {row['p_up']:>5.1%}↑  {row['p_down']:>5.1%}↓"
                f"  {row['signal']:<16}  {row['exp_ret']:>+7.1%}{note}"
            )
        print(f"  {'─' * 56}")

    # -----------------------------------------------------------------------
    # Full signal table
    # -----------------------------------------------------------------------
    print(f"\n  {'#':<3}  {'Ticker':<7}  {'P(up)':>6}  {'P(dn)':>6}  {'Signal':<16}  {ret_h:>8}  {'As of'}")
    print("  " + "-" * 70)
    for i, row in results.iterrows():
        note = "  ⚠" if not row["trained"] else ""
        print(
            f"  {i+1:<3}  {row['ticker']:<7}  {row['p_up']:>5.1%}  {row['p_down']:>6.1%}"
            f"  {row['signal']:<16}  {row['exp_ret']:>+7.1%}  {row['as_of']}{note}"
        )

    # -----------------------------------------------------------------------
    # TODAY'S CALLS
    # -----------------------------------------------------------------------
    buys  = results[results["p_up"]   >= CALL_THRESHOLD].copy()
    sells = results[results["p_down"] >= CALL_THRESHOLD].copy()
    div   = "─" * 52

    print(f"\n  {div}")
    print(f"  TODAY'S CALLS  (P≥{CALL_THRESHOLD:.0%}  ·  {cfg['precision_note']})")
    print(f"  {div}")

    if buys.empty and sells.empty:
        print(f"  No calls today — no ticker reached P≥{CALL_THRESHOLD:.0%}.")
        print(f"  Strongest up:  {results.iloc[0]['ticker']}  P(up)={results.iloc[0]['p_up']:.1%}")
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

    # -----------------------------------------------------------------------
    # Markdown report (optional)
    # -----------------------------------------------------------------------
    if report:
        path = _write_report(
            results, mode, cfg, clf_name, reg_name,
            clf_version, reg_version, buys, sells,
        )
        print(f"  Report saved: {path.resolve()}\n")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Signal scanner — LEAPS, swing, or daily mode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "tickers",
        nargs="*",
        help="Tickers to scan (default: built-in watchlist of 47 stocks)",
    )
    parser.add_argument(
        "--mode",
        choices=["leaps", "swing", "daily"],
        default="leaps",
        help="leaps=20d options (default) | swing=5d stocks | daily=3d moves",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Write a markdown report to artifacts/reports/",
    )
    args = parser.parse_args()

    tickers = (
        [t.upper().strip(".,; ") for t in args.tickers if t.strip(".,; ")]
        if args.tickers else DEFAULT_WATCHLIST
    )
    run(tickers, mode=args.mode, report=args.report)
