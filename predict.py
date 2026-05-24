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
import csv
import json
import sys
from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models import CalibratedXGB3Class  # noqa: F401
from src.constants import DEFAULT_TICKERS, TICKER_SECTOR_ETF
from src.data import pull, pull_earnings_dates, pull_macro
from src.data_massive import pull_short_volume, pull_short_interest, pull_news_sentiment, pull_option_snapshot
from src.data_fmp import pull_analyst_grades, pull_insider_trades, pull_price_target, pull_quote
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
SIGNALS_DIR      = Path(__file__).parent / "artifacts" / "signals"
PNL_LOG          = Path(__file__).parent / "artifacts" / "metrics" / "pnl_log.csv"
DATA_START       = "2015-01-01"

_MODE_HORIZON = {"leaps": 20, "swing": 5, "daily": 3}

_PNL_FIELDS = [
    "call_date", "ticker", "direction", "entry_price",
    "p_up", "p_down", "model_version", "horizon_days",
    "exit_date", "exit_price", "return_pct", "hit",
]

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

def _write_signals_json(
    mode: str,
    buys: pd.DataFrame,
    sells: pd.DataFrame,
) -> None:
    calls = []
    for _, row in buys.iterrows():
        calls.append({
            "ticker":    row["ticker"],
            "direction": "up",
            "p_up":      float(row["p_up"]),
            "p_down":    float(row["p_down"]),
            "exp_ret":   float(row["exp_ret"]),
            "as_of":     row["as_of"],
        })
    for _, row in sells.iterrows():
        calls.append({
            "ticker":    row["ticker"],
            "direction": "down",
            "p_up":      float(row["p_up"]),
            "p_down":    float(row["p_down"]),
            "exp_ret":   float(row["exp_ret"]),
            "as_of":     row["as_of"],
        })
    payload = {
        "mode":         mode,
        "horizon":      _MODE_HORIZON[mode],
        "generated_at": datetime.now().isoformat(),
        "calls":        calls,
    }
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    (SIGNALS_DIR / f"signals_{mode}.json").write_text(json.dumps(payload, indent=2))


def _save_pnl_log(rows: list[dict]) -> None:
    PNL_LOG.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        row.setdefault("direction", "up")
        row.setdefault("p_down", "")
    with open(PNL_LOG, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_PNL_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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
            sector_etf    = SECTOR_MAP.get(ticker)
            ed            = pull_earnings_dates(ticker)
            current_price = float(df["close"].iloc[-1])

            external = {}
            try:
                external["short_volume"]   = pull_short_volume(ticker, DATA_START, today)
                external["short_interest"] = pull_short_interest(ticker)
                external["news_sentiment"] = pull_news_sentiment(ticker, DATA_START, today)
            except Exception as _e:
                pass
            try:
                external["analyst_grades"] = pull_analyst_grades(ticker)
                external["insider_trades"] = pull_insider_trades(ticker)
                pt = pull_price_target(ticker)
                if pt and current_price:
                    external["price_target_upside"] = (pt - current_price) / current_price
            except Exception as _e:
                pass
            try:
                external["option_snapshot"] = pull_option_snapshot(ticker, current_price)
            except Exception as _e:
                pass

            features = compute_features(
                df, macro=macro, sector_etf=sector_etf,
                earnings_dates=ed, external=external,
            )
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
    # Signal JSON — always written (used by notify.py and --log mode)
    # -----------------------------------------------------------------------
    _write_signals_json(mode, buys, sells)

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
# Interactive trade logger
# ---------------------------------------------------------------------------

def _log_mode() -> None:
    """Show latest P≥0.65 calls, let user pick which to log to pnl_log.csv.
    Auto-closes positions that have exceeded their horizon."""
    today = date.today()

    # Load existing log
    log_rows: list[dict] = []
    if PNL_LOG.exists():
        with open(PNL_LOG, newline="") as f:
            log_rows = list(csv.DictReader(f))

    # Auto-close expired open positions
    for pos in log_rows:
        if pos.get("exit_date"):
            continue
        try:
            call_dt  = date.fromisoformat(pos["call_date"])
            horizon  = int(pos.get("horizon_days") or 20)
            if (today - call_dt).days < horizon:
                continue
            price = pull_quote(pos["ticker"])
            if not price:
                continue
            entry     = float(pos["entry_price"])
            ret_pct   = round((price - entry) / entry * 100, 2)
            direction = pos.get("direction", "up")
            hit       = int(ret_pct > 0 if direction == "up" else ret_pct < 0)
            pos["exit_date"]  = today.isoformat()
            pos["exit_price"] = round(price, 2)
            pos["return_pct"] = ret_pct
            pos["hit"]        = hit
            print(f"  Auto-closed {pos['ticker']}: {ret_pct:+.2f}% ({'hit' if hit else 'miss'})")
        except Exception as e:
            print(f"  [log] Could not close {pos['ticker']}: {e}")

    # Show open positions
    open_pos = [r for r in log_rows if not r.get("exit_date")]
    if open_pos:
        print(f"\n  Open positions ({len(open_pos)}):")
        for pos in open_pos:
            days = (today - date.fromisoformat(pos["call_date"])).days
            print(f"    {pos['call_date']}  {pos.get('direction','up').upper():4}  "
                  f"{pos['ticker']:<6}  entry={pos['entry_price']}  "
                  f"horizon={pos['horizon_days']}d  held={days}d")

    # Load latest calls from all signal JSON files
    all_calls: list[dict] = []
    for mode in ["leaps", "swing", "daily"]:
        sig_path = SIGNALS_DIR / f"signals_{mode}.json"
        if not sig_path.exists():
            continue
        try:
            data = json.loads(sig_path.read_text())
            for c in data.get("calls", []):
                c["mode"]    = mode
                c["horizon"] = data.get("horizon", _MODE_HORIZON[mode])
                all_calls.append(c)
        except Exception:
            pass

    print()
    if not all_calls:
        print(f"  No P≥{CALL_THRESHOLD:.0%} calls found. Run predict.py --report first.")
        _save_pnl_log(log_rows)
        return

    print(f"  Latest calls (P≥{CALL_THRESHOLD:.0%}):")
    for i, call in enumerate(all_calls):
        p_val  = call["p_up"] if call["direction"] == "up" else call["p_down"]
        action = "BUY " if call["direction"] == "up" else "SELL"
        print(f"  [{i+1}] {action} {call['ticker']:<6}  P={p_val:.0%}  "
              f"exp={call['exp_ret']:+.1%}  ({call['mode']}, as of {call['as_of']})")

    print()
    sel = input("  Enter numbers to log (e.g. 1,3) or Enter to skip: ").strip()
    if sel:
        try:
            indices = [int(x.strip()) - 1 for x in sel.split(",") if x.strip()]
        except ValueError:
            print("  Invalid input — skipping.")
            indices = []

        for idx in indices:
            if idx < 0 or idx >= len(all_calls):
                print(f"  [{idx+1}] out of range — skipped")
                continue
            call  = all_calls[idx]
            price = None
            try:
                price = pull_quote(call["ticker"])
                if price:
                    print(f"  Live price {call['ticker']}: ${price:.2f}")
            except Exception as e:
                print(f"  [log] Could not fetch price for {call['ticker']}: {e}")
            if not price:
                raw = input(f"  Entry price for {call['ticker']} (manual): ").strip()
                try:
                    price = float(raw)
                except ValueError:
                    print(f"  Skipping {call['ticker']} — invalid price")
                    continue

            log_rows.append({
                "call_date":     today.isoformat(),
                "ticker":        call["ticker"],
                "direction":     call["direction"],
                "entry_price":   round(price, 2),
                "p_up":          call["p_up"],
                "p_down":        call["p_down"],
                "model_version": call["mode"],
                "horizon_days":  call["horizon"],
                "exit_date":     "",
                "exit_price":    "",
                "return_pct":    "",
                "hit":           "",
            })
            action = "BUY " if call["direction"] == "up" else "SELL"
            print(f"  Logged: {action} {call['ticker']} @ ${price:.2f}")

    _save_pnl_log(log_rows)
    print(f"\n  PnL log: {PNL_LOG.resolve()}")

    # Print summary stats
    closed = [r for r in log_rows if r.get("hit") != ""]
    if closed:
        hits    = sum(int(r["hit"]) for r in closed if r.get("hit") not in ("", None))
        win_pct = hits / len(closed) * 100
        rets    = [float(r["return_pct"]) for r in closed if r.get("return_pct") not in ("", None)]
        avg_ret = sum(rets) / len(rets) if rets else 0
        print(f"\n  Performance ({len(closed)} closed): {win_pct:.0f}% hit rate  avg_ret={avg_ret:+.2f}%")


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
    parser.add_argument(
        "--log",
        action="store_true",
        help="Interactive trade logger: pick calls to track in pnl_log.csv",
    )
    args = parser.parse_args()

    if args.log:
        _log_mode()
    else:
        tickers = (
            [t.upper().strip(".,; ") for t in args.tickers if t.strip(".,; ")]
            if args.tickers else DEFAULT_WATCHLIST
        )
        run(tickers, mode=args.mode, report=args.report)
