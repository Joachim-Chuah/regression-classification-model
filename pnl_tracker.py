"""
P&L tracker for scanner calls.

Usage:
  # Log today's calls (run after predict.py):
  python pnl_tracker.py log --ticker AAPL --entry 185.50 --p-up 0.78 --model-version v1.3.0

  # Mark all open positions that have reached their 20-day horizon:
  python pnl_tracker.py mark

  # Show open positions and cumulative stats:
  python pnl_tracker.py status
"""

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

LOG_PATH = Path("artifacts/metrics/pnl_log.csv")

_COLUMNS = [
    "call_date", "ticker", "entry_price", "p_up", "model_version",
    "horizon_days", "exit_date", "exit_price", "return_pct", "hit",
]


def _load() -> pd.DataFrame:
    if LOG_PATH.exists():
        return pd.read_csv(LOG_PATH, parse_dates=["call_date", "exit_date"])
    return pd.DataFrame(columns=_COLUMNS)


def _save(df: pd.DataFrame) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LOG_PATH, index=False)


def cmd_log(args) -> None:
    df = _load()
    call_date = date.today().isoformat()
    row = {
        "call_date":      call_date,
        "ticker":         args.ticker.upper(),
        "entry_price":    args.entry,
        "p_up":           args.p_up,
        "model_version":  args.model_version,
        "horizon_days":   args.horizon,
        "exit_date":      pd.NaT,
        "exit_price":     float("nan"),
        "return_pct":     float("nan"),
        "hit":            float("nan"),
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _save(df)
    exit_date = (date.today() + timedelta(days=int(args.horizon * 1.5))).isoformat()
    print(f"  Logged {args.ticker.upper()} @ ${args.entry:.2f}  P(up)={args.p_up:.0%}  "
          f"(horizon={args.horizon}d, check ~{exit_date})")


def cmd_mark(args) -> None:
    df = _load()
    open_mask = df["exit_price"].isna()
    if not open_mask.any():
        print("  No open positions.")
        return

    today = pd.Timestamp.today().normalize()
    updated = 0
    for idx, row in df[open_mask].iterrows():
        call_date = pd.Timestamp(row["call_date"])
        horizon = int(row["horizon_days"])
        due_date = call_date + pd.offsets.BusinessDay(n=horizon)
        if today < due_date:
            continue  # not yet

        ticker = row["ticker"]
        fetch_start = (call_date + timedelta(days=1)).strftime("%Y-%m-%d")
        fetch_end   = today.strftime("%Y-%m-%d")
        try:
            price_df = yf.download(ticker, start=fetch_start, end=fetch_end,
                                   auto_adjust=True, progress=False)
            if isinstance(price_df.columns, pd.MultiIndex):
                price_df.columns = price_df.columns.get_level_values(0)
            price_df.columns = price_df.columns.str.lower()
            # Find the close on/after due_date
            after_due = price_df[price_df.index >= due_date]
            if after_due.empty:
                print(f"  {ticker}: due {due_date.date()} but no price data yet — skipping")
                continue
            exit_row = after_due.iloc[0]
            exit_price = float(exit_row["close"])
            exit_date  = after_due.index[0]
        except Exception as e:
            print(f"  {ticker}: price fetch failed ({e}) — skipping")
            continue

        entry_price = float(row["entry_price"])
        ret = (exit_price - entry_price) / entry_price
        hit = 1.0 if ret > 0 else 0.0
        df.at[idx, "exit_date"]  = exit_date
        df.at[idx, "exit_price"] = exit_price
        df.at[idx, "return_pct"] = ret
        df.at[idx, "hit"]        = hit
        print(f"  Marked {ticker}: entry=${entry_price:.2f}  exit=${exit_price:.2f}  "
              f"return={ret:+.1%}  {'WIN' if hit else 'LOSS'}")
        updated += 1

    if updated:
        _save(df)
        print(f"\n  {updated} position(s) marked.")
    else:
        print("  No positions past their horizon yet.")


def cmd_status(args) -> None:
    df = _load()
    if df.empty:
        print("  No calls logged yet. Use: python pnl_tracker.py log --help")
        return

    closed = df[df["hit"].notna()].copy()
    open_  = df[df["hit"].isna()].copy()

    print(f"\n{'='*60}")
    print(f"  OPEN POSITIONS ({len(open_)})")
    print(f"{'='*60}")
    if not open_.empty:
        for _, row in open_.iterrows():
            call_date = pd.Timestamp(row["call_date"]).date()
            due = (pd.Timestamp(row["call_date"]) +
                   pd.offsets.BusinessDay(n=int(row["horizon_days"]))).date()
            print(f"  {row['ticker']:<6} called {call_date}  P(up)={row['p_up']:.0%}  "
                  f"entry=${row['entry_price']:.2f}  due {due}")
    else:
        print("  (none)")

    print(f"\n{'='*60}")
    print(f"  CLOSED POSITIONS ({len(closed)})")
    print(f"{'='*60}")
    if not closed.empty:
        for _, row in closed.iterrows():
            hit_str = "WIN " if row["hit"] else "LOSS"
            print(f"  {row['ticker']:<6} {hit_str}  {row['return_pct']:+.1%}  "
                  f"P(up)={row['p_up']:.0%}  model={row['model_version']}")

        wins     = closed[closed["hit"] == 1.0]
        losses   = closed[closed["hit"] == 0.0]
        win_rate = len(wins) / len(closed)
        avg_ret  = closed["return_pct"].mean()
        avg_win  = wins["return_pct"].mean() if len(wins) else float("nan")
        avg_loss = losses["return_pct"].mean() if len(losses) else float("nan")

        print(f"\n  Win rate:  {win_rate:.1%}  ({len(wins)}/{len(closed)})")
        print(f"  Avg return: {avg_ret:+.2%}")
        print(f"  Avg win:    {avg_win:+.2%}   Avg loss: {avg_loss:+.2%}")
        if avg_loss != 0 and avg_loss != float("nan") and len(losses) > 0:
            profit_factor = abs(avg_win * len(wins)) / abs(avg_loss * len(losses))
            print(f"  Profit factor: {profit_factor:.2f}")

        # Breakdown by model version
        if closed["model_version"].nunique() > 1:
            print(f"\n  By model version:")
            for ver, grp in closed.groupby("model_version"):
                wr = grp["hit"].mean()
                ar = grp["return_pct"].mean()
                print(f"    {ver}: win_rate={wr:.1%}  avg_ret={ar:+.2%}  n={len(grp)}")

        # Precision at different confidence bins
        bins = [(0.65, 0.70), (0.70, 0.80), (0.80, 1.01)]
        any_bins = False
        for lo, hi in bins:
            grp = closed[(closed["p_up"] >= lo) & (closed["p_up"] < hi)]
            if len(grp) >= 2:
                if not any_bins:
                    print(f"\n  Precision by confidence:")
                    any_bins = True
                print(f"    P(up)=[{lo:.0%},{hi:.0%}): win={grp['hit'].mean():.1%}  n={len(grp)}")

    print(f"{'='*60}\n")


def main():
    p = argparse.ArgumentParser(description="P&L tracker for scanner calls")
    sub = p.add_subparsers(dest="cmd", required=True)

    log_p = sub.add_parser("log", help="Log a new scanner call")
    log_p.add_argument("--ticker",        required=True)
    log_p.add_argument("--entry",         required=True, type=float, help="Entry price")
    log_p.add_argument("--p-up",          required=True, type=float, help="Model P(up), e.g. 0.78")
    log_p.add_argument("--model-version", default="v1.3.0")
    log_p.add_argument("--horizon",       default=20, type=int, help="Trading-day horizon")

    sub.add_parser("mark",   help="Fetch exit prices for positions past their horizon")
    sub.add_parser("status", help="Show open positions and cumulative stats")

    args = p.parse_args()
    {"log": cmd_log, "mark": cmd_mark, "status": cmd_status}[args.cmd](args)


if __name__ == "__main__":
    main()
