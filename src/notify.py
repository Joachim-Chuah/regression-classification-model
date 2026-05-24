"""
Send a consolidated Gmail notification when P≥0.65 calls fire.

Reads artifacts/signals/signals_{mode}.json written by predict.py --report.
Sends one email covering all three modes (leaps, swing, daily).
No email is sent on quiet days (no calls above threshold).

Usage (called by GitHub Actions after all predict runs):
    python -m src.notify
"""

import json
import os
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
SIGNALS_DIR   = ARTIFACTS_DIR / "signals"
MODES         = ["leaps", "swing", "daily"]
THRESHOLD     = 0.65


def _load_calls() -> list[dict]:
    today  = datetime.now().strftime("%Y-%m-%d")
    result = []
    for mode in MODES:
        path = SIGNALS_DIR / f"signals_{mode}.json"
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
            if not data.get("generated_at", "").startswith(today):
                continue
            for call in data.get("calls", []):
                call["mode"]    = mode
                call["horizon"] = data.get("horizon", 20)
                result.append(call)
        except Exception as e:
            print(f"  [notify] Error reading {path.name}: {e}")
    return result


def _build_body(calls: list[dict]) -> str:
    now   = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"Signal scan complete — {now}",
        f"{len(calls)} actionable call(s) above P≥{THRESHOLD:.0%}",
        "",
    ]

    for mode in MODES:
        mode_calls = [c for c in calls if c["mode"] == mode]
        if not mode_calls:
            continue
        horizon = mode_calls[0]["horizon"]
        lines.append(f"[{mode.upper()}]  ({horizon}-day model)")
        for c in mode_calls:
            action  = "BUY " if c["direction"] == "up" else "SELL"
            p_val   = c["p_up"] if c["direction"] == "up" else c["p_down"]
            lines.append(
                f"  {action} {c['ticker']:<6}  P={p_val:.0%}  "
                f"exp_ret={c['exp_ret']:+.1%}  (as of {c['as_of']})"
            )
        lines.append("")

    lines += [
        "To log a trade:  python predict.py --log",
        "Check IV rank before entering. Model does not price volatility.",
        "",
        "-- r-c-model signal scanner",
    ]
    return "\n".join(lines)


def send_notification(calls: list[dict]) -> None:
    user     = os.environ.get("GMAIL_USER")
    password = os.environ.get("GMAIL_APP_PASSWORD")

    if not user or not password:
        print("  [notify] GMAIL_USER or GMAIL_APP_PASSWORD not set — skipping email")
        return

    if not calls:
        print("  [notify] No calls above threshold — no email sent")
        return

    today   = datetime.now().strftime("%Y-%m-%d")
    subject = f"[r-c-model] {len(calls)} signal(s) — {today}"
    body    = _build_body(calls)

    msg             = MIMEText(body, "plain")
    msg["Subject"]  = subject
    msg["From"]     = user
    msg["To"]       = user

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(user, password)
            smtp.send_message(msg)
        print(f"  [notify] Email sent to {user}  ({len(calls)} call(s))")
    except Exception as e:
        print(f"  [notify] Failed to send email: {e}")
        raise


if __name__ == "__main__":
    calls = _load_calls()
    send_notification(calls)
