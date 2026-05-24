"""
Financial Modeling Prep (FMP) data pulls with parquet caching.
Base URL: https://financialmodelingprep.com/stable
"""

import os
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

import src.env  # noqa: F401 — loads .env on import

RAW_DIR  = Path(__file__).parent.parent / "data" / "raw"
FMP_BASE = "https://financialmodelingprep.com/stable"

_GRADE_SCORE = {
    "upgrade":   1.0,
    "downgrade": -1.0,
    "initiate":  0.5,   # new coverage with bullish tone counts as mild positive
    "reiterate": 0.0,
    "maintain":  0.0,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _key() -> str:
    k = os.environ.get("FMP_API_KEY")
    if not k:
        raise RuntimeError("FMP_API_KEY not set — add it to .env")
    return k


def _get(path: str, params: dict = None) -> list | dict:
    p = {**(params or {}), "apikey": _key()}
    for attempt in range(4):
        try:
            r = requests.get(f"{FMP_BASE}{path}", params=p, timeout=30)
            if r.status_code in (400, 401, 403, 404, 422):
                r.raise_for_status()   # client error — don't retry
            r.raise_for_status()
            return r.json()
        except requests.HTTPError:
            raise
        except Exception as e:
            if attempt == 3:
                raise
            wait = 5 * (2 ** attempt)
            print(f"  [fmp] {path} attempt {attempt+1} failed ({e}), retry {wait}s...")
            time.sleep(wait)


def _today_cache(prefix: str, ticker: str) -> Path:
    safe  = ticker.upper().replace("^", "").replace("-", "_")
    today = date.today().isoformat()
    path  = RAW_DIR / f"{prefix}_{safe}_{today}.parquet"
    for old in RAW_DIR.glob(f"{prefix}_{safe}_*.parquet"):
        if old != path:
            old.unlink(missing_ok=True)
    return path


# ---------------------------------------------------------------------------
# Public pulls
# ---------------------------------------------------------------------------

def pull_analyst_grades(ticker: str) -> pd.DataFrame:
    """
    Historical analyst grade changes from FMP.
    Returns DataFrame indexed by date, column: grade_score.
      +1.0 = upgrade, -1.0 = downgrade, +0.5 = new coverage, 0.0 = maintain/reiterate.
    Cached once per day.
    """
    cache = _today_cache("fmp_grades", ticker)
    if cache.exists():
        return pd.read_parquet(cache)

    data = _get("/grades", {"symbol": ticker, "limit": 1000})

    if not data or not isinstance(data, list):
        df = pd.DataFrame({"grade_score": pd.Series(dtype=float)})
        df.index.name = "date"
        return df

    rows = [
        {"date": item["date"],
         "grade_score": _GRADE_SCORE.get(item.get("action", "").lower(), 0.0)}
        for item in data
    ]
    # Multiple firms can issue grades on the same day — sum into one daily score.
    df = (
        pd.DataFrame(rows)
        .assign(date=lambda d: pd.to_datetime(d["date"]))
        .groupby("date")["grade_score"]
        .sum()
        .to_frame()
        .sort_index()
    )
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def pull_insider_trades(ticker: str) -> pd.DataFrame:
    """
    Insider transactions from FMP (SEC Form 4).
    Returns DataFrame indexed by transaction date, column: insider_score.
      +1.0 = acquisition (buy), -1.0 = disposition (sell).
    Cached once per day.
    """
    cache = _today_cache("fmp_insider", ticker)
    if cache.exists():
        return pd.read_parquet(cache)

    data = _get("/insider-trading/search", {"symbol": ticker, "limit": 1000})

    if not data or not isinstance(data, list):
        df = pd.DataFrame({"insider_score": pd.Series(dtype=float)})
        df.index.name = "date"
        return df

    rows = []
    for item in data:
        acq = item.get("acquisitionOrDisposition", "")
        score = 1.0 if acq == "A" else -1.0 if acq == "D" else 0.0
        dt = item.get("transactionDate") or item.get("filingDate")
        if dt:
            rows.append({"date": dt, "insider_score": score})

    if not rows:
        df = pd.DataFrame({"insider_score": pd.Series(dtype=float)})
        df.index.name = "date"
        return df

    # Multiple transactions can occur on the same day — sum into one daily score.
    df = (
        pd.DataFrame(rows)
        .assign(date=lambda d: pd.to_datetime(d["date"]))
        .groupby("date")["insider_score"]
        .sum()
        .to_frame()
        .sort_index()
    )
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def pull_price_target(ticker: str) -> float | None:
    """
    Current analyst consensus price target.
    Not cached — always fetched live (used at inference time only).
    """
    try:
        data = _get("/price-target-consensus", {"symbol": ticker})
        if isinstance(data, list) and data:
            return data[0].get("targetConsensus")
        if isinstance(data, dict):
            return data.get("targetConsensus")
    except Exception as e:
        print(f"  [fmp] price_target {ticker}: {e}")
    return None
