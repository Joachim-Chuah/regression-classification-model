"""
Massive / Polygon data pulls with parquet caching.

Base URLs
---------
  Massive-branded : https://api.massive.com   (short-volume, short-interest)
  Polygon-branded : https://api.polygon.io    (news sentiment, options chain)

Both use the same API key (MASSIVE_API_KEY).
"""

import os
import time
from datetime import date
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests

import src.env  # noqa: F401 — loads .env on import

RAW_DIR      = Path(__file__).parent.parent / "data" / "raw"
MASSIVE_BASE = "https://api.massive.com"
POLYGON_BASE = "https://api.polygon.io"

_SENTIMENT = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _key() -> str:
    k = os.environ.get("MASSIVE_API_KEY")
    if not k:
        raise RuntimeError("MASSIVE_API_KEY not set — add it to .env")
    return k


def _get(base: str, path: str, params: dict) -> dict:
    p = {**params, "apiKey": _key()}
    for attempt in range(4):
        try:
            r = requests.get(f"{base}{path}", params=p, timeout=30)
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
            print(f"  [massive] {path} attempt {attempt+1} failed ({e}), retry {wait}s...")
            time.sleep(wait)


def _paginate(base: str, path: str, params: dict, max_rows: int = 50_000) -> list:
    """Follow next_url cursors to collect all results."""
    rows, current = [], dict(params)
    while True:
        data  = _get(base, path, current)
        rows += data.get("results", [])
        nxt   = data.get("next_url")
        if not nxt or len(rows) >= max_rows:
            break
        cursor = parse_qs(urlparse(nxt).query).get("cursor", [None])[0]
        if not cursor:
            break
        current = {"cursor": cursor}   # cursor replaces all other params
    return rows


def _today_cache(prefix: str, ticker: str, ext: str = "parquet") -> Path:
    """Return today's cache path, deleting stale files for the same prefix+ticker."""
    safe  = ticker.upper().replace("^", "").replace("-", "_")
    today = date.today().isoformat()
    path  = RAW_DIR / f"{prefix}_{safe}_{today}.{ext}"
    for old in RAW_DIR.glob(f"{prefix}_{safe}_*.{ext}"):
        if old != path:
            old.unlink(missing_ok=True)
    return path


# ---------------------------------------------------------------------------
# Public pulls
# ---------------------------------------------------------------------------

def pull_short_volume(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Daily short sale volume ratio from Massive FINRA feed.
    Returns DataFrame indexed by date, column: short_volume_ratio (0-100 scale).
    Cached once per day.
    """
    cache = _today_cache("massive_shortvol", ticker)
    if cache.exists():
        return pd.read_parquet(cache)

    rows = _paginate(
        MASSIVE_BASE, "/stocks/v1/short-volume",
        {"ticker": ticker, "date.gte": start, "date.lte": end,
         "limit": 1000, "sort": "date.asc"},
    )
    if not rows:
        return pd.DataFrame(columns=["short_volume_ratio"])

    df = (
        pd.DataFrame(rows)
        .assign(date=lambda d: pd.to_datetime(d["date"]))
        .set_index("date")
        .sort_index()[["short_volume_ratio"]]
    )
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def pull_short_interest(ticker: str) -> pd.DataFrame:
    """
    Biweekly FINRA short interest + days-to-cover.
    Returns DataFrame indexed by settlement_date, columns: short_interest, days_to_cover.
    Cached once per day.
    """
    cache = _today_cache("massive_shortint", ticker)
    if cache.exists():
        return pd.read_parquet(cache)

    rows = _paginate(
        MASSIVE_BASE, "/stocks/v1/short-interest",
        {"ticker": ticker, "limit": 1000, "sort": "settlement_date.asc"},
    )
    if not rows:
        return pd.DataFrame(columns=["short_interest", "days_to_cover"])

    df = (
        pd.DataFrame(rows)
        .assign(date=lambda d: pd.to_datetime(d["settlement_date"]))
        .set_index("date")
        .sort_index()[["short_interest", "days_to_cover"]]
    )
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def pull_news_sentiment(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Daily aggregated news sentiment from Polygon news API (same key).
    Scores: positive=+1, neutral=0, negative=-1, averaged per calendar day.
    Returns DataFrame indexed by date, column: news_sentiment.
    Cached once per day.
    """
    cache = _today_cache("poly_sentiment", ticker)
    if cache.exists():
        return pd.read_parquet(cache)

    raw = _paginate(
        POLYGON_BASE, "/v2/reference/news",
        {"ticker": ticker,
         "published_utc.gte": start,
         "published_utc.lte": end,
         "limit": 1000, "sort": "published_utc", "order": "asc"},
        max_rows=100_000,
    )

    records = []
    for article in raw:
        day = article.get("published_utc", "")[:10]
        for insight in article.get("insights", []):
            if insight.get("ticker") == ticker:
                records.append({
                    "date":      day,
                    "sentiment": _SENTIMENT.get(insight.get("sentiment", "neutral"), 0.0),
                })

    if not records:
        return pd.DataFrame(columns=["news_sentiment"])

    daily = (
        pd.DataFrame(records)
        .assign(date=lambda d: pd.to_datetime(d["date"]))
        .groupby("date")["sentiment"]
        .mean()
        .rename("news_sentiment")
        .to_frame()
        .sort_index()
    )
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(cache)
    return daily


def pull_option_snapshot(ticker: str, current_price: float) -> dict:
    """
    Current options chain snapshot from Polygon (same key).
    Filters to near-term contracts (7-60 days to expiry) and computes:
      - put_call_oi_ratio : put OI / call OI
      - iv_atm            : implied vol at nearest strike to current_price
      - iv_skew           : avg OTM-put IV minus avg OTM-call IV (5-15% from ATM)

    Not cached — always live. Returns dict of floats (None if unavailable).
    """
    today    = pd.Timestamp.today()
    min_exp  = (today + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    max_exp  = (today + pd.Timedelta(days=60)).strftime("%Y-%m-%d")

    rows = _paginate(
        POLYGON_BASE, f"/v3/snapshot/options/{ticker}",
        {"expiration_date.gte": min_exp,
         "expiration_date.lte": max_exp,
         "limit": 250},
        max_rows=2_000,
    )
    if not rows:
        return {"put_call_oi_ratio": None, "iv_atm": None, "iv_skew": None}

    records = []
    for r in rows:
        d = r.get("details", {})
        records.append({
            "type":   d.get("contract_type"),
            "strike": d.get("strike_price"),
            "iv":     r.get("implied_volatility"),
            "oi":     r.get("open_interest", 0) or 0,
        })

    df = pd.DataFrame(records).dropna(subset=["iv", "strike"])
    if df.empty:
        return {"put_call_oi_ratio": None, "iv_atm": None, "iv_skew": None}

    # Put/call OI ratio
    put_oi  = df[df["type"] == "put"]["oi"].sum()
    call_oi = df[df["type"] == "call"]["oi"].sum()
    pc_ratio = float(put_oi / call_oi) if call_oi > 0 else None

    # ATM IV — nearest strike to current price
    df["dist"] = (df["strike"] - current_price).abs()
    atm_idx  = df["dist"].idxmin()
    iv_atm   = float(df.loc[atm_idx, "iv"]) if atm_idx is not None else None

    # IV skew — OTM puts (85-95% of price) vs OTM calls (105-115%)
    otm_puts  = df[(df["type"] == "put")  &
                   (df["strike"] > current_price * 0.85) &
                   (df["strike"] < current_price * 0.95)]
    otm_calls = df[(df["type"] == "call") &
                   (df["strike"] > current_price * 1.05) &
                   (df["strike"] < current_price * 1.15)]
    iv_skew = None
    if len(otm_puts) > 0 and len(otm_calls) > 0:
        iv_skew = float(otm_puts["iv"].mean() - otm_calls["iv"].mean())

    return {"put_call_oi_ratio": pc_ratio, "iv_atm": iv_atm, "iv_skew": iv_skew}
