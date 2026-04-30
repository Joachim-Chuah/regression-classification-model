"""
Data freshness and caching tests.

Unit tests (no network):
    pytest tests/test_data.py -m "not integration"

Integration tests (hits yfinance — run after market close to verify EOD data):
    pytest tests/test_data.py -m integration -v
"""

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.data import RAW_DIR, pull


# ---------------------------------------------------------------------------
# Unit tests — no network required
# ---------------------------------------------------------------------------

def test_no_cache_written_for_today(tmp_path, monkeypatch):
    """pull() with end=today must not write a parquet file to disk."""
    # Redirect RAW_DIR to a temp directory so we don't touch real cache
    monkeypatch.setattr("src.data.RAW_DIR", tmp_path)

    today = date.today().isoformat()

    # Minimal fake download so the test doesn't need network
    import pandas as pd
    import numpy as np

    fake_df = pd.DataFrame(
        {
            "Open":   [100.0],
            "High":   [101.0],
            "Low":    [99.0],
            "Close":  [100.5],
            "Volume": [1_000_000.0],
        },
        index=pd.DatetimeIndex([pd.Timestamp(today)], name="Date"),
    )

    import yfinance as yf
    monkeypatch.setattr(yf, "download", lambda *a, **kw: fake_df)

    pull("SPY", start="2020-01-01", end=today)

    cache_file = tmp_path / f"SPY_2020-01-01_{today}.parquet"
    assert not cache_file.exists(), (
        f"Cache file should NOT exist for today's data (end={today}), "
        "but it was written. Check the caching logic in data.pull()."
    )


def test_cache_written_for_past_date(tmp_path, monkeypatch):
    """pull() with end < today MUST write a parquet file (so reruns are fast)."""
    monkeypatch.setattr("src.data.RAW_DIR", tmp_path)

    yesterday = (date.today() - timedelta(days=1)).isoformat()

    fake_df = pd.DataFrame(
        {
            "Open":   [100.0],
            "High":   [101.0],
            "Low":    [99.0],
            "Close":  [100.5],
            "Volume": [1_000_000.0],
        },
        index=pd.DatetimeIndex([pd.Timestamp(yesterday)], name="Date"),
    )

    import yfinance as yf
    monkeypatch.setattr(yf, "download", lambda *a, **kw: fake_df)

    pull("SPY", start="2020-01-01", end=yesterday)

    cache_file = tmp_path / f"SPY_2020-01-01_{yesterday}.parquet"
    assert cache_file.exists(), (
        f"Cache file should exist for past date (end={yesterday}), "
        "but it was not written. Historical data should be cached."
    )


# ---------------------------------------------------------------------------
# Integration tests — require network, run after market close
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_eod_data_is_fresh():
    """
    Latest row from pull() must be within 5 calendar days of today.

    Fails if:
      - yfinance is returning stale data
      - the cache policy broke and old data is being served
      - the market has been closed for an unusual stretch (> 5 days)

    Run this after 4:30pm ET on a trading day to confirm today's close is live.
    """
    today = date.today()
    start = (today - timedelta(days=30)).isoformat()   # small window — fast download

    df = pull("SPY", start=start, end=today.isoformat())

    assert not df.empty, "pull() returned an empty DataFrame for SPY."

    latest = df.index[-1].date()
    staleness_days = (today - latest).days

    assert staleness_days <= 5, (
        f"Data looks stale: latest row is {latest} ({staleness_days} days ago). "
        f"Today is {today}. Expected data within the last 5 calendar days.\n"
        "If the market has been closed for a long weekend this may be a false alarm.\n"
        "Otherwise check that the no-cache-for-today logic in data.pull() is working."
    )


@pytest.mark.integration
def test_eod_required_columns():
    """pull() must return open, high, low, close, volume columns."""
    today = date.today()
    start = (today - timedelta(days=10)).isoformat()

    df = pull("SPY", start=start, end=today.isoformat())

    for col in ["open", "high", "low", "close", "volume"]:
        assert col in df.columns, f"Missing required column: {col}"


@pytest.mark.integration
def test_eod_no_future_rows():
    """pull() must not return rows dated after today."""
    today = date.today()
    start = (today - timedelta(days=10)).isoformat()

    df = pull("SPY", start=start, end=today.isoformat())

    future_rows = df[df.index.date > today]
    assert future_rows.empty, (
        f"pull() returned {len(future_rows)} rows dated after today ({today}). "
        "This should never happen with real market data."
    )
