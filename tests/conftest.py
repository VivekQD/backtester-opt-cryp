"""Shared fixtures: build a tiny synthetic options parquet."""
import datetime as dt

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _ts(n, step_min=5):
    base = dt.datetime(2026, 6, 12, 0, 0, 0, tzinfo=dt.timezone.utc)
    return [base + dt.timedelta(minutes=step_min * i) for i in range(n)]


@pytest.fixture
def tiny_parquet(tmp_path):
    """3 bars, one liquid call (BTC_USDC linear), far expiry. Returns the file path."""
    dates = _ts(3)
    rows = []
    name = "BTC_USDC-30DEC50-50000-C"
    for i, d in enumerate(dates):
        rows.append(dict(
            date=d, instrument_name=name, option_type="call", strike=50000.0,
            index_price=60000.0 + 100 * i, underlying_price=60050.0 + 100 * i,
            mark_price=5000.0 + 50 * i, bid_price=4950.0 + 50 * i, ask_price=5050.0 + 50 * i,
            mark_iv=65.0, bid_iv=64.0, ask_iv=66.0, open_interest=10.0, volume=1.0,
            delta=0.80, gamma=0.0001, theta=-1.0, vega=10.0, currency="BTC_USDC",
        ))
    cols = {k: [r[k] for r in rows] for k in rows[0]}
    table = pa.table({
        "date": pa.array(cols["date"], type=pa.timestamp("us", tz="UTC")),
        "instrument_name": pa.array(cols["instrument_name"], type=pa.string()),
        "option_type": pa.array(cols["option_type"], type=pa.string()),
        "strike": pa.array(cols["strike"], type=pa.float64()),
        "index_price": pa.array(cols["index_price"], type=pa.float64()),
        "underlying_price": pa.array(cols["underlying_price"], type=pa.float64()),
        "mark_price": pa.array(cols["mark_price"], type=pa.float64()),
        "bid_price": pa.array(cols["bid_price"], type=pa.float64()),
        "ask_price": pa.array(cols["ask_price"], type=pa.float64()),
        "mark_iv": pa.array(cols["mark_iv"], type=pa.float64()),
        "bid_iv": pa.array(cols["bid_iv"], type=pa.float64()),
        "ask_iv": pa.array(cols["ask_iv"], type=pa.float64()),
        "open_interest": pa.array(cols["open_interest"], type=pa.float64()),
        "volume": pa.array(cols["volume"], type=pa.float64()),
        "delta": pa.array(cols["delta"], type=pa.float64()),
        "gamma": pa.array(cols["gamma"], type=pa.float64()),
        "theta": pa.array(cols["theta"], type=pa.float64()),
        "vega": pa.array(cols["vega"], type=pa.float64()),
        "currency": pa.array(cols["currency"], type=pa.string()),
    })
    p = tmp_path / "tiny.parquet"
    pq.write_table(table, p)
    return str(p)
