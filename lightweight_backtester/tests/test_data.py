import datetime as dt
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lightweight_backtester.config import ResearchConfig
from lightweight_backtester.data import MarketDataStore


def test_sparse_snapshot_is_asof_not_future(tmp_path):
    root = tmp_path / "currency=BTC" / "expiry_month=2026-05"
    root.mkdir(parents=True)
    path = root / "BTC-8MAY26-100-C.parquet"
    dates = [dt.datetime(2026, 5, 1, 7, 55, tzinfo=dt.timezone.utc),
             dt.datetime(2026, 5, 1, 8, 5, tzinfo=dt.timezone.utc)]
    pq.write_table(pa.table({
        "date": pa.array(dates, type=pa.timestamp("us", tz="UTC")),
        "instrument_name": [path.stem] * 2, "strike": [100., 100.],
        "index_price": [100., 101.], "underlying_price": [100., 101.],
        "mark_price": [.1, .2], "bid_price": [.09, .19], "ask_price": [.11, .21],
        "mark_iv": [60., 61.], "open_interest": [20., 20.], "volume": [1., 1.],
        "delta": [.5, .5], "gamma": [0., 0.], "theta": [0., 0.], "vega": [0., 0.]}), path)
    cfg = ResearchConfig(str(tmp_path / "currency=BTC"), str(tmp_path / "cache"), "BTC",
                         int(dt.datetime(2026, 5, 1, tzinfo=dt.timezone.utc).timestamp() * 1e9),
                         int(dt.datetime(2026, 5, 2, tzinfo=dt.timezone.utc).timestamp() * 1e9))
    store = MarketDataStore(cfg.data_root, cfg.cache_root)
    rows = next(iter(store.snapshots(cfg).values()))
    assert rows[0]["quote_ns"] < rows[0]["target_ns"]
    assert rows[0]["mark_price"] == .1
