import datetime as dt

import pyarrow as pa
import pyarrow.parquet as pq

from btcbt import (ChunkedDataset, DataConfig, Dataset, DefaultFill, Engine, Margin, Side,
                   Strategy, VenueConfig, market)


def _table(start, n, names):
    rows = []
    for i in range(n):
        d = start + dt.timedelta(minutes=5 * i)
        for name in names:
            rows.append(dict(
                date=d, instrument_name=name, option_type="call", strike=float(name.split("-")[2]),
                index_price=60000.0 + 10 * i, underlying_price=60010.0 + 10 * i,
                mark_price=0.05 + 0.001 * i, bid_price=0.049 + 0.001 * i,
                ask_price=0.051 + 0.001 * i, mark_iv=60.0, bid_iv=59.0, ask_iv=61.0,
                open_interest=10.0, volume=1.0, delta=0.6, gamma=0.0, theta=0.0, vega=0.0,
                currency="BTC",
            ))
    cols = {k: [r[k] for r in rows] for k in rows[0]}
    cols["date"] = pa.array(cols["date"], type=pa.timestamp("us", tz="UTC"))
    return pa.table(cols)


class BuyOnce(Strategy):
    def on_snapshot(self, snap, ctx):
        if ctx.portfolio().is_flat() and not ctx.state().get("done"):
            ctx.state()["done"] = True
            ctx.submit(market(int(snap.sym[0]), Side.BUY, 1))


def _run(ds):
    eng = Engine(ds, BuyOnce(), DefaultFill(slippage_bps=0, half_spread_bps=0),
                 VenueConfig(margin=Margin.INVERSE), cash=1.0)
    return eng.run()


def test_chunked_matches_single_file_across_boundary(tmp_path):
    names = ["BTC-26FEB21-50000-C", "BTC-26FEB21-51000-C"]
    a = _table(dt.datetime(2021, 1, 31, 23, 30, tzinfo=dt.timezone.utc), 6, names)
    b = _table(dt.datetime(2021, 2, 1, 0, 0, tzinfo=dt.timezone.utc), 6, names)
    pa_, pb, pall = tmp_path / "a.parquet", tmp_path / "b.parquet", tmp_path / "all.parquet"
    pq.write_table(a, pa_)
    pq.write_table(b, pb)
    pq.write_table(pa.concat_tables([a, b]), pall)

    single = _run(Dataset.load(DataConfig(str(pall))))
    cds = Dataset.load(DataConfig([str(pa_), str(pb)], cache_dir=str(tmp_path / "parts")))
    assert isinstance(cds, ChunkedDataset)
    chunked = _run(cds)

    assert [t for t, _ in chunked.equity] == [t for t, _ in single.equity]
    assert [e.btc for _, e in chunked.equity] == [e.btc for _, e in single.equity]
    assert len(cds.chunks()) == 2                      # Jan + Feb partitions
    assert cds.index_series.shape[0] == len(single.equity)


def test_chunked_filter_and_duplicate_files(tmp_path):
    names = ["BTC-26FEB21-50000-C", "BTC-26FEB21-90000-C"]
    t = _table(dt.datetime(2021, 2, 1, tzinfo=dt.timezone.utc), 4, names)
    p1, p2 = tmp_path / "x.parquet", tmp_path / "y.parquet"
    pq.write_table(t, p1)
    pq.write_table(t, p2)                               # same bars twice
    cfg = DataConfig([str(p1), str(p2)], filter={"strike_index_band": [0.5, 1.2]},
                     cache_dir=str(tmp_path / "parts"))
    bars = list(Dataset.load(cfg).iter_bars())
    assert len(bars) == 4                               # duplicates not replayed
    assert all(len(s) == 2 for s in bars)               # 90k strike filtered (2 copies of 50k)
