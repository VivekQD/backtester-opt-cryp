from btcbt import DataConfig, Dataset, DefaultFill, Engine, Margin, Side, Strategy, VenueConfig, market


class BuyOnce(Strategy):
    def on_snapshot(self, snap, ctx):
        if ctx.portfolio().is_flat():
            ctx.submit(market(int(snap.sym[0]), Side.BUY, 1))


def _engine(path):
    ds = Dataset.load(DataConfig(path))
    venue = VenueConfig(margin=Margin.LINEAR)
    return ds, Engine(ds, BuyOnce(), DefaultFill(slippage_bps=0, half_spread_bps=0), venue, cash=100000.0)


def test_order_fills_next_bar(tiny_parquet):
    ds, eng = _engine(tiny_parquet)
    res = eng.run()
    ts = ds.timestamps()
    assert len(res.fills) == 1                  # buys once, then no longer flat
    assert res.fills[0].ts == int(ts[1])        # submitted on bar0, fills on bar1
    assert res.fills[0].ts != int(ts[0])        # never same-bar


def test_equity_recorded_each_bar(tiny_parquet):
    ds, eng = _engine(tiny_parquet)
    res = eng.run()
    assert len(res.equity) == len(ds.timestamps())
    # bought at ask on bar1 (4900? no: ask=5050+50*1=5100) then mark rises -> position has value
    assert res.equity[-1][1].usd != res.equity[0][1].usd
