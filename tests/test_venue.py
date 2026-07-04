import datetime as dt

import numpy as np

from btcbt import Margin, OptType, VenueConfig, payoff, settle_fee, settlement_index, trade_fee
from btcbt.types import NS


def test_inverse_payoff():
    # call ITM: (S-K)/S
    assert payoff(Margin.INVERSE, OptType.CALL, 50000, 60000) == (10000 / 60000)
    # put ITM: (K-S)/S
    assert payoff(Margin.INVERSE, OptType.PUT, 60000, 50000) == (10000 / 50000)
    # OTM -> 0
    assert payoff(Margin.INVERSE, OptType.CALL, 60000, 50000) == 0.0


def test_linear_payoff():
    assert payoff(Margin.LINEAR, OptType.CALL, 50000, 60000) == 10000
    assert payoff(Margin.LINEAR, OptType.PUT, 60000, 50000) == 10000
    assert payoff(Margin.LINEAR, OptType.PUT, 50000, 60000) == 0.0


def test_fee_cap():
    cfg = VenueConfig()
    # inverse options: MIN(0.0003 BTC, 12.5% * option price) * amount
    assert trade_fee(cfg, 0.001, index_price=60000, qty=1) == 0.000125
    assert trade_fee(cfg, 1.0, index_price=60000, qty=3) == 0.0009

    lin = VenueConfig(margin=Margin.LINEAR)
    # linear options: MIN(0.0003 * index, 12.5% * option price) * contracts * size
    assert abs(trade_fee(lin, 2000, index_price=100000, qty=0.5) - 15.0) < 1e-12
    assert trade_fee(lin, 5, index_price=200, qty=1) == 0.06

    # non-Friday expiries are daily options and have no Deribit delivery fee
    monday_expiry = int(dt.datetime(2026, 6, 29, 8, tzinfo=dt.timezone.utc).timestamp()) * NS
    assert settle_fee(lin, 1000, index_price=100000, qty=1, expiry_ts=monday_expiry) == 0.0

    friday_expiry = int(dt.datetime(2026, 6, 26, 8, tzinfo=dt.timezone.utc).timestamp()) * NS
    assert abs(settle_fee(lin, 1000, index_price=100000, qty=2, expiry_ts=friday_expiry) - 30.0) < 1e-12


def test_settlement_twap():
    # index ramps; TWAP over last 30 min ending at expiry
    expiry = 1000 * NS
    ts = np.array([expiry - i * 60 * NS for i in range(60, -1, -1)], dtype=np.float64)
    idx = np.linspace(100.0, 160.0, ts.size)
    series = np.column_stack([ts, idx])
    sx = settlement_index(series, int(expiry), 30)
    # last 30 min span index 130..160 -> mean 145
    assert abs(sx - 145.0) < 1.0


def test_settlement_twap_ignores_nan_index_values():
    expiry = 1000 * NS
    ts = np.array([expiry - 31 * 60 * NS, expiry - 30 * NS, expiry], dtype=np.float64)
    idx = np.array([100.0, np.nan, 130.0])
    series = np.column_stack([ts, idx])

    assert settlement_index(series, int(expiry), 30) == 130.0
