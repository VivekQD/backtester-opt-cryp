import math

import numpy as np

from btcbt import Margin, Portfolio, VenueConfig
from btcbt.snapshot import Snapshot


def test_linear_equity_with_nan_index_keeps_usd_finite():
    cols = dict(
        sym=np.zeros(0, dtype=np.int32),
        kind=np.zeros(0, dtype=np.int8),
        otype=np.zeros(0, dtype=np.int8),
        strike=np.zeros(0),
        expiry=np.zeros(0, dtype=np.int64),
        forward=np.zeros(0),
        mark=np.zeros(0),
        bid=np.zeros(0),
        ask=np.zeros(0),
        iv=np.zeros(0),
        bid_iv=np.zeros(0),
        ask_iv=np.zeros(0),
        oi=np.zeros(0),
        volume=np.zeros(0),
        delta=np.zeros(0),
        gamma=np.zeros(0),
        vega=np.zeros(0),
        theta=np.zeros(0),
        tte=np.zeros(0),
        moneyness=np.zeros(0),
        valid=np.zeros(0, dtype=bool),
    )
    snap = Snapshot(0, np.nan, cols)
    equity = Portfolio(100.0, VenueConfig(margin=Margin.LINEAR), {}).equity(snap)

    assert equity.usd == 100.0
    assert equity.btc == 0.0
    assert math.isfinite(equity.usd)
    assert math.isfinite(equity.btc)
