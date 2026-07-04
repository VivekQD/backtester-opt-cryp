import math

import numpy as np

from btcbt import DefaultFill, InstKind, Margin, OptType, Side, VenueConfig, market, triggered
from btcbt.snapshot import Snapshot


def _snap(bid, ask, mark, index=100000.0):
    cols = dict(
        sym=np.array([0], dtype=np.int32),
        kind=np.array([int(InstKind.OPTION)], dtype=np.int8),
        otype=np.array([int(OptType.CALL)], dtype=np.int8),
        strike=np.array([100000.0]),
        expiry=np.zeros(1, dtype=np.int64),
        forward=np.array([index]),
        mark=np.array([mark]),
        bid=np.array([bid]),
        ask=np.array([ask]),
        iv=np.zeros(1),
        bid_iv=np.zeros(1),
        ask_iv=np.zeros(1),
        oi=np.ones(1),
        volume=np.ones(1),
        delta=np.zeros(1),
        gamma=np.zeros(1),
        vega=np.zeros(1),
        theta=np.zeros(1),
        tte=np.ones(1),
        moneyness=np.ones(1),
        valid=np.ones(1, dtype=bool),
    )
    return Snapshot(0, index, cols)


def test_market_fill_ignores_nan_touch_and_uses_mark():
    fill = DefaultFill(half_spread_bps=0, slippage_bps=0).fill(
        market(0, Side.SELL, 1),
        _snap(np.nan, 1100.0, 1000.0),
        VenueConfig(margin=Margin.LINEAR),
    )

    assert fill is not None
    assert fill.price == 1000.0
    assert math.isfinite(fill.fee)


def test_market_fill_rejects_non_finite_prices():
    fill = DefaultFill(half_spread_bps=0, slippage_bps=0).fill(
        market(0, Side.SELL, 1),
        _snap(np.nan, 1100.0, np.nan),
        VenueConfig(margin=Margin.LINEAR),
    )

    assert fill is None


def test_triggered_fill_uses_threshold_price():
    fill = DefaultFill(half_spread_bps=0, slippage_bps=0).fill(
        triggered(0, Side.BUY, 1, 114.0),
        _snap(95.0, 355.0, 171.0),
        VenueConfig(margin=Margin.LINEAR),
    )

    assert fill is not None
    assert fill.price == 114.0
