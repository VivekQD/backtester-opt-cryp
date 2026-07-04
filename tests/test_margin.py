from types import SimpleNamespace

import numpy as np

from btcbt import DeribitStandardMargin, InstKind, Margin, OptType, Position, VenueConfig
from btcbt.cli import build_margin
from btcbt.margin import SimpleMargin
from btcbt.snapshot import Snapshot


def _snap(index, otypes, strikes, marks, forwards=None):
    n = len(otypes)
    if forwards is None:
        forwards = [index] * n
    cols = dict(
        sym=np.arange(n, dtype=np.int32),
        kind=np.full(n, int(InstKind.OPTION), dtype=np.int8),
        otype=np.array([int(o) for o in otypes], dtype=np.int8),
        strike=np.array(strikes, dtype=np.float64),
        expiry=np.zeros(n, dtype=np.int64),
        forward=np.array(forwards, dtype=np.float64),
        mark=np.array(marks, dtype=np.float64),
        bid=np.array(marks, dtype=np.float64),
        ask=np.array(marks, dtype=np.float64),
        iv=np.zeros(n),
        bid_iv=np.zeros(n),
        ask_iv=np.zeros(n),
        oi=np.ones(n),
        volume=np.ones(n),
        delta=np.zeros(n),
        gamma=np.zeros(n),
        vega=np.zeros(n),
        theta=np.zeros(n),
        tte=np.ones(n),
        moneyness=np.ones(n),
        valid=np.ones(n, dtype=bool),
    )
    return Snapshot(0, float(index), cols)


def _portfolio(qtys):
    return SimpleNamespace(positions={sym: Position(sym, qty) for sym, qty in qtys.items()})


def test_deribit_linear_short_option_margin():
    snap = _snap(
        100000,
        [OptType.CALL, OptType.PUT],
        [100000, 100000],
        [5000, 4000],
        forwards=[95000, 120000],
    )
    portfolio = _portfolio({0: -1, 1: -1})
    venue = VenueConfig(margin=Margin.LINEAR)
    model = DeribitStandardMargin()

    assert abs(model.initial(portfolio, snap, venue) - 29000.0) < 1e-12
    assert abs(model.maintenance(portfolio, snap, venue) - 24000.0) < 1e-12


def test_deribit_inverse_short_option_margin():
    snap = _snap(
        100000,
        [OptType.CALL, OptType.PUT],
        [120000, 90000],
        [0.02, 3.0],
    )
    portfolio = _portfolio({0: -1, 1: -1})
    venue = VenueConfig(margin=Margin.INVERSE)
    model = DeribitStandardMargin()

    assert abs(model.initial(portfolio, snap, venue) - 3.345) < 1e-12
    assert abs(model.maintenance(portfolio, snap, venue) - 3.32) < 1e-12


def test_long_options_have_no_extra_standard_margin():
    snap = _snap(100000, [OptType.CALL, OptType.PUT], [100000, 100000], [5000, 4000])
    portfolio = _portfolio({0: 1, 1: 1})
    venue = VenueConfig(margin=Margin.LINEAR)
    model = DeribitStandardMargin()

    assert model.initial(portfolio, snap, venue) == 0.0
    assert model.maintenance(portfolio, snap, venue) == 0.0


def test_config_uses_deribit_standard_margin_by_default():
    assert isinstance(build_margin({"margin": {"enabled": True}}), DeribitStandardMargin)
    assert isinstance(build_margin({"margin": {"enabled": True, "model": "simple"}}), SimpleMargin)
