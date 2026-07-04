import numpy as np

from btcbt import InstKind, OptType
from btcbt.snapshot import Snapshot

_FIELDS = ("sym", "kind", "otype", "strike", "expiry", "forward", "mark", "bid", "ask", "iv",
           "bid_iv", "ask_iv", "oi", "volume", "delta", "gamma", "vega", "theta",
           "tte", "moneyness", "valid")


def _snap():
    # 2 calls, 2 puts; tte in years
    sym = np.array([0, 1, 2, 3], dtype=np.int32)
    kind = np.full(4, int(InstKind.OPTION), dtype=np.int8)
    otype = np.array([OptType.CALL, OptType.CALL, OptType.PUT, OptType.PUT], dtype=np.int8)
    strike = np.array([60000., 62000., 58000., 56000.])
    delta = np.array([0.55, 0.30, -0.40, -0.20])
    moneyness = np.array([1.0, 0.97, 1.03, 1.07])
    tte = np.array([5, 5, 12, 12]) / 365.0
    cols = dict(
        sym=sym, kind=kind, otype=otype, strike=strike, expiry=np.zeros(4, dtype=np.int64),
        forward=np.full(4, 60000.0), mark=np.full(4, 0.05), bid=np.full(4, 0.04), ask=np.full(4, 0.06),
        iv=np.full(4, 0.65), bid_iv=np.full(4, 0.64), ask_iv=np.full(4, 0.66),
        oi=np.full(4, 10.0), volume=np.full(4, 1.0),
        delta=delta, gamma=np.zeros(4), vega=np.zeros(4), theta=np.zeros(4),
        tte=tte, moneyness=moneyness, valid=np.ones(4, dtype=bool),
    )
    return Snapshot(0, 60000.0, cols)


def test_calls_puts():
    s = _snap()
    assert set(s.select().calls().syms().tolist()) == {0, 1}
    assert set(s.select().puts().syms().tolist()) == {2, 3}


def test_dte_filter():
    s = _snap()
    assert set(s.select().dte(0, 7).syms().tolist()) == {0, 1}
    assert set(s.select().dte(8, 20).syms().tolist()) == {2, 3}


def test_nearest_delta():
    s = _snap()
    # nearest call to 0.30 delta -> sym 1
    assert s.select().calls().nearest("delta", 0.30).first() == 1
    # nearest put to -0.20 -> sym 3
    assert s.select().puts().nearest("delta", -0.20).first() == 3


def test_chained_and_where():
    s = s = _snap()
    rows = s.select().options().calls().moneyness(0.95, 1.0).where(lambda sn, r: sn.delta[r] > 0.4)
    assert rows.syms().tolist() == [0]
