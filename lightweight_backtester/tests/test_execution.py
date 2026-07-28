import pytest

from lightweight_backtester.execution import ExecutionModel
from lightweight_backtester.types import Side


def test_touch_plus_slippage():
    model = ExecutionModel(10)
    assert model.fill_price(Side.BUY, .09, .11, .10) == pytest.approx(.11011)
    assert model.fill_price(Side.SELL, .09, .11, .10) == pytest.approx(.08991)
