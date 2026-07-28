import datetime as dt

import numpy as np

from lightweight_backtester.accounting import AccountingConfig, AccountingModel
from lightweight_backtester.engine import BacktestEngine
from lightweight_backtester.execution import ExecutionModel
from lightweight_backtester.types import Contract, FillEvent, LegIntent, OptionType, Side, TradeIntent


def test_stop_triggers_on_mark_and_fills_next_bar_at_ask():
    base = int(dt.datetime(2026, 5, 1, 8, tzinfo=dt.timezone.utc).timestamp() * 1e9)
    step = 5 * 60 * 1_000_000_000
    contract = Contract("BTC-8MAY26-100-C", "BTC", base + 7 * 86_400_000_000_000,
                        100, OptionType.CALL, "fake")
    path = {"date": np.array([base + step, base + 2 * step, base + 3 * step]),
            "bid_price": np.array([.09, .13, .14]), "ask_price": np.array([.11, .15, .16]),
            "mark_price": np.array([.10, .14, .15]), "index_price": np.array([100., 100., 100.])}

    class Store:
        def load_contract_path(self, *args): return path

    engine = BacktestEngine(Store(), ExecutionModel(0), AccountingModel(AccountingConfig()))
    intent = TradeIntent("t", base, (LegIntent(contract, Side.SELL, 1, .30),))
    fills = [e for e in engine._simulate_intent(intent) if isinstance(e, FillEvent)]
    assert fills[0].ts_ns == base + step
    assert fills[0].price == .09
    assert fills[1].ts_ns == base + 3 * step
    assert fills[1].price == .16
    assert fills[1].reason == "stop_loss"


def test_theoretical_stop_fills_exact_threshold_on_trigger_bar():
    base = int(dt.datetime(2026, 5, 1, 8, tzinfo=dt.timezone.utc).timestamp() * 1e9)
    step = 5 * 60 * 1_000_000_000
    contract = Contract("BTC-8MAY26-100-C", "BTC", base + 7 * 86_400_000_000_000,
                        100, OptionType.CALL, "fake")
    path = {"date": np.array([base + step, base + 2 * step, base + 3 * step]),
            "bid_price": np.array([.09, .13, .14]), "ask_price": np.array([.11, .15, .16]),
            "mark_price": np.array([.10, .14, .15]), "index_price": np.array([100., 100., 100.])}
    class Store:
        def load_contract_path(self, *args): return path
    engine = BacktestEngine(Store(), ExecutionModel(0), AccountingModel(AccountingConfig()))
    engine.stop_fill_mode = "theoretical"
    intent = TradeIntent("t", base, (LegIntent(contract, Side.SELL, 1, .30),))
    fills = [e for e in engine._simulate_intent(intent) if isinstance(e, FillEvent)]
    assert fills[1].ts_ns == base + 2 * step
    assert fills[1].price == .09 * 1.30
    assert fills[1].reason == "stop_loss_theoretical"
