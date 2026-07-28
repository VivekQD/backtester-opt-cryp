import datetime as dt

import pytest

from lightweight_backtester.accounting import AccountingConfig, AccountingModel
from lightweight_backtester.types import (Contract, FillEvent, MarkEvent, OptionType,
                                          SettlementEvent, Side)


def ns(text):
    return int(dt.datetime.fromisoformat(text).timestamp() * 1e9)


def test_inverse_short_call_settlement_and_fees():
    c = Contract("BTC-8MAY26-100-C", "BTC", ns("2026-05-08T08:00:00+00:00"),
                 100, OptionType.CALL, "unused")
    a = AccountingModel(AccountingConfig(1.0))
    events = [
        FillEvent(ns("2026-05-01T08:05:00+00:00"), "t1", c, Side.SELL, 1, .10, 100, "entry"),
        MarkEvent(ns("2026-05-01T08:05:00+00:00"), 100, {c.instrument: .10}),
        SettlementEvent(c.expiry_ns, "t1", c, 1, (120 - 100) / 120, 120),
        MarkEvent(c.expiry_ns, 120, {c.instrument: (120 - 100) / 120}),
    ]
    equity, ledger = a.process(events)
    assert len(ledger) == 1
    assert ledger[0]["gross_pnl_native"] == pytest.approx(.10 - 1 / 6)
    assert ledger[0]["fees_native"] == pytest.approx(.00045)
    assert equity[-1].native_equity == pytest.approx(1 + ledger[0]["net_pnl_native"])


def test_margin_is_reported_not_enforced():
    c = Contract("BTC-8MAY26-100-P", "BTC", ns("2026-05-08T08:00:00+00:00"),
                 100, OptionType.PUT, "unused")
    a = AccountingModel(AccountingConfig(.01, enforce_margin=False))
    events = [FillEvent(ns("2026-05-01T08:05:00+00:00"), "t", c, Side.SELL, 1,
                        .05, 100, "entry"),
              MarkEvent(ns("2026-05-01T08:05:00+00:00"), 100, {c.instrument: .05})]
    equity, _ = a.process(events)
    assert equity[-1].initial_margin_native > .01
    assert a.positions  # entry was not rejected


def test_atomic_entry_is_rejected_when_initial_margin_exceeds_equity():
    c = Contract("BTC-8MAY26-100-C", "BTC", ns("2026-05-08T08:00:00+00:00"),
                 100, OptionType.CALL, "unused")
    a = AccountingModel(AccountingConfig(.01, enforce_margin=True))
    events = [FillEvent(ns("2026-05-01T08:05:00+00:00"), "t", c, Side.SELL, 1,
                        .05, 100, "entry"),
              MarkEvent(ns("2026-05-01T08:05:00+00:00"), 100, {c.instrument: .05})]
    a.process(events)
    assert not a.positions
    assert a.rejections


def test_mixed_direction_combo_discount_and_liquidation():
    ts = ns("2026-05-01T08:05:00+00:00")
    call = Contract("BTC-8MAY26-100-C", "BTC", ns("2026-05-08T08:00:00+00:00"),
                    100, OptionType.CALL, "unused")
    put = Contract("BTC-8MAY26-90-P", "BTC", call.expiry_ns, 90, OptionType.PUT, "unused")
    combo = AccountingModel(AccountingConfig(1.0, enforce_margin=False,
                                              combo_fee_discount=True))
    combo.process([
        FillEvent(ts, "combo", call, Side.SELL, 1, .05, 100, "entry"),
        FillEvent(ts, "combo", put, Side.BUY, 1, .05, 100, "entry"),
        MarkEvent(ts, 100, {call.instrument: .05, put.instrument: .05}),
    ])
    assert combo.positions[("combo", put.instrument)].opening_fees == 0
    assert combo.positions[("combo", call.instrument)].opening_fees == pytest.approx(.0003)

    account = AccountingModel(AccountingConfig(.30, enforce_margin=True))
    account.process([
        FillEvent(ts, "short", call, Side.SELL, 1, .05, 100, "entry"),
        MarkEvent(ts, 100, {call.instrument: .05}, {call.instrument: .04}, {call.instrument: .06}),
        MarkEvent(ts + 300_000_000_000, 100, {call.instrument: .40},
                  {call.instrument: .39}, {call.instrument: .41}),
    ])
    assert account.liquidations == 1
    assert not account.positions
    assert account.leg_ledger[-1]["exit_reason"] == "liquidation"
