from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

from .types import (EquityPoint, Event, FillEvent, MarkEvent, OptionType, Position,
                    SettlementEvent, Side)


@dataclass
class AccountingConfig:
    initial_cash_native: float = 1.0
    trade_fee_rate: float = 0.0003
    settlement_fee_rate: float = 0.00015
    fee_cap_rate: float = 0.125
    initial_margin_rate: float = 0.15
    initial_margin_floor: float = 0.10
    maintenance_margin_rate: float = 0.075
    enforce_margin: bool = True
    combo_fee_discount: bool = True
    liquidation_fee_native: float = 0.0019
    liquidation_fee_cap_rate: float = 0.25
    liquidation_slippage_bps: float = 1.0


class AccountingModel:
    """Pure event reducer. Margin is calculated and reported, never enforced."""

    def __init__(self, cfg: AccountingConfig):
        self.cfg = cfg
        self.cash = cfg.initial_cash_native
        self.positions: dict[tuple[str, str], Position] = {}
        self.marks: dict[str, float] = {}
        self.spot = 0.0
        self.equity: list[EquityPoint] = []
        self.leg_ledger: list[dict] = []
        self.rejections: list[dict] = []
        self.liquidations = 0
        self.peak_initial_margin = 0.0
        self.peak_maintenance_margin = 0.0
        self.intraday_peak = cfg.initial_cash_native
        self.intraday_max_drawdown = 0.0

    def trade_fee(self, price: float, quantity: float) -> float:
        return abs(quantity) * min(self.cfg.trade_fee_rate,
                                   self.cfg.fee_cap_rate * max(price, 0.0))

    def settlement_fee(self, event: SettlementEvent) -> float:
        weekday = dt.datetime.fromtimestamp(event.ts_ns / 1e9, dt.timezone.utc).weekday()
        rate = self.cfg.settlement_fee_rate if weekday == 4 else 0.0
        return abs(event.quantity) * min(rate, self.cfg.fee_cap_rate * event.payoff)

    def _margin(self, maintenance: bool = False) -> float:
        total = 0.0
        for pos in self.positions.values():
            if pos.quantity >= 0:
                continue
            mark = self.marks.get(pos.contract.instrument, pos.average_price)
            strike, spot = pos.contract.strike, self.spot
            if spot <= 0:
                continue
            if maintenance:
                base = self.cfg.maintenance_margin_rate
                req = base + mark if pos.contract.option_type == OptionType.CALL else max(base, base * mark) + mark
            else:
                otm = ((strike - spot) if pos.contract.option_type == OptionType.CALL
                       else (spot - strike))
                req = max(self.cfg.initial_margin_rate - max(otm, 0) / spot,
                          self.cfg.initial_margin_floor) + mark
                if pos.contract.option_type == OptionType.PUT:
                    req = max(req, max(self.cfg.maintenance_margin_rate,
                                       self.cfg.maintenance_margin_rate * mark) + mark)
            total += abs(pos.quantity) * req
        return total

    def _native_equity(self) -> float:
        return self.cash + sum(pos.quantity * (self.marks.get(pos.contract.instrument,
                                                               pos.average_price) - pos.average_price)
                               for pos in self.positions.values())

    def _observe_risk(self):
        native = self._native_equity()
        initial, maintenance = self._margin(False), self._margin(True)
        self.peak_initial_margin = max(self.peak_initial_margin, initial)
        self.peak_maintenance_margin = max(self.peak_maintenance_margin, maintenance)
        self.intraday_peak = max(self.intraday_peak, native)
        if self.intraday_peak > 0:
            self.intraday_max_drawdown = min(self.intraday_max_drawdown,
                                             (native - self.intraday_peak) / self.intraday_peak)
        return native, initial, maintenance

    def _record_equity(self, ts_ns: int):
        native = self._native_equity()
        option_pnl = native - self.cfg.initial_cash_native
        self.equity.append(EquityPoint(ts_ns, native, native * self.spot,
                                      option_pnl, self._margin(False), self._margin(True)))

    def _fill(self, event: FillEvent, fee_override: float | None = None):
        key = (event.trade_id, event.contract.instrument)
        signed = event.quantity if event.side == Side.BUY else -event.quantity
        fee = self.trade_fee(event.price, event.quantity) if fee_override is None else fee_override
        self.cash -= signed * event.price + fee
        pos = self.positions.get(key)
        if pos is None:
            if event.reason != "entry":
                self.cash += signed * event.price + fee
                return
            self.positions[key] = Position(event.contract, signed, event.price,
                                           event.ts_ns, fee)
            return
        if pos.quantity * signed < 0:
            closed = min(abs(pos.quantity), abs(signed))
            direction = 1 if pos.quantity > 0 else -1
            gross = closed * (event.price - pos.average_price) * direction
            open_fee = pos.opening_fees * closed / abs(pos.quantity)
            self.leg_ledger.append({"trade_id": event.trade_id,
                "instrument": event.contract.instrument, "entry_ts_ns": pos.entry_ts_ns,
                "exit_ts_ns": event.ts_ns, "side": "long" if direction > 0 else "short",
                "asset": event.contract.asset, "option_type": event.contract.option_type.value,
                "strike": event.contract.strike,
                "quantity": closed, "entry_price": pos.average_price, "exit_price": event.price,
                "gross_pnl_native": gross, "fees_native": open_fee + fee,
                "net_pnl_native": gross - open_fee - fee, "exit_reason": event.reason})
            pos.quantity += signed
            pos.opening_fees -= open_fee
            if abs(pos.quantity) < 1e-12:
                del self.positions[key]
        else:
            new_qty = pos.quantity + signed
            pos.average_price = ((pos.quantity * pos.average_price + signed * event.price) / new_qty)
            pos.quantity = new_qty
            pos.opening_fees += fee

    def _entry_batch(self, events: list[FillEvent]):
        raw_fees = [self.trade_fee(x.price, x.quantity) for x in events]
        fees = raw_fees[:]
        if self.cfg.combo_fee_discount:
            buy_total = sum(f for f, e in zip(raw_fees, events) if e.side == Side.BUY)
            sell_total = sum(f for f, e in zip(raw_fees, events) if e.side == Side.SELL)
            if buy_total and sell_total:
                free_side = Side.BUY if buy_total <= sell_total else Side.SELL
                fees = [0.0 if e.side == free_side else f for f, e in zip(raw_fees, events)]
        # Atomic preflight on a temporary copy of account state.
        old_cash = self.cash
        old_positions = {k: Position(v.contract, v.quantity, v.average_price,
                                     v.entry_ts_ns, v.opening_fees)
                         for k, v in self.positions.items()}
        for event, fee in zip(events, fees):
            self._fill(event, fee)
            self.marks[event.contract.instrument] = event.price
        native, initial, _ = self._observe_risk()
        if self.cfg.enforce_margin and native < initial:
            self.cash, self.positions = old_cash, old_positions
            for event in events:
                self.rejections.append({"trade_id": event.trade_id, "ts_ns": event.ts_ns,
                                        "reason": "insufficient_initial_margin",
                                        "required_native": initial, "equity_native": native})

    def _liquidate(self, event: MarkEvent):
        for key, pos in list(self.positions.items()):
            instrument = pos.contract.instrument
            if pos.quantity < 0:
                side, touch = Side.BUY, event.asks.get(instrument, self.marks.get(instrument, 0))
                price = touch * (1 + self.cfg.liquidation_slippage_bps * 1e-4)
            else:
                side, touch = Side.SELL, event.bids.get(instrument, self.marks.get(instrument, 0))
                price = touch * (1 - self.cfg.liquidation_slippage_bps * 1e-4)
            if not price or price <= 0:
                continue
            quantity = abs(pos.quantity)
            fee = quantity * min(self.cfg.liquidation_fee_native,
                                 self.cfg.liquidation_fee_cap_rate * price)
            signed = quantity if side == Side.BUY else -quantity
            self.cash -= signed * price + fee
            direction = 1 if pos.quantity > 0 else -1
            gross = quantity * (price - pos.average_price) * direction
            self.leg_ledger.append({"trade_id": key[0], "instrument": instrument,
                "entry_ts_ns": pos.entry_ts_ns, "exit_ts_ns": event.ts_ns,
                "side": "long" if direction > 0 else "short", "asset": pos.contract.asset,
                "option_type": pos.contract.option_type.value, "strike": pos.contract.strike,
                "quantity": quantity, "entry_price": pos.average_price, "exit_price": price,
                "gross_pnl_native": gross, "fees_native": pos.opening_fees + fee,
                "net_pnl_native": gross - pos.opening_fees - fee, "exit_reason": "liquidation"})
            del self.positions[key]
            self.liquidations += 1

    def _settle(self, event: SettlementEvent):
        key = (event.trade_id, event.contract.instrument)
        pos = self.positions.get(key)
        if pos is None:
            return
        fee = self.settlement_fee(event)
        self.cash += pos.quantity * event.payoff - fee
        direction = 1 if pos.quantity > 0 else -1
        gross = abs(pos.quantity) * (event.payoff - pos.average_price) * direction
        self.leg_ledger.append({"trade_id": event.trade_id,
            "instrument": event.contract.instrument, "entry_ts_ns": pos.entry_ts_ns,
            "exit_ts_ns": event.ts_ns, "side": "long" if direction > 0 else "short",
            "asset": event.contract.asset, "option_type": event.contract.option_type.value,
            "strike": event.contract.strike,
            "quantity": abs(pos.quantity), "entry_price": pos.average_price,
            "exit_price": event.payoff, "gross_pnl_native": gross,
            "fees_native": pos.opening_fees + fee,
            "net_pnl_native": gross - pos.opening_fees - fee, "exit_reason": "expiry"})
        del self.positions[key]

    def process(self, events: list[Event]) -> tuple[list[EquityPoint], list[dict]]:
        priority = {SettlementEvent: 0, FillEvent: 1, MarkEvent: 2}
        ordered = sorted(events, key=lambda x: (x.ts_ns, priority[type(x)]))
        i = 0
        while i < len(ordered):
            ts = ordered[i].ts_ns
            batch = []
            while i < len(ordered) and ordered[i].ts_ns == ts:
                batch.append(ordered[i]); i += 1
            for event in batch:
                self.spot = float(event.spot)
                if isinstance(event, SettlementEvent):
                    self._settle(event)
            entries = [x for x in batch if isinstance(x, FillEvent) and x.reason == "entry"]
            for trade_id in dict.fromkeys(x.trade_id for x in entries):
                self._entry_batch([x for x in entries if x.trade_id == trade_id])
            for event in batch:
                if isinstance(event, FillEvent) and event.reason != "entry":
                    self._fill(event)
            for event in batch:
                if isinstance(event, MarkEvent):
                    self.spot = float(event.spot)
                    self.marks.update(event.marks)
                    native, _, maintenance = self._observe_risk()
                    if self.cfg.enforce_margin and self.positions and native < maintenance:
                        self._liquidate(event)
                        self._observe_risk()
                    if event.record_equity:
                        self._record_equity(event.ts_ns)
        # A contract can settle after its source file's last 5-minute quote.
        # Record the post-settlement state even when no later MarkEvent exists.
        if ordered and (not self.equity or self.equity[-1].ts_ns < ordered[-1].ts_ns):
            self._record_equity(ordered[-1].ts_ns)
        return self.equity, self.leg_ledger
